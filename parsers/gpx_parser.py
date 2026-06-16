"""GPX file parser: extracts track points, computes metrics, detects laps."""

import io
import xml.etree.ElementTree as ET
from typing import Optional

import gpxpy
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ── Haversine distance ────────────────────────────────────────────────────────

def _haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── GPX namespace helpers ─────────────────────────────────────────────────────

_HR_TAGS = [
    "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr",
    "{http://www.garmin.com/xmlschemas/TrackPointExtension/v2}hr",
    "ns3:hr",
    "gpxtpx:hr",
    "hr",
]

_CADENCE_TAGS = [
    "{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}cad",
    "{http://www.garmin.com/xmlschemas/TrackPointExtension/v2}cad",
    "cad",
]


def _get_extension_value(point, tags):
    """Extract a numeric value from GPX extensions by trying multiple tag names."""
    if not hasattr(point, "extensions") or not point.extensions:
        return None
    raw_xml = getattr(point, "_extensions_xml", None)
    if raw_xml is None:
        return None
    try:
        root = ET.fromstring(f"<root>{raw_xml}</root>")
        for tag in tags:
            els = root.findall(f".//{tag}")
            if els:
                try:
                    return float(els[0].text)
                except (TypeError, ValueError):
                    pass
    except ET.ParseError:
        pass
    return None


def _extract_hr_from_gpx_point(point):
    """Pull HR from gpxpy extension data."""
    for ext in point.extensions:
        # gpxpy exposes extensions as lxml or ElementTree elements
        try:
            tag = ext.tag.split("}")[-1] if "}" in ext.tag else ext.tag
            if tag.lower() in ("hr", "heartrate"):
                return float(ext.text)
            for child in ext:
                ctag = child.tag.split("}")[-1].lower()
                if ctag in ("hr", "heartrate"):
                    return float(child.text)
        except (TypeError, ValueError, AttributeError):
            pass
    return None


def _extract_cadence_from_gpx_point(point):
    for ext in point.extensions:
        try:
            tag = ext.tag.split("}")[-1] if "}" in ext.tag else ext.tag
            if tag.lower() in ("cad", "cadence"):
                return float(ext.text)
            for child in ext:
                ctag = child.tag.split("}")[-1].lower()
                if ctag in ("cad", "cadence"):
                    return float(child.text)
        except (TypeError, ValueError, AttributeError):
            pass
    return None


# ── Track point extraction ────────────────────────────────────────────────────

def _extract_track_points(gpx) -> pd.DataFrame:
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                hr = _extract_hr_from_gpx_point(pt)
                cad = _extract_cadence_from_gpx_point(pt)
                rows.append({
                    "timestamp": pt.time,
                    "lat": pt.latitude,
                    "lon": pt.longitude,
                    "elevation_m": pt.elevation,
                    "hr": hr,
                    "cadence": cad,
                })

    if not rows:
        raise ValueError("GPX file contains no track points.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Distance computation ──────────────────────────────────────────────────────

def _compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    lat = df["lat"].values
    lon = df["lon"].values
    dist = np.zeros(len(df))
    if len(df) > 1:
        step = _haversine_vectorized(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dist[1:] = step
    df["step_m"] = dist
    df["distance_m"] = dist.cumsum()
    return df


# ── Spike / pause detection ───────────────────────────────────────────────────

def _detect_gps_spikes(df: pd.DataFrame) -> pd.DataFrame:
    time_delta_s = df["timestamp"].diff().dt.total_seconds().fillna(1.0)
    time_delta_s = time_delta_s.replace(0, 0.001)
    velocity = df["step_m"] / time_delta_s

    spike = (velocity > 50) | (df["step_m"] > 200)
    # Also mark the recovery point after a spike
    spike = spike | spike.shift(1, fill_value=False)

    df["is_spike"] = spike

    # Interpolate lat/lon/elevation across spikes
    for col in ["lat", "lon", "elevation_m"]:
        if col in df.columns:
            df.loc[spike, col] = np.nan
            df[col] = df[col].interpolate(method="linear")

    # Recompute step_m and distance_m after interpolation
    df = _compute_distances(df.drop(columns=["step_m", "distance_m"]))
    return df


def _detect_pauses(df: pd.DataFrame) -> pd.DataFrame:
    time_delta_s = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
    step = df["step_m"]
    time_delta_safe = time_delta_s.replace(0, 0.001)
    velocity = step / time_delta_safe

    is_moving = (velocity >= 0.3) & (time_delta_s < 30)
    df["is_moving"] = is_moving
    df["time_delta_s"] = time_delta_s
    return df


# ── Pace computation ──────────────────────────────────────────────────────────

def _compute_instantaneous_pace(df: pd.DataFrame) -> pd.DataFrame:
    time_delta_s = df["time_delta_s"]
    step_km = df["step_m"] / 1000.0

    with np.errstate(divide="ignore", invalid="ignore"):
        pace = np.where(
            (step_km > 0) & df["is_moving"] & ~df["is_spike"],
            (time_delta_s / 60.0) / step_km,
            np.nan,
        )

    pace = np.where(np.isfinite(pace), np.clip(pace, 0, 30), np.nan)
    df["pace_min_km"] = pace
    return df


def _smooth_pace(df: pd.DataFrame, window_s: int = 30) -> pd.DataFrame:
    ts_indexed = df.set_index("timestamp")
    rolled = ts_indexed["pace_min_km"].rolling(f"{window_s}s", min_periods=1).mean()

    valid = ~np.isnan(rolled.values)
    smoothed = rolled.values.copy()
    if valid.sum() > 5:
        filled = pd.Series(smoothed).interpolate(method="linear").values
        smoothed = gaussian_filter1d(filled, sigma=3)
        smoothed = np.where(valid | (np.arange(len(smoothed)) < valid.argmax()), smoothed, np.nan)

    df["smoothed_pace_min_km"] = smoothed
    return df


# ── Elevation ─────────────────────────────────────────────────────────────────

def _compute_elevation_gain_loss(elevation: pd.Series) -> tuple[float, float]:
    smoothed = elevation.rolling(5, min_periods=1, center=True).mean()
    diff = smoothed.diff().dropna()
    gain = float(diff[diff > 1.0].sum())
    loss = float(abs(diff[diff < -1.0].sum()))
    return gain, loss


# ── Riegel race projections ───────────────────────────────────────────────────

def _riegel_projection(distance_m: float, time_s: float, target_m: float) -> float:
    if distance_m <= 0 or time_s <= 0:
        return 0.0
    return time_s * (target_m / distance_m) ** 1.06


# ── Lap extraction ────────────────────────────────────────────────────────────

def _extract_gpx_laps(gpx) -> Optional[pd.DataFrame]:
    """Try to extract Garmin-style lap metadata from GPX extensions."""
    lap_rows = []

    for track in gpx.tracks:
        for i, segment in enumerate(track.segments):
            if len(track.segments) > 1:
                pts = segment.points
                if not pts:
                    continue
                start = pd.to_datetime(pts[0].time, utc=True)
                end = pd.to_datetime(pts[-1].time, utc=True)
                duration_s = (end - start).total_seconds()
                lap_rows.append({
                    "lap_num": i + 1,
                    "start_time": start,
                    "end_time": end,
                    "duration_s": duration_s,
                    "distance_m": None,
                    "avg_pace_min_km": None,
                    "avg_hr": None,
                    "max_hr": None,
                    "elevation_gain": None,
                })

    if lap_rows:
        return pd.DataFrame(lap_rows)
    return None


def _compute_laps_from_distance(df: pd.DataFrame, lap_distance_m: float = 1000.0) -> pd.DataFrame:
    """Fallback: generate per-km laps from track points."""
    max_dist = df["distance_m"].max()
    if max_dist < lap_distance_m:
        return pd.DataFrame()

    laps = []
    km_boundaries = np.arange(lap_distance_m, max_dist + lap_distance_m, lap_distance_m)

    prev_idx = 0
    for lap_num, boundary in enumerate(km_boundaries, start=1):
        mask = df["distance_m"] <= boundary
        group = df[mask].iloc[prev_idx:]
        if group.empty:
            break

        start_time = group["timestamp"].iloc[0]
        end_time = group["timestamp"].iloc[-1]
        duration_s = (end_time - start_time).total_seconds()
        dist = group["distance_m"].iloc[-1] - group["distance_m"].iloc[0]
        avg_pace = (duration_s / 60.0) / (dist / 1000.0) if dist > 0 else None
        avg_hr = group["hr"].mean() if "hr" in group and group["hr"].notna().any() else None
        max_hr = group["hr"].max() if "hr" in group and group["hr"].notna().any() else None

        elev = group["elevation_m"].dropna()
        gain = 0.0
        if len(elev) > 1:
            diff = elev.diff().dropna()
            gain = float(diff[diff > 1.0].sum())

        laps.append({
            "lap_num": lap_num,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "distance_m": dist,
            "avg_pace_min_km": avg_pace,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "elevation_gain": gain,
        })
        prev_idx += len(group)

    return pd.DataFrame(laps)


# ── Summary metrics ───────────────────────────────────────────────────────────

def _build_summary_metrics(df: pd.DataFrame, laps: pd.DataFrame) -> dict:
    total_distance_m = float(df["distance_m"].max())
    first_ts = df["timestamp"].min()
    last_ts = df["timestamp"].max()
    total_time_s = (last_ts - first_ts).total_seconds()

    moving = df[df["is_moving"]]
    moving_time_s = float(moving["time_delta_s"].sum())

    avg_pace = (moving_time_s / 60.0) / (total_distance_m / 1000.0) if total_distance_m > 0 else None

    hr_col = df["hr"].dropna()
    avg_hr = float(hr_col.mean()) if not hr_col.empty else None
    max_hr = float(hr_col.max()) if not hr_col.empty else None

    gain, loss = _compute_elevation_gain_loss(df["elevation_m"].dropna())

    projections = {}
    if avg_pace and total_distance_m > 0:
        for name, target_m in [("5K", 5000), ("10K", 10000), ("HM", 21097), ("Marathon", 42195)]:
            proj_s = _riegel_projection(total_distance_m, moving_time_s, target_m)
            proj_pace = (proj_s / 60.0) / (target_m / 1000.0) if proj_s > 0 else None
            projections[name] = {"time_s": proj_s, "pace_min_km": proj_pace}

    return {
        "total_distance_m": total_distance_m,
        "total_time_s": total_time_s,
        "moving_time_s": moving_time_s,
        "avg_pace_min_km": avg_pace,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "elevation_gain_m": gain,
        "elevation_loss_m": loss,
        "race_projections": projections,
        "start_time": first_ts,
        "end_time": last_ts,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def parse_gpx(file_obj) -> dict:
    """
    Parse a GPX file and return all computed data structures.

    Returns:
        dict with keys: workout_points, laps, summary_metrics
    """
    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)

    try:
        gpx = gpxpy.parse(file_obj)
    except Exception as e:
        raise ValueError(f"Failed to parse GPX file: {e}") from e

    if not gpx.tracks:
        raise ValueError("GPX file contains no tracks.")

    df = _extract_track_points(gpx)
    df = _compute_distances(df)
    df = _detect_gps_spikes(df)
    df = _detect_pauses(df)
    df = _compute_instantaneous_pace(df)
    df = _smooth_pace(df)

    # Attempt to extract lap metadata from GPX structure
    laps = _extract_gpx_laps(gpx)
    if laps is None or laps.empty:
        laps = _compute_laps_from_distance(df)

    # Enrich laps with computed metrics from workout_points
    if not laps.empty and "start_time" in laps.columns:
        enriched_laps = []
        for _, lap in laps.iterrows():
            row = lap.to_dict()
            mask = (df["timestamp"] >= lap["start_time"]) & (df["timestamp"] <= lap["end_time"])
            seg = df[mask]
            if not seg.empty:
                dist = seg["distance_m"].max() - seg["distance_m"].min()
                dur = (seg["timestamp"].max() - seg["timestamp"].min()).total_seconds()
                row["distance_m"] = dist
                row["duration_s"] = dur
                row["avg_pace_min_km"] = (dur / 60.0) / (dist / 1000.0) if dist > 0 else None
                if seg["hr"].notna().any():
                    row["avg_hr"] = float(seg["hr"].mean())
                    row["max_hr"] = float(seg["hr"].max())
                elev = seg["elevation_m"].dropna()
                if len(elev) > 1:
                    d = elev.diff().dropna()
                    row["elevation_gain"] = float(d[d > 1.0].sum())
            enriched_laps.append(row)
        laps = pd.DataFrame(enriched_laps)

    summary_metrics = _build_summary_metrics(df, laps)

    # Final column ordering
    cols = ["timestamp", "lat", "lon", "elevation_m", "hr", "cadence",
            "distance_m", "step_m", "pace_min_km", "smoothed_pace_min_km",
            "is_moving", "is_spike", "time_delta_s"]
    df = df[[c for c in cols if c in df.columns]]

    return {
        "workout_points": df,
        "laps": laps,
        "summary_metrics": summary_metrics,
    }
