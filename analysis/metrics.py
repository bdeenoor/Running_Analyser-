"""Pure metric computations: splits, averages, formatting."""

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_pace(pace_min_km) -> str:
    """Convert float min/km to mm:ss string."""
    if pace_min_km is None:
        return "--:--"
    try:
        v = float(pace_min_km)
    except (TypeError, ValueError):
        return "--:--"
    if not math.isfinite(v) or v <= 0:
        return "--:--"
    if v >= 30:
        return "30:00+"
    minutes = int(v)
    seconds = round((v - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def format_duration(seconds) -> str:
    """Convert total seconds to H:MM:SS or MM:SS."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "--"
    s = int(abs(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def pace_to_float(pace_str: str) -> Optional[float]:
    """Convert 'mm:ss' string to float min/km. Returns None on failure."""
    if not pace_str or pace_str in ("--:--", "--"):
        return None
    parts = pace_str.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]) + int(parts[1]) / 60.0
    except ValueError:
        return None


# ── KM splits ────────────────────────────────────────────────────────────────

def compute_km_splits(workout_points: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-km splits from workout_points.
    Uses true elapsed time / distance for each km segment (not mean of instantaneous pace).
    """
    if workout_points is None or workout_points.empty:
        return pd.DataFrame()

    df = workout_points.copy()
    df["km_bin"] = np.floor(df["distance_m"] / 1000.0).astype(int)

    splits = []
    for km_bin, group in df.groupby("km_bin"):
        if group.empty:
            continue
        start_dist = group["distance_m"].iloc[0]
        end_dist = group["distance_m"].iloc[-1]
        dist_km = (end_dist - start_dist) / 1000.0

        # Skip phantom zero-length bins (can appear at exact km boundaries)
        if dist_km < 0.01:
            continue

        start_ts = group["timestamp"].iloc[0]
        end_ts = group["timestamp"].iloc[-1]
        duration_s = (end_ts - start_ts).total_seconds()

        split_pace = (duration_s / 60.0) / dist_km if dist_km > 0 else None

        hr_vals = group["hr"].dropna() if "hr" in group else pd.Series(dtype=float)
        avg_hr = float(hr_vals.mean()) if not hr_vals.empty else None

        elev = group["elevation_m"].dropna()
        gain = 0.0
        if len(elev) > 1:
            diff = elev.diff().dropna()
            gain = float(diff[diff > 1.0].sum())

        splits.append({
            "km": km_bin + 1,
            "split_pace_min_km": split_pace,
            "avg_hr": avg_hr,
            "elevation_gain": gain,
            "distance_m": end_dist - start_dist,
            "duration_s": duration_s,
        })

    return pd.DataFrame(splits)


# ── Summary metrics ───────────────────────────────────────────────────────────

def compute_summary_metrics(
    workout_points: pd.DataFrame,
    laps: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute scalar summary metrics, preferring laps data when available."""
    if workout_points is None or workout_points.empty:
        return {}

    total_distance_m = float(workout_points["distance_m"].max())
    first_ts = workout_points["timestamp"].min()
    last_ts = workout_points["timestamp"].max()
    total_time_s = (last_ts - first_ts).total_seconds()

    moving = workout_points[workout_points["is_moving"]] if "is_moving" in workout_points.columns else workout_points
    moving_time_s = float(moving["time_delta_s"].sum()) if "time_delta_s" in moving.columns else total_time_s

    avg_pace = (moving_time_s / 60.0) / (total_distance_m / 1000.0) if total_distance_m > 0 else None

    hr_col = workout_points["hr"].dropna() if "hr" in workout_points.columns else pd.Series(dtype=float)
    avg_hr = float(hr_col.mean()) if not hr_col.empty else None
    max_hr = float(hr_col.max()) if not hr_col.empty else None

    elev = workout_points["elevation_m"].dropna() if "elevation_m" in workout_points.columns else pd.Series(dtype=float)
    gain, loss = 0.0, 0.0
    if len(elev) > 1:
        sm = elev.rolling(5, min_periods=1, center=True).mean()
        diff = sm.diff().dropna()
        gain = float(diff[diff > 1.0].sum())
        loss = float(abs(diff[diff < -1.0].sum()))

    from parsers.gpx_parser import _riegel_projection
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


# ── Moving averages ───────────────────────────────────────────────────────────

def compute_moving_averages(workout_points: pd.DataFrame, window_s: int = 30) -> pd.DataFrame:
    """Add pace_ma and hr_ma columns using time-based rolling window."""
    df = workout_points.copy()
    ts_indexed = df.set_index("timestamp")

    if "pace_min_km" in ts_indexed.columns:
        df["pace_ma"] = ts_indexed["pace_min_km"].rolling(f"{window_s}s", min_periods=1).mean().values
    if "hr" in ts_indexed.columns:
        df["hr_ma"] = ts_indexed["hr"].rolling(f"{window_s}s", min_periods=1).mean().values

    return df


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(state: dict) -> list[str]:
    """Cross-validate data sources and return list of warning strings."""
    warnings = []
    wp = state.get("workout_points")
    sm = state.get("summary_metrics", {})
    laps = state.get("laps")
    matched = state.get("matched_segments")

    if wp is not None and not wp.empty and sm:
        gpx_dist = wp["distance_m"].max()
        ocr_dist = sm.get("total_distance_m_ocr")
        if ocr_dist and abs(gpx_dist - ocr_dist) / ocr_dist > 0.02:
            warnings.append(
                f"⚠️ Distance mismatch: GPX = {gpx_dist/1000:.2f} km, "
                f"Screenshot = {ocr_dist/1000:.2f} km"
            )

        max_hr = sm.get("max_hr")
        if max_hr is not None:
            if max_hr > 220:
                warnings.append(f"⚠️ Max HR = {max_hr:.0f} bpm — suspiciously high (>220).")
            elif max_hr < 40:
                warnings.append(f"⚠️ Max HR = {max_hr:.0f} bpm — suspiciously low (<40).")

        recomputed_pace = (
            (sm.get("moving_time_s", 0) / 60.0) / (gpx_dist / 1000.0)
            if gpx_dist > 0 else None
        )
        stored_pace = sm.get("avg_pace_min_km")
        if recomputed_pace and stored_pace:
            if abs(recomputed_pace - stored_pace) > 0.1:
                warnings.append(
                    f"⚠️ Avg pace discrepancy: computed = {format_pace(recomputed_pace)}, "
                    f"stored = {format_pace(stored_pace)}"
                )

    if matched is not None and not matched.empty and sm:
        seg_total = matched["actual_duration_s"].sum() if "actual_duration_s" in matched.columns else 0
        workout_total = sm.get("total_time_s", 0)
        if workout_total and seg_total and abs(seg_total - workout_total) / workout_total > 0.05:
            warnings.append(
                f"⚠️ Segment durations ({format_duration(seg_total)}) don't sum to "
                f"total workout time ({format_duration(workout_total)}) — >5% discrepancy."
            )

    if laps is not None and not laps.empty and "distance_m" in laps.columns:
        lap_total = laps["distance_m"].dropna().sum()
        if sm.get("total_distance_m") and lap_total > 0:
            if abs(lap_total - sm["total_distance_m"]) / sm["total_distance_m"] > 0.05:
                warnings.append(
                    f"⚠️ Lap distances sum ({lap_total/1000:.2f} km) ≠ "
                    f"total GPX distance ({sm['total_distance_m']/1000:.2f} km)."
                )

    return warnings
