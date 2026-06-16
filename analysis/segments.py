"""Segment matching logic: planned vs actual, confidence scoring, coaching text."""

import math
from typing import Optional

import numpy as np
import pandas as pd

from analysis.metrics import format_pace, format_duration


# ── Segment type constants ────────────────────────────────────────────────────

SEGMENT_TYPES = ["warmup", "easy", "hard", "tempo", "marathon_pace", "cooldown", "recovery"]

HEBREW_TERMS = {
    "overall_summary": "סיכום כללי",
    "plan_execution": "ביצוע התכנית",
    "best_segment": "הקטע הטוב ביותר",
    "hardest_segment": "הקטע הקשה ביותר",
    "hr_drift": "סחף דופק",
    "race_projections": "תחזיות תחרות",
    "next_workout": "המלצה לאימון הבא",
    "effort_control": "שליטה במאמץ",
    "warmup": "חימום",
    "easy": "קל",
    "hard": "קשה",
    "tempo": "טמפו",
    "cooldown": "צינון",
    "recovery": "התאוששות",
    "marathon_pace": "קצב מרתון",
    "faster_than_plan": "מהיר מהתכנית",
    "slower_than_plan": "איטי מהתכנית",
    "on_target": "בדיוק לפי התכנית",
    "km": "ק\"מ",
    "min_km": "דק/ק\"מ",
    "bpm": "פעימות/דקה",
    "avg_hr": "דופק ממוצע",
    "max_hr": "דופק מקסימלי",
    "avg_pace": "קצב ממוצע",
    "total_distance": "מרחק כולל",
    "total_time": "זמן כולל",
    "elevation_gain": "עלייה מצטברת",
}

ENGLISH_TERMS = {k: k.replace("_", " ").title() for k in HEBREW_TERMS}
ENGLISH_TERMS.update({
    "warmup": "Warm-Up",
    "easy": "Easy",
    "hard": "Hard",
    "tempo": "Tempo",
    "cooldown": "Cool-Down",
    "recovery": "Recovery",
    "marathon_pace": "Marathon Pace",
    "min_km": "min/km",
    "bpm": "bpm",
    "km": "km",
})


# ── Actual metrics from a workout_points slice ────────────────────────────────

def _compute_segment_actuals(wp_slice: pd.DataFrame) -> dict:
    if wp_slice is None or wp_slice.empty:
        return {}

    dist = wp_slice["distance_m"].max() - wp_slice["distance_m"].min()
    start_ts = wp_slice["timestamp"].min()
    end_ts = wp_slice["timestamp"].max()
    duration_s = (end_ts - start_ts).total_seconds()
    pace = (duration_s / 60.0) / (dist / 1000.0) if dist > 0 else None

    hr = wp_slice["hr"].dropna() if "hr" in wp_slice.columns else pd.Series(dtype=float)
    avg_hr = float(hr.mean()) if not hr.empty else None
    max_hr = float(hr.max()) if not hr.empty else None

    cad = wp_slice["cadence"].dropna() if "cadence" in wp_slice.columns else pd.Series(dtype=float)
    avg_cadence = float(cad.mean()) if not cad.empty else None

    elev = wp_slice["elevation_m"].dropna()
    elevation_change = float(elev.iloc[-1] - elev.iloc[0]) if len(elev) > 1 else None

    return {
        "actual_distance_m": dist,
        "actual_duration_s": duration_s,
        "actual_pace_min_km": pace,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "avg_cadence": avg_cadence,
        "elevation_change_m": elevation_change,
    }


def _compute_delta_pace(planned_pace, actual_pace) -> Optional[float]:
    """Delta in sec/km: positive = slower than plan, negative = faster."""
    if planned_pace is None or actual_pace is None:
        return None
    if not (math.isfinite(float(planned_pace)) and math.isfinite(float(actual_pace))):
        return None
    return (float(actual_pace) - float(planned_pace)) * 60.0


# ── Strategy 1: Lap Boundary Matching ────────────────────────────────────────

def _match_lap_boundary(
    planned: pd.DataFrame,
    laps: pd.DataFrame,
    workout_points: pd.DataFrame,
) -> pd.DataFrame:
    """Match planned segments to actual laps by time-window overlap."""
    if laps.empty or "start_time" not in laps.columns:
        return pd.DataFrame()

    # Build cumulative planned timeline
    plan_start = workout_points["timestamp"].min() if workout_points is not None and not workout_points.empty else laps["start_time"].min()
    plan_starts = []
    plan_ends = []
    t = 0.0
    for _, seg in planned.iterrows():
        plan_starts.append(t)
        t += float(seg["duration_s"])
        plan_ends.append(t)

    # Build lap timeline — use tz-aware Series for consistent comparison
    lap_starts_abs = pd.to_datetime(laps["start_time"]).dt.tz_localize("UTC") if pd.to_datetime(laps["start_time"]).dt.tz is None else pd.to_datetime(laps["start_time"])
    if "end_time" in laps.columns and laps["end_time"].notna().any():
        lap_ends_abs = pd.to_datetime(laps["end_time"]).dt.tz_localize("UTC") if pd.to_datetime(laps["end_time"]).dt.tz is None else pd.to_datetime(laps["end_time"])
    else:
        lap_ends_abs = lap_starts_abs + pd.to_timedelta(laps["duration_s"].fillna(0), unit="s")

    results = []
    for i, seg in planned.iterrows():
        seg_start_s = plan_starts[i]
        seg_end_s = plan_ends[i]
        seg_start_abs = plan_start + pd.Timedelta(seconds=seg_start_s)
        seg_end_abs = plan_start + pd.Timedelta(seconds=seg_end_s)

        # Find overlapping laps
        overlap_mask = (lap_starts_abs < seg_end_abs) & (lap_ends_abs > seg_start_abs)
        matching_laps = laps[overlap_mask]

        if matching_laps.empty:
            results.append(_build_matched_row(seg, {}, "timestamp"))
            continue

        # Aggregate matching laps
        agg_dist = matching_laps["distance_m"].dropna().sum()
        agg_dur = matching_laps["duration_s"].dropna().sum()
        agg_pace = (agg_dur / 60.0) / (agg_dist / 1000.0) if agg_dist > 0 else None
        agg_hr = matching_laps["avg_hr"].dropna().mean() if "avg_hr" in matching_laps else None
        agg_max_hr = matching_laps["max_hr"].dropna().max() if "max_hr" in matching_laps else None
        agg_elev = matching_laps["elevation_gain"].dropna().sum() if "elevation_gain" in matching_laps else None

        actuals = {
            "actual_distance_m": agg_dist,
            "actual_duration_s": agg_dur,
            "actual_pace_min_km": agg_pace,
            "avg_hr": agg_hr,
            "max_hr": agg_max_hr,
            "elevation_change_m": agg_elev,
            "avg_cadence": None,
        }

        source = "lap_boundary" if len(matching_laps) == 1 else "lap_boundary_merged"
        results.append(_build_matched_row(seg, actuals, source))

    return pd.DataFrame(results) if results else pd.DataFrame()


# ── Strategy 2: Timestamp Inference ──────────────────────────────────────────

def _match_timestamp(
    planned: pd.DataFrame,
    workout_points: pd.DataFrame,
) -> pd.DataFrame:
    """Slice workout_points by planned segment duration offsets from workout start."""
    if workout_points is None or workout_points.empty:
        return pd.DataFrame()

    start_ts = workout_points["timestamp"].min()
    results = []
    t = 0.0

    for _, seg in planned.iterrows():
        seg_start = start_ts + pd.Timedelta(seconds=t)
        seg_end = start_ts + pd.Timedelta(seconds=t + float(seg["duration_s"]))
        t += float(seg["duration_s"])

        mask = (workout_points["timestamp"] >= seg_start) & (workout_points["timestamp"] < seg_end)
        wp_slice = workout_points[mask]

        actuals = _compute_segment_actuals(wp_slice) if not wp_slice.empty else {}
        results.append(_build_matched_row(seg, actuals, "timestamp"))

    return pd.DataFrame(results) if results else pd.DataFrame()


# ── Shared row builder ────────────────────────────────────────────────────────

def _build_matched_row(planned_seg, actuals: dict, source: str) -> dict:
    planned_pace = planned_seg.get("target_pace_min_km") if hasattr(planned_seg, "get") else None
    actual_pace = actuals.get("actual_pace_min_km")
    delta = _compute_delta_pace(planned_pace, actual_pace)

    return {
        "segment_num": planned_seg.get("segment_num", 0) if hasattr(planned_seg, "get") else planned_seg["segment_num"],
        "name": planned_seg.get("name", "") if hasattr(planned_seg, "get") else planned_seg["name"],
        "type": planned_seg.get("type", "easy") if hasattr(planned_seg, "get") else planned_seg.get("type", "easy"),
        "planned_duration_s": planned_seg.get("duration_s") if hasattr(planned_seg, "get") else planned_seg["duration_s"],
        "planned_pace_min_km": planned_pace,
        "actual_duration_s": actuals.get("actual_duration_s"),
        "actual_distance_m": actuals.get("actual_distance_m"),
        "actual_pace_min_km": actual_pace,
        "delta_pace_sec_km": delta,
        "avg_hr": actuals.get("avg_hr"),
        "max_hr": actuals.get("max_hr"),
        "avg_cadence": actuals.get("avg_cadence"),
        "elevation_change_m": actuals.get("elevation_change_m"),
        "source": source,
        "confidence": "medium",
    }


# ── Confidence scoring ────────────────────────────────────────────────────────

def score_confidence(matched: pd.DataFrame) -> pd.DataFrame:
    df = matched.copy()

    def _score(row):
        delta = row.get("delta_pace_sec_km")
        dur_planned = row.get("planned_duration_s")
        dur_actual = row.get("actual_duration_s")
        source = row.get("source", "")

        if delta is None or not math.isfinite(float(delta) if delta is not None else float("nan")):
            return "low"

        within_10s = abs(delta) <= 10
        dur_ok = False
        if dur_planned and dur_actual and dur_planned > 0:
            dur_ok = abs(dur_actual - dur_planned) / dur_planned <= 0.05

        if within_10s and dur_ok and "lap_boundary" in source:
            return "high"
        if abs(delta) <= 20 or source == "timestamp":
            return "medium"
        return "low"

    df["confidence"] = df.apply(_score, axis=1)
    return df


# ── Public: match_segments ────────────────────────────────────────────────────

def match_segments(
    planned_segments: pd.DataFrame,
    laps: Optional[pd.DataFrame],
    workout_points: Optional[pd.DataFrame],
    strategy: str = "auto",
) -> pd.DataFrame:
    """
    Match planned segments to actual workout data.

    Strategy auto-selection:
    - "lap_boundary" if laps available and count is close to planned count
    - "timestamp" otherwise
    """
    if planned_segments is None or planned_segments.empty:
        return pd.DataFrame()

    has_laps = laps is not None and not laps.empty
    has_wp = workout_points is not None and not workout_points.empty

    if strategy == "auto":
        if has_laps and has_wp:
            count_diff = abs(len(laps) - len(planned_segments))
            strategy = "lap_boundary" if count_diff <= max(2, len(planned_segments) // 3) else "timestamp"
        elif has_wp:
            strategy = "timestamp"
        else:
            return pd.DataFrame()

    if strategy == "lap_boundary" and has_laps:
        result = _match_lap_boundary(planned_segments, laps, workout_points)
        if not result.empty:
            return score_confidence(result)

    if has_wp:
        result = _match_timestamp(planned_segments, workout_points)
        return score_confidence(result)

    return pd.DataFrame()


# ── Coaching text generation ──────────────────────────────────────────────────

def generate_coaching_text(
    matched_segments: pd.DataFrame,
    summary_metrics: dict,
    language: str = "he",
) -> str:
    terms = HEBREW_TERMS if language == "he" else ENGLISH_TERMS
    rtl_mark = "‏" if language == "he" else ""

    lines = []

    def h(section_key: str) -> str:
        return f"\n{'─' * 40}\n{rtl_mark}🏃 {terms.get(section_key, section_key).upper()}\n{'─' * 40}"

    # ── Overall summary ───────────────────────────────────────────────────────
    lines.append(h("overall_summary"))
    dist = summary_metrics.get("total_distance_m", 0) / 1000
    moving_s = summary_metrics.get("moving_time_s", 0)
    avg_pace = summary_metrics.get("avg_pace_min_km")
    avg_hr = summary_metrics.get("avg_hr")
    max_hr = summary_metrics.get("max_hr")
    gain = summary_metrics.get("elevation_gain_m", 0)

    if language == "he":
        lines.append(f"{rtl_mark}מרחק: {dist:.2f} {terms['km']}")
        lines.append(f"{rtl_mark}זמן ריצה: {format_duration(moving_s)}")
        lines.append(f"{rtl_mark}{terms['avg_pace']}: {format_pace(avg_pace)} {terms['min_km']}")
        if avg_hr:
            lines.append(f"{rtl_mark}{terms['avg_hr']}: {avg_hr:.0f} {terms['bpm']}")
        if max_hr:
            lines.append(f"{rtl_mark}{terms['max_hr']}: {max_hr:.0f} {terms['bpm']}")
        if gain:
            lines.append(f"{rtl_mark}{terms['elevation_gain']}: {gain:.0f} מטר")
    else:
        lines.append(f"Distance: {dist:.2f} km")
        lines.append(f"Moving Time: {format_duration(moving_s)}")
        lines.append(f"Avg Pace: {format_pace(avg_pace)} min/km")
        if avg_hr:
            lines.append(f"Avg HR: {avg_hr:.0f} bpm")
        if max_hr:
            lines.append(f"Max HR: {max_hr:.0f} bpm")
        if gain:
            lines.append(f"Elevation Gain: {gain:.0f} m")

    # ── Plan execution ────────────────────────────────────────────────────────
    if matched_segments is not None and not matched_segments.empty:
        lines.append(h("plan_execution"))

        has_delta = matched_segments["delta_pace_sec_km"].notna()
        n_total = has_delta.sum()

        if n_total > 0:
            deltas = matched_segments.loc[has_delta, "delta_pace_sec_km"]
            within_10 = (abs(deltas) <= 10).sum()
            within_20 = (abs(deltas) <= 20).sum()
            pct_10 = 100 * within_10 / n_total
            pct_20 = 100 * within_20 / n_total

            if language == "he":
                lines.append(f"{rtl_mark}{within_10} מתוך {n_total} קטעים בטווח ±10 שניות/ק\"מ מהתכנית ({pct_10:.0f}%)")
                lines.append(f"{rtl_mark}{within_20} מתוך {n_total} קטעים בטווח ±20 שניות/ק\"מ ({pct_20:.0f}%)")
            else:
                lines.append(f"{within_10}/{n_total} segments within ±10 sec/km of plan ({pct_10:.0f}%)")
                lines.append(f"{within_20}/{n_total} segments within ±20 sec/km ({pct_20:.0f}%)")

        # ── Best segment ──────────────────────────────────────────────────────
        lines.append(h("best_segment"))
        valid = matched_segments[matched_segments["delta_pace_sec_km"].notna()]
        if not valid.empty:
            best = valid.loc[valid["delta_pace_sec_km"].idxmin()]
            best_name = best.get("name", f"Segment {best.get('segment_num', '?')}")
            best_delta = best["delta_pace_sec_km"]
            best_pace = best.get("actual_pace_min_km")
            sign = "⬆️" if best_delta < 0 else "⬇️"
            if language == "he":
                lines.append(f"{rtl_mark}{best_name}: {format_pace(best_pace)} {terms['min_km']}")
                lines.append(f"{rtl_mark}{sign} {abs(best_delta):.0f} שניות/ק\"מ {terms['faster_than_plan'] if best_delta < 0 else terms['slower_than_plan']}")
            else:
                direction = "faster" if best_delta < 0 else "slower"
                lines.append(f"{best_name}: {format_pace(best_pace)} min/km")
                lines.append(f"{sign} {abs(best_delta):.0f} sec/km {direction} than plan")

        # ── Hardest segment ───────────────────────────────────────────────────
        lines.append(h("hardest_segment"))
        hr_valid = matched_segments[matched_segments["avg_hr"].notna()]
        if not hr_valid.empty:
            hardest = hr_valid.loc[hr_valid["avg_hr"].idxmax()]
            hard_name = hardest.get("name", f"Segment {hardest.get('segment_num', '?')}")
            hard_hr = hardest["avg_hr"]
            hard_pace = hardest.get("actual_pace_min_km")
            if language == "he":
                lines.append(f"{rtl_mark}{hard_name}: דופק ממוצע {hard_hr:.0f} {terms['bpm']}, קצב {format_pace(hard_pace)}")
            else:
                lines.append(f"{hard_name}: avg HR {hard_hr:.0f} bpm, pace {format_pace(hard_pace)}")

        # ── HR drift ─────────────────────────────────────────────────────────
        lines.append(h("hr_drift"))
        hr_segs = matched_segments[matched_segments["avg_hr"].notna()]
        if len(hr_segs) >= 4:
            half = len(hr_segs) // 2
            first_half_hr = hr_segs.iloc[:half]["avg_hr"].mean()
            second_half_hr = hr_segs.iloc[half:]["avg_hr"].mean()
            drift = second_half_hr - first_half_hr

            if language == "he":
                lines.append(f"{rtl_mark}דופק חצי ראשון: {first_half_hr:.0f} {terms['bpm']}")
                lines.append(f"{rtl_mark}דופק חצי שני: {second_half_hr:.0f} {terms['bpm']}")
                if drift > 10:
                    lines.append(f"{rtl_mark}⚠️ סחף דופק משמעותי ({drift:.0f} פעימות) — עייפות אפשרית.")
                elif drift > 5:
                    lines.append(f"{rtl_mark}📊 סחף דופק קל ({drift:.0f} פעימות) — תגובה נורמלית לאימון.")
                else:
                    lines.append(f"{rtl_mark}✅ יציבות דופק מצוינת ({drift:+.0f} פעימות).")
            else:
                lines.append(f"First half avg HR: {first_half_hr:.0f} bpm")
                lines.append(f"Second half avg HR: {second_half_hr:.0f} bpm")
                if drift > 10:
                    lines.append(f"⚠️ Significant HR drift ({drift:.0f} bpm) — possible fatigue.")
                elif drift > 5:
                    lines.append(f"📊 Mild HR drift ({drift:.0f} bpm) — normal training response.")
                else:
                    lines.append(f"✅ Excellent HR stability ({drift:+.0f} bpm).")

        # ── Effort control ────────────────────────────────────────────────────
        lines.append(h("effort_control"))
        pace_vals = matched_segments["actual_pace_min_km"].dropna()
        if len(pace_vals) > 1:
            cv = pace_vals.std() / pace_vals.mean() * 100
            if language == "he":
                if cv < 5:
                    lines.append(f"{rtl_mark}✅ שליטה מצוינת בקצב — עקביות גבוהה (CV={cv:.1f}%)")
                elif cv < 10:
                    lines.append(f"{rtl_mark}📊 שליטה טובה בקצב (CV={cv:.1f}%)")
                else:
                    lines.append(f"{rtl_mark}⚠️ שונות קצב גבוהה (CV={cv:.1f}%) — עבוד על עקביות")
            else:
                if cv < 5:
                    lines.append(f"✅ Excellent pace control — high consistency (CV={cv:.1f}%)")
                elif cv < 10:
                    lines.append(f"📊 Good pace control (CV={cv:.1f}%)")
                else:
                    lines.append(f"⚠️ High pace variability (CV={cv:.1f}%) — work on consistency")

    # ── Race projections ──────────────────────────────────────────────────────
    projections = summary_metrics.get("race_projections", {})
    if projections:
        lines.append(h("race_projections"))
        for race, data in projections.items():
            t = data.get("time_s", 0)
            p = data.get("pace_min_km")
            if t > 0:
                if language == "he":
                    lines.append(f"{rtl_mark}{race}: {format_duration(t)} ({format_pace(p)} {terms['min_km']})")
                else:
                    lines.append(f"{race}: {format_duration(t)} ({format_pace(p)} min/km)")

    # ── Next workout ──────────────────────────────────────────────────────────
    lines.append(h("next_workout"))
    avg_hr_val = summary_metrics.get("avg_hr")
    avg_pace_val = summary_metrics.get("avg_pace_min_km")

    if matched_segments is not None and not matched_segments.empty:
        deltas = matched_segments["delta_pace_sec_km"].dropna()
        hr_segs_all = matched_segments["avg_hr"].dropna()
        hr_drift_val = 0.0
        if len(hr_segs_all) >= 4:
            half = len(hr_segs_all) // 2
            hr_drift_val = hr_segs_all.iloc[half:].mean() - hr_segs_all.iloc[:half].mean()

        if hr_drift_val > 10:
            if language == "he":
                lines.append(f"{rtl_mark}🔵 סחף דופק משמעותי — מומלץ שבוע התאוששות לפני האימון הבא הקשה.")
            else:
                lines.append("🔵 Significant HR drift — recommend a recovery week before next hard session.")
        elif len(deltas) > 0 and (abs(deltas) <= 10).mean() > 0.8:
            if language == "he":
                lines.append(f"{rtl_mark}🟢 ביצוע מצוין! ניתן להגדיל את נפח הקטעים הקשים ב-10% באימון הבא.")
            else:
                lines.append("🟢 Excellent execution! You can increase hard segment volume by 10% next session.")
        elif avg_hr_val and avg_pace_val and avg_hr_val > 165 and avg_pace_val < 5.5:
            if language == "he":
                lines.append(f"{rtl_mark}🟡 דופק גבוה ביחס לקצב — הוסף ריצות קלות לפיתוח הבסיס האירובי.")
            else:
                lines.append("🟡 High HR relative to pace — add easy runs to build aerobic base.")
        else:
            if language == "he":
                lines.append(f"{rtl_mark}🟢 אימון טוב. המשך עם אותה עצימות — עקביות היא המפתח.")
            else:
                lines.append("🟢 Good training. Maintain this intensity — consistency is key.")

    return "\n".join(lines)
