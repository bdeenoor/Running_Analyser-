"""Tests for analysis/segments.py — segment matching and coaching text."""

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.segments import (
    match_segments,
    score_confidence,
    generate_coaching_text,
    _compute_delta_pace,
    _compute_segment_actuals,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_planned_segments(types=("warmup", "hard", "easy", "cooldown"), durations=(1200, 300, 180, 600)):
    rows = []
    for i, (t, d) in enumerate(zip(types, durations)):
        rows.append({
            "segment_num": i + 1,
            "name": f"{t.capitalize()} {i+1}",
            "type": t,
            "duration_s": d,
            "target_pace_min_km": 4.5 if t == "hard" else (5.5 if t == "easy" else None),
        })
    return pd.DataFrame(rows)


def make_laps(n: int = 4, base_ts: datetime = None):
    if base_ts is None:
        base_ts = datetime(2024, 1, 1, 7, 0, 0, tzinfo=timezone.utc)
    rows = []
    t = base_ts
    durations = [1200, 300, 180, 600]
    for i in range(n):
        dur = durations[i % len(durations)]
        end = t + timedelta(seconds=dur)
        rows.append({
            "lap_num": i + 1,
            "start_time": t,
            "end_time": end,
            "duration_s": dur,
            "distance_m": dur / 60 * 1000 / 5.0,  # roughly 5 min/km
            "avg_pace_min_km": 5.0,
            "avg_hr": 150.0 + i * 5,
            "max_hr": 160.0 + i * 5,
            "elevation_gain": 5.0,
        })
        t = end
    return pd.DataFrame(rows)


def make_workout_points(n: int = 400, start_time: datetime = None):
    if start_time is None:
        start_time = datetime(2024, 1, 1, 7, 0, 0, tzinfo=timezone.utc)
    total_s = 2280  # 38 min (sum of fixture durations)
    timestamps = pd.date_range(start_time, periods=n, freq=f"{total_s/n:.1f}s", tz="UTC")
    distances = np.linspace(0, 7500, n)
    paces = np.full(n, 5.0) + np.random.normal(0, 0.1, n)
    hrs = np.full(n, 150.0) + np.random.normal(0, 3, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "lat": np.linspace(51.5, 51.57, n),
        "lon": np.full(n, -0.1278),
        "elevation_m": np.linspace(0, 30, n),
        "hr": hrs,
        "cadence": np.full(n, 170.0),
        "distance_m": distances,
        "step_m": np.full(n, 7500 / n),
        "pace_min_km": paces,
        "smoothed_pace_min_km": paces,
        "is_moving": np.ones(n, dtype=bool),
        "is_spike": np.zeros(n, dtype=bool),
        "time_delta_s": np.full(n, total_s / n),
    })


# ── _compute_delta_pace ───────────────────────────────────────────────────────

class TestDeltaPace:
    def test_faster_than_plan(self):
        delta = _compute_delta_pace(5.0, 4.5)
        assert delta < 0  # faster = negative
        assert abs(delta - (-30.0)) < 0.1

    def test_slower_than_plan(self):
        delta = _compute_delta_pace(5.0, 5.5)
        assert delta > 0
        assert abs(delta - 30.0) < 0.1

    def test_on_target(self):
        delta = _compute_delta_pace(5.0, 5.0)
        assert abs(delta) < 0.01

    def test_none_planned(self):
        assert _compute_delta_pace(None, 5.0) is None

    def test_none_actual(self):
        assert _compute_delta_pace(5.0, None) is None

    def test_both_none(self):
        assert _compute_delta_pace(None, None) is None


# ── _compute_segment_actuals ──────────────────────────────────────────────────

class TestSegmentActuals:
    def test_basic(self):
        wp = make_workout_points(n=100)
        result = _compute_segment_actuals(wp)
        assert "actual_distance_m" in result
        assert "actual_duration_s" in result
        assert "actual_pace_min_km" in result
        assert result["actual_distance_m"] > 0
        assert result["actual_duration_s"] > 0
        assert 3.0 <= result["actual_pace_min_km"] <= 10.0

    def test_hr_extracted(self):
        wp = make_workout_points(n=50)
        result = _compute_segment_actuals(wp)
        assert result["avg_hr"] is not None
        assert 130 < result["avg_hr"] < 170

    def test_empty_slice(self):
        result = _compute_segment_actuals(pd.DataFrame())
        assert result == {}


# ── match_segments ────────────────────────────────────────────────────────────

class TestSegmentMatching:
    def test_timestamp_strategy(self):
        planned = make_planned_segments()
        wp = make_workout_points(n=500)
        result = match_segments(planned, None, wp, strategy="timestamp")
        assert not result.empty
        assert len(result) == len(planned)

    def test_lap_boundary_strategy(self):
        planned = make_planned_segments()
        laps = make_laps(n=4)
        wp = make_workout_points(n=500)
        result = match_segments(planned, laps, wp, strategy="lap_boundary")
        assert not result.empty
        assert len(result) == len(planned)

    def test_auto_selects_lap_when_available(self):
        planned = make_planned_segments()
        laps = make_laps(n=4)
        wp = make_workout_points(n=500)
        result = match_segments(planned, laps, wp, strategy="auto")
        assert not result.empty
        assert "source" in result.columns

    def test_auto_falls_back_to_timestamp(self):
        planned = make_planned_segments()
        wp = make_workout_points(n=500)
        result = match_segments(planned, None, wp, strategy="auto")
        assert not result.empty
        assert result["source"].iloc[0] == "timestamp"

    def test_empty_planned_returns_empty(self):
        wp = make_workout_points()
        result = match_segments(pd.DataFrame(), None, wp)
        assert result.empty

    def test_segment_count_matches(self):
        planned = make_planned_segments(
            types=("warmup", "hard", "easy", "hard", "easy", "cooldown"),
            durations=(1200, 120, 120, 240, 120, 600),
        )
        wp = make_workout_points(n=600)
        result = match_segments(planned, None, wp, strategy="timestamp")
        assert len(result) == len(planned)

    def test_all_required_columns_present(self):
        planned = make_planned_segments()
        wp = make_workout_points(n=400)
        result = match_segments(planned, None, wp, strategy="timestamp")
        required = [
            "segment_num", "name", "type",
            "planned_duration_s", "actual_duration_s",
            "actual_pace_min_km", "delta_pace_sec_km",
            "confidence",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"


# ── score_confidence ──────────────────────────────────────────────────────────

class TestConfidenceScoring:
    def _make_row(self, delta, dur_planned, dur_actual, source):
        return {
            "segment_num": 1, "name": "Test", "type": "hard",
            "planned_duration_s": dur_planned, "actual_duration_s": dur_actual,
            "planned_pace_min_km": 5.0, "actual_pace_min_km": 5.0 + delta / 60,
            "delta_pace_sec_km": delta, "avg_hr": 150.0, "max_hr": 165.0,
            "elevation_change_m": 0.0, "avg_cadence": 170.0,
            "actual_distance_m": dur_actual / 60 * 200, "source": source,
            "confidence": "medium",
        }

    def test_high_confidence(self):
        df = pd.DataFrame([self._make_row(5, 300, 310, "lap_boundary")])
        result = score_confidence(df)
        assert result.iloc[0]["confidence"] == "high"

    def test_medium_confidence_by_delta(self):
        df = pd.DataFrame([self._make_row(15, 300, 320, "timestamp")])
        result = score_confidence(df)
        assert result.iloc[0]["confidence"] == "medium"

    def test_low_confidence_large_delta(self):
        df = pd.DataFrame([self._make_row(60, 300, 500, "dtw")])
        result = score_confidence(df)
        assert result.iloc[0]["confidence"] == "low"

    def test_none_delta_is_low(self):
        row = self._make_row(0, 300, 300, "timestamp")
        row["delta_pace_sec_km"] = None
        df = pd.DataFrame([row])
        result = score_confidence(df)
        assert result.iloc[0]["confidence"] == "low"


# ── generate_coaching_text ────────────────────────────────────────────────────

class TestCoachingText:
    def _make_matched(self):
        planned = make_planned_segments()
        rows = []
        for _, seg in planned.iterrows():
            rows.append({
                "segment_num": seg["segment_num"],
                "name": seg["name"],
                "type": seg["type"],
                "planned_duration_s": seg["duration_s"],
                "planned_pace_min_km": seg["target_pace_min_km"],
                "actual_duration_s": seg["duration_s"] * 1.02,
                "actual_distance_m": seg["duration_s"] / 60 * 1000 / 5.0,
                "actual_pace_min_km": 5.0,
                "delta_pace_sec_km": 5.0,
                "avg_hr": 150.0 + seg["segment_num"] * 3,
                "max_hr": 165.0,
                "avg_cadence": 170.0,
                "elevation_change_m": 5.0,
                "source": "timestamp",
                "confidence": "medium",
            })
        return pd.DataFrame(rows)

    def _make_summary(self, distance_km=10.0, pace=5.0, avg_hr=155.0, max_hr=175.0):
        from parsers.gpx_parser import _riegel_projection
        moving_s = distance_km * pace * 60
        proj = {}
        for name, target in [("5K", 5000), ("10K", 10000), ("HM", 21097), ("Marathon", 42195)]:
            t = _riegel_projection(distance_km * 1000, moving_s, target)
            proj[name] = {"time_s": t, "pace_min_km": (t / 60) / (target / 1000)}
        return {
            "total_distance_m": distance_km * 1000,
            "moving_time_s": moving_s,
            "total_time_s": moving_s * 1.05,
            "avg_pace_min_km": pace,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "elevation_gain_m": 50,
            "elevation_loss_m": 45,
            "race_projections": proj,
        }

    def test_hebrew_output_not_empty(self):
        matched = self._make_matched()
        sm = self._make_summary()
        text = generate_coaching_text(matched, sm, language="he")
        assert len(text) > 100
        assert "km" in text.lower() or "ק" in text

    def test_english_output_not_empty(self):
        matched = self._make_matched()
        sm = self._make_summary()
        text = generate_coaching_text(matched, sm, language="en")
        assert len(text) > 100
        assert "Distance" in text or "km" in text.lower()

    def test_race_projections_in_output(self):
        matched = self._make_matched()
        sm = self._make_summary()
        text = generate_coaching_text(matched, sm, language="en")
        assert "Marathon" in text
        assert "10K" in text

    def test_hr_drift_detected(self):
        matched = self._make_matched()
        # Make second half have much higher HR
        half = len(matched) // 2
        matched.loc[half:, "avg_hr"] = 175.0  # big drift
        sm = self._make_summary(avg_hr=155.0)
        text = generate_coaching_text(matched, sm, language="en")
        # Should detect drift and mention fatigue
        assert "drift" in text.lower() or "fatigue" in text.lower() or "סחף" in text

    def test_hr_stable_not_flagged(self):
        matched = self._make_matched()
        # Stable HR throughout
        matched["avg_hr"] = 152.0
        sm = self._make_summary(avg_hr=152.0)
        text = generate_coaching_text(matched, sm, language="en")
        assert "stability" in text.lower() or "✅" in text

    def test_empty_matched_still_generates_summary(self):
        sm = self._make_summary()
        text = generate_coaching_text(pd.DataFrame(), sm, language="en")
        assert "Distance" in text or "km" in text.lower()
