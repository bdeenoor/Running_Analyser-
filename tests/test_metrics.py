"""Tests for analysis/metrics.py — pure function tests with synthetic data."""

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.metrics import (
    format_pace,
    format_duration,
    pace_to_float,
    compute_km_splits,
    compute_summary_metrics,
    compute_moving_averages,
)
from parsers.gpx_parser import _haversine_vectorized, _riegel_projection


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_workout_points(
    n: int = 200,
    distance_km: float = 5.0,
    pace_min_km: float = 5.0,
    avg_hr: float = 150.0,
    start_time: datetime = None,
) -> pd.DataFrame:
    """Generate a synthetic workout_points DataFrame."""
    if start_time is None:
        start_time = datetime(2024, 1, 1, 7, 0, 0, tzinfo=timezone.utc)

    total_s = distance_km * pace_min_km * 60
    timestamps = pd.date_range(start_time, periods=n, freq=f"{total_s/n:.1f}s", tz="UTC")
    distances = np.linspace(0, distance_km * 1000, n)
    paces = np.full(n, pace_min_km) + np.random.normal(0, 0.05, n)
    paces = np.clip(paces, 1, 30)
    hrs = np.full(n, avg_hr) + np.random.normal(0, 3, n)
    hrs = np.clip(hrs, 40, 220)

    # Build lat/lon along a straight line (heading north from London)
    lat_start, lon_start = 51.5074, -0.1278
    lats = lat_start + np.linspace(0, distance_km / 111.0, n)
    lons = np.full(n, lon_start)
    elevs = np.linspace(0, 50, n) + np.random.normal(0, 0.5, n)

    time_deltas = np.full(n, total_s / n)
    time_deltas[0] = 0.0

    return pd.DataFrame({
        "timestamp": timestamps,
        "lat": lats,
        "lon": lons,
        "elevation_m": elevs,
        "hr": hrs,
        "cadence": np.full(n, 170.0),
        "distance_m": distances,
        "step_m": distances,
        "pace_min_km": paces,
        "smoothed_pace_min_km": paces,
        "is_moving": np.ones(n, dtype=bool),
        "is_spike": np.zeros(n, dtype=bool),
        "time_delta_s": time_deltas,
    })


# ── format_pace ───────────────────────────────────────────────────────────────

class TestFormatPace:
    def test_normal(self):
        assert format_pace(5.383) == "5:23"

    def test_exact_minutes(self):
        assert format_pace(5.0) == "5:00"

    def test_zero(self):
        result = format_pace(0.0)
        assert result == "--:--"

    def test_negative(self):
        assert format_pace(-1.0) == "--:--"

    def test_nan(self):
        assert format_pace(float("nan")) == "--:--"

    def test_none(self):
        assert format_pace(None) == "--:--"

    def test_over_30(self):
        assert format_pace(35.0) == "30:00+"

    def test_exactly_30(self):
        result = format_pace(30.0)
        assert result == "30:00+"

    def test_second_rollover(self):
        # 5:60 should become 6:00
        result = format_pace(5 + 59.5 / 60)
        assert result == "6:00"

    def test_string_input(self):
        # Should handle numeric strings
        result = format_pace("5.5")
        assert result == "5:30"


# ── format_duration ───────────────────────────────────────────────────────────

class TestFormatDuration:
    def test_under_hour(self):
        assert format_duration(3723) == "1:02:03"

    def test_minutes_only(self):
        assert format_duration(125) == "02:05"

    def test_zero(self):
        assert format_duration(0) == "00:00"

    def test_one_hour(self):
        assert format_duration(3600) == "1:00:00"

    def test_none(self):
        assert format_duration(None) == "--"


# ── pace_to_float ─────────────────────────────────────────────────────────────

class TestPaceToFloat:
    def test_normal(self):
        assert abs(pace_to_float("5:23") - (5 + 23/60)) < 0.001

    def test_zero_secs(self):
        assert abs(pace_to_float("5:00") - 5.0) < 0.001

    def test_invalid(self):
        assert pace_to_float("not a pace") is None

    def test_empty(self):
        assert pace_to_float("") is None

    def test_placeholder(self):
        assert pace_to_float("--:--") is None


# ── Haversine ─────────────────────────────────────────────────────────────────

class TestHaversine:
    def test_zero_distance(self):
        d = _haversine_vectorized(
            np.array([51.5]), np.array([-0.1]),
            np.array([51.5]), np.array([-0.1]),
        )
        assert float(d[0]) == pytest.approx(0.0, abs=1.0)

    def test_known_100m(self):
        # ~100m north at London latitude
        d = _haversine_vectorized(
            np.array([51.5074]), np.array([-0.1278]),
            np.array([51.5083]), np.array([-0.1278]),
        )
        assert 90 < float(d[0]) < 110

    def test_london_paris_approx(self):
        # London to Paris: ~340 km
        d = _haversine_vectorized(
            np.array([51.5074]), np.array([-0.1278]),
            np.array([48.8566]), np.array([2.3522]),
        )
        assert 330_000 < float(d[0]) < 350_000


# ── Riegel projections ────────────────────────────────────────────────────────

class TestRiegelProjection:
    def test_same_distance(self):
        t = _riegel_projection(10000, 3000, 10000)
        assert t == pytest.approx(3000, rel=0.001)

    def test_10k_from_5k(self):
        t = _riegel_projection(5000, 25 * 60, 10000)
        # Should be slightly more than double
        assert 49 * 60 < t < 55 * 60

    def test_marathon_reasonable(self):
        # A 4:00/km runner over 10km
        t = _riegel_projection(10000, 40 * 60, 42195)
        # Expect ~3h range
        assert 160 * 60 < t < 200 * 60

    def test_zero_distance(self):
        assert _riegel_projection(0, 3600, 10000) == 0.0


# ── km splits ─────────────────────────────────────────────────────────────────

class TestKmSplits:
    def test_basic_splits(self):
        wp = make_workout_points(n=300, distance_km=3.0, pace_min_km=5.0)
        splits = compute_km_splits(wp)
        # 3km workout → 3 splits (km bins 0, 1, 2 → labels 1, 2, 3)
        assert len(splits) >= 3
        assert splits["km"].min() >= 1

    def test_partial_last_km(self):
        wp = make_workout_points(n=250, distance_km=2.5, pace_min_km=5.0)
        splits = compute_km_splits(wp)
        assert len(splits) == 3  # km 0–1, 1–2, 2–2.5

    def test_hr_average(self):
        wp = make_workout_points(n=200, distance_km=2.0, avg_hr=155.0)
        splits = compute_km_splits(wp)
        for _, row in splits.iterrows():
            if row["avg_hr"] is not None:
                assert 130 < row["avg_hr"] < 180

    def test_elevation_gain_nonnegative(self):
        wp = make_workout_points(n=200, distance_km=2.0)
        splits = compute_km_splits(wp)
        for _, row in splits.iterrows():
            assert row["elevation_gain"] >= 0

    def test_empty_input(self):
        result = compute_km_splits(pd.DataFrame())
        assert result.empty

    def test_pace_reasonable(self):
        wp = make_workout_points(n=300, distance_km=5.0, pace_min_km=4.5)
        splits = compute_km_splits(wp)
        valid = splits.dropna(subset=["split_pace_min_km"])
        for _, row in valid.iterrows():
            assert 3.0 <= row["split_pace_min_km"] <= 8.0


# ── Summary metrics ───────────────────────────────────────────────────────────

class TestSummaryMetrics:
    def test_total_distance(self):
        wp = make_workout_points(n=200, distance_km=10.0)
        sm = compute_summary_metrics(wp)
        assert abs(sm["total_distance_m"] - 10_000) < 100

    def test_avg_pace_reasonable(self):
        wp = make_workout_points(n=200, distance_km=5.0, pace_min_km=5.0)
        sm = compute_summary_metrics(wp)
        assert 4.0 <= sm["avg_pace_min_km"] <= 7.0

    def test_hr_stats(self):
        wp = make_workout_points(n=200, distance_km=5.0, avg_hr=155.0)
        sm = compute_summary_metrics(wp)
        assert 140 < sm["avg_hr"] < 170
        assert sm["max_hr"] >= sm["avg_hr"]

    def test_race_projections_present(self):
        wp = make_workout_points(n=300, distance_km=10.0, pace_min_km=5.0)
        sm = compute_summary_metrics(wp)
        assert "race_projections" in sm
        assert "5K" in sm["race_projections"]
        assert "Marathon" in sm["race_projections"]

    def test_projections_reasonable(self):
        wp = make_workout_points(n=300, distance_km=10.0, pace_min_km=4.0)
        sm = compute_summary_metrics(wp)
        marathon_t = sm["race_projections"]["Marathon"]["time_s"]
        # 4:00/km runner should project 2h50–3h30 marathon
        assert 170 * 60 < marathon_t < 210 * 60

    def test_empty_returns_empty_dict(self):
        sm = compute_summary_metrics(pd.DataFrame())
        assert sm == {}

    def test_moving_time_less_than_total(self):
        wp = make_workout_points(n=200, distance_km=5.0)
        # Add some pauses
        wp.loc[50:60, "is_moving"] = False
        wp.loc[50:60, "time_delta_s"] = 35.0  # long gap
        sm = compute_summary_metrics(wp)
        assert sm["moving_time_s"] <= sm["total_time_s"]
