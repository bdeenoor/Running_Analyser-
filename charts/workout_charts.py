"""All chart generation: Plotly interactive charts + Matplotlib static dashboard."""

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.metrics import format_pace, format_duration

# ── Design system ─────────────────────────────────────────────────────────────

CHART_THEME = {
    "bg_color": "#0F1117",
    "paper_color": "#161B27",
    "grid_color": "#1E2130",
    "text_color": "#FAFAFA",
    "subtext_color": "#8B9EC3",
    "accent_1": "#00D4FF",   # pace — cyan
    "accent_2": "#FF6B6B",   # HR — coral
    "accent_3": "#FFD93D",   # planned — yellow
    "accent_4": "#6BCB77",   # actual — green
    "segment_colors": {
        "warmup": "#4ECDC4",
        "easy": "#95E1D3",
        "hard": "#F38181",
        "tempo": "#F9844A",
        "marathon_pace": "#9B5DE5",
        "cooldown": "#A8D8EA",
        "recovery": "#B5E7A0",
        "unknown": "#8B9EC3",
    },
    "confidence_colors": {
        "high": "#6BCB77",
        "medium": "#FFD93D",
        "low": "#FF6B6B",
    },
    "line_width": 2.5,
    "font_family": "Inter, Arial, sans-serif",
    "font_size_title": 16,
    "font_size_axis": 12,
}

MUTED_ALPHA = 0.3


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=CHART_THEME["paper_color"],
        plot_bgcolor=CHART_THEME["bg_color"],
        font=dict(family=CHART_THEME["font_family"], color=CHART_THEME["text_color"], size=12),
        xaxis=dict(
            gridcolor=CHART_THEME["grid_color"],
            linecolor=CHART_THEME["grid_color"],
            zerolinecolor=CHART_THEME["grid_color"],
        ),
        yaxis=dict(
            gridcolor=CHART_THEME["grid_color"],
            linecolor=CHART_THEME["grid_color"],
            zerolinecolor=CHART_THEME["grid_color"],
        ),
        legend=dict(
            bgcolor=CHART_THEME["paper_color"],
            bordercolor=CHART_THEME["grid_color"],
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def _format_pace_axis(fig: go.Figure, axis_name: str = "yaxis") -> go.Figure:
    """Configure a pace axis: inverted, mm:ss tick labels, proper range."""
    axis_config = dict(
        autorange="reversed",
        gridcolor=CHART_THEME["grid_color"],
        linecolor=CHART_THEME["grid_color"],
        title_font=dict(color=CHART_THEME["text_color"]),
        tickfont=dict(color=CHART_THEME["text_color"]),
    )
    fig.update_layout(**{axis_name: axis_config})
    return fig


def _elapsed_seconds_to_label(s: float) -> str:
    return format_duration(s)


def _build_x_axis(wp: pd.DataFrame, x_axis: str) -> tuple[pd.Series, str, str]:
    """Return (x_values, x_label, x_tickformat)."""
    if x_axis == "distance":
        x = wp["distance_m"] / 1000.0
        return x, "Distance (km)", ".1f"
    else:
        start = wp["timestamp"].min()
        elapsed = (wp["timestamp"] - start).dt.total_seconds()
        return elapsed, "Elapsed Time", ""


def _x_tickvals_time(elapsed_max_s: float, n: int = 10):
    step = max(60, int(elapsed_max_s / n / 60) * 60)
    vals = np.arange(0, elapsed_max_s + step, step)
    texts = [format_duration(v) for v in vals]
    return vals.tolist(), texts


def _add_segment_shading(
    fig: go.Figure,
    matched_segments: Optional[pd.DataFrame],
    x_axis: str = "time",
    wp: Optional[pd.DataFrame] = None,
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    if matched_segments is None or matched_segments.empty:
        return fig

    start_ts = wp["timestamp"].min() if wp is not None and not wp.empty else None
    t_offset = 0.0

    for _, seg in matched_segments.iterrows():
        seg_type = seg.get("type", "unknown")
        color = CHART_THEME["segment_colors"].get(seg_type, CHART_THEME["segment_colors"]["unknown"])
        dur = seg.get("planned_duration_s") or seg.get("actual_duration_s", 0)

        if x_axis == "distance" and seg.get("actual_distance_m"):
            x0 = seg.get("start_distance_km", t_offset / 60.0)
            x1 = x0 + seg["actual_distance_m"] / 1000.0
        else:
            x0 = t_offset
            x1 = t_offset + (dur or 0)

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=0, y1=1,
            xref=f"x{'' if col == 1 else col}",
            yref="paper",
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
            row=row, col=col,
        )

        # Segment label
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.97,
            xref=f"x{'' if col == 1 else col}",
            yref="paper",
            text=seg.get("name", seg_type)[:8],
            showarrow=False,
            font=dict(size=9, color=color),
            opacity=0.7,
        )

        t_offset += dur or 0

    return fig


# ── Chart 1: Pace over time/distance ──────────────────────────────────────────

def make_pace_chart(
    workout_points: pd.DataFrame,
    x_axis: str = "time",
    matched_segments: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    if workout_points is None or workout_points.empty:
        return fig

    x, xlabel, xfmt = _build_x_axis(workout_points, x_axis)

    # Raw pace (thin, muted)
    if "pace_min_km" in workout_points.columns:
        fig.add_trace(go.Scatter(
            x=x,
            y=workout_points["pace_min_km"],
            name="Raw Pace",
            line=dict(color=CHART_THEME["accent_1"], width=1),
            opacity=MUTED_ALPHA,
            mode="lines",
            hovertemplate=f"{xlabel}: %{{x}}<br>Pace: %{{y:.2f}} min/km<extra></extra>",
        ))

    # Smoothed pace (thick)
    if "smoothed_pace_min_km" in workout_points.columns:
        fig.add_trace(go.Scatter(
            x=x,
            y=workout_points["smoothed_pace_min_km"],
            name="Smoothed Pace",
            line=dict(color=CHART_THEME["accent_1"], width=CHART_THEME["line_width"]),
            mode="lines",
            hovertemplate=f"{xlabel}: %{{x}}<br>Pace: %{{y:.2f}} min/km<extra></extra>",
        ))

    _add_segment_shading(fig, matched_segments, x_axis, workout_points)
    fig = _apply_dark_theme(fig)
    fig = _format_pace_axis(fig, "yaxis")

    fig.update_layout(
        title=dict(text="Pace Over Time", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title=xlabel,
        yaxis_title="Pace (min/km)",
    )

    if x_axis == "time" and len(x) > 0:
        tvals, ttexts = _x_tickvals_time(float(x.max()))
        fig.update_xaxes(tickvals=tvals, ticktext=ttexts)

    return fig


# ── Chart 2: HR over time/distance ────────────────────────────────────────────

def make_hr_chart(
    workout_points: pd.DataFrame,
    x_axis: str = "time",
    matched_segments: Optional[pd.DataFrame] = None,
    max_hr_override: Optional[float] = None,
) -> go.Figure:
    fig = go.Figure()
    if workout_points is None or workout_points.empty or "hr" not in workout_points.columns:
        return fig

    x, xlabel, _ = _build_x_axis(workout_points, x_axis)
    hr = workout_points["hr"]

    fig.add_trace(go.Scatter(
        x=x,
        y=hr,
        name="Heart Rate",
        fill="tozeroy",
        fillcolor=f"rgba(255,107,107,0.15)",
        line=dict(color=CHART_THEME["accent_2"], width=CHART_THEME["line_width"]),
        mode="lines",
        hovertemplate=f"{xlabel}: %{{x}}<br>HR: %{{y:.0f}} bpm<extra></extra>",
    ))

    # HR zone lines
    max_hr = max_hr_override or (hr.max() if hr.notna().any() else 200)
    if max_hr and max_hr > 0:
        for pct, label, color in [
            (0.9, "Z5 90%", "#FF3333"),
            (0.8, "Z4 80%", "#FF8800"),
            (0.7, "Z3 70%", "#FFCC00"),
            (0.6, "Z2 60%", "#00CC88"),
        ]:
            fig.add_hline(
                y=max_hr * pct,
                line_dash="dot",
                line_color=color,
                opacity=0.5,
                annotation_text=f"{label} ({max_hr * pct:.0f})",
                annotation_font_size=10,
                annotation_font_color=color,
            )

    _add_segment_shading(fig, matched_segments, x_axis, workout_points)
    fig = _apply_dark_theme(fig)
    fig.update_layout(
        title=dict(text="Heart Rate Over Time", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title=xlabel,
        yaxis_title="Heart Rate (bpm)",
    )

    if x_axis == "time" and len(x) > 0:
        tvals, ttexts = _x_tickvals_time(float(x.max()))
        fig.update_xaxes(tickvals=tvals, ticktext=ttexts)

    return fig


# ── Chart 3: Pace + HR combined ───────────────────────────────────────────────

def make_pace_hr_combined(
    workout_points: pd.DataFrame,
    x_axis: str = "time",
    matched_segments: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if workout_points is None or workout_points.empty:
        return _apply_dark_theme(fig)

    x, xlabel, _ = _build_x_axis(workout_points, x_axis)

    # Smoothed pace on primary (inverted) axis
    if "smoothed_pace_min_km" in workout_points.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=workout_points["smoothed_pace_min_km"],
                name="Pace",
                line=dict(color=CHART_THEME["accent_1"], width=CHART_THEME["line_width"]),
                hovertemplate="Pace: %{y:.2f} min/km<extra></extra>",
            ),
            secondary_y=False,
        )

    # HR on secondary axis
    if "hr" in workout_points.columns and workout_points["hr"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=workout_points["hr"],
                name="Heart Rate",
                line=dict(color=CHART_THEME["accent_2"], width=CHART_THEME["line_width"]),
                hovertemplate="HR: %{y:.0f} bpm<extra></extra>",
            ),
            secondary_y=True,
        )

    _add_segment_shading(fig, matched_segments, x_axis, workout_points)
    fig = _apply_dark_theme(fig)

    fig.update_yaxes(
        title_text="Pace (min/km)",
        autorange="reversed",
        gridcolor=CHART_THEME["grid_color"],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Heart Rate (bpm)",
        gridcolor=CHART_THEME["grid_color"],
        secondary_y=True,
    )
    fig.update_layout(
        title=dict(text="Pace + Heart Rate", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title=xlabel,
    )

    if x_axis == "time" and len(x) > 0:
        tvals, ttexts = _x_tickvals_time(float(x.max()))
        fig.update_xaxes(tickvals=tvals, ticktext=ttexts)

    return fig


# ── Chart 4: Planned vs Actual pace ───────────────────────────────────────────

def make_planned_vs_actual(matched_segments: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if matched_segments is None or matched_segments.empty:
        return _apply_dark_theme(fig)

    seg_names = matched_segments["name"].tolist()
    # Use coach override pace when available, fall back to OCR/entered target
    if "effective_target_pace_min_km" in matched_segments.columns:
        planned = matched_segments["effective_target_pace_min_km"].tolist()
    else:
        planned = matched_segments["planned_pace_min_km"].tolist()
    actual = matched_segments["actual_pace_min_km"].tolist()
    deltas = matched_segments["delta_pace_sec_km"].tolist()
    confidences = matched_segments.get("confidence", ["medium"] * len(matched_segments)).tolist()

    # Planned bars
    fig.add_trace(go.Bar(
        name="Planned",
        x=seg_names,
        y=planned,
        marker_color=CHART_THEME["accent_3"],
        opacity=0.8,
        hovertemplate="Planned: %{y:.2f} min/km<extra></extra>",
    ))

    # Actual bars with confidence-colored borders
    conf_colors = [CHART_THEME["confidence_colors"].get(c, "#8B9EC3") for c in confidences]
    fig.add_trace(go.Bar(
        name="Actual",
        x=seg_names,
        y=actual,
        marker=dict(
            color=CHART_THEME["accent_4"],
            line=dict(color=conf_colors, width=2.5),
        ),
        opacity=0.9,
        hovertemplate="Actual: %{y:.2f} min/km<extra></extra>",
    ))

    # Delta annotations
    for i, (name, d) in enumerate(zip(seg_names, deltas)):
        if d is not None and math.isfinite(d):
            sign = "+" if d > 0 else ""
            color = CHART_THEME["accent_2"] if d > 5 else (CHART_THEME["accent_4"] if d < -5 else CHART_THEME["text_color"])
            fig.add_annotation(
                x=name,
                y=(actual[i] or 0) + 0.05,
                text=f"{sign}{d:.0f}s",
                showarrow=False,
                font=dict(color=color, size=10),
                xanchor="center",
            )

    fig = _apply_dark_theme(fig)
    fig = _format_pace_axis(fig, "yaxis")
    fig.update_layout(
        title=dict(text="Planned vs Actual Pace by Segment", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title="Segment",
        yaxis_title="Pace (min/km)",
        barmode="group",
        xaxis=dict(tickangle=-30),
    )
    return fig


# ── Chart 5: HR by segment ────────────────────────────────────────────────────

def make_hr_by_segment(matched_segments: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if matched_segments is None or matched_segments.empty:
        return _apply_dark_theme(fig)

    seg_names = matched_segments["name"].tolist()
    avg_hr = matched_segments["avg_hr"].tolist()
    max_hr = matched_segments["max_hr"].tolist()
    types = matched_segments.get("type", ["easy"] * len(matched_segments)).tolist()
    bar_colors = [CHART_THEME["segment_colors"].get(t, "#8B9EC3") for t in types]

    # Avg HR bars
    fig.add_trace(go.Bar(
        name="Avg HR",
        x=seg_names,
        y=avg_hr,
        marker_color=bar_colors,
        opacity=0.85,
        hovertemplate="Avg HR: %{y:.0f} bpm<extra></extra>",
    ))

    # Max HR as scatter overlay
    valid_max = [(n, v) for n, v in zip(seg_names, max_hr) if v is not None]
    if valid_max:
        names_v, vals_v = zip(*valid_max)
        fig.add_trace(go.Scatter(
            name="Max HR",
            x=list(names_v),
            y=list(vals_v),
            mode="markers",
            marker=dict(symbol="line-ew", size=14, color=CHART_THEME["accent_2"], line=dict(width=3, color=CHART_THEME["accent_2"])),
            hovertemplate="Max HR: %{y:.0f} bpm<extra></extra>",
        ))

    # Overall avg HR reference line
    all_hr = [v for v in avg_hr if v is not None]
    if all_hr:
        overall = np.mean(all_hr)
        fig.add_hline(
            y=overall,
            line_dash="dash",
            line_color=CHART_THEME["subtext_color"],
            annotation_text=f"Avg {overall:.0f} bpm",
            annotation_font_color=CHART_THEME["subtext_color"],
        )

    fig = _apply_dark_theme(fig)
    fig.update_layout(
        title=dict(text="Heart Rate by Segment", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title="Segment",
        yaxis_title="Heart Rate (bpm)",
        xaxis=dict(tickangle=-30),
    )
    return fig


# ── Chart 6: KM splits ────────────────────────────────────────────────────────

def make_km_splits_chart(km_splits: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if km_splits is None or km_splits.empty:
        return _apply_dark_theme(fig)

    kms = km_splits["km"].tolist()
    paces = km_splits["split_pace_min_km"].tolist()
    hrs = km_splits["avg_hr"].tolist() if "avg_hr" in km_splits.columns else []

    mean_pace = np.nanmean([p for p in paces if p is not None])
    bar_colors = []
    for p in paces:
        if p is None:
            bar_colors.append(CHART_THEME["subtext_color"])
        elif p < mean_pace - 0.1:
            bar_colors.append(CHART_THEME["accent_4"])   # faster = green
        elif p > mean_pace + 0.1:
            bar_colors.append(CHART_THEME["accent_2"])   # slower = red
        else:
            bar_colors.append(CHART_THEME["accent_1"])   # on target = cyan

    fig.add_trace(
        go.Bar(
            name="Pace",
            x=[f"km {k}" for k in kms],
            y=paces,
            marker_color=bar_colors,
            hovertemplate="km %{x}: %{y:.2f} min/km<extra></extra>",
        ),
        secondary_y=False,
    )

    valid_hr = [(k, h) for k, h in zip(kms, hrs) if h is not None]
    if valid_hr:
        ks_v, hs_v = zip(*valid_hr)
        fig.add_trace(
            go.Scatter(
                name="Avg HR",
                x=[f"km {k}" for k in ks_v],
                y=list(hs_v),
                line=dict(color=CHART_THEME["accent_2"], width=CHART_THEME["line_width"]),
                mode="lines+markers",
                hovertemplate="km %{x} HR: %{y:.0f} bpm<extra></extra>",
            ),
            secondary_y=True,
        )

    fig = _apply_dark_theme(fig)
    fig.update_yaxes(title_text="Pace (min/km)", autorange="reversed", secondary_y=False)
    fig.update_yaxes(title_text="Heart Rate (bpm)", secondary_y=True)
    fig.update_layout(
        title=dict(text="Pace per Kilometer", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title="Kilometer",
    )
    return fig


# ── Chart 7: Pace-HR scatter ──────────────────────────────────────────────────

def make_pace_hr_scatter(workout_points: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if workout_points is None or workout_points.empty:
        return _apply_dark_theme(fig)

    cols_needed = ["hr", "smoothed_pace_min_km", "timestamp"]
    if not all(c in workout_points.columns for c in cols_needed):
        return _apply_dark_theme(fig)

    valid = workout_points[
        workout_points["hr"].notna() & workout_points["smoothed_pace_min_km"].notna()
    ].copy()
    if valid.empty:
        return _apply_dark_theme(fig)

    start = valid["timestamp"].min()
    elapsed = (valid["timestamp"] - start).dt.total_seconds()

    fig.add_trace(go.Scatter(
        x=valid["hr"],
        y=valid["smoothed_pace_min_km"],
        mode="markers",
        marker=dict(
            color=elapsed,
            colorscale="Plasma",
            size=4,
            opacity=0.7,
            colorbar=dict(title="Elapsed (s)", tickfont=dict(color=CHART_THEME["text_color"])),
        ),
        hovertemplate="HR: %{x:.0f} bpm<br>Pace: %{y:.2f} min/km<extra></extra>",
        name="Data Points",
    ))

    # Regression line
    hr_vals = valid["hr"].values
    pace_vals = valid["smoothed_pace_min_km"].values
    if len(hr_vals) > 10:
        coeffs = np.polyfit(hr_vals, pace_vals, 1)
        x_line = np.linspace(hr_vals.min(), hr_vals.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color=CHART_THEME["accent_3"], width=2, dash="dash"),
        ))

    fig = _apply_dark_theme(fig)
    fig = _format_pace_axis(fig, "yaxis")
    fig.update_layout(
        title=dict(text="Running Efficiency: Pace vs Heart Rate", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title="Heart Rate (bpm)",
        yaxis_title="Pace (min/km)",
    )
    return fig


# ── Chart 8: Interval performance ────────────────────────────────────────────

def make_interval_performance(matched_segments: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if matched_segments is None or matched_segments.empty:
        return _apply_dark_theme(fig)

    hard_segs = matched_segments[matched_segments["type"].isin(["hard", "tempo", "marathon_pace"])].copy()
    if hard_segs.empty:
        return _apply_dark_theme(fig)

    hard_segs = hard_segs.reset_index(drop=True)
    x = list(range(1, len(hard_segs) + 1))
    actual_paces = hard_segs["actual_pace_min_km"].tolist()
    planned_paces = hard_segs["planned_pace_min_km"].tolist()
    names = hard_segs["name"].tolist()

    # Actual paces line
    fig.add_trace(go.Scatter(
        x=x,
        y=actual_paces,
        mode="lines+markers",
        name="Actual Pace",
        line=dict(color=CHART_THEME["accent_4"], width=CHART_THEME["line_width"]),
        marker=dict(size=10, symbol="circle"),
        hovertemplate="Rep %{x}: %{y:.2f} min/km<extra></extra>",
        text=names,
    ))

    # Planned pace as dashed per-segment lines
    valid_planned = [(i+1, p) for i, p in enumerate(planned_paces) if p is not None]
    if valid_planned:
        xs_p, ys_p = zip(*valid_planned)
        fig.add_trace(go.Scatter(
            x=list(xs_p),
            y=list(ys_p),
            mode="markers",
            name="Planned Pace",
            marker=dict(size=12, symbol="line-ew", color=CHART_THEME["accent_3"],
                        line=dict(width=3, color=CHART_THEME["accent_3"])),
            hovertemplate="Planned Rep %{x}: %{y:.2f} min/km<extra></extra>",
        ))
        # Horizontal planned line across all reps
        first_planned = [p for p in planned_paces if p is not None]
        if first_planned and len(set(first_planned)) == 1:
            fig.add_hline(
                y=first_planned[0],
                line_dash="dash",
                line_color=CHART_THEME["accent_3"],
                opacity=0.5,
                annotation_text=f"Target {format_pace(first_planned[0])}",
            )

    fig = _apply_dark_theme(fig)
    fig = _format_pace_axis(fig, "yaxis")
    fig.update_layout(
        title=dict(text="Interval / Hard Segment Performance", font=dict(size=CHART_THEME["font_size_title"])),
        xaxis_title="Repetition",
        yaxis_title="Pace (min/km)",
        xaxis=dict(dtick=1),
    )
    return fig


# ── Summary dashboard (Matplotlib, 300 DPI) ────────────────────────────────────

def make_summary_dashboard(
    workout_points: pd.DataFrame,
    matched_segments: Optional[pd.DataFrame],
    summary_metrics: dict,
    km_splits: Optional[pd.DataFrame],
) -> plt.Figure:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), facecolor=CHART_THEME["bg_color"])
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.07, right=0.97,
        top=0.92, bottom=0.07,
    )

    _CYAN = CHART_THEME["accent_1"]
    _RED = CHART_THEME["accent_2"]
    _YELLOW = CHART_THEME["accent_3"]
    _GREEN = CHART_THEME["accent_4"]
    _BG = CHART_THEME["bg_color"]
    _GRID = CHART_THEME["grid_color"]

    def _style_ax(ax, title: str):
        ax.set_facecolor(_BG)
        ax.set_title(title, color=CHART_THEME["text_color"], fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=CHART_THEME["subtext_color"], labelsize=8)
        ax.spines["bottom"].set_color(_GRID)
        ax.spines["left"].set_color(_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color(CHART_THEME["subtext_color"])
        ax.yaxis.label.set_color(CHART_THEME["subtext_color"])

    # ── 1: Pace timeline ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _style_ax(ax1, "Pace Over Time")
    if workout_points is not None and not workout_points.empty:
        start = workout_points["timestamp"].min()
        elapsed = (workout_points["timestamp"] - start).dt.total_seconds().values / 60.0
        if "pace_min_km" in workout_points.columns:
            ax1.plot(elapsed, workout_points["pace_min_km"].values, color=_CYAN, alpha=0.25, lw=0.8)
        if "smoothed_pace_min_km" in workout_points.columns:
            ax1.plot(elapsed, workout_points["smoothed_pace_min_km"].values, color=_CYAN, lw=2)
        ax1.set_ylabel("Pace (min/km)")
        ax1.set_xlabel("Time (min)")
        ax1.invert_yaxis()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format_pace(v)))
        ax1.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.5)

    # ── 2: HR timeline ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    _style_ax(ax2, "Heart Rate Over Time")
    if workout_points is not None and not workout_points.empty and "hr" in workout_points.columns:
        start = workout_points["timestamp"].min()
        elapsed = (workout_points["timestamp"] - start).dt.total_seconds().values / 60.0
        hr_vals = workout_points["hr"].values
        ax2.fill_between(elapsed, hr_vals, alpha=0.3, color=_RED)
        ax2.plot(elapsed, hr_vals, color=_RED, lw=1.5)
        ax2.set_ylabel("HR (bpm)")
        ax2.set_xlabel("Time (min)")
        ax2.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.5)

    # ── 3: KM splits ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    _style_ax(ax3, "Pace per Kilometer")
    if km_splits is not None and not km_splits.empty:
        kms = km_splits["km"].values
        paces = km_splits["split_pace_min_km"].values
        mean_p = np.nanmean(paces)
        colors = [_GREEN if p < mean_p - 0.1 else (_RED if p > mean_p + 0.1 else _CYAN) for p in paces]
        ax3.bar([f"km {k}" for k in kms], paces, color=colors)
        ax3.invert_yaxis()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format_pace(v)))
        ax3.set_ylabel("Pace (min/km)")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.5, axis="y")

    # ── 4: Summary metrics text ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_facecolor(CHART_THEME["paper_color"])
    ax4.axis("off")
    ax4.set_title("Summary", color=CHART_THEME["text_color"], fontsize=11, fontweight="bold")

    sm = summary_metrics or {}
    lines_text = [
        ("Distance", f"{sm.get('total_distance_m', 0)/1000:.2f} km"),
        ("Moving Time", format_duration(sm.get("moving_time_s", 0))),
        ("Avg Pace", format_pace(sm.get("avg_pace_min_km"))),
        ("Avg HR", f"{sm.get('avg_hr', 0):.0f} bpm" if sm.get("avg_hr") else "—"),
        ("Max HR", f"{sm.get('max_hr', 0):.0f} bpm" if sm.get("max_hr") else "—"),
        ("Elev Gain", f"{sm.get('elevation_gain_m', 0):.0f} m"),
    ]
    for j, (label, val) in enumerate(lines_text):
        y_pos = 0.88 - j * 0.14
        ax4.text(0.05, y_pos, label + ":", transform=ax4.transAxes,
                 color=CHART_THEME["subtext_color"], fontsize=10, ha="left")
        ax4.text(0.95, y_pos, val, transform=ax4.transAxes,
                 color=CHART_THEME["text_color"], fontsize=10, fontweight="bold", ha="right")

    # ── 5: Planned vs Actual ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style_ax(ax5, "Planned vs Actual")
    if matched_segments is not None and not matched_segments.empty:
        names = [n[:10] for n in matched_segments["name"].tolist()]
        planned = matched_segments["planned_pace_min_km"].tolist()
        actual = matched_segments["actual_pace_min_km"].tolist()
        x_pos = np.arange(len(names))
        w = 0.35
        ax5.bar(x_pos - w/2, planned, w, color=_YELLOW, alpha=0.8, label="Planned")
        ax5.bar(x_pos + w/2, actual, w, color=_GREEN, alpha=0.8, label="Actual")
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(names, rotation=45, fontsize=7)
        ax5.invert_yaxis()
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format_pace(v)))
        ax5.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID)
        ax5.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.5, axis="y")

    # ── 6: Race projections ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(CHART_THEME["paper_color"])
    ax6.axis("off")
    ax6.set_title("Race Projections", color=CHART_THEME["text_color"], fontsize=11, fontweight="bold")

    projections = sm.get("race_projections", {})
    proj_lines = list(projections.items())
    for j, (race, data) in enumerate(proj_lines):
        y_pos = 0.88 - j * 0.22
        t = data.get("time_s", 0)
        p = data.get("pace_min_km")
        ax6.text(0.05, y_pos, race, transform=ax6.transAxes,
                 color=CHART_THEME["subtext_color"], fontsize=10)
        ax6.text(0.95, y_pos, f"{format_duration(t)}", transform=ax6.transAxes,
                 color=_CYAN, fontsize=10, fontweight="bold", ha="right")

    # Title
    fig.suptitle(
        "Running Workout Analysis Dashboard",
        color=CHART_THEME["text_color"],
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    return fig
