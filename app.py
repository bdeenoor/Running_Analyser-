"""Running Workout Analyzer — main Streamlit application."""

import io
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Running Workout Analyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state initialization ──────────────────────────────────────────────

INITIAL_STATE = {
    "workout_points": None,
    "laps": None,
    "km_splits": None,
    "planned_segments": None,
    "matched_segments": None,
    "summary_metrics": {},
    "source": None,
    "ocr_raw_text": {},
    "validation_warnings": [],
    "chart_x_axis": "time",
    "coaching_language": "he",
    "analysis_done": False,
    "ocr_confidence_notes": [],
    "screenshots_status": None,
    "last_screenshots_hash": None,
}

for key, default in INITIAL_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .stApp { background-color: #0F1117; color: #FAFAFA; }
    .metric-card {
        background: #161B27;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #1E2130;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00D4FF; }
    .metric-label { font-size: 0.85rem; color: #8B9EC3; margin-top: 4px; }
    .confidence-high { color: #6BCB77; font-weight: 600; }
    .confidence-medium { color: #FFD93D; font-weight: 600; }
    .confidence-low { color: #FF6B6B; font-weight: 600; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #1E2130;
    }
    .warning-box {
        background: #2A1F0F;
        border-left: 4px solid #FFD93D;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 6px 0;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Imports (lazy to keep startup fast) ───────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_parse_gpx(file_bytes: bytes) -> dict:
    from parsers.gpx_parser import parse_gpx
    return parse_gpx(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _cached_parse_screenshots(files_bytes: list[bytes]) -> dict:
    from parsers.ocr_parser import parse_screenshots
    return parse_screenshots([io.BytesIO(b) for b in files_bytes])


# ── Analysis orchestration ────────────────────────────────────────────────────

def run_full_analysis():
    """Recompute everything downstream from current session state."""
    from analysis.metrics import compute_km_splits, compute_summary_metrics, run_validation
    from analysis.segments import match_segments, generate_coaching_text

    wp = st.session_state.get("workout_points")
    laps = st.session_state.get("laps")
    planned = st.session_state.get("planned_segments")
    sm = st.session_state.get("summary_metrics", {})

    if wp is not None and not wp.empty:
        st.session_state["km_splits"] = compute_km_splits(wp)
        fresh_sm = compute_summary_metrics(wp, laps)
        # Merge OCR-extracted summary values
        ocr_sm = {k: v for k, v in sm.items() if k.endswith("_ocr")}
        fresh_sm.update(ocr_sm)
        st.session_state["summary_metrics"] = fresh_sm
        sm = fresh_sm

    if planned is not None and not planned.empty:
        matched = match_segments(
            planned, laps, wp, strategy="auto"
        )
        st.session_state["matched_segments"] = matched
    else:
        st.session_state["matched_segments"] = None

    st.session_state["validation_warnings"] = run_validation(st.session_state)

    matched = st.session_state.get("matched_segments")
    lang = st.session_state.get("coaching_language", "he")
    if sm:
        text = generate_coaching_text(
            matched if matched is not None else pd.DataFrame(),
            sm,
            language=lang,
        )
        st.session_state["coaching_text"] = text

    st.session_state["analysis_done"] = True


# ── Helper: metric card HTML ──────────────────────────────────────────────────

def _metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def _confidence_badge(conf: str) -> str:
    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
    return f'<span class="confidence-{conf}">{icon} {conf.capitalize()}</span>'


# ── Format helpers ────────────────────────────────────────────────────────────

def _fmt_pace(v) -> str:
    from analysis.metrics import format_pace
    return format_pace(v)


def _fmt_dur(v) -> str:
    from analysis.metrics import format_duration
    return format_duration(v)


# ── App header ────────────────────────────────────────────────────────────────

st.markdown("## 🏃 Running Workout Analyzer")
st.markdown("*Professional coach-quality insights from GPX files and watch screenshots*")

if st.session_state.get("validation_warnings"):
    for w in st.session_state["validation_warnings"]:
        st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload",
    "📋 Data Review",
    "🔗 Segment Alignment",
    "📊 Charts",
    "🧑‍💼 Coach Summary",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">Upload Workout Data</div>', unsafe_allow_html=True)

    col_gpx, col_img = st.columns(2)

    with col_gpx:
        st.markdown("#### GPX File")
        gpx_file = st.file_uploader(
            "Upload a GPX file from your watch or app",
            type=["gpx"],
            key="gpx_uploader",
        )

        if gpx_file is not None:
            with st.spinner("Parsing GPX file…"):
                try:
                    result = _cached_parse_gpx(gpx_file.read())
                    st.session_state["workout_points"] = result["workout_points"]
                    if result.get("laps") is not None and not result["laps"].empty:
                        st.session_state["laps"] = result["laps"]
                    sm_gpx = result.get("summary_metrics", {})
                    st.session_state["summary_metrics"].update(sm_gpx)
                    st.session_state["source"] = "gpx"
                    st.success(f"✅ Parsed {len(result['workout_points'])} GPS points.")
                    run_full_analysis()
                except ValueError as e:
                    st.error(f"❌ GPX parse error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    with col_img:
        st.markdown("#### Screenshots")

        # Show persistent status banner from previous upload (survives rerun)
        scr_status = st.session_state.get("screenshots_status")
        if scr_status:
            if scr_status.get("ok") is True:
                n_seg = scr_status.get("segments", 0)
                n_lap = scr_status.get("laps", 0)
                parts = []
                if n_seg:
                    parts.append(f"{n_seg} segment{'s' if n_seg != 1 else ''}")
                if n_lap:
                    parts.append(f"{n_lap} lap{'s' if n_lap != 1 else ''}")
                if parts:
                    st.success(f"✅ Extracted {', '.join(parts)} — review and edit in the **Data Review** tab →")
                else:
                    st.warning("⚠️ Screenshots processed but no data extracted. Check image quality or enter data manually in **Data Review**.")
            elif scr_status.get("reason") == "ocr_unavailable":
                detail = scr_status.get("backend_error", "")
                msg = (
                    "ℹ️ OCR is not available on this deployment. "
                    "Please enter your workout plan and lap data manually in the **Data Review** tab →"
                )
                if detail:
                    msg += f"\n\n_(Technical detail: {detail})_"
                st.info(msg)
            else:
                st.error(f"❌ OCR error: {scr_status.get('msg', 'unknown error')}")

        img_files = st.file_uploader(
            "Upload watch/app screenshots (plan, laps, summary, HR, pace)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="img_uploader",
        )

        if img_files:
            # Hash guard — only process if files changed (prevents rerun loop)
            files_hash = tuple(f.name + str(f.size) for f in img_files)
            if st.session_state.get("last_screenshots_hash") != files_hash:
                with st.spinner("Running OCR on screenshots…"):
                    try:
                        files_bytes = [f.read() for f in img_files]
                        ocr_result = _cached_parse_screenshots(files_bytes)

                        if not ocr_result.get("ocr_available", True):
                            st.session_state["screenshots_status"] = {
                                "ok": None,
                                "reason": "ocr_unavailable",
                                "backend_error": ocr_result.get("backend_error", ""),
                            }
                        else:
                            n_seg, n_lap = 0, 0
                            if ocr_result.get("planned_segments") is not None:
                                st.session_state["planned_segments"] = ocr_result["planned_segments"]
                                n_seg = len(ocr_result["planned_segments"])
                            if ocr_result.get("laps") is not None:
                                if st.session_state.get("laps") is None:
                                    st.session_state["laps"] = ocr_result["laps"]
                                n_lap = len(ocr_result["laps"])
                            if ocr_result.get("summary_metrics"):
                                st.session_state["summary_metrics"].update(ocr_result["summary_metrics"])

                            st.session_state["ocr_raw_text"] = ocr_result.get("raw_extractions", {})
                            st.session_state["ocr_confidence_notes"] = ocr_result.get("confidence_notes", [])
                            st.session_state["source"] = "both" if st.session_state.get("source") == "gpx" else "ocr"
                            st.session_state["screenshots_status"] = {"ok": True, "segments": n_seg, "laps": n_lap}
                            run_full_analysis()

                        st.session_state["last_screenshots_hash"] = files_hash
                        st.rerun()

                    except Exception as e:
                        st.session_state["screenshots_status"] = {"ok": False, "reason": "error", "msg": str(e)}
                        st.session_state["last_screenshots_hash"] = files_hash
                        st.rerun()

    # ── Quick stats after upload ──────────────────────────────────────────────
    sm = st.session_state.get("summary_metrics", {})
    if sm:
        st.markdown("---")
        st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        stats = [
            ("Distance", f"{sm.get('total_distance_m', 0)/1000:.2f} km"),
            ("Moving Time", _fmt_dur(sm.get("moving_time_s"))),
            ("Avg Pace", _fmt_pace(sm.get("avg_pace_min_km"))),
            ("Avg HR", f"{sm.get('avg_hr', 0):.0f} bpm" if sm.get("avg_hr") else "—"),
            ("Max HR", f"{sm.get('max_hr', 0):.0f} bpm" if sm.get("max_hr") else "—"),
            ("Elev Gain", f"{sm.get('elevation_gain_m', 0):.0f} m"),
        ]
        for col, (label, value) in zip(cols, stats):
            with col:
                st.markdown(_metric_card(label, value), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: DATA REVIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Review & Edit Extracted Data</div>', unsafe_allow_html=True)
    st.caption("All data extracted from GPX and OCR is shown below. Edit any values before running analysis.")

    # ── GPS Points preview ────────────────────────────────────────────────────
    with st.expander("📍 GPS Track Points (preview)", expanded=False):
        wp = st.session_state.get("workout_points")
        if wp is not None and not wp.empty:
            display_cols = ["timestamp", "distance_m", "pace_min_km", "smoothed_pace_min_km", "hr", "elevation_m"]
            display_cols = [c for c in display_cols if c in wp.columns]
            preview = wp[display_cols].head(100).copy()
            if "pace_min_km" in preview:
                preview["pace_min_km"] = preview["pace_min_km"].apply(_fmt_pace)
            if "smoothed_pace_min_km" in preview:
                preview["smoothed_pace_min_km"] = preview["smoothed_pace_min_km"].apply(_fmt_pace)
            st.dataframe(preview, use_container_width=True, height=300)
            st.caption(f"Showing first 100 of {len(wp)} points.")
        else:
            st.info("No GPS data loaded yet. Upload a GPX file in the Upload tab.")

    # ── Lap data ──────────────────────────────────────────────────────────────
    with st.expander("🏁 Lap Data (editable)", expanded=True):
        laps = st.session_state.get("laps")
        if laps is not None and not laps.empty:
            edit_cols = ["lap_num", "duration_s", "distance_m", "avg_pace_min_km", "avg_hr", "max_hr", "elevation_gain"]
            edit_cols = [c for c in edit_cols if c in laps.columns]
            edited_laps = st.data_editor(
                laps[edit_cols],
                num_rows="dynamic",
                use_container_width=True,
                key="laps_editor",
                column_config={
                    "lap_num": st.column_config.NumberColumn("Lap #", min_value=1),
                    "duration_s": st.column_config.NumberColumn("Duration (s)", min_value=0),
                    "distance_m": st.column_config.NumberColumn("Distance (m)", min_value=0),
                    "avg_pace_min_km": st.column_config.NumberColumn("Avg Pace (min/km)", min_value=0, max_value=30, format="%.2f"),
                    "avg_hr": st.column_config.NumberColumn("Avg HR (bpm)", min_value=40, max_value=220),
                    "max_hr": st.column_config.NumberColumn("Max HR (bpm)", min_value=40, max_value=220),
                    "elevation_gain": st.column_config.NumberColumn("Elev Gain (m)"),
                },
            )
            if st.button("💾 Save lap edits", key="save_laps"):
                updated = laps.copy()
                for col in edit_cols:
                    if col in edited_laps.columns:
                        updated[col] = edited_laps[col].values
                st.session_state["laps"] = updated
                run_full_analysis()
                st.success("Laps updated.")
        else:
            st.info("No lap data. Upload a GPX with lap metadata or screenshots with a lap table.")

    # ── Planned segments ──────────────────────────────────────────────────────
    with st.expander("📋 Planned Workout Segments (editable)", expanded=True):
        st.markdown("Enter or edit your planned workout. Each row = one segment.")

        planned = st.session_state.get("planned_segments")
        if planned is None or planned.empty:
            planned = pd.DataFrame(columns=["segment_num", "name", "type", "duration_s", "target_pace_min_km"])

        # Ensure all required columns present
        for col, default in [("segment_num", 1), ("name", ""), ("type", "easy"),
                              ("duration_s", 300), ("target_pace_min_km", None),
                              ("coach_target_pace_min_km", None), ("coach_notes", "")]:
            if col not in planned.columns:
                planned[col] = default

        edited_plan = st.data_editor(
            planned,
            num_rows="dynamic",
            use_container_width=True,
            key="plan_editor",
            column_config={
                "segment_num": st.column_config.NumberColumn("Seg #", min_value=1),
                "name": st.column_config.TextColumn("Name"),
                "type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["warmup", "easy", "hard", "tempo", "marathon_pace", "cooldown", "recovery"],
                ),
                "duration_s": st.column_config.NumberColumn("Duration (s)", min_value=1),
                "target_pace_min_km": st.column_config.NumberColumn(
                    "OCR Pace (min/km)", min_value=2.0, max_value=30.0, format="%.2f",
                    help="Target pace extracted from screenshot. Edit below to override with coach instruction."
                ),
                "coach_target_pace_min_km": st.column_config.NumberColumn(
                    "Coach Pace Override (min/km)", min_value=2.0, max_value=30.0, format="%.2f",
                    help="Leave blank to use OCR pace. Set this when your coach gave a different pace target."
                ),
                "coach_notes": st.column_config.TextColumn(
                    "Coach Notes",
                    help="Optional coach instructions for this segment (e.g. 'stay relaxed', 'negative split').",
                    max_chars=200,
                ),
            },
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Save plan & re-analyse", key="save_plan"):
                edited_plan["segment_num"] = range(1, len(edited_plan) + 1)
                st.session_state["planned_segments"] = edited_plan
                run_full_analysis()
                st.success("Plan saved and analysis updated.")

        with col_btn2:
            with st.expander("⚡ Quick-load pyramid template"):
                if st.button("Load 2-4-6-8-6-4-2 min pyramid"):
                    pyramid_rows = [
                        (1, "Warm Up", "warmup", 1200, None),
                        (2, "Hard 2min", "hard", 120, 4.5),
                        (3, "Easy 2min", "easy", 120, None),
                        (4, "Hard 4min", "hard", 240, 4.5),
                        (5, "Easy 2min", "easy", 120, None),
                        (6, "Hard 6min", "hard", 360, 4.5),
                        (7, "Easy 2min", "easy", 120, None),
                        (8, "Hard 8min", "hard", 480, 4.5),
                        (9, "Easy 2min", "easy", 120, None),
                        (10, "Hard 6min", "hard", 360, 4.5),
                        (11, "Easy 2min", "easy", 120, None),
                        (12, "Hard 4min", "hard", 240, 4.5),
                        (13, "Easy 2min", "easy", 120, None),
                        (14, "Hard 2min", "hard", 120, 4.5),
                        (15, "Easy 2min", "easy", 120, None),
                        (16, "Cool Down", "cooldown", 600, None),
                    ]
                    df_pyr = pd.DataFrame(pyramid_rows, columns=["segment_num", "name", "type", "duration_s", "target_pace_min_km"])
                    st.session_state["planned_segments"] = df_pyr
                    run_full_analysis()
                    st.success("Pyramid plan loaded.")
                    st.rerun()

    # ── OCR raw extractions ───────────────────────────────────────────────────
    ocr_raw = st.session_state.get("ocr_raw_text", {})
    if ocr_raw:
        with st.expander("🔍 OCR Raw Extractions (for debugging)", expanded=False):
            for img_name, items in ocr_raw.items():
                st.markdown(f"**{img_name}**")
                df_raw = pd.DataFrame(items)
                st.dataframe(df_raw, use_container_width=True, height=200)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: SEGMENT ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Segment Alignment</div>', unsafe_allow_html=True)
    st.caption(
        "Review how planned segments were matched to actual performance. "
        "Confidence: 🟢 High (lap-aligned) · 🟡 Medium (timestamp) · 🔴 Low (needs review)"
    )

    matched = st.session_state.get("matched_segments")

    if matched is None or matched.empty:
        if st.session_state.get("planned_segments") is not None:
            st.warning("No segment alignment yet. Add lap data or ensure workout_points are loaded.")
        else:
            st.info("Add a workout plan in the Data Review tab to see segment alignment.")
    else:
        # ── Styled alignment table ────────────────────────────────────────────
        display = matched.copy()
        display["Planned Pace"] = display["planned_pace_min_km"].apply(_fmt_pace)
        display["Actual Pace"] = display["actual_pace_min_km"].apply(_fmt_pace)
        display["Delta (s/km)"] = display["delta_pace_sec_km"].apply(
            lambda x: f"{x:+.0f}" if x is not None and math.isfinite(x) else "—"
        )
        display["Duration Plan"] = display["planned_duration_s"].apply(_fmt_dur)
        display["Duration Actual"] = display["actual_duration_s"].apply(_fmt_dur)
        display["Avg HR"] = display["avg_hr"].apply(lambda x: f"{x:.0f}" if x else "—")
        display["Max HR"] = display["max_hr"].apply(lambda x: f"{x:.0f}" if x else "—")

        show_cols = ["segment_num", "name", "type", "Duration Plan", "Duration Actual",
                     "Planned Pace", "Actual Pace", "Delta (s/km)", "Avg HR", "Max HR", "confidence"]

        # Color-code confidence
        def _style_confidence(row):
            colors = {"high": "background-color: rgba(107,203,119,0.15)",
                      "medium": "background-color: rgba(255,217,61,0.10)",
                      "low": "background-color: rgba(255,107,107,0.15)"}
            conf = row.get("confidence", "medium")
            color = colors.get(conf, "")
            return [color] * len(row)

        styled = display[show_cols].style.apply(_style_confidence, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

        # ── Confidence summary ────────────────────────────────────────────────
        conf_counts = matched["confidence"].value_counts()
        col_h, col_m, col_l = st.columns(3)
        with col_h:
            st.markdown(_metric_card("🟢 High Confidence", str(conf_counts.get("high", 0))), unsafe_allow_html=True)
        with col_m:
            st.markdown(_metric_card("🟡 Medium Confidence", str(conf_counts.get("medium", 0))), unsafe_allow_html=True)
        with col_l:
            st.markdown(_metric_card("🔴 Low Confidence", str(conf_counts.get("low", 0))), unsafe_allow_html=True)

        st.markdown("---")

        # ── Manual override ───────────────────────────────────────────────────
        with st.expander("✏️ Manually override segment boundaries"):
            st.caption("Override matching strategy and re-run analysis.")
            strategy = st.selectbox(
                "Matching strategy",
                ["auto", "lap_boundary", "timestamp"],
                key="strategy_select",
            )
            if st.button("🔄 Re-run with selected strategy"):
                from analysis.segments import match_segments
                new_matched = match_segments(
                    st.session_state.get("planned_segments"),
                    st.session_state.get("laps"),
                    st.session_state.get("workout_points"),
                    strategy=strategy,
                )
                st.session_state["matched_segments"] = new_matched
                run_full_analysis()
                st.success(f"Re-matched using strategy: {strategy}")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: CHARTS
# ════════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">Workout Charts</div>', unsafe_allow_html=True)

    wp = st.session_state.get("workout_points")
    km_splits = st.session_state.get("km_splits")
    matched = st.session_state.get("matched_segments")
    sm = st.session_state.get("summary_metrics", {})

    if wp is None or wp.empty:
        st.info("Upload a GPX file to see charts.")
    else:
        from charts.workout_charts import (
            make_pace_chart, make_hr_chart, make_pace_hr_combined,
            make_planned_vs_actual, make_hr_by_segment,
            make_km_splits_chart, make_pace_hr_scatter, make_interval_performance,
        )

        # Controls
        col_ctrl1, col_ctrl2 = st.columns([2, 4])
        with col_ctrl1:
            x_axis = st.radio("X-axis", ["time", "distance"], horizontal=True, key="x_axis_radio")
            st.session_state["chart_x_axis"] = x_axis

        # ── Chart 1 + 2: Pace and HR side by side ────────────────────────────
        st.markdown("#### Pace & Heart Rate")
        col_p, col_hr = st.columns(2)
        with col_p:
            fig_pace = make_pace_chart(wp, x_axis=x_axis, matched_segments=matched)
            st.plotly_chart(fig_pace, use_container_width=True)
        with col_hr:
            fig_hr = make_hr_chart(
                wp, x_axis=x_axis, matched_segments=matched,
                max_hr_override=sm.get("max_hr"),
            )
            st.plotly_chart(fig_hr, use_container_width=True)

        # ── Chart 3: Combined ─────────────────────────────────────────────────
        st.markdown("#### Pace + HR Combined")
        fig_combined = make_pace_hr_combined(wp, x_axis=x_axis, matched_segments=matched)
        st.plotly_chart(fig_combined, use_container_width=True)

        # ── Chart 4 + 5: Segment analysis ────────────────────────────────────
        if matched is not None and not matched.empty:
            st.markdown("#### Segment Analysis")
            col_pva, col_hr_seg = st.columns(2)
            with col_pva:
                fig_pva = make_planned_vs_actual(matched)
                st.plotly_chart(fig_pva, use_container_width=True)
            with col_hr_seg:
                fig_hr_seg = make_hr_by_segment(matched)
                st.plotly_chart(fig_hr_seg, use_container_width=True)

            # Interval performance (only if hard segments exist)
            hard_count = (matched["type"].isin(["hard", "tempo", "marathon_pace"])).sum()
            if hard_count >= 2:
                st.markdown("#### Interval Performance")
                fig_interval = make_interval_performance(matched)
                st.plotly_chart(fig_interval, use_container_width=True)

        # ── Chart 6: KM Splits ────────────────────────────────────────────────
        st.markdown("#### Kilometer Splits")
        if km_splits is not None and not km_splits.empty:
            fig_km = make_km_splits_chart(km_splits)
            st.plotly_chart(fig_km, use_container_width=True)

        # ── Chart 7: Pace-HR Scatter ──────────────────────────────────────────
        if "hr" in wp.columns and wp["hr"].notna().any():
            st.markdown("#### Running Efficiency: Pace vs HR")
            fig_scatter = make_pace_hr_scatter(wp)
            st.plotly_chart(fig_scatter, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: COACH SUMMARY & EXPORT
# ════════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">Coach Summary & Export</div>', unsafe_allow_html=True)

    sm = st.session_state.get("summary_metrics", {})
    matched = st.session_state.get("matched_segments")
    wp = st.session_state.get("workout_points")

    if not sm and (wp is None or wp.empty):
        st.info("Upload workout data to generate a coaching summary.")
    else:
        # ── Language toggle ───────────────────────────────────────────────────
        lang_col, _ = st.columns([1, 3])
        with lang_col:
            lang = st.radio(
                "Language",
                ["Hebrew (עברית)", "English"],
                key="lang_toggle",
                horizontal=True,
            )
            new_lang = "he" if lang.startswith("Hebrew") else "en"
            if new_lang != st.session_state.get("coaching_language"):
                st.session_state["coaching_language"] = new_lang
                run_full_analysis()

        coaching_text = st.session_state.get("coaching_text", "")
        if not coaching_text:
            from analysis.segments import generate_coaching_text
            coaching_text = generate_coaching_text(
                matched if matched is not None else pd.DataFrame(),
                sm,
                language=st.session_state.get("coaching_language", "he"),
            )
            st.session_state["coaching_text"] = coaching_text

        # ── Coaching text display ─────────────────────────────────────────────
        is_hebrew = st.session_state.get("coaching_language", "he") == "he"
        st.markdown(
            f"""<style>
            .coaching-text {{
                direction: {'rtl' if is_hebrew else 'ltr'};
                text-align: {'right' if is_hebrew else 'left'};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 1rem;
                line-height: 1.8;
                background: #161B27;
                padding: 20px 24px;
                border-radius: 12px;
                border: 1px solid #1E2130;
                white-space: pre-wrap;
                color: #FAFAFA;
            }}
            </style>
            <div class="coaching-text">{coaching_text}</div>
            """,
            unsafe_allow_html=True,
        )

        edited_text = st.text_area(
            "✏️ Edit coaching notes (optional)",
            value=coaching_text,
            height=300,
            key="coaching_edit",
        )
        if edited_text != coaching_text:
            st.session_state["coaching_text"] = edited_text
            coaching_text = edited_text

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📦 Export Package")

        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            export_png_opt = st.checkbox("PNG charts", value=True)
        with col_opt2:
            export_svg_opt = st.checkbox("SVG charts", value=False)
        with col_opt3:
            export_pdf_opt = st.checkbox("PDF report", value=True)
        with col_opt4:
            export_zip_opt = st.checkbox("ZIP archive", value=True)

        if st.button("🚀 Generate export package", type="primary"):
            with st.spinner("Generating charts and report…"):
                try:
                    from charts.workout_charts import (
                        make_pace_chart, make_hr_chart, make_pace_hr_combined,
                        make_planned_vs_actual, make_hr_by_segment,
                        make_km_splits_chart, make_pace_hr_scatter,
                        make_interval_performance, make_summary_dashboard,
                    )
                    from export.exporter import export_png, export_svg, export_pdf_reportlab, create_zip

                    km_splits = st.session_state.get("km_splits")
                    x_axis = st.session_state.get("chart_x_axis", "time")

                    # Build all chart figures
                    figures = {}
                    if wp is not None and not wp.empty:
                        figures["pace_chart"] = make_pace_chart(wp, x_axis, matched)
                        figures["hr_chart"] = make_hr_chart(wp, x_axis, matched, sm.get("max_hr"))
                        figures["pace_hr_combined"] = make_pace_hr_combined(wp, x_axis, matched)
                        figures["pace_hr_scatter"] = make_pace_hr_scatter(wp)

                    if matched is not None and not matched.empty:
                        figures["planned_vs_actual"] = make_planned_vs_actual(matched)
                        figures["hr_by_segment"] = make_hr_by_segment(matched)
                        hard_count = (matched["type"].isin(["hard", "tempo", "marathon_pace"])).sum()
                        if hard_count >= 2:
                            figures["interval_performance"] = make_interval_performance(matched)

                    if km_splits is not None and not km_splits.empty:
                        figures["km_splits"] = make_km_splits_chart(km_splits)

                    # Dashboard (matplotlib)
                    dashboard_fig = make_summary_dashboard(wp, matched, sm, km_splits)

                    charts_png = {}
                    charts_svg = {}
                    png_errors = []

                    if export_png_opt:
                        for name, fig in figures.items():
                            try:
                                charts_png[name] = export_png(fig)
                            except Exception as e:
                                png_errors.append(f"{name}: {e}")
                        try:
                            charts_png["summary_dashboard"] = export_png(dashboard_fig)
                        except Exception as e:
                            png_errors.append(f"dashboard: {e}")

                    if export_svg_opt:
                        for name, fig in figures.items():
                            try:
                                charts_svg[name] = export_svg(fig)
                            except Exception as e:
                                pass

                    if png_errors:
                        for err in png_errors:
                            st.warning(f"PNG export issue ({err}). Ensure kaleido is installed.")

                    # PDF
                    pdf_bytes = None
                    if export_pdf_opt:
                        try:
                            pdf_bytes = export_pdf_reportlab(
                                charts_png,
                                sm,
                                coaching_text,
                                dashboard_png=charts_png.get("summary_dashboard"),
                            )
                        except Exception as e:
                            st.warning(f"PDF generation issue: {e}")

                    # Download buttons
                    if charts_png:
                        import zipfile
                        # Individual PNG downloads
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            if "summary_dashboard" in charts_png:
                                st.download_button(
                                    "⬇️ Download Dashboard PNG",
                                    data=charts_png["summary_dashboard"],
                                    file_name="running_dashboard.png",
                                    mime="image/png",
                                )
                        with dl_col2:
                            if pdf_bytes:
                                st.download_button(
                                    "⬇️ Download PDF Report",
                                    data=pdf_bytes,
                                    file_name="running_report.pdf",
                                    mime="application/pdf",
                                )

                    if export_zip_opt:
                        zip_bytes = create_zip(
                            charts_png,
                            charts_svg,
                            pdf_bytes or b"",
                            coaching_text,
                        )
                        st.download_button(
                            "📦 Download Full ZIP Package",
                            data=zip_bytes,
                            file_name="running_analysis.zip",
                            mime="application/zip",
                            type="primary",
                        )

                    st.success(f"✅ Export ready: {len(charts_png)} PNG charts, PDF report, coaching notes.")

                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8B9EC3; font-size:0.8rem;'>"
    "Running Workout Analyzer · GPX + OCR Analysis · Coach-quality insights"
    "</div>",
    unsafe_allow_html=True,
)
