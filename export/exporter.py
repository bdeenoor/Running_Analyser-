"""Export charts and reports to PNG, SVG, PDF, and ZIP."""

import io
import zipfile
from typing import Union

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from analysis.metrics import format_pace, format_duration


# ── PNG export ────────────────────────────────────────────────────────────────

def export_png(fig, dpi: int = 300) -> bytes:
    """Export a Plotly or Matplotlib figure to PNG bytes."""
    if isinstance(fig, go.Figure):
        try:
            return fig.to_image(format="png", width=1920, height=1080, scale=300 / 96)
        except Exception as e:
            # Fallback: kaleido not installed or error
            raise RuntimeError(f"Plotly PNG export failed (is kaleido installed?): {e}") from e

    if isinstance(fig, plt.Figure):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        return buf.getvalue()

    raise TypeError(f"Unsupported figure type: {type(fig)}")


# ── SVG export ────────────────────────────────────────────────────────────────

def export_svg(fig) -> bytes:
    """Export a Plotly or Matplotlib figure to SVG bytes."""
    if isinstance(fig, go.Figure):
        try:
            return fig.to_image(format="svg", width=1920, height=1080)
        except Exception as e:
            raise RuntimeError(f"Plotly SVG export failed: {e}") from e

    if isinstance(fig, plt.Figure):
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        return buf.getvalue()

    raise TypeError(f"Unsupported figure type: {type(fig)}")


# ── PDF export (ReportLab) ────────────────────────────────────────────────────

def export_pdf_reportlab(
    figures_png: dict,            # {"chart_name": png_bytes}
    summary_metrics: dict,
    coaching_text: str,
    dashboard_png: bytes = None,
) -> bytes:
    """Build a multi-page PDF report with ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
            Table, TableStyle, PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        raise ImportError("reportlab is required: pip install reportlab")

    try:
        from bidi.algorithm import get_display
    except ImportError:
        def get_display(text):
            return text

    import os
    font_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "NotoSansHebrew-Regular.ttf")
    hebrew_font = "Helvetica"
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("NotoHebrew", font_path))
            hebrew_font = "NotoHebrew"
        except Exception:
            pass

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=18, alignment=TA_CENTER,
        textColor=colors.HexColor("#00D4FF"), spaceAfter=12,
    )
    style_heading = ParagraphStyle(
        "Heading", fontName="Helvetica-Bold", fontSize=13,
        textColor=colors.HexColor("#FFD93D"), spaceAfter=6,
    )
    style_body = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=10,
        textColor=colors.black, spaceAfter=4, leading=14,
    )
    style_hebrew = ParagraphStyle(
        "Hebrew", fontName=hebrew_font, fontSize=11,
        textColor=colors.black, spaceAfter=4, leading=16,
        alignment=TA_RIGHT,
    )

    story = []
    pw, ph = landscape(A4)
    usable_w = pw - 3 * cm
    usable_h = ph - 3 * cm

    # ── Page 1: Dashboard ────────────────────────────────────────────────────
    story.append(Paragraph("Running Workout Analysis", style_title))
    story.append(Spacer(1, 0.3*cm))

    if dashboard_png:
        img_buf = io.BytesIO(dashboard_png)
        img = RLImage(img_buf, width=usable_w, height=usable_h * 0.85)
        story.append(img)
    story.append(PageBreak())

    # ── Page 2: Summary metrics table ────────────────────────────────────────
    story.append(Paragraph("Workout Summary", style_heading))
    story.append(Spacer(1, 0.3*cm))

    sm = summary_metrics or {}
    proj = sm.get("race_projections", {})
    table_data = [
        ["Metric", "Value"],
        ["Total Distance", f"{sm.get('total_distance_m', 0)/1000:.2f} km"],
        ["Moving Time", format_duration(sm.get("moving_time_s", 0))],
        ["Total Time", format_duration(sm.get("total_time_s", 0))],
        ["Avg Pace", format_pace(sm.get("avg_pace_min_km"))],
        ["Avg Heart Rate", f"{sm.get('avg_hr', 0):.0f} bpm" if sm.get("avg_hr") else "—"],
        ["Max Heart Rate", f"{sm.get('max_hr', 0):.0f} bpm" if sm.get("max_hr") else "—"],
        ["Elevation Gain", f"{sm.get('elevation_gain_m', 0):.0f} m"],
        ["Elevation Loss", f"{sm.get('elevation_loss_m', 0):.0f} m"],
    ]
    for race, data in proj.items():
        t = data.get("time_s", 0)
        p = data.get("pace_min_km")
        table_data.append([f"{race} Projection", f"{format_duration(t)} ({format_pace(p)}/km)"])

    tbl = Table(table_data, colWidths=[8*cm, 10*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161B27")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00D4FF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F5F5"), colors.white]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # ── Pages 3–N: Charts (2 per page) ───────────────────────────────────────
    chart_items = list(figures_png.items())
    for i in range(0, len(chart_items), 2):
        for j in range(2):
            if i + j >= len(chart_items):
                break
            name, png_bytes = chart_items[i + j]
            story.append(Paragraph(name.replace("_", " ").title(), style_heading))
            img_buf = io.BytesIO(png_bytes)
            img = RLImage(img_buf, width=usable_w, height=usable_h * 0.42)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        story.append(PageBreak())

    # ── Last page: Coaching text ──────────────────────────────────────────────
    story.append(Paragraph("Coach Notes", style_heading))
    story.append(Spacer(1, 0.3*cm))

    for line in coaching_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.2*cm))
            continue
        try:
            display_line = get_display(line)
        except Exception:
            display_line = line
        # Use Hebrew style for lines with Hebrew characters
        has_hebrew = any("֐" <= c <= "׿" for c in line)
        style = style_hebrew if has_hebrew else style_body
        try:
            story.append(Paragraph(display_line, style))
        except Exception:
            story.append(Paragraph(line.encode("ascii", "replace").decode(), style_body))

    doc.build(story)
    return buf.getvalue()


# ── ZIP archive ───────────────────────────────────────────────────────────────

def create_zip(
    charts_png: dict,      # {"name": bytes}
    charts_svg: dict,      # {"name": bytes}
    pdf_bytes: bytes,
    coaching_text: str,
) -> bytes:
    """Bundle all outputs into a ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in charts_png.items():
            safe_name = name.replace(" ", "_").replace("/", "-")
            zf.writestr(f"charts/png/{safe_name}.png", data)

        for name, data in charts_svg.items():
            safe_name = name.replace(" ", "_").replace("/", "-")
            zf.writestr(f"charts/svg/{safe_name}.svg", data)

        if pdf_bytes:
            zf.writestr("report.pdf", pdf_bytes)

        if coaching_text:
            zf.writestr("coaching_notes.txt", coaching_text.encode("utf-8"))

    return buf.getvalue()
