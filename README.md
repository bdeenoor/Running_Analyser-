# 🏃 Running Workout Analyzer

A professional running coach dashboard built with Streamlit. Analyzes GPX files and watch screenshots to produce coach-quality insights and high-resolution charts — with Hebrew coaching summaries.

## Features

- **GPX parsing**: GPS points, heart rate, elevation, pace, splits, lap detection
- **Screenshot OCR**: Extract workout plans, lap tables, and summary metrics from watch screenshots
- **Segment matching**: Align planned workout segments to actual performance with confidence scoring
- **8 interactive charts**: Pace, HR, combined, planned vs actual, km splits, interval performance, pace-HR scatter
- **Hebrew coaching analysis**: Rule-based coaching text with HR drift analysis, race projections, and next workout recommendations
- **Export**: PNG (300 DPI), SVG, PDF report, ZIP archive

## Installation

```bash
# Clone or create the project directory
cd Running_Analyser-

# Install dependencies
pip install -r requirements.txt
```

> **Note on EasyOCR**: First run downloads ~1.5 GB of model weights. Set `EASYOCR_GPU=false` (default) for CPU-only mode.

## Quick Start

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

### Workflow 1: GPX Only
1. Upload your `.gpx` file in the **Upload** tab
2. Review detected GPS points and auto-generated km splits in **Data Review**
3. Enter your planned workout segments manually in **Data Review** → Planned Segments
4. View segment alignment in **Segment Alignment**
5. Explore charts in **Charts**
6. Read and export the coaching summary in **Coach Summary**

### Workflow 2: Screenshots Only
1. Upload watch screenshots (plan, laps, summary) in the **Upload** tab
2. Review OCR-extracted values — edit any errors in **Data Review**
3. Confirm segment alignment in **Segment Alignment**
4. Export in **Coach Summary**

### Workflow 3: GPX + Screenshots (Recommended)
1. Upload both GPX file and screenshots
2. GPX provides precise GPS/pace/HR data; screenshots fill in planned workout and lap metadata
3. Cross-validation flags discrepancies automatically

## Supported Workout Types

- Easy run / Long run
- Tempo
- Marathon pace
- Intervals (e.g., 6 × 1km)
- Pyramid workouts (e.g., 2-4-6-8-6-4-2 min)
- Structured warm-up / intervals / cool-down

## Sample Data

```bash
# Generate a sample GPX file
cd sample_data
python generate_sample_gpx.py
```

Use `sample_run.gpx` and `sample_workout_plan.json` for testing.

## Architecture

```
app.py                    # Streamlit UI orchestrator
parsers/
  gpx_parser.py           # GPX → DataFrame pipeline
  ocr_parser.py           # Screenshot OCR extraction
analysis/
  metrics.py              # Pace, splits, summary, projections
  segments.py             # Segment matching + coaching text
charts/
  workout_charts.py       # Plotly + Matplotlib charts
export/
  exporter.py             # PNG/SVG/PDF/ZIP export
tests/
  test_metrics.py         # Unit tests for metrics
  test_segments.py        # Unit tests for segment matching
sample_data/
  sample_run.gpx
  sample_workout_plan.json
```

## Running Tests

```bash
pytest tests/ -v --cov=parsers --cov=analysis --cov=charts --cov=export
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EASYOCR_GPU` | `false` | Set to `true` to use GPU for OCR |

## GPX Compatibility

Tested with exports from:
- Garmin Connect (`.gpx` with HR extensions)
- Strava (standard GPX)
- Apple Watch / Health app GPX exports
- Polar Flow GPX

Heart rate data is extracted from `gpxtpx:hr` or `ns3:hr` extensions.

## Tech Stack

- Python 3.11+
- [Streamlit](https://streamlit.io/) — UI
- [gpxpy](https://github.com/tkrajina/gpxpy) — GPX parsing
- [Plotly](https://plotly.com/python/) — interactive charts
- [Matplotlib](https://matplotlib.org/) — static dashboard export
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — screenshot text extraction
- [ReportLab](https://www.reportlab.com/) — PDF generation
- [scipy](https://scipy.org/) — signal smoothing

## Pace Conventions

- Pace is in **min/km** (lower = faster)
- On pace charts: Y-axis is **inverted** (faster pace appears higher)
- Delta pace: **negative = faster than plan**, positive = slower than plan
- Format: `mm:ss` (e.g., `4:30` = 4 minutes 30 seconds per km)
