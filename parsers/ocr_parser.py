"""OCR-based screenshot parser: extracts workout plan, laps, and summary from watch screenshots."""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


# ── OCR backend detection ─────────────────────────────────────────────────────
# Priority: EasyOCR (best accuracy) → pytesseract (lightweight, works on free hosting)

_ocr_reader = None
_ocr_init_attempted = False
_detected_backend = None   # cached after first successful detection
_backend_error = None      # last error message if no backend found

try:
    import easyocr as _easyocr_module
    _EASYOCR_IMPORTABLE = True
except Exception:
    _easyocr_module = None
    _EASYOCR_IMPORTABLE = False

try:
    import pytesseract as _pytesseract_module
    from PIL import Image as _PIL_Image
    _TESSERACT_IMPORTABLE = True
except Exception:
    _pytesseract_module = None
    _TESSERACT_IMPORTABLE = False


def _get_backend() -> str:
    """Return which OCR backend to use: 'tesseract', 'easyocr', or 'none'.

    Tesseract is checked first — it's lightweight (no model download) and
    works on Streamlit Cloud free tier via packages.txt. EasyOCR is only
    tried when Tesseract binary is not installed (e.g. local dev without
    tesseract-ocr apt package). Set EASYOCR_ENABLED=false to skip EasyOCR.
    """
    global _detected_backend, _backend_error, _ocr_reader, _ocr_init_attempted

    if _detected_backend is not None:
        return _detected_backend

    errors = []

    # 1. Tesseract — fast, no model download, works on Streamlit Cloud
    if _TESSERACT_IMPORTABLE:
        try:
            _pytesseract_module.get_tesseract_version()
            _detected_backend = "tesseract"
            return "tesseract"
        except Exception as e:
            errors.append(f"Tesseract binary not found: {e}")

    # 2. EasyOCR — accurate but heavy (~200 MB model download on first use)
    easyocr_enabled = os.getenv("EASYOCR_ENABLED", "true").lower() != "false"
    if not _ocr_init_attempted and _EASYOCR_IMPORTABLE and easyocr_enabled:
        _ocr_init_attempted = True
        try:
            gpu = os.getenv("EASYOCR_GPU", "false").lower() == "true"
            _ocr_reader = _easyocr_module.Reader(["en"], gpu=gpu, verbose=False)
            _detected_backend = "easyocr"
            return "easyocr"
        except Exception as e:
            _ocr_reader = None
            errors.append(f"EasyOCR failed to load: {e}")

    _backend_error = "; ".join(errors) if errors else "pytesseract and easyocr not importable"
    _detected_backend = "none"
    return "none"


# Keep for backward compatibility
def get_ocr_reader():
    backend = _get_backend()
    return _ocr_reader if backend == "easyocr" else (True if backend == "tesseract" else None)

OCR_AVAILABLE = _EASYOCR_IMPORTABLE or _TESSERACT_IMPORTABLE


# ── Image preprocessing ───────────────────────────────────────────────────────

def _open_pil_image(image_file) -> "Image":
    """Open image file as PIL Image."""
    if isinstance(image_file, bytes):
        import io
        return Image.open(io.BytesIO(image_file)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def _preprocess_pil(img: "Image") -> "Image":
    """Resize if too large, boost contrast for dark watch screens."""
    max_width = 2000
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    return ImageEnhance.Contrast(img).enhance(1.5)


def _preprocess_image(image_file) -> np.ndarray:
    """Legacy helper — returns numpy array for EasyOCR."""
    return np.array(_preprocess_pil(_open_pil_image(image_file)))


def _ocr_image_easyocr(image_file) -> list[tuple]:
    """EasyOCR backend → [(bbox, text, confidence), ...]."""
    img_array = _preprocess_image(image_file)
    results = _ocr_reader.readtext(img_array, detail=1)
    return [(bbox, text, conf) for bbox, text, conf in results if conf > 0.3]


def _ocr_image_tesseract(image_file) -> list[tuple]:
    """Tesseract backend → same [(bbox, text, confidence), ...] format."""
    img = _preprocess_pil(_open_pil_image(image_file))
    data = _pytesseract_module.image_to_data(img, output_type=_pytesseract_module.Output.DICT)
    results = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 30:  # tesseract uses 0-100, equivalent to 0.3 in easyocr
            continue
        left, top, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = [[left, top], [left + w, top], [left + w, top + h], [left, top + h]]
        results.append((bbox, text, conf / 100.0))
    return results


def _ocr_image(image_file) -> list[tuple]:
    """Run OCR with best available backend. Returns [(bbox, text, confidence), ...]."""
    backend = _get_backend()
    if backend == "easyocr":
        return _ocr_image_easyocr(image_file)
    if backend == "tesseract":
        return _ocr_image_tesseract(image_file)
    raise RuntimeError("No OCR backend available (EasyOCR and Tesseract both unavailable)")


# ── Screenshot classifier ─────────────────────────────────────────────────────

_PLAN_KEYWORDS = re.compile(
    r"\b(warmup|warm.up|cooldown|cool.down|interval|tempo|easy|hard|recovery|"
    r"marathon|repeat|rest|jog|חימום|קצב|קל|קשה|התאוששות)\b",
    re.IGNORECASE,
)
_LAP_KEYWORDS = re.compile(r"\b(lap|split|pace|hr|bpm|km|mile|avg|max)\b", re.IGNORECASE)
_SUMMARY_KEYWORDS = re.compile(
    r"\b(total|distance|duration|time|average|calories|elevation|gain|summary)\b",
    re.IGNORECASE,
)


def _classify_screenshot_type(ocr_results: list) -> str:
    text = " ".join(t for _, t, _ in ocr_results)
    plan_hits = len(_PLAN_KEYWORDS.findall(text))
    lap_hits = len(_LAP_KEYWORDS.findall(text))
    summary_hits = len(_SUMMARY_KEYWORDS.findall(text))

    if plan_hits > 2:
        return "planned_workout"
    if lap_hits > 3:
        return "lap_summary"
    if summary_hits > 2:
        return "overall_summary"
    return "unknown"


# ── Pace / time parsers ───────────────────────────────────────────────────────

def _parse_pace_string(s: str) -> Optional[float]:
    """Convert '5:23' or '5:23/km' → 5.383 (min/km as float)."""
    m = re.search(r"(\d{1,2}):(\d{2})(?:\s*/\s*k(?:m)?)?", s)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if 0 <= secs < 60 and 1 <= mins <= 30:
            return mins + secs / 60.0
    return None


def _parse_time_string(s: str) -> Optional[int]:
    """Convert 'H:MM:SS' or 'MM:SS' → total seconds."""
    m3 = re.search(r"(\d+):(\d{2}):(\d{2})", s)
    if m3:
        return int(m3.group(1)) * 3600 + int(m3.group(2)) * 60 + int(m3.group(3))
    m2 = re.search(r"(\d{1,2}):(\d{2})", s)
    if m2:
        return int(m2.group(1)) * 60 + int(m2.group(2))
    return None


def _parse_distance_string(s: str) -> Optional[float]:
    """Extract distance in meters from '10.5 km', '5.00 KM', '10500 m'."""
    m_km = re.search(r"(\d+\.?\d*)\s*k(?:m)?", s, re.IGNORECASE)
    if m_km:
        return float(m_km.group(1)) * 1000
    m_m = re.search(r"(\d+\.?\d*)\s*m\b", s, re.IGNORECASE)
    if m_m:
        return float(m_m.group(1))
    return None


def _parse_hr_string(s: str) -> Optional[float]:
    m = re.search(r"(\d{2,3})\s*(?:bpm)?", s, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 40 <= val <= 220:
            return float(val)
    return None


# ── Segment line parser ───────────────────────────────────────────────────────

_SEGMENT_TYPE_MAP = {
    "warmup": "warmup", "warm": "warmup", "warm-up": "warmup", "warm_up": "warmup",
    "cooldown": "cooldown", "cool": "cooldown", "cool-down": "cooldown", "cool_down": "cooldown",
    "easy": "easy", "jog": "easy", "recovery": "easy", "rest": "easy",
    "hard": "hard", "interval": "hard", "tempo": "tempo", "marathon": "marathon_pace",
    "mp": "marathon_pace",
}


def _parse_segment_line(line: str) -> Optional[dict]:
    """Try to parse a text line as a workout segment."""
    line = line.strip()
    if len(line) < 3:
        return None

    # Detect segment type
    seg_type = "easy"
    for kw, stype in _SEGMENT_TYPE_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", line, re.IGNORECASE):
            seg_type = stype
            break

    # Duration: "20:00", "20 min", "2:00"
    duration_s = None
    m_dur = re.search(r"(\d{1,3}):(\d{2})", line)
    if m_dur:
        mins, secs = int(m_dur.group(1)), int(m_dur.group(2))
        duration_s = mins * 60 + secs
    else:
        m_min = re.search(r"(\d{1,3})\s*min", line, re.IGNORECASE)
        if m_min:
            duration_s = int(m_min.group(1)) * 60

    if duration_s is None or duration_s <= 0:
        return None

    # Target pace: "@ 4:30/km", "4:30 min/km"
    pace = _parse_pace_string(line) if "@" in line or "km" in line.lower() else None

    return {
        "type": seg_type,
        "duration_s": duration_s,
        "target_pace_min_km": pace,
    }


# ── Planned workout extractor ─────────────────────────────────────────────────

def _extract_planned_workout(ocr_results: list) -> pd.DataFrame:
    texts = [text for _, text, _ in ocr_results]
    segments = []
    seg_num = 1

    for text in texts:
        parsed = _parse_segment_line(text)
        if parsed:
            name = text[:40].strip()
            parsed["segment_num"] = seg_num
            parsed["name"] = name
            segments.append(parsed)
            seg_num += 1

    if not segments:
        return pd.DataFrame()

    return pd.DataFrame(segments)[["segment_num", "name", "type", "duration_s", "target_pace_min_km"]]


# ── Lap table extractor ───────────────────────────────────────────────────────

def _group_by_row(ocr_results: list, y_tolerance: int = 20) -> list[list]:
    """Group OCR results into rows by Y-coordinate proximity."""
    if not ocr_results:
        return []

    def y_center(bbox):
        ys = [pt[1] for pt in bbox]
        return (min(ys) + max(ys)) / 2

    sorted_results = sorted(ocr_results, key=lambda r: y_center(r[0]))
    rows = []
    current_row = [sorted_results[0]]

    for item in sorted_results[1:]:
        if abs(y_center(item[0]) - y_center(current_row[-1][0])) <= y_tolerance:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=lambda r: r[0][0][0]))
            current_row = [item]
    rows.append(sorted(current_row, key=lambda r: r[0][0][0]))

    return rows


def _extract_lap_table(ocr_results: list) -> pd.DataFrame:
    rows = _group_by_row(ocr_results)
    if not rows:
        return pd.DataFrame()

    # Find header row
    header_idx = None
    for i, row in enumerate(rows):
        row_text = " ".join(t for _, t, _ in row).lower()
        if any(kw in row_text for kw in ["lap", "pace", "time", "hr", "distance"]):
            header_idx = i
            break

    data_rows = rows[header_idx + 1:] if header_idx is not None else rows

    laps = []
    for row_idx, row in enumerate(data_rows):
        texts = [t for _, t, _ in row]
        row_text = " ".join(texts)

        lap_num = row_idx + 1
        pace = None
        avg_hr = None
        distance_m = None
        duration_s = None

        # Try to extract metrics from the row
        for t in texts:
            if pace is None:
                pace = _parse_pace_string(t)
            if avg_hr is None:
                avg_hr = _parse_hr_string(t)
            if distance_m is None:
                distance_m = _parse_distance_string(t)
            if duration_s is None and pace is None:
                duration_s = _parse_time_string(t)

        if pace is not None or avg_hr is not None:
            laps.append({
                "lap_num": lap_num,
                "start_time": None,
                "end_time": None,
                "duration_s": duration_s,
                "distance_m": distance_m,
                "avg_pace_min_km": pace,
                "avg_hr": avg_hr,
                "max_hr": None,
                "elevation_gain": None,
            })

    return pd.DataFrame(laps) if laps else pd.DataFrame()


# ── Summary extractor ─────────────────────────────────────────────────────────

def _extract_summary_values(ocr_results: list) -> dict:
    texts = [t for _, t, _ in ocr_results]
    full_text = " ".join(texts)

    result = {}

    dist = _parse_distance_string(full_text)
    if dist:
        result["total_distance_m_ocr"] = dist

    # Total time: look for longer time strings
    m_time = re.search(r"(\d{1,2}:\d{2}:\d{2})", full_text)
    if m_time:
        result["total_time_s_ocr"] = _parse_time_string(m_time.group(1))

    # Average pace
    m_pace = re.search(r"(?:avg|average|pace)[^\d]*(\d{1,2}:\d{2})", full_text, re.IGNORECASE)
    if m_pace:
        result["avg_pace_min_km_ocr"] = _parse_pace_string(m_pace.group(1))

    # HR
    m_avg_hr = re.search(r"(?:avg|average)\s*(?:hr|heart)[^\d]*(\d{2,3})", full_text, re.IGNORECASE)
    if m_avg_hr:
        result["avg_hr_ocr"] = float(m_avg_hr.group(1))

    m_max_hr = re.search(r"(?:max)\s*(?:hr|heart)[^\d]*(\d{2,3})", full_text, re.IGNORECASE)
    if m_max_hr:
        result["max_hr_ocr"] = float(m_max_hr.group(1))

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def parse_screenshots(image_files: list) -> dict:
    """
    Process multiple screenshots and extract workout data.

    Returns:
        dict with keys: planned_segments, laps, summary_metrics, raw_extractions, confidence_notes
    """
    planned_segments = pd.DataFrame()
    laps = pd.DataFrame()
    summary_metrics = {}
    raw_extractions = {}
    confidence_notes = []

    backend = _get_backend()
    if backend == "none":
        return {
            "planned_segments": None,
            "laps": None,
            "summary_metrics": {},
            "raw_extractions": {},
            "confidence_notes": [],
            "ocr_available": False,
            "backend_error": _backend_error,
        }

    confidence_notes.append(f"OCR backend: {backend}")

    for i, image_file in enumerate(image_files):
        try:
            ocr_results = _ocr_image(image_file)
        except Exception as e:
            confidence_notes.append(f"Image {i+1}: OCR failed — {e}")
            continue

        img_name = getattr(image_file, "name", f"image_{i+1}")
        raw_extractions[img_name] = [
            {"text": t, "confidence": f"{c:.2f}"} for _, t, c in ocr_results
        ]

        screen_type = _classify_screenshot_type(ocr_results)

        if screen_type == "planned_workout":
            extracted = _extract_planned_workout(ocr_results)
            if not extracted.empty:
                planned_segments = extracted
                confidence_notes.append(
                    f"{img_name}: Detected workout plan with {len(extracted)} segments (review before use)."
                )
            else:
                confidence_notes.append(f"{img_name}: Detected as plan but could not parse segments.")

        elif screen_type == "lap_summary":
            extracted = _extract_lap_table(ocr_results)
            if not extracted.empty:
                laps = extracted
                confidence_notes.append(
                    f"{img_name}: Detected lap table with {len(extracted)} laps (review before use)."
                )
            else:
                confidence_notes.append(f"{img_name}: Detected as lap table but could not parse rows.")

        elif screen_type == "overall_summary":
            extracted = _extract_summary_values(ocr_results)
            summary_metrics.update(extracted)
            confidence_notes.append(f"{img_name}: Detected summary screen — {len(extracted)} values extracted.")

        else:
            # Try all extractors
            plan = _extract_planned_workout(ocr_results)
            if not plan.empty and planned_segments.empty:
                planned_segments = plan
            lap = _extract_lap_table(ocr_results)
            if not lap.empty and laps.empty:
                laps = lap
            summ = _extract_summary_values(ocr_results)
            summary_metrics.update(summ)
            confidence_notes.append(f"{img_name}: Unknown type — tried all extractors.")

    if planned_segments.empty and laps.empty and not summary_metrics:
        confidence_notes.append(
            "⚠️ No data could be extracted from screenshots. "
            "Please use manual entry or check image quality."
        )

    return {
        "planned_segments": planned_segments if not planned_segments.empty else None,
        "laps": laps if not laps.empty else None,
        "summary_metrics": summary_metrics,
        "raw_extractions": raw_extractions,
        "confidence_notes": confidence_notes,
        "ocr_available": True,
    }
