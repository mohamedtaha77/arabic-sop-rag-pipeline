"""Project-wide paths and extraction settings."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"

TEXTLAYER_OUTPUT = PROCESSED_DIR / "01_documents_textlayer.json"
OCR_OUTPUT = PROCESSED_DIR / "01_documents_ocr.json"
LAYOUT_OUTPUT = PROCESSED_DIR / "01_documents_layout.json"

# Page render resolution for OCR. 72 DPI (the PDF default) loses the dots that
# distinguish ب ت ث ن; 600 DPI quadruples runtime for no measured accuracy gain.
RENDER_DPI = 300

# EasyOCR language codes. Arabic first: it drives the recognition model and the
# corpus is ~85% Arabic. English is needed for embedded system names and codes.
OCR_LANGUAGES = ["ar", "en"]
