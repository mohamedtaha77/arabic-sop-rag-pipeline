"""Project-wide paths, extraction settings and local model configuration."""

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

CHUNKS_OUTPUT = PROCESSED_DIR / "02_chunks.json"

# Page render resolution for OCR. 72 DPI (the PDF default) loses the dots that
# distinguish ب ت ث ن; 600 DPI quadruples runtime for no measured accuracy gain.
RENDER_DPI = 300

# EasyOCR language codes. Arabic first: it drives the recognition model and the
# corpus is ~85% Arabic. English is needed for embedded system names and codes.
OCR_LANGUAGES = ["ar", "en"]


# --- local model layer ------------------------------------------------------

LLM_CACHE_DIR = DATA_DIR / "llm_cache"

# Ollama's OpenAI-compatible endpoint, kept as a base URL rather than a full
# path. llama.cpp's server speaks the same dialect, so the documented fallback
# is a one-line change here rather than a rewrite of the client.
LLM_BASE_URL = "http://localhost:11434/v1"

# Seconds. Generous on purpose: a 4 GB card holds only part of even a 3B model,
# so the remaining layers run on CPU and a long-context generation is measured
# in minutes. A timeout tight enough to feel responsive would fire on every
# real synthesiser call and look like a bug in the client.
LLM_TIMEOUT = 300

# Two models, from two different families. A judge sharing weights with the
# generator grades its own work generously; a judge sharing only a
# quantisation still shares every bias worth catching.
#
# 3B rather than the 7B the plan first assumed, because this machine measured
# 11.8 GB of system RAM beside the 4 GB card. Promotion to 7B for synthesis is
# a decision for the probe's numbers, not for this file.
GENERATOR_MODEL = "qwen2.5:3b-instruct-q4_K_M"
JUDGE_MODEL = "llama3.2:3b-instruct-q4_K_M"

# Tokens of context this pipeline needs, and what probe.py verifies the server
# actually provides rather than trusting.
#
# Ollama defaults to far less and truncates the overflow in silence. Eight
# retrieved chunks at the p90 size measured in chunking are roughly 6,000
# characters, near 2,000 Arabic tokens, before the prompt and the context
# prefixes stage 4 adds. The default would cut that mid-context and return a
# fluent answer built on half the evidence, which is quality.py's failure mode
# exactly: it returns, it reads well, and nothing appears in the logs.
LLM_CONTEXT = 8192
