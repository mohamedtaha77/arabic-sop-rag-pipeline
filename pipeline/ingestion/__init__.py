"""Ingestion stage: PDF sources into cleaned, assessed Document objects.

Two extraction routes are provided. They fail in opposite directions, so both
are kept and the right one is chosen per task rather than globally.

    textlayer   Fast, around 50ms per page. Preserves layout order, so label
                and value stay adjacent in tables. Accuracy depends entirely on
                the PDF's ToUnicode table being correct.

    ocr         Around 10s per page on a GPU. Reads rendered pixels, so a
                damaged ToUnicode table cannot affect it. Flattens tables and
                loses column structure.

Run textlayer first: it is cheap and its quality assessment determines whether
OCR is needed at all.
"""

from .cleaning import clean_text
from .document import Document
from .metadata import apply_version_fields, extract_version_fields
from .quality import assess_quality
from .storage import load_documents, save_documents

__all__ = [
    "Document",
    "clean_text",
    "assess_quality",
    "extract_version_fields",
    "apply_version_fields",
    "load_documents",
    "save_documents",
]
