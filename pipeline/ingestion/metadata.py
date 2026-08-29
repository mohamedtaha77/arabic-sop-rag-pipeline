"""Version control fields parsed from the per page header table.

Source documents are living procedure manuals with a revision log. Answering
from a superseded version is a silent correctness failure, so the version,
issue date and review date travel with every chunk.

These fields are parsed from the text layer route, not from OCR. The header is
a three row table, and pypdf reads in layout order so each value stays adjacent
to its label. EasyOCR groups the three labels onto one line and places the
values elsewhere on the page, which destroys the binding. Measured on the
reference corpus: the patterns below match on 28 of 29 and 19 of 20 text layer
pages, and on zero OCR pages.
"""

from __future__ import annotations

import re
from collections import Counter

from .document import Document

VERSION_FIELD_PATTERNS = {
    "doc_version": re.compile(r"(\d{1,2})\s*رقم النسخة"),
    "issue_date": re.compile(r"(\d{2}/\d{4})\s*تاريخ الإصدار"),
    "review_date": re.compile(r"(\d{2}/\d{4})\s*تاريخ آخر مراجعة"),
}

FIELD_NAMES = tuple(VERSION_FIELD_PATTERNS)


def extract_version_fields(documents: list[Document]) -> dict[str, dict]:
    """Return version control fields keyed by source filename.

    The header repeats on every page, so the value is taken by majority vote
    across pages rather than from the first match. A single misread page cannot
    then set the version for an entire document.

    Fields that cannot be read are recorded as None rather than omitted. A
    missing key is easy to overlook downstream; an explicit None is not.
    """
    per_source: dict[str, dict] = {}

    for source in sorted({d.metadata["source"] for d in documents}):
        pages = [d for d in documents if d.metadata["source"] == source]
        fields: dict[str, object] = {}

        for name, pattern in VERSION_FIELD_PATTERNS.items():
            votes: Counter[str] = Counter()
            for document in pages:
                votes.update(pattern.findall(document.text))

            if votes:
                value, count = votes.most_common(1)[0]
                fields[name] = value
                fields[f"{name}_page_count"] = count
            else:
                fields[name] = None
                fields[f"{name}_page_count"] = 0

        per_source[source] = fields

    return per_source


def apply_version_fields(
    documents: list[Document], fields_by_source: dict[str, dict]
) -> None:
    """Stamp version fields onto every page of their source, in place."""
    for document in documents:
        fields = fields_by_source.get(document.metadata["source"], {})
        for name in FIELD_NAMES:
            document.metadata[name] = fields.get(name)
