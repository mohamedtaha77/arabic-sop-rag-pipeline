"""Version control fields parsed from the per page header table.

Source documents are living procedure manuals with a revision log. Answering
from a superseded version is a silent correctness failure, so the version,
issue date and review date travel with every chunk.

These fields are parsed from the text layer route, not from OCR. The header is
a three row table, and pypdf reads in layout order so each value stays adjacent
to its label. EasyOCR groups the three labels onto one line and places the
values elsewhere on the page, which destroys the binding. Measured on the
reference corpus: the patterns below match the version on 28 of 29, 19 of 20
and 28 of 29 text layer pages, and on zero OCR pages.

The two date patterns still find nothing in the third manual, whose labels are
too damaged to anchor on without guessing which date is which. layout.py reads
those two from the header table's geometry instead, and the two routes agree on
every field they both recover.
"""

from __future__ import annotations

import re
from collections import Counter

from .document import Document

# The version label is matched on its opening رقم ال rather than in full.
# Central Mail's text layer truncates it to رقم الن, which is why this file
# originally reported all three of its fields as None: the digit was sitting
# right there, correctly extracted, with nothing the pattern recognised beside
# it. Digits are the one thing the broken ToUnicode table does not corrupt, so
# the value was never in doubt, only the anchor.
#
# Measured against the full corpus: the shortened anchor matches on exactly the
# same 28 and 19 pages as the full label did for the other two manuals and
# returns the same value, adds 28 of Central Mail's 29, and never matches more
# than once on any page in the corpus, so it has not become loose enough to
# catch a different رقم.
#
# The two dates keep their full labels deliberately. Central Mail truncates
# both تاريخ الإصدار and تاريخ آخر مراجعة to the same تار, so a shortened
# anchor could not tell issue from review; it would only appear to work here
# because both happen to read 02/2026. layout.py already recovers those two by
# their position in the header table, which is a real distinction rather than a
# coincidence, so this route leaves them alone.
VERSION_FIELD_PATTERNS = {
    "doc_version": re.compile(r"(\d{1,2})\s*رقم\s+ال"),
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
