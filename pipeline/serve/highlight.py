"""Where on the page a cited chunk actually came from.

Position: the source pane's whole reason to exist. An answer that says
"[1]" is only checkable if a reader can see the passage it points at,
on the page it points at, without hunting for it. This module turns a
chunk id into rectangles in rendered-image pixel coordinates; app.py
serves them, and the browser draws them over the page image.

What makes this possible at all, and what limits it, both measured
rather than assumed:

  * chunk metadata carries ``source``, ``page``, ``table_id`` and
    ``row_range`` (chunk.py's own Chunk contract), but no geometry.
  * ``01_documents_layout.json`` carries, per page, a ``tables`` list
    where every entry has a ``bbox`` in PDF points. That is the
    geometry, and ``table_id`` is its index.
  * 316 of the corpus's 357 chunks (88.5%) carry a ``table_id``. The
    other 41 are ``prose``, which never went through table detection
    and therefore has no stored geometry of any kind. Those return no
    rectangle and say why, rather than boxing an arbitrary region and
    implying a precision that does not exist. That is the same habit
    quality.py's own fragment_ratio and stage 9's own uncited residual
    already follow.

The one real trap, found by checking rather than trusting the obvious
reading: **``table_id`` indexes the STORED list, not what
``find_tables()`` returns today.** Measured on Central Alarm page 6,
live detection finds 4 tables while the stored list holds 2: layout.py
lifts the version-header band into metadata and drops 2-column
full-height layout containers (its own finding E), so the two lists
are offset by whatever that page happened to contain. Indexing live
detection with a stored ``table_id`` would silently highlight a
different table. So the stored bbox is authoritative for WHICH table,
and live detection is consulted only to sharpen WHERE inside it.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from ..chunking.chunk import source_slug
from ..config import CONTEXT_OUTPUTS, LAYOUT_OUTPUT, RAW_DIR

# How close two bboxes must be, in PDF points, to count as the same
# table across the stored list and a live find_tables() pass. Generous
# on purpose: the two come from the same detector on the same file, so
# they agree closely or not at all, and a near-miss here costs only the
# row-level refinement, never correctness of which table was cited.
_BBOX_TOLERANCE_PT = 2.0


@functools.lru_cache(maxsize=1)
def _chunk_index() -> dict[str, dict[str, Any]]:
    """chunk_id -> metadata, over the shipping variant.

    The template variant is the one stage 7 shipped
    (05_retrieval_decision.json), and chunk ids are identical across
    all three variants anyway: only the prefixed text differs, never
    the id or the geometry this module reads.
    """
    raw = json.loads(CONTEXT_OUTPUTS["template"].read_text(encoding="utf-8"))
    return {c["metadata"]["chunk_id"]: c["metadata"] for c in raw}


@functools.lru_cache(maxsize=1)
def _layout_index() -> dict[tuple[str, int], list[dict[str, Any]]]:
    """(source, page) -> that page's stored tables, geometry included."""
    raw = json.loads(LAYOUT_OUTPUT.read_text(encoding="utf-8"))
    index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in raw:
        meta = entry["metadata"]
        index[(meta["source"], meta["page"])] = meta.get("tables") or []
    return index


@functools.lru_cache(maxsize=1)
def _slug_to_pdf() -> dict[str, Any]:
    """Slug -> PDF path, derived by scanning RAW_DIR exactly the way
    worksheet.slug_to_path does, rather than a second hardcoded table.
    """
    return {source_slug(p.name): p for p in sorted(RAW_DIR.glob("*.pdf"))}


def _bboxes_match(a: list[float], b: list[float]) -> bool:
    return all(abs(x - y) <= _BBOX_TOLERANCE_PT for x, y in zip(a, b))


def _live_rows(source: str, page: int, stored_bbox: list[float]) -> list[list[float]] | None:
    """Per-row bboxes for the stored table, from a live detection pass,
    or None when this page's live detection does not offer a table that
    matches the stored one.

    Matched by bbox rather than by index, for the offset reason in the
    module docstring. Row heights genuinely vary within one table
    (measured on Central Alarm p6: 17.5, 26.9, 34.3, 51.0 points), so
    this refinement is worth the page parse: proportional subdivision
    of the table bbox would put a highlight in the wrong place on any
    table whose rows are not uniform.
    """
    import pymupdf

    pdf_path = _slug_to_pdf().get(source_slug(source))
    if pdf_path is None:
        return None

    document = pymupdf.open(pdf_path)
    try:
        found = document[page - 1].find_tables()
        for table in found.tables:
            if _bboxes_match(list(table.bbox), stored_bbox):
                return [list(row.bbox) for row in table.rows]
    finally:
        document.close()
    return None


def _proportional_rows(bbox: list[float], row_count: int) -> list[list[float]]:
    """Fallback: split the table bbox into equal horizontal bands.

    Only used when live detection offers no matching table. Equal bands
    are wrong for a table with varied row heights, which is why this is
    the fallback and not the primary path; it still puts the highlight
    inside the right table, which is the property that matters most.
    """
    x0, y0, x1, y1 = bbox
    if row_count <= 0:
        return [list(bbox)]
    height = (y1 - y0) / row_count
    return [[x0, y0 + i * height, x1, y0 + (i + 1) * height] for i in range(row_count)]


def _union(rects: list[list[float]]) -> list[float]:
    return [
        min(r[0] for r in rects), min(r[1] for r in rects),
        max(r[2] for r in rects), max(r[3] for r in rects),
    ]


def resolve(chunk_id: str, dpi: int) -> dict[str, Any]:
    """Where this chunk sits on its own page, in image pixels.

        source     the PDF this chunk came from
        page       1-based page number
        rects      [[x, y, w, h], ...] in pixels at `dpi`, empty when
                    this chunk carries no geometry
        covered    whether a real rectangle was produced
        precision  "row" when live detection placed the exact rows,
                    "table" when the whole stored table bbox is used,
                    "none" when there is nothing to draw
        reason     why, when covered is False, in the plain words the
                    UI can show a reader directly
    """
    meta = _chunk_index().get(chunk_id)
    if meta is None:
        return {"source": None, "page": None, "rects": [], "covered": False,
                "precision": "none", "reason": f"unknown chunk id {chunk_id!r}"}

    source, page = meta["source"], meta["page"]
    base = {"source": source, "page": page, "chunk_type": meta.get("chunk_type")}

    table_id, row_range = meta.get("table_id"), meta.get("row_range")
    if table_id is None or row_range is None:
        return {**base, "rects": [], "covered": False, "precision": "none",
                "reason": "prose section: this chunk never went through table "
                          "detection, so the corpus holds no geometry for it"}

    tables = _layout_index().get((source, page)) or []
    if table_id >= len(tables):
        return {**base, "rects": [], "covered": False, "precision": "none",
                "reason": f"table {table_id} is not on the stored page"}

    stored = tables[table_id]
    stored_bbox = stored["bbox"]
    stored_row_count = len(stored.get("rows") or [])

    rows = _live_rows(source, page, stored_bbox)
    precision = "row"
    if rows is None or not rows:
        rows = _proportional_rows(stored_bbox, stored_row_count)
        precision = "table" if stored_row_count <= 1 else "row"

    # row_range is [first, last] INCLUSIVE (chunk.py's own contract, and
    # tables.py writes [0, len(rows) - 1] for a whole table), so the
    # slice runs to last + 1. Clamped rather than trusted: stored rows
    # are the merged, post-processing rows and a live pass can disagree
    # about the count, which must degrade to a slightly wider highlight
    # rather than an IndexError on a user's screen.
    first, last = row_range
    selected = rows[max(0, first):min(len(rows), last + 1)]
    if not selected:
        selected = [stored_bbox]
        precision = "table"

    scale = dpi / 72.0
    x0, y0, x1, y1 = _union(selected)
    return {
        **base,
        "rects": [[round(x0 * scale), round(y0 * scale),
                   round((x1 - x0) * scale), round((y1 - y0) * scale)]],
        "covered": True,
        "precision": precision,
        "reason": "",
    }
