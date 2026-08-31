"""Table and prose reconstruction from page geometry.

ocr.py reads every page with EasyOCR's ``detail=0`` option, which returns
joined strings and throws away where each word sits on the page. That is
exactly why a table row comes back as one flattened line instead of separate
cells: OCR is right about the characters and blind to the columns.

This module keeps OCR for characters and uses PyMuPDF's ``find_tables`` for
geometry. Each page is OCR'd once at ``detail=1``, which returns a bounding
box per recognised fragment, and every fragment is then assigned to the table
cell its centre falls inside. Text that lands outside every table is prose.

Three things came out of measuring this on the real corpus rather than
assuming the shape of a "table" in advance, recorded here because each one
would produce a working-looking but wrong result if skipped:

* PyMuPDF's own ``Table.extract()`` returns text that is both ToUnicode-
  corrupted, the same font-mapping damage described in ingestion.md, and in
  visual (mirror) order rather than logical reading order. Its geometry is
  trustworthy; its text is not. Cell text has to come from OCR.
* A detected "table" of two columns spanning most of the page height is not
  data. It is the numbered-list frame Word draws around a block of procedure
  steps, and it usually contains a genuine table nested inside it. Building
  rows for it truncates real content into two wide, meaningless cells.
* Most content tables in this corpus are not uniform grids. A row can be a
  single wide heading cell, a label-value pair such as ``الجهة المنفذة``
  naming the executing unit, or a step-number-and-description pair, all
  inside the same PyMuPDF table, because PyMuPDF represents a merged cell
  once and marks every grid position it spans with ``None``. Rows are
  emitted as variable-length lists rather than padded to a fixed column
  count, so a two-cell label row is not forced to look like a seven-cell
  grid row with five empty strings in it.

What this module does not do: decide whether a row is a heading, a label
pair, or a step. That is a question about what a row means, and belongs to
chunking, not to geometry. This module's job ends at correct, ordered text.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import (
    LAYOUT_OUTPUT,
    OCR_CACHE_DIR,
    RAW_DIR,
    RENDER_DPI,
    TEXTLAYER_OUTPUT,
)
from .cleaning import clean_text, correct_ocr_misreads
from .document import Document
from .metadata import FIELD_NAMES, extract_version_fields
from .ocr import file_fingerprint, get_reader, render_page
from .quality import assess_quality
from .storage import load_documents, save_documents

EXTRACTION_METHOD = "layout"

# --- table classification ---------------------------------------------

# The version-control band repeats on nearly every page: 3 rows, bottom edge
# in the top 12% of the page. Measured on all 78 pages: 28, 19 and 28 matches
# per source file, so a handful of pages (covers, annexes) will not have one,
# which downstream code has to tolerate rather than assume away.
HEADER_ROW_COUNT = 3
HEADER_Y_MAX_FRAC = 0.12

# A table this narrow and this tall is not data. It is the numbered-list
# layout frame Word draws around a block of steps, with a real table usually
# nested inside it. Measured: 3 in Assets, 3 in Central Alarm, 0 in Central
# Mail, and every one confirmed by eye to be a false table.
CONTAINER_MAX_COLS = 2
CONTAINER_MIN_HEIGHT_FRAC = 0.6

# The footer band is not a table at all, so find_tables() never sees it and
# the header handling above never touches it. Measured directly on three
# pages: the page number, "Internal Use" and the preparing department's name
# sit at y 0.972-0.974 every time, with the nearest real body text over 0.2
# of the page higher. A fixed threshold below that cluster and well above the
# body text is a safe cut, the same reasoning as the header band above.
FOOTER_Y_MIN_FRAC = 0.95

# --- page-spanning merge ------------------------------------------------

# Measured across all 78 pages: 15 genuine continuations, all of them the
# procedure grids naming an executing unit and its steps. A table ending here
# or lower, met by one starting here or higher on the next page, with the
# same column count and matching column x positions, is one table cut by a
# page break rather than two tables that happen to be adjacent. An earlier
# draft of this measurement treated the Central Alarm approval matrix on
# pages 6 and 7 as a spanning pair to verify this against; it is not one.
# Both tables there carry their own header row and the first ends with room
# to spare. Re-measuring against the real 15 is what fixed these numbers.
SPAN_PREV_END_MIN_FRAC = 0.80
SPAN_NEXT_START_MAX_FRAC = 0.15
SPAN_X_TOLERANCE_FRAC = 0.02

# --- fragment ordering ---------------------------------------------------

# Two OCR fragments belong to the same line if their vertical centres sit
# within this many points of each other. Measured at 15 pixels on a 300 DPI
# render; expressed in points so it holds at a different render resolution.
LINE_GROUP_TOLERANCE_PT = 15 * 72 / RENDER_DPI

# Arabic label text for the version-control fields, read structurally instead
# of by regex on flattened text. Keys match pipeline/ingestion/metadata.py so
# the two extraction paths agree on what a field is called.
HEADER_LABELS = {
    "رقم النسخة": "doc_version",
    "تاريخ الإصدار": "issue_date",
    "تاريخ آخر مراجعة": "review_date",
}


@dataclass
class Fragment:
    """One OCR-recognised word or short phrase, with its position.

    EasyOCR returns a four-corner polygon per fragment. Pages in this corpus
    are not rotated or skewed, so the axis-aligned box bounding it is a safe
    simplification and far easier to reason about than four points.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    confidence: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @classmethod
    def from_easyocr(cls, box: list[list[float]], text: str,
                      confidence: float) -> "Fragment":
        # EasyOCR's box coordinates are numpy scalars (int32 or float32), not
        # native Python floats, and json.dumps rejects those outright.
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return cls(min(xs), min(ys), max(xs), max(ys), text, float(confidence))


# --- OCR at fragment level ------------------------------------------------

def ocr_fragments(image_bytes: bytes) -> list[Fragment]:
    """Recognise text in one rendered page, keeping each fragment's position.

    ``detail=1`` returns (box, text, confidence) instead of joined strings.
    ``paragraph=False`` keeps fragments small: paragraph grouping is exactly
    the EasyOCR behaviour that destroyed table row and column binding in
    ocr.py, and cell assignment needs fragments smaller than a whole cell.
    """
    raw = get_reader().readtext(image_bytes, detail=1, paragraph=False)
    return [Fragment.from_easyocr(box, text, conf) for box, text, conf in raw]


def _fragments_cache_path(fingerprint: str, page_number: int, dpi: int) -> Path:
    """Cache location for one page's fragment-level OCR result.

    Named distinctly from ocr.py's cache because the two calls return
    different shapes: joined text there, a fragment list here. Same
    directory, same fingerprint-plus-dpi keying, so replacing a source PDF
    invalidates both caches together.
    """
    return OCR_CACHE_DIR / f"{fingerprint}_p{page_number:03d}_{dpi}dpi_fragments.json"


def fragments_for_page(page: pymupdf.Page, page_number: int,
                        fingerprint: str, dpi: int) -> list[Fragment]:
    """Fragment-level OCR for one page, cached like ocr.py's page cache."""
    cache_file = _fragments_cache_path(fingerprint, page_number, dpi)
    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Fragment(*row) for row in raw]

    fragments = ocr_fragments(render_page(page, dpi))
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [[f.x0, f.y0, f.x1, f.y1, f.text, f.confidence] for f in fragments]
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return fragments


# --- assigning fragments to geometry --------------------------------------

def _scale_for_dpi(dpi: int) -> float:
    """PDF points to render-pixel ratio. A PDF's native unit is 72 DPI."""
    return dpi / 72


def _order_fragments(fragments: list[Fragment], tolerance_px: float) -> str:
    """Join fragments into right-to-left, top-to-bottom reading order.

    Fragments are grouped into lines by vertical proximity, lines are sorted
    top to bottom, and fragments within a line are sorted by descending x so
    the physically rightmost fragment, the start of an Arabic line, comes
    first. OCR emits fragments in whatever order its own detector found them,
    which is not reading order, so trusting that order is what scrambles
    words within a cell.

    ``tolerance_px`` is ``LINE_GROUP_TOLERANCE_PT`` converted to render-pixel
    space by the caller. Fragment coordinates never leave pixel space, so
    comparing them directly against the points-based constant silently made
    the tolerance about four times tighter than intended at 300 DPI, which
    is what first fractured a single visual line into several and scrambled
    their order rather than simply grouping them.
    """
    if not fragments:
        return ""

    ordered = sorted(fragments, key=lambda f: f.cy)
    lines: list[list[Fragment]] = []
    for frag in ordered:
        if lines and abs(frag.cy - lines[-1][-1].cy) <= tolerance_px:
            lines[-1].append(frag)
        else:
            lines.append([frag])

    line_texts = []
    for line in lines:
        line.sort(key=lambda f: f.cx, reverse=True)
        line_texts.append(" ".join(f.text for f in line))
    return "\n".join(line_texts)


def _claim_in_rect(fragments: list[Fragment], claimed: list[bool],
                    rect: tuple[float, float, float, float],
                    scale: float) -> list[Fragment]:
    """Mark and return the unclaimed fragments whose centre falls in ``rect``.

    ``rect`` is in PDF point space, from PyMuPDF; fragment coordinates are in
    render-pixel space, from EasyOCR. ``scale`` converts one to the other.
    Each fragment is claimed by at most one cell: once matched here it is
    unavailable to any later call, which is what lets prose collection at the
    end simply take whatever fragments no table claimed.
    """
    x0, y0, x1, y1 = (v * scale for v in rect)
    matched = []
    for i, frag in enumerate(fragments):
        if claimed[i]:
            continue
        if x0 <= frag.cx <= x1 and y0 <= frag.cy <= y1:
            claimed[i] = True
            matched.append(frag)
    return matched


# --- table classification and row construction ----------------------------

def _table_y_frac(table, page_height: float) -> tuple[float, float]:
    _, y0, _, y1 = table.bbox
    return y0 / page_height, y1 / page_height


def classify_table(table, page_height: float) -> str:
    """Sort a detected table into "header", "container" or "data"."""
    y0_frac, y1_frac = _table_y_frac(table, page_height)
    if table.row_count == HEADER_ROW_COUNT and y1_frac < HEADER_Y_MAX_FRAC:
        return "header"
    if (table.col_count <= CONTAINER_MAX_COLS
            and (y1_frac - y0_frac) > CONTAINER_MIN_HEIGHT_FRAC):
        return "container"
    return "data"


def _distinct_cells(row) -> list[tuple[float, float, float, float]]:
    """A row's real cell rectangles.

    PyMuPDF represents a merged cell once, at its own rectangle, and marks
    every grid position it spans with ``None``. Dropping the ``None``
    entries turns the nominal, union-of-all-rows column count into the row's
    actual cell count, which is not constant: a heading row can be one wide
    cell where a data row below it is six.
    """
    return [cell for cell in row.cells if cell is not None]


def build_rows(table, fragments: list[Fragment], claimed: list[bool],
                scale: float) -> list[list[str]]:
    """Reconstruct one table's rows as variable-length, RTL-ordered text.

    Each cell's text comes from OCR fragments assigned by geometry, never
    from ``Table.extract()``, whose own text is both corrupted and reversed.
    Empty cells are dropped rather than kept as padding, and each row is
    then reversed: a row's cells are laid out left to right in PDF space,
    but the rightmost cell is the first one read on an Arabic page.
    """
    tolerance_px = LINE_GROUP_TOLERANCE_PT * scale
    rows: list[list[str]] = []
    for row in table.rows:
        cells = []
        for rect in _distinct_cells(row):
            frags = _claim_in_rect(fragments, claimed, rect, scale)
            text = correct_ocr_misreads(clean_text(_order_fragments(frags, tolerance_px)))
            if text:
                cells.append(text)
        if cells:
            cells.reverse()
            rows.append(cells)
    return rows


def _fold_alef_maksura(text: str) -> str:
    """Fold alef maksura (ى) to yeh (ي) for matching a header label.

    Measured on Central Mail: EasyOCR reads تاريخ آخر مراجعة's first word as
    تارىخ, with a dotless ى, on every page checked, never the standard ي. An
    exact-match lookup against HEADER_LABELS silently missed every one of
    them, which is what left review_date at None even though the label was
    right there. The version-issue label happened to OCR cleanly and needed
    no help, which is why only this one word gets folded rather than the
    whole cell going through a general fuzzy match.
    """
    return text.replace("ى", "ي")


def extract_header_fields(rows: list[list[str]]) -> dict[str, str | None]:
    """Read version-control fields from a classified header table's rows.

    Reads structurally rather than by regex on flattened text, so it does
    not depend on a label and its value staying adjacent in extraction
    order. On Central Mail, whose text-layer labels are themselves
    corrupted, this is the only route to these fields; see metadata.py for
    why the text-layer regex route cannot reach it.

    A field with no clean single value beside its label is recorded as
    ``None``, matching metadata.py's rule that a missing field is an
    explicit question, not a silently absent key.
    """
    fields: dict[str, str | None] = {name: None for name in HEADER_LABELS.values()}
    for row in rows:
        for i, cell in enumerate(row):
            field_name = HEADER_LABELS.get(_fold_alef_maksura(cell))
            if field_name is None:
                continue
            others = [c for j, c in enumerate(row) if j != i]
            if len(others) == 1:
                fields[field_name] = others[0]
    return fields


# --- per-page and per-document orchestration -------------------------------

@dataclass
class TableBlock:
    """One reconstructed table.

    ``rows`` are already in logical reading order. ``page`` and ``end_page``
    are equal for an ordinary table and differ only when the page-spanning
    merge below folded a continuation table into this one.
    """

    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    col_count: int
    page: int
    end_page: int


def _claim_footer(fragments: list[Fragment], claimed: list[bool],
                   page_height: float, scale: float) -> bool:
    """Exclude the bottom-margin footer band from prose.

    Not a table, so ``find_tables()`` never sees it and the header handling
    above never touches it. Measured directly rather than assumed: the page
    number, "Internal Use" and the preparing department's name sit at
    y 0.972-0.974 on every page checked, with the nearest real body text over
    0.2 of the page higher. Returns whether the band carried the "Internal
    Use" marker, worth keeping as metadata rather than silently discarding it
    with the rest of the footer.
    """
    threshold_px = FOOTER_Y_MIN_FRAC * page_height * scale
    restricted = False
    for i, frag in enumerate(fragments):
        if claimed[i]:
            continue
        if frag.cy >= threshold_px:
            claimed[i] = True
            if "Internal Use" in frag.text:
                restricted = True
    return restricted


@dataclass
class PageLayout:
    """One page's reconstructed content: prose text plus its data tables."""

    page: int
    prose_text: str
    tables: list[TableBlock]
    header_fields: dict[str, str | None]
    restricted: bool


def page_layout(page: pymupdf.Page, page_number: int,
                 fragments: list[Fragment], dpi: int) -> PageLayout:
    """Classify every table on a page and split its fragments accordingly.

    Header and container tables never call ``build_rows``: a header table's
    fragments are claimed only to keep the version-control band out of
    prose, and a container's fragments are never claimed at all, which is
    what lets its real, nested table and its surrounding list text fall
    through to normal processing without any containment logic of its own.
    The footer band is claimed the same way, separately, since it is not a
    table at all. Whatever nothing claims by the end is prose.
    """
    page_height = page.rect.height
    scale = _scale_for_dpi(dpi)
    claimed = [False] * len(fragments)

    tables: list[TableBlock] = []
    header_fields: dict[str, str | None] = {name: None for name in HEADER_LABELS.values()}

    detected = sorted(page.find_tables().tables, key=lambda t: t.bbox[1])
    for table in detected:
        kind = classify_table(table, page_height)
        if kind == "container":
            continue
        rows = build_rows(table, fragments, claimed, scale)
        if kind == "header":
            header_fields = extract_header_fields(rows)
            continue
        if rows:
            tables.append(
                TableBlock(rows, table.bbox, table.col_count, page_number, page_number)
            )

    restricted = _claim_footer(fragments, claimed, page_height, scale)

    prose_fragments = [f for f, used in zip(fragments, claimed) if not used]
    prose_text = correct_ocr_misreads(clean_text(
        _order_fragments(prose_fragments, LINE_GROUP_TOLERANCE_PT * scale)
    ))

    return PageLayout(page_number, prose_text, tables, header_fields, restricted)


def merge_page_spans(layouts: list[PageLayout], page_height: float,
                      page_width: float) -> None:
    """Chain a table across as many pages as it actually spans, in place.

    A pairwise scan that always re-reads ``current.tables[-1]`` breaks a
    three-page chain: once a continuation is popped off the middle page's
    table list, that page has nothing left for the next boundary to compare
    against, even though the accumulating table is still open. ``open_target``
    carries the real, mutating table object forward across boundaries instead
    of re-deriving it from list state that the merge itself just emptied.

    A continuation's first row is dropped only if it duplicates the target's
    own first row once both are cleaned. That is a self-verifying check
    rather than an assumed behaviour: nothing measured on this corpus showed
    these tables repeating a header row, so the check costs nothing when
    there is none to drop and only acts when there genuinely is one.
    """
    x_tolerance = page_width * SPAN_X_TOLERANCE_FRAC
    open_target: TableBlock | None = None

    for i in range(len(layouts) - 1):
        current, following = layouts[i], layouts[i + 1]

        target = open_target
        if target is None:
            if not current.tables:
                continue
            target = current.tables[-1]

        if not following.tables:
            open_target = None
            continue

        continuation = following.tables[0]
        aligned = (
            target.col_count == continuation.col_count
            and abs(target.bbox[0] - continuation.bbox[0]) <= x_tolerance
            and abs(target.bbox[2] - continuation.bbox[2]) <= x_tolerance
        )
        reaches_bottom = target.bbox[3] / page_height >= SPAN_PREV_END_MIN_FRAC
        starts_top = continuation.bbox[1] / page_height <= SPAN_NEXT_START_MAX_FRAC

        if aligned and reaches_bottom and starts_top:
            continuation_rows = continuation.rows
            if (continuation_rows and target.rows
                    and continuation_rows[0] == target.rows[0]):
                continuation_rows = continuation_rows[1:]
            target.rows.extend(continuation_rows)
            target.bbox = (target.bbox[0], target.bbox[1],
                            target.bbox[2], continuation.bbox[3])
            target.end_page = continuation.end_page
            following.tables.pop(0)
            open_target = target
        else:
            open_target = None


def structural_version_fields(
    layouts_by_source: dict[str, list[PageLayout]]
) -> dict[str, dict]:
    """Vote across a document's pages for its version-control fields.

    Mirrors metadata.extract_version_fields's majority-vote rule: a single
    page misreading a digit, like the missing version number measured on a
    Central Alarm page during development, must not set the value for the
    whole document.
    """
    result: dict[str, dict] = {}
    for source, layouts in layouts_by_source.items():
        fields: dict[str, object] = {}
        for name in FIELD_NAMES:
            votes: Counter[str] = Counter(
                lp.header_fields[name] for lp in layouts if lp.header_fields.get(name)
            )
            if votes:
                value, count = votes.most_common(1)[0]
                fields[name] = value
                fields[f"{name}_page_count"] = count
            else:
                fields[name] = None
                fields[f"{name}_page_count"] = 0
        result[source] = fields
    return result


def layout_pdf(pdf_path: Path, dpi: int = RENDER_DPI,
                verbose: bool = True) -> list[PageLayout]:
    """Reconstruct every page of one PDF: prose plus its tables.

    Page height and width are read once from the first page and reused for
    the whole-document span merge. Every page in this corpus measures the
    same 612x792, so that is a safe simplification here rather than a
    general assumption about any future PDF.
    """
    fingerprint = file_fingerprint(pdf_path)
    doc = pymupdf.open(pdf_path)
    layouts: list[PageLayout] = []

    try:
        total_pages = len(doc)
        page_height = doc[0].rect.height
        page_width = doc[0].rect.width

        for page_number in range(1, total_pages + 1):
            page = doc[page_number - 1]
            started = time.time()
            fragments = fragments_for_page(page, page_number, fingerprint, dpi)
            layouts.append(page_layout(page, page_number, fragments, dpi))
            if verbose:
                print(f"    page {page_number:>2}/{total_pages} "
                      f"{time.time() - started:>5.1f}s "
                      f"{len(layouts[-1].tables)} tables")

        merge_page_spans(layouts, page_height, page_width)
    finally:
        doc.close()

    return layouts


def run(directory: Path = RAW_DIR, output: Path = LAYOUT_OUTPUT,
        dpi: int = RENDER_DPI) -> int:
    """Reconstruct tables and prose for every PDF in a directory."""
    print(f"Layout extraction from {directory}")
    print(f"cache {OCR_CACHE_DIR}, render {dpi} DPI\n")

    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {directory}")
        return 0

    layouts_by_source: dict[str, list[PageLayout]] = {}
    started = time.time()
    for pdf_path in pdf_paths:
        print(f"  {pdf_path.name}")
        layouts_by_source[pdf_path.name] = layout_pdf(pdf_path, dpi)
        print()

    structural_fields = structural_version_fields(layouts_by_source)
    textlayer_fields = (
        extract_version_fields(load_documents(TEXTLAYER_OUTPUT))
        if TEXTLAYER_OUTPUT.exists() else {}
    )

    documents: list[Document] = []
    for source, layouts in layouts_by_source.items():
        text_fields = textlayer_fields.get(source, {})
        struct_fields = structural_fields.get(source, {})

        for lp in layouts:
            quality = assess_quality(lp.prose_text)
            metadata = {
                "source": source,
                "page": lp.page,
                "total_pages": len(layouts),
                "char_count": len(lp.prose_text),
                "extraction_method": EXTRACTION_METHOD,
                "extraction_quality": quality["verdict"],
                "arabic_ratio": quality["arabic_ratio"],
                "fragment_ratio": quality["fragment_ratio"],
                "ocr_dpi": dpi,
                "restricted": lp.restricted,
                "tables": [
                    {"rows": t.rows, "bbox": t.bbox, "col_count": t.col_count,
                     "page": t.page, "end_page": t.end_page}
                    for t in lp.tables
                ],
            }
            for name in FIELD_NAMES:
                from_text = text_fields.get(name)
                from_structure = struct_fields.get(name)
                metadata[name] = from_text if from_text is not None else from_structure
                metadata[f"{name}_source"] = (
                    "textlayer" if from_text is not None
                    else "structural" if from_structure is not None
                    else None
                )
            documents.append(Document(text=lp.prose_text, metadata=metadata))

    if not documents:
        print("Layout extraction produced no documents.")
        return 0

    save_documents(documents, output)

    total_chars = sum(d.metadata["char_count"] for d in documents)
    total_tables = sum(len(d.metadata["tables"]) for d in documents)
    print(f"{len(documents)} documents, {total_chars:,} prose characters, "
          f"{total_tables} tables")
    print(f"elapsed {(time.time() - started) / 60:.1f} min")
    print(f"written to {output}")

    print("\nPer source")
    for source in sorted(layouts_by_source):
        pages = [d for d in documents if d.metadata["source"] == source]
        meta = pages[0].metadata
        print(f"  {source[:44]:<44} {len(pages):>3}p "
              f"{sum(len(p.metadata['tables']) for p in pages):>3} tables")
        print(f"    version {meta.get('doc_version') or 'unknown'} "
              f"({meta.get('doc_version_source')}), "
              f"issued {meta.get('issue_date') or 'unknown'} "
              f"({meta.get('issue_date_source')})")

    return len(documents)


if __name__ == "__main__":
    run()
