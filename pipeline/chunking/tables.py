"""One reconstructed table into chunks.

This is where the stage's row gate is met or missed. Every path below consumes
whole rows and records which ones in ``row_range``, so chunker.py can check the
gate against the output rather than trust it.

A detected table is not one thing. Measured across the corpus's 60 tables, five
kinds exist, and a chunker that treats them alike works on the approval
matrices and destroys the procedure grids, or the reverse:

    procedure   an executing unit, an executor, and their numbered steps.
                52 tables, and the bulk of the corpus.
    grid        a true data grid, three or more real cells per row. 9 tables.
    accounting  the journal-entry annex. 1 table, 66 rows.
    approval    the signature page naming who approved the manual. 3 tables.
    revision    the جدول التعديلات log, recognised by its header.

The procedure path splits on the actor swimlane rather than a token count. A
step number and a description mean little without knowing which unit performs
them, and the document already draws that boundary.

This module decides no section path and assigns no chunk ids. It asks the
tracker for the path, and chunker.py assigns ids in one place so the per-page
counter has a single owner. It also writes no path into chunk text: stage 4
compares three prefixing strategies against a no-prefix baseline, and a path
baked in here would quietly make that baseline something else.
"""

from __future__ import annotations

import re
from typing import Any

from .chunk import Chunk
from .rows import classify_row, fold_for_match, match_key, step_number
from .sections import SectionTracker

# Page 1 is the cover, one cell holding the bank's name and the manual's title.
# Page 2 is the contents, whose entries near-duplicate the real procedure
# headings; retrieving one returns something that looks like an answer and
# contains none.
FURNITURE_PAGES = frozenset({1, 2})

# A grid small enough to keep whole. Above this its rows go out individually
# with the header re-attached, so a question about one row does not drag in the
# rest. This keeps the 5-row approval matrix whole and splits the 9-row one,
# which is the split Q5 needs to reconcile a row on each page.
GRID_WHOLE_MAX_ROWS = 6

# Two tables whose shape gives them away as something else: the signature page
# is majority one-cell rows, exactly like a procedure table, and the revision
# log is a small grid, exactly like a reports table. Both answer real
# questions, so both are kept and identified by their header row instead.
APPROVAL_HEADER = ("الدائرة", "الاسم", "التاريخ", "التوقيع")
REVISION_HEADER = ("النسخة", "الجهة طالبة التعديل", "التاريخ", "أسباب التعديل")

_APPROVAL_KEY = tuple(match_key(c) for c in APPROVAL_HEADER)
_REVISION_KEY = tuple(match_key(c) for c in REVISION_HEADER)

# Both name their own section rather than inheriting one, because an exact
# header match identifies them with more certainty than the surrounding page
# offers. Inheriting gave the wrong answer: one page carries four section
# headings in its prose and the tracker holds only the last, so its revision
# log came out under الملاحق. A sweep against the source PDFs found the word
# جدول in no chunk's text or path at all.
SELF_NAMING_SECTIONS = {
    "approval": "الموافقات",
    "revision": "جدول التعديلات",
}

# The two labels in their source spelling, for writing back into chunk text.
# rows.py is what matches them tolerantly.
UNIT_LABEL = "الجهة المنفذة"
ACTOR_LABEL = "المنفذ"

# Markers inside the accounting annex that name no entry: مذكورين is "sundry
# accounts", and a cell opening with ح is a debit or credit line whose account
# name did not split into its own cell.
_ENTRY_STRUCTURE = re.compile(
    fold_for_match(r"^\s*(?:(?:من|إلى|إل)\s+)?(?:مذكورين|ح\s*[/_])")
)


def _is_entry_structure(cell: str) -> bool:
    return bool(_ENTRY_STRUCTURE.match(fold_for_match(cell)))


def _join(parts: list[str]) -> str:
    return "\n".join(part for part in parts if part)


# --- table classification ---------------------------------------------------

def _header_row(rows: list[list[str]]) -> list[str] | None:
    """The first row wide enough to be a grid header, if any."""
    for row in rows:
        if len(row) >= 3:
            return row
    return None


def table_kind(rows: list[list[str]]) -> str:
    """Sort a table into one of the five kinds.

    Header matches go first. The signature page is majority one-cell and would
    otherwise fall to the procedure path, where it has no steps and emits
    nothing at all.
    """
    header = _header_row(rows)
    if header is not None:
        key = tuple(match_key(c) for c in header)
        if key == _APPROVAL_KEY:
            return "approval"
        if key == _REVISION_KEY:
            return "revision"

    kinds = [classify_row(row) for row in rows]
    if any(kind == "entry_line" for kind in kinds):
        return "accounting"
    if sum(kind == "grid" for kind in kinds) > len(rows) / 2:
        return "grid"
    return "procedure"


# --- the five paths ---------------------------------------------------------

def _grid_chunks(rows: list[list[str]], meta: dict[str, Any],
                  chunk_type: str) -> list[Chunk]:
    """A data grid, with the header re-attached to every row.

    The header travelling with each row is what makes a row answerable alone.
    "مدير الفرع" by itself is a job title; under the header
    "الموافقة المطلوبة من طرف الجهة الطالبة" it answers who approves a request.

    One deliberate exception to the row-coverage gate lives here. A split
    grid's header appears in the text of every chunk and in no chunk's
    ``row_range``, leaving 3 rows corpus-wide covered by nothing. Listing the
    header index in each range would instead report it as covered nine times
    over, turning a clean no-double-counting check into a permanently noisy
    one. chunker.py names grid headers as the expected exception rather than
    tolerating any uncovered row.
    """
    header = _header_row(rows)
    if header is None:
        return []

    header_index = rows.index(header)
    data = [(i, r) for i, r in enumerate(rows) if i != header_index and r]
    if not data:
        return []

    header_text = " | ".join(header)

    if len(data) <= GRID_WHOLE_MAX_ROWS:
        body = _join([header_text] + [" | ".join(r) for _, r in data])
        return [Chunk(body, {**meta, "chunk_type": chunk_type,
                             "row_range": [0, len(rows) - 1]})]

    return [
        Chunk(_join([header_text, " | ".join(row)]),
              {**meta, "chunk_type": chunk_type, "row_range": [i, i]})
        for i, row in data
    ]


def _approval_chunk(rows: list[list[str]], meta: dict[str, Any]) -> list[Chunk]:
    """The signature page, kept whole.

    Seven one-cell rows of job titles under a four-column header. Split per
    row, each chunk holds one job title and no indication of what was approved.
    That is worse than useless: those same titles are real answers elsewhere.
    """
    body = _join([" | ".join(row) for row in rows if row])
    if not body:
        return []
    return [Chunk(body, {**meta, "chunk_type": "approval",
                         "row_range": [0, len(rows) - 1]})]


def _procedure_chunks(rows: list[list[str]], meta: dict[str, Any],
                       tracker: SectionTracker,
                       prose_by_page: dict[int, str],
                       page_breaks: dict[int, int]) -> list[Chunk]:
    """Consecutive steps under one actor, one chunk each.

    A block closes when the executor changes, when the executing unit changes,
    or when a heading opens a new procedure. The document draws all three, so
    none of them is a tunable number.

    A ``continuation`` row with no block open still opens one. Those 74 rows
    are step text whose number cell did not survive OCR, and dropping them to
    keep the block shape tidy would lose real procedure content.

    The executor comes from the tracker, not a local, so it survives a page
    break that split one procedure across two tables.
    """
    chunks: list[Chunk] = []
    lines: list[str] = []
    first_row: int | None = None
    last_row = 0

    def flush() -> None:
        nonlocal lines, first_row
        if lines and first_row is not None:
            header = []
            if tracker.unit:
                header.append(f"{UNIT_LABEL}: {tracker.unit}")
            if tracker.actor:
                header.append(f"{ACTOR_LABEL}: {tracker.actor}")
            chunks.append(Chunk(
                _join(header + lines),
                {**meta, "chunk_type": "procedure_block", "actor": tracker.actor,
                 "unit": tracker.unit, "section_path": tracker.path(),
                 "row_range": [first_row, last_row]},
            ))
        lines = []
        first_row = None

    for index, row in enumerate(rows):
        if index in page_breaks:
            # A merged table crossing onto a new page. That page's prose may
            # announce the next procedure, and the heading has to land before
            # the rows it governs rather than after the whole table.
            flush()
            tracker.observe_prose(prose_by_page.get(page_breaks[index], ""))

        if tracker.observe_row(row):
            flush()
            continue

        kind = classify_row(row)
        if kind == "actor":
            # Nothing closes while nothing is bound, so content above a table's
            # first label row picks up the binding declared below it. One page
            # opens with a full-width row naming what the procedure covers,
            # printed above the الجهة المنفذة row; closing there left it
            # attributed to nobody. Holding it lets it take the executing unit,
            # and the المنفذ row below closes it as a scope line rather than
            # folding it into step 1.
            if tracker.actor or tracker.unit:
                flush()
            tracker.set_actor(row[0], row[1])
            continue

        if kind == "step":
            number = step_number(row[0])
            lines.append(f"{number}. {row[1]}")
        elif kind == "grid":
            lines.append(" | ".join(row))
        else:
            lines.append(row[-1])

        if first_row is None:
            first_row = index
        last_row = index

    flush()

    # A table of numbered rows naming no executor, and inheriting none from the
    # procedure above it, is a reference list rather than a procedure: the
    # forms and annex indexes, and the مقدمة's general rules. All three used to
    # go out as procedure blocks with a null actor, telling a retriever they
    # were steps somebody performs.
    #
    # The test only works because the actor survives a page break. Without
    # that, the three tables opening partway through a procedure would be
    # mislabelled here too.
    if chunks and not any(c.metadata["actor"] or c.metadata["unit"] for c in chunks):
        for chunk in chunks:
            chunk.metadata["chunk_type"] = "reference"
    return chunks


def _accounting_chunks(rows: list[list[str]], meta: dict[str, Any],
                        tracker: SectionTracker) -> list[Chunk]:
    """One journal entry per chunk.

    The largest table in the corpus at 66 rows. Emitting it whole would mean a
    question about one entry always retrieves fourteen irrelevant ones.

    A block ends at the next standalone cell that follows at least one account
    line, and that qualification carries the rule. Opening a new entry at every
    standalone cell lost content twice: two rows are account names printed
    without the ح marker their neighbours carry, so each looked like a title,
    closed a block holding nothing, and vanished. Requiring an account line
    first keeps a run of bare account names inside the entry they belong to.

    Every row lands in ``lines``, including the one that names the entry, so
    ``row_range`` covers the whole block. Holding the title outside the text
    left 17 rows present in the output but absent from every range.
    """
    chunks: list[Chunk] = []
    lines: list[str] = []
    first_row: int | None = None
    last_row = 0
    has_account_line = False

    def flush() -> None:
        nonlocal lines, first_row, has_account_line
        if lines and first_row is not None:
            chunks.append(Chunk(
                _join(lines),
                {**meta, "chunk_type": "accounting_entry",
                 "section_path": tracker.path(),
                 "row_range": [first_row, last_row]},
            ))
        lines = []
        first_row = None
        has_account_line = False

    for index, row in enumerate(rows):
        is_account_line = len(row) > 1 or _is_entry_structure(row[0])
        if not is_account_line and has_account_line:
            flush()

        lines.append(" ".join(row) if len(row) > 1 else row[0])
        has_account_line = has_account_line or is_account_line
        if first_row is None:
            first_row = index
        last_row = index

    flush()
    return chunks


# --- entry point ------------------------------------------------------------

def table_chunks(table: dict[str, Any], table_index: int,
                  base: dict[str, Any], tracker: SectionTracker,
                  prose_by_page: dict[int, str]) -> list[Chunk]:
    """Every chunk one table produces.

    ``base`` holds what is true of the whole page, source and version metadata;
    each path adds what is true of the chunk.
    """
    rows: list[list[str]] = table["rows"]
    if not rows or base["page"] in FURNITURE_PAGES:
        return []

    meta = {
        **base,
        "end_page": table.get("end_page", base["page"]),
        "table_id": table_index,
        "section_path": tracker.path(),
        "actor": None,
        "unit": None,
        "row_range": None,
    }
    page_breaks = {index: page for index, page in table.get("row_page_breaks", [])}
    kind = table_kind(rows)

    if kind in SELF_NAMING_SECTIONS:
        meta = {**meta, "section_path": SELF_NAMING_SECTIONS[kind]}
    if kind == "approval":
        return _approval_chunk(rows, meta)
    if kind == "revision":
        return _grid_chunks(rows, meta, "revision")
    if kind == "grid":
        return _grid_chunks(rows, meta, "grid_row" if len(rows) - 1 > GRID_WHOLE_MAX_ROWS
                            else "grid_table")
    if kind == "accounting":
        return _accounting_chunks(rows, meta, tracker)
    return _procedure_chunks(rows, meta, tracker, prose_by_page, page_breaks)
