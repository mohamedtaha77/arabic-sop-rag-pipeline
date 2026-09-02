"""The Chunk contract shared by every chunking path.

Ingestion's unit is the page, the best citation anchor a PDF offers. Retrieval
needs something smaller: a page here holds a whole procedure, and embedding it
whole dilutes the one step a question asks about.

Chunk mirrors Document, text plus a metadata dict, so downstream stages learn
one shape rather than two, and never need to know which of the five builders
produced a given chunk. Only ``chunk_type`` says which.

Nothing here decides what a chunk is. The splitting rules live in rows.py,
sections.py, tables.py and prose.py; this file is the envelope.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Every value chunk_type takes. Explicit rather than implicit in whatever the
# builders emit, because stage 8's router filters on it and a typo would return
# nothing rather than raise. Each is a shape measured in the corpus:
#
#   procedure_block   heading + executing actor + its consecutive steps
#   reference         a numbered list naming no executor: the forms and annex
#                     indexes, and the مقدمة's general rules
#   grid_row          one row of a data grid, header re-attached
#   grid_table        a whole data grid, when small enough to keep whole
#   accounting_entry  one journal entry from the annex
#   approval          the signature page naming who approved the manual
#   revision          a row of the جدول التعديلات log
#   prose             a split of the running text in each manual's front matter
CHUNK_TYPES = (
    "procedure_block",
    "reference",
    "grid_row",
    "grid_table",
    "accounting_entry",
    "approval",
    "revision",
    "prose",
)

# One color per chunk_type, shared by every local viewer this project builds
# (store/browse.py, embedding/visualize.py) so a color means the same thing
# wherever it appears. Chosen to stay in the family of Housing Bank's own two
# brand colors, navy and gold, each given a lighter and a deeper shade, with
# four further hues added only so all eight types stay told apart at a
# glance. Never read by any gate or any retrieval code; it is display only.
CHUNK_TYPE_COLORS = {
    "procedure_block": "#005295",
    "prose": "#5b8fb9",
    "grid_row": "#c8b18b",
    "grid_table": "#a9823f",
    "accounting_entry": "#3c7a5c",
    "approval": "#6b4c9a",
    "reference": "#7d7d7d",
    "revision": "#b6484f",
}


@dataclass
class Chunk:
    """One retrievable unit of source text with its provenance.

    Metadata keys, documented rather than enforced for the reason Document
    gives: a dataclass with fifteen optional fields reads worse than a dict
    with a stated contract, and the check that matters (no row lost or
    duplicated) spans the whole corpus, which no per-object validator sees.

        chunk_id            stable, readable identifier; see make_chunk_id
        source              PDF filename, carried through from ingestion
        page                first page this chunk's content appears on
        end_page            last page; differs only for a table layout.py
                            merged across a page break
        section_path        where this sits, "الإجراءات > 3. ..."
        chunk_type          one of CHUNK_TYPES
        actor               the role performing these steps, or None
        unit                the department they belong to, or None
        table_id            which table on the page, or None for prose
        row_range           [first, last] row indices consumed, or None
        doc_version         version fields, carried from ingestion
        issue_date
        review_date
        extraction_quality  assess_quality's verdict on this chunk's own
                            text, computed once chunker.py knows what
                            that text is; see chunker._finalise
        char_count          length of text

    ``context_prefix`` is absent on purpose. Stage 4 adds it, and a placeholder
    now would let something depend on a key that means nothing yet.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# --- identifiers ------------------------------------------------------------

# Dropped when building a source slug. Long enough to turn three near-identical
# manual titles into three legible prefixes: two of them share four words.
_SLUG_STOPWORDS = frozenset({
    "and", "the", "of", "for", "unit", "tasks", "procedures", "manual",
    "operation", "operations",
})

_NON_WORD = re.compile(r"[^a-z0-9]+")


def source_slug(source: str, words: int = 2) -> str:
    """Shorten a PDF filename into a readable id prefix.

    Derived rather than looked up, so a fourth manual needs no code change. Two
    filenames could in principle slug alike; ``source`` stays in the metadata
    in full, and chunker.py asserts ids are unique across the corpus, which is
    where a collision would surface.
    """
    stem = Path(source).stem.lower()
    significant = [w for w in _NON_WORD.split(stem) if w and w not in _SLUG_STOPWORDS]
    return "_".join(significant[:words]) or "doc"


def make_chunk_id(source: str, page: int, chunk_type: str, index: int) -> str:
    """Build a stable id, e.g. ``assets_wearhouse_p10_procedure_block_02``.

    A corpus-wide sequence would be shorter and would break stage 5. Its golden
    set names gold chunk ids read off the rendered pages by hand, and under
    sequential numbering, inserting one chunk on page 8 renumbers everything
    after it and invalidates the set with nothing raising.

    Scoping the counter to one page and one type contains that: editing how
    prose splits cannot move a procedure block's id, and editing page 10 cannot
    move page 11's. The id also states its own provenance, so a citation can be
    checked against the PDF by eye.
    """
    return f"{source_slug(source)}_p{page:02d}_{chunk_type}_{index:02d}"


# --- storage ----------------------------------------------------------------

def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to JSON.

    UTF-8 is required rather than stylistic: the Windows default cannot
    represent Arabic and raises on the first chunk. ensure_ascii stays off so
    the file remains readable in an editor, which is the only practical way to
    check Arabic output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(chunk) for chunk in chunks]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks back from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(text=c["text"], metadata=c["metadata"]) for c in raw]
