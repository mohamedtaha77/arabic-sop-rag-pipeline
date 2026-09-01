"""Running text into chunks.

The small path. After layout.py lifts the tables out, 20,524 characters of
prose remain against 110,793 in table cells — prose is a tenth of this corpus,
and 15,994 characters of it survive heading removal to be chunked here.

It is not a tenth of the value. Every general rule the manuals state lives in
prose, in the القواعد والأحكام العامة and مقدمة sections, including the FIFO
storage rule and the periodic-inventory rationale two evaluation questions ask
for directly. The tables say who does what in which order; the prose says why.

The split follows the document's own numbering. A general-rules page is a
numbered list and each item is a self-contained rule, so a rule is the natural
unit here just as the actor swimlane is for a procedure: 33 numbered items
across 9 pages, median 212 characters. The three مقدمة pages carry no numbering
and are short enough to keep whole.

sections.py has already taken the headings off the top of the page and hands
over what is left, so one definition of where a heading stops serves both
modules.
"""

from __future__ import annotations

import re
from typing import Any

from .chunk import Chunk
from .sections import SectionTracker

# A numbered rule opening a line. The leading hyphen is the reversed form of
# the source's "1-", the same artifact rows.py records for step numbers: the
# page reads right to left, so the number's trailing dash arrives in front of
# it. Measured at 33 matches, every one opening a general rule.
RULE_START = re.compile(r"^\s*[-_]\s*\d{1,2}\s")

# Above this, a segment splits. Set from measurement rather than from a model's
# context window, which is not the binding constraint at 1,686 characters: the
# table paths already emit chunks up to 2,903 and cannot split them without
# breaking a row, so prose is held to something comparable rather than
# stricter. Two segments in the corpus exceed it.
#
# Revisit at stage 6. No tokenizer is installed yet, so this is a character
# count standing in for a token count at roughly 3 to 4 characters per Arabic
# token, and BGE-M3's tokenizer is what should settle it.
MAX_CHARS = 1200

# Below this, a leftover is dropped. After the headings come off, most pages of
# one manual leave nothing and a few leave a fragment of punctuation. A chunk
# of two characters answers nothing and only dilutes a retrieval score.
MIN_CHARS = 40


def split_segments(body: str) -> list[str]:
    """Break a prose body at each numbered rule.

    A page with no numbering comes back whole, which is right for the three
    مقدمة pages: 666 to 1,160 characters of continuous introduction with no
    internal structure, where inventing a split point would cut a sentence to
    hit a size nothing requires.
    """
    segments: list[str] = []
    current: list[str] = []
    for line in body.split("\n"):
        if RULE_START.match(line) and current:
            segments.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        segments.append("\n".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def pack_lines(segment: str, limit: int = MAX_CHARS) -> list[str]:
    """Split an oversized segment on line boundaries, never mid-line.

    Packing whole lines beats splitting sentences here. An Arabic sentence
    splitter has to decide what a full stop means in text that uses it as a
    list marker, inside numbers and after abbreviations, and OCR sprinkles
    stray periods on top of that. The line structure is real: each line of a
    rule is a sub-clause the document set on its own line, so the boundary is
    already drawn.

    A single line longer than the limit goes out whole. An oversized chunk is a
    size problem; a chunk cut mid-clause is a meaning problem.
    """
    if len(segment) <= limit:
        return [segment]

    packed: list[str] = []
    current: list[str] = []
    length = 0
    for line in segment.split("\n"):
        if current and length + len(line) + 1 > limit:
            packed.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        packed.append("\n".join(current))
    return packed


def prose_chunks(body: str, meta: dict[str, Any],
                  tracker: SectionTracker) -> list[Chunk]:
    """Every chunk one page's prose body produces.

    ``body`` is what sections.py returned after taking the page's headings, not
    the page's raw prose, so the path read here reflects them.
    """
    if len(body) < MIN_CHARS:
        return []

    path = tracker.path()
    chunks: list[Chunk] = []
    for segment in split_segments(body):
        for piece in pack_lines(segment):
            if len(piece) < MIN_CHARS:
                continue
            chunks.append(Chunk(piece, {
                **meta,
                "chunk_type": "prose",
                "section_path": path,
                "end_page": meta["page"],
                "table_id": None,
                "row_range": None,
                "actor": None,
                "unit": None,
            }))
    return chunks
