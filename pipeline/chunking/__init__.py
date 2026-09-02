"""Chunking stage: pages into retrievable units.

Ingestion's unit is the page, the best citation anchor a PDF offers. Retrieval
needs something smaller, and this stage decides how much smaller and where the
cuts fall.

The corpus is nine tenths table by volume, so row shape drives almost every
decision. layout.py recovers rows as variable-length lists and stops short of
saying what they mean; five modules pick that up:

    chunk       the Chunk contract, stable ids, storage
    rows        what one row is: heading, actor, step, account line, grid
    sections    where a chunk sits, from two heading sources
    tables      a table's five kinds and the chunks each produces
    prose       running text, split on the document's own numbering

Three gates run on every pass, in chunker.verify: no chunk splits a table row,
every chunk has a section path, every procedure block binds an actor.

Run `python cli.py chunk`, after `python cli.py layout`.

A sixth module sits downstream of all five rather than beside them:

    context     three context-prefix variants of the chunks above, none,
                template and llm, for the Contextual Retrieval comparison

Run `python cli.py context`, after `python cli.py chunk`.
"""

from .chunk import CHUNK_TYPES, Chunk, load_chunks, make_chunk_id, save_chunks
from .chunker import chunk_documents, verify
from .context import (
    build_llm,
    build_none,
    build_template,
    manual_title,
    manual_titles,
    template_prefix,
)
from .context import run as run_context
from .context import verify as verify_context
from .rows import ROW_KINDS, classify_row
from .sections import UNCLASSIFIED, SectionTracker

__all__ = [
    "Chunk",
    "CHUNK_TYPES",
    "make_chunk_id",
    "save_chunks",
    "load_chunks",
    "classify_row",
    "ROW_KINDS",
    "SectionTracker",
    "UNCLASSIFIED",
    "chunk_documents",
    "verify",
    "build_none",
    "build_template",
    "build_llm",
    "manual_title",
    "manual_titles",
    "template_prefix",
    "run_context",
    "verify_context",
]
