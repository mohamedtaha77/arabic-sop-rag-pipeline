"""Pages into chunks, and the gates that check the result.

The five modules before this one each answer a narrow question. This one walks
the corpus in reading order, hands each page to the right path, and assigns the
two fields nothing local can decide: a chunk id, which needs a per-page
counter, and a character count.

Prose comes before tables on a page, because a page's prose sits above and
around its tables and reading it first puts the heading in the tracker's hands
before any row is seen. Pages run source by source, ascending, because the
tracker carries state down a document and shuffling would scramble it.

Three gates run on every chunking run rather than in a separate script, since a
gate nobody remembers to run is not a gate: no chunk splits a table row, every
chunk has a section path, and every procedure block binds an executing actor.

This module knows nothing about what a row means or how a table becomes chunks.
It routes and it counts.
"""

from __future__ import annotations

import collections
import statistics
import time
from pathlib import Path
from typing import Any

from ..config import CHUNKS_OUTPUT, LAYOUT_OUTPUT
from ..ingestion.document import Document
from ..ingestion.storage import load_documents
from .chunk import Chunk, make_chunk_id, save_chunks
from .prose import prose_chunks
from .rows import classify_row
from .sections import UNCLASSIFIED, SectionTracker
from .tables import FURNITURE_PAGES, table_chunks

# Row kinds that belong in no chunk's row_range. Headings fold into the section
# path, actor labels into the block header their steps carry, and a split
# grid's header travels in the text of every row. Anything else left uncovered
# is dropped content, which is what the gate exists to catch.
EXPECTED_UNCOVERED = frozenset({"heading", "sub_heading", "actor", "grid"})

# Metadata carried unchanged from the page a chunk came from.
_INHERITED = ("doc_version", "issue_date", "review_date", "extraction_quality")


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Every chunk the corpus produces, in reading order."""
    by_source: dict[str, list[Document]] = collections.defaultdict(list)
    for document in documents:
        by_source[document.metadata["source"]].append(document)

    tracker = SectionTracker()
    chunks: list[Chunk] = []

    for source in sorted(by_source):
        tracker.reset()
        pages = sorted(by_source[source], key=lambda d: d.metadata["page"])
        prose_by_page = {d.metadata["page"]: d.text for d in pages}

        for document in pages:
            meta = document.metadata
            body = tracker.observe_prose(document.text)
            if meta["page"] in FURNITURE_PAGES:
                continue

            base: dict[str, Any] = {
                "source": source,
                "page": meta["page"],
                **{name: meta.get(name) for name in _INHERITED},
            }

            chunks.extend(prose_chunks(body, base, tracker))
            for table_index, table in enumerate(meta["tables"]):
                chunks.extend(
                    table_chunks(table, table_index, base, tracker, prose_by_page)
                )

    return _finalise(chunks)


def _finalise(chunks: list[Chunk]) -> list[Chunk]:
    """Assign chunk ids and character counts.

    Here rather than in the builders, so the counter an id depends on has one
    owner. Scoped to a page and a type for the reason make_chunk_id gives.
    """
    counters: collections.Counter[tuple[str, int, str]] = collections.Counter()
    for chunk in chunks:
        meta = chunk.metadata
        key = (meta["source"], meta["page"], meta["chunk_type"])
        meta["chunk_id"] = make_chunk_id(
            meta["source"], meta["page"], meta["chunk_type"], counters[key]
        )
        counters[key] += 1
        meta["char_count"] = len(chunk.text)
    return chunks


# --- verification -----------------------------------------------------------

def verify(documents: list[Document], chunks: list[Chunk]) -> list[str]:
    """Check the three stage gates. Returns failures, empty when clean."""
    failures: list[str] = []

    ids = [c.metadata["chunk_id"] for c in chunks]
    duplicates = [i for i, n in collections.Counter(ids).items() if n > 1]
    if duplicates:
        failures.append(f"duplicate chunk ids: {duplicates[:5]}")

    missing = [c.metadata["chunk_id"] for c in chunks
               if not c.metadata.get("section_path")]
    if missing:
        failures.append(f"gate 2: {len(missing)} chunks with no section_path")

    # A procedure block naming nobody is a numbered instruction with no idea
    # which unit performs it. Once the actor survives a page break and an
    # unattributed numbered list goes out as a reference, none remain, which is
    # what makes this checkable rather than merely true today.
    unbound = [c.metadata["chunk_id"] for c in chunks
               if c.metadata["chunk_type"] == "procedure_block"
               and not (c.metadata.get("actor") or c.metadata.get("unit"))]
    if unbound:
        failures.append(f"gate 3: {len(unbound)} procedure blocks bind no actor "
                        f"({unbound[:3]})")

    covered: dict[tuple[str, int, int], collections.Counter[int]] = (
        collections.defaultdict(collections.Counter)
    )
    for chunk in chunks:
        meta = chunk.metadata
        if meta.get("row_range") is None:
            continue
        key = (meta["source"], meta["page"], meta["table_id"])
        low, high = meta["row_range"]
        for index in range(low, high + 1):
            covered[key][index] += 1

    overlapping = 0
    dropped: list[str] = []
    for document in documents:
        meta = document.metadata
        if meta["page"] in FURNITURE_PAGES:
            continue
        for table_index, table in enumerate(meta["tables"]):
            key = (meta["source"], meta["page"], table_index)
            seen = covered.get(key, collections.Counter())
            overlapping += sum(1 for count in seen.values() if count > 1)
            for index, row in enumerate(table["rows"]):
                if seen[index] == 0 and classify_row(row) not in EXPECTED_UNCOVERED:
                    dropped.append(f"{meta['source'][:20]} p{meta['page']} "
                                   f"t{table_index} row{index}")

    if overlapping:
        failures.append(f"gate 1: {overlapping} rows in more than one chunk")
    if dropped:
        failures.append(f"gate 1: {len(dropped)} content rows in no chunk "
                        f"({dropped[:3]})")
    return failures


# --- entry point ------------------------------------------------------------

def run(source: Path = LAYOUT_OUTPUT, output: Path = CHUNKS_OUTPUT) -> int:
    """Chunk the layout-extracted corpus and report on the result."""
    if not source.exists():
        print(f"{source} not found. Run `python cli.py layout` first.")
        return 0

    print(f"Chunking {source.name}")
    started = time.time()

    documents = load_documents(source)
    chunks = chunk_documents(documents)
    if not chunks:
        print("Chunking produced no chunks.")
        return 0

    save_chunks(chunks, output)

    sizes = sorted(c.metadata["char_count"] for c in chunks)
    print(f"\n{len(chunks)} chunks from {len(documents)} pages, "
          f"{sum(sizes):,} characters")
    print(f"elapsed {time.time() - started:.1f}s")
    print(f"written to {output}")

    print("\nBy type")
    for name, count in collections.Counter(
        c.metadata["chunk_type"] for c in chunks
    ).most_common():
        subset = [c.metadata["char_count"] for c in chunks
                  if c.metadata["chunk_type"] == name]
        print(f"  {name:<18} {count:>4}   median {int(statistics.median(subset)):>5} "
              f"max {max(subset):>5} chars")

    print("\nPer source")
    for name in sorted({c.metadata["source"] for c in chunks}):
        subset = [c for c in chunks if c.metadata["source"] == name]
        actors = sum(1 for c in subset if c.metadata.get("actor"))
        paths = len({c.metadata["section_path"] for c in subset})
        print(f"  {name[:44]:<44} {len(subset):>4} chunks, "
              f"{paths:>3} section paths, {actors:>4} with an actor")

    print("\nSize")
    print(f"  median {int(statistics.median(sizes))}, "
          f"p90 {sizes[int(len(sizes) * 0.9)]}, max {sizes[-1]} characters")

    unclassified = sum(1 for c in chunks
                       if c.metadata["section_path"] == UNCLASSIFIED)
    print(f"  {unclassified} chunks with no section path ({UNCLASSIFIED})")

    print("\nGates")
    failures = verify(documents, chunks)
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
    else:
        print("  ok  no chunk splits or drops a table row")
        print("  ok  every chunk has a section path")
        print("  ok  every procedure block binds an executing actor")

    return len(chunks)


if __name__ == "__main__":
    run()
