"""Renders pages and lays out the worksheet a person reads to fill the golden set.

Nothing here decides an answer. This module gets a rendered page in front of a
person and a chunk's full text beside it; the reading is done outside this
file entirely, in an editor, against the render.

Two outputs, both for a person rather than for code:

    render_pages    one PNG per page reference, under data/golden/pages/
    write_worksheet a single text file, ordered by page rather than by
                     question, so a page is read once and everything on it is
                     answered together in one pass
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..chunking.chunk import Chunk, load_chunks, source_slug
from ..config import (
    CHUNKS_OUTPUT,
    GOLDEN_PAGES_DIR,
    GOLDEN_RENDER_DPI,
    GOLDEN_SET,
    GOLDEN_WORKSHEET,
    RAW_DIR,
)
from ..ingestion.ocr import file_fingerprint, render_page
from .question import Question, load_golden, page_ref, parse_page_ref


# --- source resolution ---------------------------------------------------------

def slug_to_path() -> dict[str, Path]:
    """Map each corpus source's slug to its PDF, built by scanning RAW_DIR.

    Derived rather than hardcoded, the same reason chunk.source_slug itself is
    derived rather than a lookup table: a fourth manual dropped into data/raw
    needs no code change here either.
    """
    return {source_slug(p.name): p for p in sorted(RAW_DIR.glob("*.pdf"))}


def resolve_slug(token: str, slugs: dict) -> str:
    """Resolve a CLI shorthand like "alarm" to its full slug, "central_alarm".

    The canonical page_ref format inside golden_set.json always carries the
    full slug, so a page reference is an unambiguous prefix of the chunk ids
    on that page with no lookup needed. This function exists only for typing
    speed at the command line: the token has to be an unambiguous substring of
    exactly one known slug, never a guess.
    """
    token = token.lower()
    if token in slugs:
        return token
    matches = [s for s in slugs if token in s]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"{token!r} matches no source slug: {sorted(slugs)}")
    raise ValueError(f"{token!r} matches more than one source slug: {matches}")


# --- rendering -------------------------------------------------------------------

def golden_page_path(slug: str, fingerprint: str, page: int, dpi: int) -> Path:
    """Location of one rendered page.

    Mirrors ocr.py's cache_path: the fingerprint is part of the name so a
    replaced PDF produces a new render rather than silently reusing a stale
    one, and dpi is part of the name because GOLDEN_RENDER_DPI is itself a
    considered decision that might later change.
    """
    return GOLDEN_PAGES_DIR / f"{slug}_{fingerprint}_p{page:02d}_{dpi}dpi.png"


def render_pages(
    refs: list[str], dpi: int = GOLDEN_RENDER_DPI, verbose: bool = True
) -> dict[str, Path]:
    """Render each page reference to a PNG, skipping ones already rendered.

    A PDF is opened once and reused across every page requested from it,
    rather than reopened per page, the same care ocr_pdf takes with the try
    and finally around one open document.

    Returns a dict from page reference to its render path, so callers that
    need both the set of files and which reference produced which file, the
    worksheet writer among them, do not have to recompute golden_page_path a
    second time.
    """
    slugs = slug_to_path()
    GOLDEN_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    open_docs: dict[str, pymupdf.Document] = {}
    try:
        for ref in refs:
            slug, page_number = parse_page_ref(ref)
            if slug not in slugs:
                raise ValueError(
                    f"unknown source slug {slug!r} in page reference {ref!r}, "
                    f"known slugs: {sorted(slugs)}"
                )
            pdf_path = slugs[slug]
            fingerprint = file_fingerprint(pdf_path)
            out_path = golden_page_path(slug, fingerprint, page_number, dpi)

            if not out_path.exists():
                if slug not in open_docs:
                    open_docs[slug] = pymupdf.open(pdf_path)
                doc = open_docs[slug]
                image_bytes = render_page(doc[page_number - 1], dpi=dpi)
                out_path.write_bytes(image_bytes)
                if verbose:
                    print(f"  rendered {out_path.name}")

            written[ref] = out_path
    finally:
        for doc in open_docs.values():
            doc.close()
    return written


def parse_range_arg(arg: str) -> list[str]:
    """Parse a --pages shorthand like "alarm:5-8" or "assets:10" into refs.

    The part before the colon is resolved through resolve_slug, so "alarm" and
    "central_alarm" both work; the part after is one page or an inclusive
    range. This is what lets a page turning out to hold an answer get rendered
    on the spot, without first editing the skeleton to add it.
    """
    token, sep, page_spec = arg.partition(":")
    if not sep or not page_spec:
        raise ValueError(f"expected SLUG:PAGE or SLUG:START-END, got {arg!r}")

    slug = resolve_slug(token, slug_to_path())
    if "-" in page_spec:
        start_s, end_s = page_spec.split("-", 1)
        start, end = int(start_s), int(end_s)
    else:
        start = end = int(page_spec)

    return [page_ref(slug, p) for p in range(start, end + 1)]


# --- the worksheet -----------------------------------------------------------

def collect_pages(questions: list[Question]) -> list[str]:
    """Every page reference any question points at, first-seen order.

    First-seen across Q1 to Q20 rather than sorted, since the question order
    already sweeps the corpus front to back, and a reader working through the
    worksheet top to bottom effectively reads the manuals in page order too.
    """
    seen: list[str] = []
    for question in questions:
        for ref in question.pages_to_read:
            if ref not in seen:
                seen.append(ref)
    return seen


def chunks_on_page(chunks: list[Chunk], slug: str, page: int) -> list[Chunk]:
    """Every chunk whose page range includes this page.

    A page-span merged table is emitted once, at its first page, but its rows
    physically sit on every page it spans. Checking the whole [page, end_page]
    window rather than only metadata["page"] is what makes that table show up
    on the worksheet block for its later pages too, where a reader would
    otherwise see rows on the render with no chunk beside them.
    """
    hits = []
    for chunk in chunks:
        meta = chunk.metadata
        if source_slug(meta["source"]) != slug:
            continue
        start = meta["page"]
        end = meta.get("end_page") or start
        if start <= page <= end:
            hits.append(chunk)
    return hits


def build_worksheet(
    questions: list[Question], chunks: list[Chunk], rendered: dict[str, Path]
) -> str:
    """Compose the worksheet text, one block per page reference.

    Returns a string rather than writing directly, so a caller building it in
    a test can compare the text without touching disk.
    """
    pages = collect_pages(questions)
    lines: list[str] = []

    for ref in pages:
        slug, page_number = parse_page_ref(ref)
        pointing = [q.id for q in questions if ref in q.pages_to_read]
        page_chunks = chunks_on_page(chunks, slug, page_number)

        lines.append("=" * 80)
        lines.append(f"{ref}")
        lines.append(f"render: {rendered.get(ref, '(not rendered)')}")
        lines.append(f"questions: {', '.join(pointing) if pointing else 'none'}")
        lines.append("")

        if not page_chunks:
            lines.append("  (no chunks on this page; furniture, e.g. cover or "
                          "table of contents)")
        for chunk in page_chunks:
            meta = chunk.metadata
            lines.append(
                f"  --- {meta['chunk_id']}  type={meta['chunk_type']}  "
                f"table={meta['table_id']}  rows={meta['row_range']}  "
                f"chars={meta['char_count']}"
            )
            lines.append(f"      section: {meta['section_path']}")
            for text_line in chunk.text.splitlines():
                lines.append(f"      {text_line}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


def write_worksheet(
    questions: list[Question],
    chunks: list[Chunk],
    rendered: dict[str, Path],
    path: Path = GOLDEN_WORKSHEET,
) -> None:
    """Write the worksheet to disk.

    UTF-8 required, the same reason chunk.save_chunks and question.save_golden
    both give: the Windows console default cannot represent Arabic. Open this
    file in an editor, never a console.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_worksheet(questions, chunks, rendered), encoding="utf-8")


def run(questions_path: Path = GOLDEN_SET, chunks_path: Path = CHUNKS_OUTPUT) -> bool:
    """Render every page any question in the golden set points at, and write
    the worksheet.
    """
    if not questions_path.exists():
        print(f"Missing {questions_path.name}. The skeleton has to exist first.")
        return False

    questions, _ = load_golden(questions_path)
    chunks = load_chunks(chunks_path)

    refs = collect_pages(questions)
    print(f"rendering {len(refs)} page(s)")
    rendered = render_pages(refs)

    write_worksheet(questions, chunks, rendered)
    print(f"wrote {GOLDEN_WORKSHEET}")
    return True


def run_pages(page_args: list[str]) -> bool:
    """Render an arbitrary set of --pages ranges on demand, no worksheet.

    This is the "a page turning out to hold the answer" path: render it and
    look at it without editing the skeleton or regenerating the whole
    worksheet first.
    """
    refs: list[str] = []
    for arg in page_args:
        refs.extend(parse_range_arg(arg))

    rendered = render_pages(refs)
    for ref, path in rendered.items():
        print(f"  {ref} -> {path}")
    return True
