"""Extraction by rendering pages to images and recognising the pixels.

Required when a PDF's text layer is unreliable. Rendering resolves fonts the
same way a viewer does, so a damaged ToUnicode table is bypassed entirely: the
drawn glyphs are correct even when the characters behind them are not.

On the reference corpus this raised fidelity against known correct spellings
from 44% to 97%. It does not recover table structure. A row of cells is
flattened into one line, so column meaning is lost. The bounding boxes
discarded by detail=0 in ocr_image are where a fix for that would begin.

OCR runs locally. The source corpus is classified for internal use, which rules
out cloud OCR and vision APIs regardless of their accuracy.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pymupdf

from ..config import (
    OCR_CACHE_DIR,
    OCR_LANGUAGES,
    OCR_OUTPUT,
    RAW_DIR,
    RENDER_DPI,
    TEXTLAYER_OUTPUT,
)
from .cleaning import clean_text, correct_ocr_misreads
from .document import Document
from .metadata import apply_version_fields, extract_version_fields
from .quality import assess_quality
from .storage import load_documents, save_documents

EXTRACTION_METHOD = "ocr"

# The model costs roughly 90 seconds and a gigabyte to load, so it is built
# once and reused for every page.
_reader = None


def get_reader():
    """Build the EasyOCR reader on first use and reuse it thereafter."""
    global _reader
    if _reader is None:
        import easyocr
        import torch

        use_gpu = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        print(f"  loading OCR model ({'+'.join(OCR_LANGUAGES)}) on {device}")
        if not use_gpu:
            print("  no CUDA device found, expect roughly 10x slower runtime")

        started = time.time()
        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu, verbose=False)
        print(f"  model ready in {time.time() - started:.0f}s")
    return _reader


def file_fingerprint(path: Path) -> str:
    """Short hash of a file's contents, used to key its cache entries.

    Hashing contents rather than the filename means replacing a source PDF
    with a corrected export invalidates its cached pages automatically, even if
    the filename is unchanged.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def cache_path(fingerprint: str, page_number: int, dpi: int) -> Path:
    """Location of one page's cached OCR result.

    DPI is part of the name because output differs by resolution. Changing it
    re-runs OCR instead of serving results produced at the previous setting.
    """
    return OCR_CACHE_DIR / f"{fingerprint}_p{page_number:03d}_{dpi}dpi.txt"


def render_page(page: pymupdf.Page, dpi: int = RENDER_DPI) -> bytes:
    """Rasterise one page to PNG bytes.

    Returned in memory because EasyOCR accepts a byte stream directly.
    """
    return page.get_pixmap(dpi=dpi).tobytes("png")


def ocr_image(image_bytes: bytes) -> str:
    """Recognise text in one rendered page.

    detail=0 returns plain strings instead of (box, text, confidence) tuples.
    paragraph=True groups nearby fragments into blocks rather than returning
    loose words.
    """
    blocks = get_reader().readtext(image_bytes, detail=0, paragraph=True)
    return "\n".join(blocks)


def ocr_pdf(pdf_path: Path, dpi: int = RENDER_DPI,
            verbose: bool = True) -> list[Document]:
    """OCR every page of one PDF, using cached results where available.

    Every page is processed rather than only those the quality gate flagged.
    The gate detects undecodable bytes and wrong script characters, but the
    dominant failure on this corpus produces valid Arabic letters spelling the
    wrong word, which no character level check can identify. When the damaged
    subset cannot be determined, all pages are re-read.
    """
    fingerprint = file_fingerprint(pdf_path)
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    documents: list[Document] = []
    cached = 0

    try:
        total_pages = len(doc)
        for page_number in range(1, total_pages + 1):
            cache_file = cache_path(fingerprint, page_number, dpi)

            if cache_file.exists():
                raw_text = cache_file.read_text(encoding="utf-8")
                cached += 1
            else:
                started = time.time()
                raw_text = ocr_image(render_page(doc[page_number - 1], dpi))
                cache_file.write_text(raw_text, encoding="utf-8")
                if verbose:
                    print(f"    page {page_number:>2}/{total_pages} "
                          f"{time.time() - started:>5.1f}s "
                          f"{len(raw_text):>5} chars")

            text = correct_ocr_misreads(clean_text(raw_text))
            if not text:
                if verbose:
                    print(f"    page {page_number}: OCR returned nothing")
                continue

            quality = assess_quality(text)
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                        "total_pages": total_pages,
                        "char_count": len(text),
                        "extraction_method": EXTRACTION_METHOD,
                        "extraction_quality": quality["verdict"],
                        "arabic_ratio": quality["arabic_ratio"],
                        "fragment_ratio": quality["fragment_ratio"],
                        "ocr_dpi": dpi,
                    },
                )
            )
    finally:
        doc.close()

    if verbose and cached:
        print(f"    {cached} pages served from cache")
    return documents


def version_fields_from_textlayer() -> dict[str, dict]:
    """Read version control fields from the text layer output.

    Not parsed from OCR text. Paragraph grouping collapses the header table's
    labels onto one line and scatters their values, so the label to value
    binding is gone. See pipeline/ingestion/metadata.py for the measurements.
    """
    if not TEXTLAYER_OUTPUT.exists():
        print(f"  {TEXTLAYER_OUTPUT.name} not found. Run the text layer route "
              f"first to populate version metadata.")
        return {}
    return extract_version_fields(load_documents(TEXTLAYER_OUTPUT))


def run(directory: Path = RAW_DIR, output: Path = OCR_OUTPUT,
        dpi: int = RENDER_DPI) -> int:
    """OCR every PDF in a directory and write the result to disk."""
    print(f"OCR extraction from {directory}")
    print(f"cache {OCR_CACHE_DIR}, render {dpi} DPI\n")

    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {directory}")
        return 0

    documents: list[Document] = []
    started = time.time()
    for pdf_path in pdf_paths:
        print(f"  {pdf_path.name}")
        documents.extend(ocr_pdf(pdf_path, dpi))
        print()

    if not documents:
        print("OCR produced no text.")
        return 0

    apply_version_fields(documents, version_fields_from_textlayer())
    save_documents(documents, output)

    total_chars = sum(d.metadata["char_count"] for d in documents)
    print(f"{len(documents)} documents, {total_chars:,} characters, "
          f"{total_chars // len(documents):,} per page")
    print(f"elapsed {(time.time() - started) / 60:.1f} min")
    print(f"written to {output}")

    print("\nPer source")
    for source in sorted({d.metadata["source"] for d in documents}):
        pages = [d for d in documents if d.metadata["source"] == source]
        chars = sum(d.metadata["char_count"] for d in pages)
        fragmentation = sum(d.metadata["fragment_ratio"] for d in pages)
        meta = pages[0].metadata
        print(f"  {source[:44]:<44} {len(pages):>3}p "
              f"{chars // len(pages):>5} ch/p "
              f"frag {fragmentation / len(pages):>5.1%}")
        print(f"    version {meta.get('doc_version') or 'unknown'}, "
              f"issued {meta.get('issue_date') or 'unknown'}, "
              f"reviewed {meta.get('review_date') or 'unknown'}")

    return len(documents)


if __name__ == "__main__":
    run()
