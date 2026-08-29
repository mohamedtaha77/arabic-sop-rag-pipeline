"""Extraction from a PDF's embedded text layer, via pypdf.

This is the fast route, around 50ms per page against roughly 10s per page for
OCR, and it is the correct default for any well formed PDF. It also produces
the quality assessment that determines whether OCR is needed at all.

On the reference corpus the text layer is unreliable for prose because the
source PDFs have damaged ToUnicode tables. It remains the better route for the
version control header, where layout order preserves the binding between each
label and its value. See pipeline/ingestion/metadata.py.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ..config import RAW_DIR, TEXTLAYER_OUTPUT
from .cleaning import clean_text
from .document import Document
from .metadata import apply_version_fields, extract_version_fields
from .quality import assess_quality
from .storage import save_documents

EXTRACTION_METHOD = "textlayer"


def load_pdf(pdf_path: Path, verbose: bool = True) -> list[Document]:
    """Extract one Document per page that yields text."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text returns None on some malformed pages.
        text = clean_text(page.extract_text() or "")

        if not text:
            # Either a genuinely blank page or a scanned image with no text
            # layer. The second case is silent and destructive: the page looks
            # correct in a viewer while the pipeline receives nothing.
            if verbose:
                print(f"    page {page_number}: no extractable text, skipped. "
                      f"Blank page, or a scan requiring OCR.")
            continue

        quality = assess_quality(text)
        if verbose and quality["verdict"] != "ok":
            print(f"    page {page_number}: extraction {quality['verdict']}, "
                  f"off script {quality['off_script_ratio']:.1%}, "
                  f"{quality['replacement_chars']} undecodable characters")

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
                },
            )
        )

    return documents


def load_directory(directory: Path = RAW_DIR,
                   verbose: bool = True) -> list[Document]:
    """Extract every PDF in a directory into one flat list of Documents.

    Sorted for deterministic ordering across runs.
    """
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {directory}")
        return []

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        if verbose:
            print(f"  {pdf_path.name}")
        pages = load_pdf(pdf_path, verbose=verbose)
        if verbose:
            print(f"    {len(pages)} pages with text")
        documents.extend(pages)

    return documents


def run(directory: Path = RAW_DIR, output: Path = TEXTLAYER_OUTPUT) -> int:
    """Extract, assess, stamp version fields, and write to disk."""
    print(f"Text layer extraction from {directory}")

    documents = load_directory(directory)
    if not documents:
        return 0

    version_fields = extract_version_fields(documents)
    apply_version_fields(documents, version_fields)
    save_documents(documents, output)

    _report(documents, version_fields, output)
    return len(documents)


def _report(documents: list[Document], version_fields: dict[str, dict],
            output: Path) -> None:
    total_chars = sum(d.metadata["char_count"] for d in documents)
    degraded = [d for d in documents
                if d.metadata["extraction_quality"] != "ok"]

    print(f"\n{len(documents)} documents, {total_chars:,} characters, "
          f"{total_chars // len(documents):,} per page")
    print(f"written to {output}")

    print("\nPer source")
    for source in sorted({d.metadata["source"] for d in documents}):
        pages = [d for d in documents if d.metadata["source"] == source]
        chars = sum(d.metadata["char_count"] for d in pages)
        fragmentation = sum(d.metadata["fragment_ratio"] for d in pages)
        bad = sum(1 for d in pages
                  if d.metadata["extraction_quality"] != "ok")
        fields = version_fields[source]
        print(f"  {source[:44]:<44} {len(pages):>3}p "
              f"{chars // len(pages):>5} ch/p "
              f"frag {fragmentation / len(pages):>5.1%}"
              f"{f'  {bad} degraded' if bad else ''}")
        print(f"    version {fields['doc_version'] or 'unknown'}, "
              f"issued {fields['issue_date'] or 'unknown'}, "
              f"reviewed {fields['review_date'] or 'unknown'}")

    if degraded:
        print(f"\n{len(degraded)} of {len(documents)} pages failed the quality "
              f"gate. Their text contains undecodable bytes or characters from "
              f"the wrong script.")
    print("Run the OCR route and compare before relying on this output.")


if __name__ == "__main__":
    run()
