"""Reading and writing Document collections as JSON.

Each stage writes its output to disk and the next stage reads it. Ingestion is
the slow part of the pipeline, so caching its result allows the chunking and
embedding stages to be re-run without re-parsing any PDF.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .document import Document


def save_documents(documents: list[Document], path: Path) -> None:
    """Write documents to JSON.

    UTF-8 is required rather than stylistic: the Windows default encoding
    cannot represent Arabic and raises on the first page. ensure_ascii is off
    so the file stays readable in an editor.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(document) for document in documents]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_documents(path: Path) -> list[Document]:
    """Read documents back from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Document(text=d["text"], metadata=d["metadata"]) for d in raw]
