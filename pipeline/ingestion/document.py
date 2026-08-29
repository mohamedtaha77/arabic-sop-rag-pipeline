"""The Document contract shared by every extraction route."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of source text with its provenance.

    One Document per PDF page. The page is the most useful citation unit a PDF
    offers, and it gives the chunker a boundary it must not cross.

    Downstream stages depend only on ``text`` and ``metadata``, never on the
    source format, so adding a new input type means adding a loader and nothing
    else.

    Metadata is populated at extraction time because it cannot be recovered
    later: page numbers drive citations, source and version drive filtering,
    and extraction-quality fields let later stages down-rank unreliable text.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
