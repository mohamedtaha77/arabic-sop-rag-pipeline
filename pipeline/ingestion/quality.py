"""Extraction quality assessment.

Corrupt text is the most expensive failure mode in a retrieval system: it
embeds, stores and searches without error, but never matches a real query. The
system then reports no answer for documents that plainly contain one, with
nothing in the logs. Detection has to happen at extraction time, which is here.

The corpus this was built against has damaged ToUnicode tables in all three
source PDFs. Glyphs decode to the wrong codepoint, often producing a valid
Arabic word that is the wrong word. Both pypdf and PyMuPDF return identical
output on the affected pages, so the damage is in the files rather than the
reader.
"""

from __future__ import annotations

import re
from typing import Any

# Script ranges considered legitimate for an Arabic/English corpus.
ARABIC_BLOCK = (0x0600, 0x06FF)
ARABIC_SUPPLEMENT = (0x0750, 0x077F)

# Punctuation that appears legitimately and should not count as off-script.
ALLOWED_PUNCTUATION = frozenset(
    chr(cp) for cp in (
        0x2013, 0x2014, 0x2022, 0x2026, 0x00AB, 0x00BB, 0x201C, 0x201D,
        0x2018, 0x2019, 0x00B0, 0x00B1, 0x00D7, 0x00F7, 0x00A7, 0x00B6,
    )
)

# U+FFFD REPLACEMENT CHARACTER: proof that a byte could not be decoded at all.
REPLACEMENT_CHAR = chr(0xFFFD)

# A run of Arabic letters, i.e. one word.
ARABIC_TOKEN = re.compile(
    "[" + chr(ARABIC_BLOCK[0]) + "-" + chr(ARABIC_SUPPLEMENT[1]) + "]+"
)

# Gate thresholds, measured on the reference corpus: intact pages sit near 0.1%
# off-script, damaged pages an order of magnitude higher. Re-measure before
# applying these to a different corpus.
OFF_SCRIPT_LIMIT = 0.01
MIN_ARABIC_RATIO = 0.10
MIN_TOKENS_FOR_FRAGMENTATION = 20


def assess_quality(text: str) -> dict[str, Any]:
    """Score one page of extracted text.

    Returns the metrics plus a verdict of ``ok``, ``suspicious``, ``degraded``
    or ``empty``.

    Two signals gate the verdict because they are unambiguous:

    * replacement characters, which prove a decode failure
    * off-script characters, letters from scripts absent from this corpus

    A third, ``fragment_ratio``, is reported but deliberately does not gate.
    Dropped letters split words into stubs, so the share of one and two letter
    Arabic tokens rises with corruption. On the reference corpus it measured
    13.6% to 17.6% before OCR and 5.0% to 5.9% after, which confirms it tracks
    the damage. It still does not gate, because the least damaged file sat only
    marginally below the worst. Without a known clean Arabic PDF to calibrate
    against, any threshold would be invented rather than derived.
    """
    if not text:
        return {
            "chars": 0,
            "arabic_ratio": 0.0,
            "off_script_ratio": 0.0,
            "fragment_ratio": 0.0,
            "replacement_chars": 0,
            "verdict": "empty",
        }

    arabic = 0
    off_script = 0
    replacements = text.count(REPLACEMENT_CHAR)

    for char in text:
        if char.isspace() or char in ALLOWED_PUNCTUATION:
            continue
        code = ord(char)
        if (ARABIC_BLOCK[0] <= code <= ARABIC_BLOCK[1]
                or ARABIC_SUPPLEMENT[0] <= code <= ARABIC_SUPPLEMENT[1]):
            arabic += 1
        elif not char.isascii():
            off_script += 1

    total = len(text)
    arabic_ratio = arabic / total
    off_script_ratio = off_script / total

    tokens = ARABIC_TOKEN.findall(text)
    if len(tokens) >= MIN_TOKENS_FOR_FRAGMENTATION:
        fragment_ratio = sum(1 for t in tokens if len(t) <= 2) / len(tokens)
    else:
        fragment_ratio = 0.0

    if replacements > 0 or off_script_ratio > OFF_SCRIPT_LIMIT:
        verdict = "degraded"
    elif arabic_ratio < MIN_ARABIC_RATIO and total > 200:
        verdict = "suspicious"
    else:
        verdict = "ok"

    return {
        "chars": total,
        "arabic_ratio": round(arabic_ratio, 3),
        "off_script_ratio": round(off_script_ratio, 4),
        "fragment_ratio": round(fragment_ratio, 3),
        "replacement_chars": replacements,
        "verdict": verdict,
    }
