"""Normalisation applied to extracted text before it reaches the chunker.

Each rule below was added in response to a measurement on the source corpus,
noted inline. Applied to both extraction routes so their outputs stay
comparable.

Unicode targets are declared as hex codepoint ranges rather than literal
characters. Most of them are invisible or combining marks, so pasting them into
source produces lines that cannot be reviewed, grepped, or safely edited.
"""

from __future__ import annotations

import re
import unicodedata

# --- characters deleted outright -------------------------------------------
# (first, last) inclusive codepoint ranges.

ARABIC_DIACRITICS = [
    (0x064B, 0x065F),   # harakat: fatha, damma, kasra, sukun, shadda, ...
    (0x0670, 0x0670),   # superscript alef
]

TATWEEL = [
    (0x0640, 0x0640),   # kashida: stretches a word to justify a line
]

BIDI_CONTROLS = [
    (0x200B, 0x200F),   # zero-width space/joiners, LRM, RLM
    (0x202A, 0x202E),   # embedding and override marks
    (0x2066, 0x2069),   # isolates
]

PRIVATE_USE = [
    (0xE000, 0xF8FF),   # symbol fonts (Wingdings and similar) live here
]

# A single translation table is faster than four regex passes and states the
# intent plainly: every one of these characters is removed, none is replaced.
_DELETION_TABLE = {
    codepoint: None
    for ranges in (ARABIC_DIACRITICS, TATWEEL, BIDI_CONTROLS, PRIVATE_USE)
    for first, last in ranges
    for codepoint in range(first, last + 1)
}

# --- characters replaced ----------------------------------------------------

# Table-of-contents leaders ("....... 12"). Measured at 4-8% of every source
# file. U+2026 is the single-character ellipsis.
DOT_LEADERS = re.compile("[." + chr(0x2026) + "]{4,}")

HYPHENATED_LINEBREAK = re.compile(r"(\w)-\n(\w)")
HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
PADDED_NEWLINE = re.compile(r" *\n *")
BLANK_LINE_RUN = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Normalise extracted text. Safe to apply to any extraction route."""

    # NFKC folds Arabic Presentation Forms (U+FB50-FEFF) back to standard
    # letters. Some PDF producers store positional letter *shapes* rather than
    # letters; those codepoints are absent from embedding tokenizers and never
    # match a normally typed query. Measured: 8,990 such characters in one
    # source file, roughly a quarter of its content.
    text = unicodedata.normalize("NFKC", raw)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_DELETION_TABLE)

    # Replaced with a space, not removed, so the words either side stay apart.
    text = DOT_LEADERS.sub(" ", text)

    # Rejoin words hyphenated across a line break. Arabic does not hyphenate,
    # but the corpus contains English system names and codes.
    text = HYPHENATED_LINEBREAK.sub(r"\1\2", text)

    # Space padding is 8-24% of each source file: right-to-left layout encoded
    # as literal spaces. The class is [ \t] rather than \s so newlines survive
    # for the paragraph rule below.
    text = HORIZONTAL_WHITESPACE.sub(" ", text)
    text = PADDED_NEWLINE.sub("\n", text)

    # Collapse blank-line runs to a single paragraph break. Near no-op on this
    # corpus: blank lines contain no glyphs, so extraction rarely emits them.
    # Downstream code must not assume paragraph breaks are present.
    text = BLANK_LINE_RUN.sub("\n\n", text)

    return text.strip()
