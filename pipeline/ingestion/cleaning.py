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


# --- known OCR misreads ------------------------------------------------

# Confirmed by direct measurement, not assumed, and kept separate from
# clean_text above: these are visual misreadings EasyOCR makes, a different
# defect class from the text-layer route's ToUnicode corruption. None of these
# forms appears anywhere in the text-layer route's own output, so the
# correction is scoped to OCR-produced text only.
#
# Deliberately a short, explicit list rather than fuzzy matching. Sweeping
# the corpus for edit-distance-1 variants of the structural anchor words
# found 14 "near-miss" forms for الإجراءات alone; all but one were legitimate
# Arabic, a conjunction prefix (والإجراءات), a dropped hamza seat
# (الاجراءات), not corruption. A fuzzy matcher would have "corrected" those
# into the wrong word. An explicit list corrects only what was actually
# confirmed wrong.
#
# A wider sweep, run when chunking exposed these forms inside section paths,
# put a number on that. Every corpus token used at most twice that sits one
# edit from a token used five times or more: 610 candidates, and the large
# majority are ordinary Arabic carrying a prefix, واداري, للمستودعات, ومسؤول,
# بالاتفاقيات, or a hamza variant, إستلام. Bulk correction from that list
# would damage far more than it repaired, which is why every entry below was
# confirmed by reading it in context on its own page.
#
# Each entry is a whole token, matched with Arabic-letter boundaries by
# correct_ocr_misreads, never as a bare substring. That distinction is load
# bearing now that the list is long: لتعامل sits inside the perfectly good
# التعامل, which occurs 11 times, and لجد sits inside الجدول and الجديد. A
# substring pass would have turned all fifteen of those into nonsense while
# fixing two real errors.
KNOWN_OCR_MISREADS = {
    # ر misread as د, the most common single defect in this corpus.
    "الإجداءات": "الإجراءات",
    "البديد": "البريد",
    "للبديد": "للبريد",
    "الصاددة": "الصادرة",
    "الخادجية": "الخارجية",
    "وادساله": "وارساله",
    "الدسمية": "الرسمية",
    "والدسائل": "والرسائل",
    "الصغيدة": "الصغيرة",
    "السديع": "السريع",

    # ر dropped entirely.
    "الديد": "البريد",
    "البيد": "البريد",
    "الواد": "الوارد",
    "الصغدرة": "الصغيرة",

    # A letter dropped from the definite article or the stem.
    "امنفذ": "المنفذ",
    "املفات": "الملفات",
    "للفات": "لملفات",
    "الستلم": "المستلم",
    "لحتوبات": "المحتويات",
    "لجد": "الجرد",
    "لتعامل": "التعامل",
    "لللحلية": "المحلية",
    "ايوميا": "يوميا",
    "شعدبا": "شهريا",
    "متابعتا": "متابعة",

    # Letters transposed.
    "علميات": "عمليات",

    # ه misread as ع.
    "الجعات": "الجهات",

    # A trailing letter dropped, where the truncation is not itself a word.
    "النماذ": "النماذج",

    # A word merged with the following one, where the join also lost a letter.
    # Expanded back into two words, so the space is restored with the letter.
    "البيدمن": "البريد من",
    "الوادمن": "الوارد من",

    # The Assets manual's own cover title, page 1, confirmed by rendering the
    # page at 250 DPI and reading it: "دليل إجراءات وحدة الموجودات وعمليات
    # المستودعات". Each of the three forms below occurs exactly once in the
    # whole corpus, on that one page, in both extraction routes, and none of
    # the three correct forms ever collides with it elsewhere; stage 4 needs
    # this title read correctly for the template context prefix.
    "إجدءات": "إجراءات",       # ر misread as د, same class as الإجداءات above
    "لوجودات": "الموجودات",     # ال and م both dropped from الموجودات
    "الستودعات": "المستودعات",  # م dropped from المستودعات
}

# Arabic block. A key is a correction only when what surrounds it is not more
# Arabic, which is what stops a short key matching inside a longer real word.
_ARABIC = r"[؀-ۿ]"

# One alternation, longest key first. The boundaries already make order
# irrelevant, since البيد cannot match inside البيدمن when م follows it, but
# sorting by length keeps that independence from resting on the lookahead
# alone.
_MISREAD_PATTERN = re.compile(
    rf"(?<!{_ARABIC})(?:"
    + "|".join(re.escape(k) for k in sorted(KNOWN_OCR_MISREADS, key=len, reverse=True))
    + rf")(?!{_ARABIC})"
)

# An underscore standing alone between spaces is a space EasyOCR read as a
# character, not punctuation the document contains. Measured: 114 occurrences
# in the OCR route's prose and zero in the text layer's, which is what
# identifies it as a recognition artifact rather than something on the page.
# It is why the approval matrix reads مدير _ الفرع where the page says
# مدير الفرع. A stray character inside a cell threatens nothing downstream on
# its own, but it does put a word no query can match into a chunk.
#
# Whitespace before, and either whitespace or an Arabic letter after. The
# second half was added after a page-by-page sweep against the source PDFs:
# requiring whitespace on both sides fixed 87 cases and left 22 of the form
# "تقرير _الجرد", where the underscore replaced the space but closed up
# against the following word. Reading the page shows those to be the identical
# artifact, and the earlier rule simply described the first shape it happened
# to look at.
#
# The three it still leaves are the ones that belong to somebody else: two
# cells opening "_1" and "_4", where the underscore stands in for a step
# number's dash, and one trailing "1_". A digit after the underscore is what
# separates them, and rows.py already reads that form; rewriting it here would
# take the decision away from the module that measured it.
LONE_UNDERSCORE = re.compile(r"(?<=\s)_(?=[\s؀-ۿ])")


def correct_ocr_misreads(text: str) -> str:
    """Fix known, individually confirmed OCR misreadings.

    Call after clean_text, on OCR-produced text only (ocr.py, layout.py).
    The text-layer route has its own, unrelated corruption and is handled by
    OCR replacing it entirely rather than by any correction here.

    Matches whole Arabic tokens rather than substrings. The earlier version
    used str.replace, which was safe while the list held two long, distinctive
    keys and stopped being safe the moment it held short ones.
    """
    text = _MISREAD_PATTERN.sub(lambda m: KNOWN_OCR_MISREADS[m.group(0)], text)
    return HORIZONTAL_WHITESPACE.sub(" ", LONE_UNDERSCORE.sub(" ", text))
