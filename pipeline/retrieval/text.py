"""Arabic lexical normalisation for the BM25 leg, and nothing else.

Position: bm25.py is the only caller. Nothing here ever reaches an
embedding model, which tokenizes for itself and would have its own
vocabulary damaged by a clitic stripped off for a term-frequency count's
sake.

Three normalisation levels, not one, because the probe that shaped this
stage's plan measured a real, sizeable difference between them: plain
tokenisation scored 0.630 Recall@10 on the golden set, adding a measured
stopword list left that unchanged at 0.630 (it moves MRR, not recall, on
this corpus), and adding clitic stripping on top lifted it to 0.713. Each
level is exposed as its own function so bm25.py, and evaluate.py's grid,
can score under all three rather than one being chosen by preference
before the numbers existed.

Nothing here is a stemmer. Stemming that reached the root would collapse
"مدير" and "مدير مساعد" into the same token, and those are different
actors on the same page; cleaning.py's own KNOWN_OCR_MISREADS list and
rows.py's fold_for_match both already draw this same line, correcting or
folding only what was specifically confirmed safe rather than applying a
general rule and hoping the corpus tolerates it.
"""

from __future__ import annotations

import collections
import re

from ..chunking.rows import fold_for_match

# --- digit folding -----------------------------------------------------------

# Arabic-Indic digits, hex per the project's rule for anything outside plain
# ASCII: pasting ٠١٢٣٤٥٦٧٨٩ as literals produces a line that is not safely
# reviewable or editable. Measured on the corpus: 14 occurrences against 848
# ASCII digits, so this is a small, real, and cheap normalisation rather than
# a hypothetical one. The Extended Arabic-Indic block (Persian/Urdu digits,
# 0x06F0-0x06F9) was checked too and measured zero occurrences, so it is left
# out rather than handled on spec.
_ARABIC_INDIC_DIGITS = (0x0660, 0x0669)

_DIGIT_FOLD_TABLE = {
    codepoint: chr(ord("0") + (codepoint - _ARABIC_INDIC_DIGITS[0]))
    for codepoint in range(_ARABIC_INDIC_DIGITS[0], _ARABIC_INDIC_DIGITS[1] + 1)
}


def fold_digits(text: str) -> str:
    """Arabic-Indic digits to ASCII. Never applied to storage, only to a
    copy made for lexical matching, the same discipline fold_for_match
    states for letter folding.
    """
    return text.translate(_DIGIT_FOLD_TABLE)


# --- tokenising ---------------------------------------------------------------

# Two runs, kept apart rather than one pattern for "a word": an Arabic-script
# run, and a Latin-or-digit run. Splitting them is what keeps a system name
# like BPM intact inside Arabic text instead of losing its case or getting
# merged with neighbouring Arabic letters that happen to sit beside it with
# no separator. Q11 exists specifically to exercise this, a literal Latin
# system name inside a sentence otherwise in Arabic.
#
# This range is not quality.py's ARABIC_BLOCK, and reusing that constant was
# tried first and measured wrong for this job. quality.py asks "is this
# character legitimate Arabic script", and Arabic punctuation correctly
# counts as legitimate there. This asks "is this character part of the same
# token", and a comma is not, so the same range answers two different
# questions incorrectly if shared. Confirmed by enumerating every character
# 03_chunks_none.json's 141,276 characters actually put in the Arabic block
# after cleaning.py's own diacritic and tatweel stripping: 28 real letters,
# the three Arabic-Indic digits fold_digits already converts before this
# pattern ever runs, and exactly one punctuation mark, U+060C, the Arabic
# comma, at 5 occurrences. The Arabic Supplement block, 0x0750 to 0x077F,
# measured zero occurrences of anything at all. 0x0621 to 0x064A is hamza
# through yeh, the real letters, nothing else, on this corpus.
_ARABIC_LETTERS = (0x0621, 0x064A)

_ARABIC_RUN = "[" + chr(_ARABIC_LETTERS[0]) + "-" + chr(_ARABIC_LETTERS[1]) + "]+"
_TOKEN_PATTERN = re.compile(_ARABIC_RUN + "|[A-Za-z0-9]+")


def _is_arabic_token(token: str) -> bool:
    first = ord(token[0])
    return _ARABIC_LETTERS[0] <= first <= _ARABIC_LETTERS[1]


def tokenize_plain(text: str) -> list[str]:
    """Fold letters and digits, then split into Arabic-script and
    Latin/digit runs. The floor every other level builds on.
    """
    folded = fold_digits(fold_for_match(text))
    return _TOKEN_PATTERN.findall(folded)


# --- stopwords -----------------------------------------------------------------

# Measured by document frequency over 03_chunks_none.json's 357 chunks, the
# plain corpus text with no context prefix: the template variant was tried
# first and rejected for this specific purpose, because its own prefix
# ("دليل اجراءات ...") puts words like دليل and اجراءات in literally 100% of
# chunks, and اجراءات ("procedures") is a real content word a query can
# contain, not a stopword the template variant merely made look like one.
# 03_context_samples.txt and 04_token_census.txt both name the none variant
# as the unprefixed baseline for exactly this kind of check.
#
# Every entry below is a closed-class function word: a preposition,
# conjunction, relative pronoun, demonstrative, negation particle, or the
# passive auxiliary یتم ("is carried out"). None is a noun, verb of
# obligation, or role name; مدير ("director") and المنفذ ("the executor")
# measured far higher document frequency than several entries here (0.305
# and 0.745) and are deliberately excluded, because Q13's own "tests" field
# is "actor binding", which means an actor's name is exactly the kind of
# term a real query can and does contain.
#
# A handful of entries carry a document frequency of zero or near it in the
# corpus body, ما and هل specifically. They stay in the list anyway, for a
# different and equally measured reason: a term that is common in a written
# question but rare or absent in the corpus gets the highest possible IDF
# under BM25, and if it happens to land in one unrelated chunk by chance,
# that chunk is rewarded out of proportion to anything it actually answers.
# Removing a genuine function word before scoring costs nothing on the
# query side and closes that hole on the corpus side.
#
# Stored here already folded (fold_for_match applied), since that is the
# form tokenize_plain produces; a canonical unfolded spelling is noted in
# each comment for a human reader. df values are out of 357.
STOPWORDS = frozenset({
    # prepositions
    "من",     # from, df 250 (.700)
    "الي",    # إلى, to/until, df 130 (.364)
    "علي",    # على, on/upon, df 189 (.529)
    "في",     # in, df 93 (.261)
    "مع",     # with, df 77 (.216)
    "عن",     # about/from, df 34 (.095)
    "قبل",    # before, df 52 (.146)
    "بعد",    # after, df 37 (.104)
    "خلال",   # during, df 43 (.120)
    "حسب",    # according to, df 46 (.129)
    "لدي",    # at/possessing, df 59 (.165)
    "دون",    # without, df 12 (.034)
    "وفق",    # according to, df 9 (.025)
    "بين",    # between, df 10 (.028)
    "عند",    # at/when, df 11 (.031)
    # conjunctions and relative pronouns
    "ان",     # that/if, df 67 (.188)
    "التي",   # which (fem.), df 53 (.148)
    "الذي",   # which (masc.), df 6 (.017)
    "او",     # or, df 23 (.064)
    "اذا",    # إذا, if, df 2 (.006)
    "كما",    # as/likewise, df 4 (.011)
    "حيث",    # where/whereas, df 9 (.025)
    "الا",    # إلا, except, df 5 (.014)
    "انه",    # أنه, that it/he, df 2 (.006)
    # demonstratives
    "هذا",    # this (masc.), df 7 (.020)
    "هذه",    # this (fem.), df 27 (.076)
    "ذلك",    # that (masc.), df 22 (.062)
    # negation and modal particles
    "لا", "لم", "لن", "ليس", "غير",
    # light verb and adverbial
    "يتم",    # is carried out, passive auxiliary, df 83 (.232)
    "بشكل",   # in a manner, df 59 (.165)
    # interrogatives, kept for the reason stated above
    "ما", "هل",
})


def tokenize_stopwords(text: str) -> list[str]:
    """tokenize_plain, minus closed-class function words."""
    return [t for t in tokenize_plain(text) if t not in STOPWORDS]


# --- clitic stripping ----------------------------------------------------------

# Prefix clitics only, never a suffix and never a root-finding stemmer.
# Longest first, so والمنفذ tries وال before it can fall through to و and
# strip only one letter of a three-letter attachment. Latin tokens never
# reach this function; _is_arabic_token gates it.
_CLITIC_PREFIXES = ("وال", "بال", "كال", "فال", "ال", "و", "ب", "ك", "ف", "ل")

# A stripped token has to still be a plausible stem. Measured at 3: shorter
# than that and a clitic strip starts eating real short words rather than
# attachments, which is the same failure mode an aggressive stemmer has on
# this corpus's role names.
_MIN_STEM_LENGTH = 3


def _strip_clitics(token: str) -> str:
    for prefix in _CLITIC_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= _MIN_STEM_LENGTH:
            return token[len(prefix):]
    return token


def tokenize_clitics(text: str) -> list[str]:
    """tokenize_stopwords, then a conservative prefix strip on Arabic
    tokens only. The level that measured 0.713 Recall@10 against plain
    tokenisation's 0.630 on the golden set.
    """
    tokens = tokenize_stopwords(text)
    return [
        _strip_clitics(t) if _is_arabic_token(t) else t
        for t in tokens
    ]


# --- the report the frozen list above was read from ---------------------------

def document_frequency_report(texts: list[str], top_n: int = 100) -> str:
    """The measurement `cli.py retrieve --stopwords` writes to a file.

    Recomputes the same count STOPWORDS above was chosen from, so a
    re-chunked corpus can be checked against the frozen list rather than
    trusted to still match it. This never changes STOPWORDS itself; that
    stays a decision made once and read in source, the way
    KNOWN_OCR_MISREADS in cleaning.py stays a fixed, reviewed list rather
    than something recomputed and applied silently on every run.
    """
    document_frequency: collections.Counter[str] = collections.Counter()
    for text in texts:
        document_frequency.update(set(tokenize_plain(text)))

    total = len(texts)
    lines = [
        f"Document frequency over {total} chunks (no context prefix).",
        f"Top {top_n} tokens by document frequency, folded (fold_for_match "
        f"plus Arabic-Indic digit folding), before stopword removal.",
        "",
        f"{'token':<20} {'df':>5} {'df/n':>7}  {'stopword?'}",
    ]
    for token, count in document_frequency.most_common(top_n):
        marker = "yes" if token in STOPWORDS else ""
        lines.append(f"{token:<20} {count:>5} {count / total:>7.3f}  {marker}")
    return "\n".join(lines) + "\n"
