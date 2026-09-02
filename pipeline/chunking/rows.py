"""Row shape classification.

layout.py rebuilds a table row as a variable-length list of cells and stops
there, because what a row *means* is a chunking question. This module answers
it: heading, actor label, step, account line, grid row, or continuation.

The measurement that shaped the module: a one-cell row is not reliably a
heading. Of 107 one-cell rows in the corpus, 37 are headings. The rest are
cover titles, approver names, account lines, and step text that lost its number
cell. Reading cell count as meaning produces section paths that look right on
one manual and fill another's with fragments of step prose.

This module sees one row and nothing else: no page, no table, no document.
That limit costs something real. An accounting-entry title and an orphaned step
continuation are both a single cell of ordinary Arabic, and only the
surrounding table tells them apart. So both come back as ``continuation`` and
tables.py promotes them where it can; guessing here would be right about one
manual and wrong about another.
"""

from __future__ import annotations

import re

# Plain strings rather than an Enum, matching layout.py's classify_table.
#
#   heading        a numbered procedure title, opens a section
#   sub_heading    an أولا/ثانيا ordinal, opens a subsection
#   actor          المنفذ or الجهة المنفذة naming who executes what follows
#   step           a numbered procedure step, the real retrievable content
#   entry_line     a debit/credit line in the accounting annex
#   grid           a row of a true data grid, three or more real cells
#   continuation   content belonging to whatever came before it
ROW_KINDS = (
    "heading",
    "sub_heading",
    "actor",
    "step",
    "entry_line",
    "grid",
    "continuation",
)


# --- match-scoped normalisation ---------------------------------------------

# Arabic letters that differ by codepoint but not by meaning when matching a
# label. Declared as hex for the reason cleaning.py gives: four alef seats
# differ by a mark a few pixels tall and ى differs from ي by two dots, so as
# literals they produce lines nobody can review.
#
# Not optional here. الإجراءات appears 6 times in the prose and الاجراءات, the
# same word with the hamza seat dropped, appears 6 more. Both are ordinary
# Arabic, and exact matching finds half the anchors.
_ALEF_SEATS = [
    (0x0623, 0x0627),   # أ alef with hamza above  -> ا alef
    (0x0625, 0x0627),   # إ alef with hamza below  -> ا
    (0x0622, 0x0627),   # آ alef with madda above  -> ا
    (0x0671, 0x0627),   # ٱ alef wasla            -> ا
    (0x0649, 0x064A),   # ى alef maksura          -> ي yeh
]

_FOLD_TABLE = {source: chr(target) for source, target in _ALEF_SEATS}

# Folding ة to ه, the other standard Arabic normalisation, is deliberately
# absent: no anchor or label in this corpus varies on it, and it is the
# aggressive half of the pair. الجهة المنفذة ends in ة twice.

# Punctuation that clings to a label without changing what it says.
_EDGE_PUNCTUATION = " \t:.،-_)("


def fold_for_match(text: str) -> str:
    """Fold confusable Arabic letters. For comparison only, never for storage.

    Folding at the point of comparison changes what matches without changing
    what gets written out, so a chunk keeps the spelling that was on the page.
    """
    return text.translate(_FOLD_TABLE)


def match_key(cell: str) -> str:
    """Reduce a cell to what it says, for comparing against a label."""
    return fold_for_match(cell).strip(_EDGE_PUNCTUATION).strip()


# --- step numbers -----------------------------------------------------------

# A step number fills its own cell and holds nothing else. Measured forms, most
# frequent first: -1 through -18 (533 rows), then .5, 10, 2, 1, _1, _4.
#
# The leading hyphen is neither a minus sign nor an error. layout.py reads
# cells right to left, so the source's "1-" arrives as "-1". EasyOCR also reads
# some spaces as underscores, which is where _1 comes from.
#
# Anchored at both ends, and that is what separates a step from a heading:
# ".2" alone is step 2, ".2 اجراءات ترحيل محاضر الاستلام" titles procedure 2.
STEP_NUMBER = re.compile(r"^\s*[.\-_]?\s*(\d{1,2})\s*[.\-_)]?\s*$")


def step_number(cell: str) -> int | None:
    """The step number a cell holds alone, or None.

    An int rather than the raw string, so row_range and step order compare
    arithmetically instead of putting "10" before "2".
    """
    match = STEP_NUMBER.match(cell)
    return int(match.group(1)) if match else None


# --- headings ---------------------------------------------------------------

# A number followed by the title it numbers. Measured: 23 rows, 28 to 114
# characters, always on one line.
#
# Neither a leading nor a trailing hyphen counts, and both exclusions come from
# a real failure rather than caution. Assets page 6 opens "-8 بشكل شهري..." and
# Central Alarm page 6 opens "-3 الموافقات المطلوبة...", general-rule items
# carrying a reversed step number. Central Mail page 10 holds "8- ترحيل
# المغلفات الواردة...", a step whose number merged into its text. Accept either
# form and a step overwrites the page's real heading, taking the section path
# for the rest of the procedure with it.
#
# All 23 genuine headings separate number from title with a space or a dot. The
# reversed "-N" form marks an item inside a list; a bare or dot-prefixed number
# marks the procedure the list belongs to.
HEADING_NUMBER = re.compile(r"^\s*(?![-_])[.]?\s*(\d{1,2})\s*[.)]?\s+\S")

# A closed list rather than a pattern, for the reason KNOWN_OCR_MISREADS is a
# list: a pattern loose enough to catch these catches ordinary words that start
# the same way. Measured: 14 rows, all أولا, ثانيا or ثالثا. The rest of the
# series costs nothing to include.
ORDINALS = (
    "أولا", "ثانيا", "ثالثا", "رابعا", "خامسا", "سادسا",
)

_FOLDED_ORDINALS = tuple(fold_for_match(word) for word in ORDINALS)


def heading_number(cell: str) -> int | None:
    """The procedure number a heading opens with, or None."""
    match = HEADING_NUMBER.match(cell)
    return int(match.group(1)) if match else None


def is_sub_heading(cell: str) -> bool:
    """Whether a cell opens with an Arabic ordinal."""
    folded = fold_for_match(cell).lstrip(_EDGE_PUNCTUATION)
    return any(folded.startswith(word) for word in _FOLDED_ORDINALS)


# --- actors -----------------------------------------------------------------

# The two labels binding a run of steps to whoever performs them. Without this
# a retrieved step is an instruction with no idea which unit executes it, which
# for procedure documents is the failure nobody notices until an auditor asks.
#
# Measured: المنفذ 256 rows, الجهة المنفذة 47, and المنفذ with a trailing colon
# 7 more. Those 7 are why comparison runs through match_key rather than ==.
ACTOR_LABELS = frozenset({"المنفذ", "الجهة المنفذة"})

_FOLDED_ACTOR_LABELS = frozenset(match_key(label) for label in ACTOR_LABELS)


def is_actor_label(cell: str) -> bool:
    """Whether a cell is one of the two executing-role labels."""
    return match_key(cell) in _FOLDED_ACTOR_LABELS


# --- accounting entries -----------------------------------------------------

# Debit and credit lines in the Assets manual's annex. ح abbreviates حساب,
# account, and the slash after it is what the source prints. Measured: 37 rows
# spelled من ح /, ح /, إل ح /, إلى ح / and الى ح /.
#
# Folding the pattern before compiling is not decoration. Written with unfolded
# literals it matched 31 of the 37: is_entry_line folds its input, so الى
# arrives as الي and an الى inside the pattern never meets it. Folding one side
# of a comparison and not the other fails silently, because the misses look
# exactly like rows that were never meant to match. Compiling from folded
# source makes both sides agree by construction.
_ENTRY_LINE_SOURCE = r"^\s*(?:(?:من|إلى|إل)\s+)?ح\s*[/_]\s*$"

ENTRY_LINE = re.compile(fold_for_match(_ENTRY_LINE_SOURCE))


def is_entry_line(cell: str) -> bool:
    """Whether a cell is an accounting-entry debit or credit marker."""
    return bool(ENTRY_LINE.match(fold_for_match(cell)))


# --- the classifier ---------------------------------------------------------

# Three or more real cells means a genuine data grid: the approval matrices,
# the reports tables, the revision logs. Measured at 52 rows across 9 tables,
# so this is the corpus's smallest path rather than its largest.
GRID_MIN_CELLS = 3


def classify_row(cells: list[str]) -> str:
    """Decide what one row is. Returns a value from ROW_KINDS.

    Cell count settles grid rows first, since a three-cell row cannot be a
    label pair whatever its first cell holds. Among two-cell rows the actor
    labels go first, being a closed set of two strings that cannot collide.
    """
    if not cells:
        return "continuation"

    if len(cells) >= GRID_MIN_CELLS:
        return "grid"

    if len(cells) == 2:
        first = cells[0]
        if is_actor_label(first):
            return "actor"
        if step_number(first) is not None:
            return "step"
        if is_entry_line(first):
            return "entry_line"
        return "continuation"

    only = cells[0]
    if heading_number(only) is not None:
        return "heading"
    if is_sub_heading(only):
        return "sub_heading"
    return "continuation"
