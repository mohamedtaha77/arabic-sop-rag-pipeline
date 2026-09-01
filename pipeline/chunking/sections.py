"""Section paths, tracked down a document.

Each manual restarts its own numbering, so "step 9" means different things in
different files. A chunk carrying only a source and a page cannot tell a
retriever which one it came from. A section path can:

    الإجراءات > 3. اجراءات الجرد السنوي > أولا . المطابقات الشهرية
    ^ anchor     ^ numbered heading       ^ ordinal sub-heading

The three manuals disagree about where a heading lives. Two of them put a
procedure title in a one-cell row at the top of its table. The third prints it
in prose on the page above, and its tables carry no title at all. Reading only
table rows loses 17 procedures; reading only prose loses nearly all of the
other two manuals'. So this tracker reads both into one stack.

When content arrives before any anchor, the path is UNCLASSIFIED rather than an
invented root — the same choice metadata.py makes in recording a missing
version field as an explicit None. A fabricated path would satisfy the gate and
hide the gap; a marked one is countable, and chunker.py counts it.
"""

from __future__ import annotations

import re

from .rows import (
    classify_row,
    fold_for_match,
    heading_number,
    is_sub_heading,
    match_key,
)

SEPARATOR = " > "

# A visible value rather than an empty string, so "every chunk has a
# section_path" cannot pass by an accident of formatting.
UNCLASSIFIED = "غير مصنف"


# --- top-level anchors ------------------------------------------------------

# Section headers measured across the corpus. مرفق القيود المحاسبية is specific
# to one manual's annex, and it sits here on the same terms as
# KNOWN_OCR_MISREADS in cleaning.py: an explicit list of what was measured,
# grown by hand when a new document arrives, rather than a pattern guessing at
# titles it has never seen. It earns its place by giving the annex's entries a
# real path instead of UNCLASSIFIED.
#
# One manual has no revision-log or annex header at all. A parser that requires
# every anchor fails on the file that needs it most, so absence is normal here.
ANCHORS = (
    "المحتويات",
    "الموافقات",
    "مقدمة",
    "القواعد والأحكام العامة",
    "الإجراءات",
    "التقارير",
    "النماذج",
    "جدول التعديلات",
    "الملاحق",
    "مرفق القيود المحاسبية",
)

_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCTUATION = " \t:.،-_)("

# Tells a heading or a line of body text from a stray date or page number.
_ARABIC_LETTER = re.compile(r"[؀-ۿ]")


def anchor_key(text: str) -> str:
    """Reduce a line to a form comparable against ANCHORS.

    Removes whitespace rather than collapsing it, because EasyOCR reads
    الموافقات as المو افقات, split mid-word, on page 3 of all three manuals.
    Anchors go through the same reduction, so a phrase like
    القواعد والأحكام العامة still matches itself and the looser key costs
    nothing across a closed list of ten.
    """
    folded = fold_for_match(text).strip(_EDGE_PUNCTUATION)
    return _WHITESPACE.sub("", folded)


# Anchors are canonicalised on the way into a path, so it reads الموافقات
# rather than the page's damaged المو افقات. Headings are not: they are an open
# set with no canonical form, and the page's own words are the honest answer.
_ANCHOR_LOOKUP = {anchor_key(name): name for name in ANCHORS}


def anchor_for(text: str) -> str | None:
    """The canonical anchor a line names, or None."""
    return _ANCHOR_LOOKUP.get(anchor_key(text))


# --- the tracker ------------------------------------------------------------

_ANCHOR, _HEADING, _SUB = 0, 1, 2
_DEPTH = 3

# The executing unit, as distinct from the executor. rows.py classifies both as
# "actor"; only this one replaces the unit and clears the executor beneath it.
_UNIT_LABEL_KEY = match_key("الجهة المنفذة")


class SectionTracker:
    """Heading state, carried down a document one page at a time.

    State belongs to the document, not the table. A procedure table on page 16
    carries no anchor of its own — الإجراءات was announced on page 8 and holds
    until something replaces it.

    Setting a level clears everything below it. A new anchor drops the heading
    and sub-heading with it, because أولا under القواعد والأحكام العامة is a
    different أولا from the one under الإجراءات.
    """

    def __init__(self) -> None:
        self._levels: list[str | None] = [None] * _DEPTH
        self._prose_names_headings = False
        self.actor: str | None = None
        self.unit: str | None = None

    def reset(self) -> None:
        """Clear all state. Call between documents."""
        self._levels = [None] * _DEPTH
        self._prose_names_headings = False
        self.actor = None
        self.unit = None

    def _set(self, level: int, text: str) -> None:
        self._levels[level] = text.strip()
        for below in range(level + 1, _DEPTH):
            self._levels[below] = None
        if level <= _HEADING:
            # A new procedure or section ends the previous executor's run of
            # steps. A sub-heading does not: أولا and ثانيا divide one
            # procedure, and the unit performing it carries across the divide.
            self.actor = None
            self.unit = None

    def set_actor(self, label: str, value: str) -> None:
        """Record who performs the steps that follow.

        The binding outlives the table that declares it. A page break splits a
        procedure into two separate tables three times in this corpus, and the
        second table opens partway through with its الجهة المنفذة row two pages
        back. Holding the actor beside the section state expires it when the
        section changes, which is the boundary that actually governs it.
        """
        if match_key(label) == _UNIT_LABEL_KEY:
            self.unit, self.actor = value, None
        else:
            self.actor = value

    def path(self) -> str:
        """The current section path, or UNCLASSIFIED if nothing is open."""
        parts = [part for part in self._levels if part]
        return SEPARATOR.join(parts) if parts else UNCLASSIFIED

    # --- the two heading sources --------------------------------------------

    def observe_prose(self, prose_text: str) -> str:
        """Take headings off the top of a page's prose, return the rest.

        Consumes leading lines while each is an anchor, a numbered heading or
        an ordinal, then stops. Stopping is what makes it safe: a general-rules
        page is full of numbered items that look exactly like headings, and the
        only thing separating them is that a real heading comes first, above
        any body text. Across the 37 pages carrying prose this takes every
        anchor and all 17 prose headings, and takes nothing from the four pages
        that would otherwise offer a false one.

        Call once per page before that page's rows, and again mid-table at each
        of a merged table's row_page_breaks. A table spanning pages 16 to 18 is
        emitted once, at page 16, so a heading announced in page 17's prose
        would otherwise arrive after every row it governs.

        The returned remainder is what prose.py chunks. Returning it from here
        keeps one definition of where a heading stops, so the two modules
        cannot drift apart on whether a line was a title or content.
        """
        lines = prose_text.split("\n")
        consumed = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                break
            if not _ARABIC_LETTER.search(stripped):
                # A line with no Arabic is a stray date or page number, never
                # body text. One page opens with two bare 02/2026 fragments
                # above four real headings; breaking on the first of them left
                # the whole page under a procedure three pages earlier. Skip
                # rather than consume, so it cannot become a heading itself.
                consumed += 1
                continue
            anchor = anchor_for(stripped)
            if anchor is not None:
                self._set(_ANCHOR, anchor)
            elif heading_number(stripped) is not None:
                self._set(_HEADING, stripped)
                self._prose_names_headings = True
            elif is_sub_heading(stripped):
                self._set(_SUB, stripped)
            else:
                break
            consumed += 1
        return "\n".join(lines[consumed:]).strip()

    def observe_row(self, cells: list[str]) -> bool:
        """Take a heading from a one-cell table row.

        Returns whether the row became a heading. A consumed row joins the path
        and must not also become chunk content, which is the caller's cue to
        skip it.

        Once a document announces a procedure in its prose, numbered rows stop
        counting as headings for the rest of that document. Nothing in the row
        itself can settle this. One manual holds the one-cell row
        "1 استلام البريد المسجل الوارد من شركة البريد الاردني بواسطة قسم
        الحركة .", a step whose number merged into its text, character for
        character the same shape as another manual's genuine
        "1 إجراءات الجرد السنوي لمستودعات القرطاسية".

        The manual's convention separates them: one prints procedure titles in
        prose above the table, so a numbered row inside that table is content;
        the other two print them in the table's own first row and their prose
        offers no heading anywhere. Prose names a heading on 17 pages, all from
        the same manual, so the latch never arms for the other two.

        The latch belongs to the document, not the page, and that distinction
        was measured. Scoped to the page it broke on the very case it was for:
        a table spanning onto a page with no prose cleared the flag, and a step
        became a heading again.
        """
        kind = classify_row(cells)
        if kind == "heading":
            if self._prose_names_headings:
                return False
            self._set(_HEADING, cells[0])
            return True
        if kind == "sub_heading":
            self._set(_SUB, cells[0])
            return True
        return False
