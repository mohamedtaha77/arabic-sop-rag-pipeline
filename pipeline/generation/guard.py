"""The grounding guard: a deterministic diff, no model, no cost.

Position: called from generation/run.py after present.py, on the
presenter's own output against the synthesiser's output it was built
from. This is the second enforcement layer the plan requires. synthesise.py
is trusted to reconcile retrieved context into an answer; present.py is
not trusted at all, and this file is what makes that distrust real
rather than declared. Every number, date, citation marker and
Latin-script token the presenter's text contains has to already be
present in the synthesiser's own text, or the presenter's output is
rejected outright and the synthesiser's text ships instead. That
rejection, and how often it fires, is the number this stage exists to
report: the difference between telling a model not to add facts and
actually stopping it when it does.

Why one check, applied uniformly, rather than three. A number, a date
written digit-first ("08/2024"), and a system code ("BPM") are all the
same shape of problem from this file's point of view: a short token
that either exists verbatim in the trusted text or does not. Folding
Arabic-Indic digits to ASCII first, the same fold_digits retrieval.text
already exports for lexical matching, means "٣" and "3" count as the
same token rather than as a synthesiser writing one and a presenter
inventing the other. A citation marker gets its own check for a
different reason: [5] is not a fact about the corpus, it is a pointer,
and a presenter writing a pointer the synthesiser never wrote has
nothing behind it to point at, which is exactly as fabricated as a
number with no source.

What this file does not do, and says why rather than pretending
coverage it does not have. It does not check whether a *sentence*, as
opposed to a token, is faithfully preserved: a presenter that keeps
every number and every citation but reverses what a sentence claims
about them ("لا يتجاوز" rewritten as "يتجاوز") would pass this diff
clean, since no new token was introduced. That is not a gap this file
is meant to close; entail.py's own sentence-level entailment check,
built next, is the layer meant for exactly that shape of failure, and
splitting the two concerns keeps this file's own contract checkable by
inspection: a token is either present or it is not.

Arabic-entity drift, meaning a presenter that swaps in a different role
name or noun that a token-level diff cannot see because nothing here
attempts Arabic named-entity recognition, is measured and reported as a
soft signal, never gated on. Extracting Arabic named entities without an
NER model is guesswork, the same reasoning quality.py's own
`fragment_ratio` is reported and explicitly not gated on: a wrong gate
built on a guess would reject a legitimate rephrasing as often as it
caught a real one. What is reported instead is a segment-level word
count, comma and sentence-ending marks splitting the text into smaller
pieces so the report can say roughly where wording changed, without
claiming to know whether that change was faithful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..retrieval.text import fold_digits

# A citation marker, the same shape synthesise.py's own _CITATION_MARKER
# matches. Redefined here rather than imported: each file in this package
# owns the one regex it needs, the same way compress.py's own _MAX_TOKENS
# and rerank.py's own _MAX_LENGTH stay local rather than shared, and a
# citation marker is a one-line pattern, not worth a cross-module import
# for.
_CITATION_MARKER = re.compile(r"\[(\d+)\]")

# A run of Latin letters or ASCII digits, one token, matching text.py's
# own reasoning for keeping a system name like BPM intact rather than
# splitting it: this file needs the same unit for a different job, not
# "is this part of the same search token" but "does this exact string
# appear in the trusted text", so numbers, dates written digit-first, and
# Latin system codes are all caught by the one pattern uniformly. Digit
# folding runs first, so this pattern never has to know about
# Arabic-Indic digits at all.
_ALNUM_TOKEN = re.compile(r"[A-Za-z0-9]+")

# hex codepoints, per the project's own rule: a literal Arabic mark in
# source is not safely reviewable. Arabic question mark (U+061F), Arabic
# comma (U+060C), Arabic semicolon (U+061B), alongside the ASCII marks
# Arabic prose in this corpus also uses. This splits text into segments
# for the soft drift report's own granularity, not into grammatically
# exact sentences: a comma-separated clause is a fine enough unit to say
# roughly where wording changed, and no gate in this file reads the
# split any more strictly than that.
_SEGMENT_SPLIT = re.compile(
    "[.!؟،؛\n]+"
)

# hex codepoints again, the same range synthesise.py's own _ARABIC_LETTERS
# duplicates for a different job than retrieval.text's private range
# answers: this counts Arabic words for the soft drift signal, not
# tokens for BM25.
_ARABIC_LETTERS = (0x0621, 0x064A)
_ARABIC_WORD = re.compile(f"[{chr(_ARABIC_LETTERS[0])}-{chr(_ARABIC_LETTERS[1])}]+")


@dataclass
class GuardReport:
    """What one guard check found. Section 16's per-technique analysis
    reads this for how often the presenter tried to add something and
    was blocked, the number this whole split-generation design exists to
    produce.

        passed              True when nothing in the presented text is
                             outside the allowed set the synthesised
                             text established. False means the presenter
                             is rejected and the synthesised text ships
                             instead
        added_tokens         alnum tokens (a number, a date, a Latin
                             name) present in the presented text but not
                             in the synthesised text, after digit
                             folding. Empty when passed is True
        added_citations      citation markers present in the presented
                             text but not in the synthesised text, e.g. a
                             [4] the presenter wrote when the synthesiser
                             never cited a fourth source. Empty when
                             passed is True
        drift_segments       the soft signal, never gated on: segments
                             (comma- and sentence-mark-split) where the
                             presented text's own Arabic words are not a
                             subset of the synthesised text's, paired
                             with the words that changed. Can be
                             non-empty even when passed is True, since
                             rewording without adding a new fact is
                             exactly what the presenter is for
    """

    passed: bool
    added_tokens: tuple[str, ...]
    added_citations: tuple[str, ...]
    drift_segments: tuple[tuple[str, tuple[str, ...]], ...]


def _allowed_alnum(text: str) -> set[str]:
    return set(_ALNUM_TOKEN.findall(fold_digits(text)))


def _allowed_citations(text: str) -> set[str]:
    return set(_CITATION_MARKER.findall(text))


def _arabic_words(segment: str) -> set[str]:
    return set(_ARABIC_WORD.findall(segment))


def _drift_segments(
    synthesised: str, presented: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Segment the presented text and report, per segment, which Arabic
    words are not covered by the synthesised text's own Arabic
    vocabulary as a whole. Compared against the whole synthesised text
    rather than an aligned segment, since the presenter is free to
    reorder and merge sentences, and an aligned, order-sensitive
    comparison would flag reordering as drift, which it is not.
    """
    allowed_words = _arabic_words(synthesised)
    drift = []
    for segment in _SEGMENT_SPLIT.split(presented):
        segment = segment.strip()
        if not segment:
            continue
        new_words = tuple(sorted(_arabic_words(segment) - allowed_words))
        if new_words:
            drift.append((segment, new_words))
    return tuple(drift)


def apply(synthesised: str, presented: str) -> GuardReport:
    """Diff the presenter's output against the synthesiser's output it
    was built from. The only inputs are two strings; this file never
    touches retrieved context, a chunk, or a model, which is what makes
    it a check on the presenter specifically, not a second grounding
    check against the corpus (entail.py's own job, checking the
    synthesiser's sentences against the chunks they cite, is that).
    """
    allowed_tokens = _allowed_alnum(synthesised)
    allowed_citations = _allowed_citations(synthesised)

    presented_tokens = _allowed_alnum(presented)
    presented_citations = _allowed_citations(presented)

    added_tokens = tuple(sorted(presented_tokens - allowed_tokens))
    added_citations = tuple(sorted(presented_citations - allowed_citations))
    drift = _drift_segments(synthesised, presented)

    return GuardReport(
        passed=not added_tokens and not added_citations,
        added_tokens=added_tokens, added_citations=added_citations,
        drift_segments=drift,
    )


def apply_to_refusal(question: str, refusal_text: str) -> GuardReport:
    """The same diff, for the one other place generation/run.py needs
    it: a generated refusal, checked against the question itself rather
    than against a synthesiser's own text, since a refusal has no
    trusted synthesised text to diff against, there was no retrieval to
    trust.

    Two differences from apply() follow directly from that. The allowed
    alnum set comes from the question, not from an answer, since the
    only numbers or dates a refusal has any legitimate reason to
    mention are ones the user themselves already named. And the allowed
    citation set is the empty set, always, never read from anywhere: a
    refusal is a statement that nothing in the corpus was used, so a
    citation marker of any kind, even one that would otherwise resolve,
    has no business appearing in one at all.
    """
    allowed_tokens = _allowed_alnum(question)
    refusal_tokens = _allowed_alnum(refusal_text)
    refusal_citations = _allowed_citations(refusal_text)

    added_tokens = tuple(sorted(refusal_tokens - allowed_tokens))
    added_citations = tuple(sorted(refusal_citations))

    return GuardReport(
        passed=not added_tokens and not added_citations,
        added_tokens=added_tokens, added_citations=added_citations,
        drift_segments=(),
    )


# --- verification --------------------------------------------------------------

def verify_rejects_fabrication_not_genuine(synthesised: str) -> dict:
    """The build order's own gate: a fabricated number injected into
    presenter input is rejected, and the same synthesised text, merely
    reformatted, is not. Needs no real presenter call, since the
    injection and the genuine reformatting are both constructed directly
    from one real synthesised answer; that is also why this file comes
    before present.py rather than after it.
    """
    fabricated = synthesised.rstrip() + " كما تم رصد 47 حالة إضافية [9]."
    fabricated_report = apply(synthesised, fabricated)

    # A genuine reformatting: reordered into a one-line list marker, same
    # facts, same citation, nothing added. Presenters are expected to do
    # exactly this, restructure without inventing.
    genuine = "• " + synthesised.strip().replace("\n\n", "\n• ")
    genuine_report = apply(synthesised, genuine)

    return {
        "fabricated_rejected": not fabricated_report.passed,
        "fabricated_added_tokens": list(fabricated_report.added_tokens),
        "fabricated_added_citations": list(fabricated_report.added_citations),
        "genuine_accepted": genuine_report.passed,
    }


def verify_refusal_guard(question: str) -> dict:
    """The refusal variant's own gate: a refusal that invents a citation
    or a number the question never named is rejected, and a plain,
    honest refusal is not.
    """
    fabricated = "هذا السؤال خارج النطاق، راجع المقطع [3] لمزيد من التفاصيل."
    fabricated_report = apply_to_refusal(question, fabricated)

    genuine = "هذا السؤال يقع خارج نطاق الأدلة الثلاثة ولا يمكنني الإجابة عليه."
    genuine_report = apply_to_refusal(question, genuine)

    return {
        "fabricated_rejected": not fabricated_report.passed,
        "fabricated_added_citations": list(fabricated_report.added_citations),
        "genuine_accepted": genuine_report.passed,
    }


if __name__ == "__main__":
    from ..config import PROCESSED_DIR

    # A real synthesised-shaped answer, built by hand rather than run
    # live through synthesise.py: this gate only needs one representative
    # answer with a number, a date, a Latin name and a citation, and
    # constructing it directly keeps this file runnable without the
    # store or the LLM endpoint, the same independence rerank.py's own
    # worker keeps from retriever.py.
    sample_synthesised = (
        "يتوجب على وحدة الانذار المركزي فحص تسجيلات المراقبة حسب الحاجة "
        "وبحد أقصى كل 3 أشهر [1]. صدر الدليل بتاريخ 08/2024 وتم استخدام "
        "نظام BPM لتوثيق الإجراء [2]."
    )
    sample_question = "ما هي أفضل طريقة لاستثمار مدخراتي؟"

    outcome = verify_rejects_fabrication_not_genuine(sample_synthesised)
    refusal_outcome = verify_refusal_guard(sample_question)

    lines = [
        f"fabricated input rejected: {outcome['fabricated_rejected']}",
        f"  added_tokens: {outcome['fabricated_added_tokens']}",
        f"  added_citations: {outcome['fabricated_added_citations']}",
        f"genuine reformatting accepted: {outcome['genuine_accepted']}",
        "",
        f"refusal, fabricated citation rejected: {refusal_outcome['fabricated_rejected']}",
        f"  added_citations: {refusal_outcome['fabricated_added_citations']}",
        f"refusal, genuine wording accepted: {refusal_outcome['genuine_accepted']}",
    ]

    out_path = PROCESSED_DIR / "16_guard_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    ok = (
        outcome["fabricated_rejected"] and outcome["genuine_accepted"]
        and refusal_outcome["fabricated_rejected"] and refusal_outcome["genuine_accepted"]
    )
    print(f"{'ok' if ok else 'FAIL'}: written to {out_path}")
