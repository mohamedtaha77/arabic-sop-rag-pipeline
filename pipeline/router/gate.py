"""The deterministic pre-gate: catches chitchat before any model is called.

Position: gate.check runs first, before router.py ever spends a call. It
answers one narrow question, "is this obviously not a real question about
the corpus at all", and answers nothing else.

A real out-of-domain question, Q9's shape, is a different case entirely: a
well-formed sentence asking for something the corpus does not cover, and it
belongs to router.py's own model judgement, not to a regex here. A gate that
caught Q9 would pass its own test for the wrong reason, the exact mistake
advanced-rag-plan.md's finding C warns against: a component verified against
evidence that happens to agree with it rather than evidence that actually
exercises what it claims to do.

What this module does not do: it never returns a refusal. A pre-gate match
always means route "simple", refusal_kind None, an LLM answers a greeting
normally with no retrieval and no judgement about scope. Scope refusal is
router.py's decision to make, on a question it actually read.
"""

from __future__ import annotations

from pathlib import Path

from ..chunking.rows import fold_for_match
from ..config import GOLDEN_SET, PROCESSED_DIR
from ..golden.question import load_golden
from .schema import RouteDecision

# Whole-message greetings and closers, checked against the entire folded
# message rather than as a substring: a real question can legitimately open
# with a courtesy word ("من فضلك") without being a greeting, and matching a
# substring would catch it.
_GREETINGS_AR = frozenset({
    "مرحبا", "مرحبا بك", "اهلا", "اهلا وسهلا", "اهلا وسهلا بك",
    "السلام عليكم", "السلام عليكم ورحمة الله وبركاته",
    "صباح الخير", "مساء الخير", "كيف حالك", "كيفك", "شلونك",
    "شكرا", "شكرا لك", "شكرا جزيلا", "تسلم", "تسلم ايدك",
    "يعطيك العافية", "الى اللقاء", "مع السلامة", "وداعا",
})

_GREETINGS_EN = frozenset({
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "how are you", "thanks", "thank you", "thanks a lot", "bye", "goodbye",
    "ok", "okay", "cool", "nice",
})

# Punctuation trimmed off the ends of a message before comparing it against
# the greeting sets above. Arabic and Latin marks both, since a message can
# arrive in either script's punctuation regardless of which script the text
# itself is in.
_EDGE_PUNCTUATION = " \t.!,;:،؛-_)(\n"

_ARABIC_QUESTION_MARK = "؟"

# Below this many folded, whitespace-split words, and carrying no question
# mark of either script, a message is short enough that it is almost
# certainly not a real question about a 357-chunk procedure corpus. Measured
# against the golden set rather than guessed: every one of its 20 questions
# runs to at least six words once split, except Q3 at three words, which
# carries a question mark and is exempted by that check running first and
# independently. See verify_against_golden below, which checks this
# threshold against the real set rather than trusting the arithmetic.
_MIN_QUESTION_WORDS = 3


def _fold(text: str) -> str:
    """fold_for_match plus casefold, stripped of edge punctuation.

    fold_for_match only folds Arabic letter variants; it does nothing for
    English case, and "HELLO" would otherwise miss the greeting set that
    "hello" is written in. casefold is a no-op on Arabic script, which has
    no case, so applying both in sequence is safe for either script.
    """
    return fold_for_match(text).strip(_EDGE_PUNCTUATION).casefold()


def _is_greeting_or_closer(folded: str) -> bool:
    return folded in _GREETINGS_AR or folded in _GREETINGS_EN


def _has_question_mark(text: str) -> bool:
    return "?" in text or _ARABIC_QUESTION_MARK in text


def check(message: str) -> RouteDecision | None:
    """A RouteDecision for an obvious non-question, or None to hand the
    message to router.py.

    None is the common case: this function exists to shortcut the uncommon
    one, a greeting, a closer, or an empty or fragmentary message, at zero
    cost and with no model call.
    """
    stripped = message.strip()
    if not stripped:
        return RouteDecision(
            route="simple", reason="pre-gate: empty message", refusal_kind=None,
        )

    folded = _fold(stripped)
    if _is_greeting_or_closer(folded):
        return RouteDecision(
            route="simple",
            reason=f"pre-gate: greeting or closer ({folded!r})",
            refusal_kind=None,
        )

    if _has_question_mark(stripped):
        return None

    word_count = len(folded.split())
    if word_count < _MIN_QUESTION_WORDS:
        return RouteDecision(
            route="simple",
            reason=f"pre-gate: {word_count} word(s), no question mark",
            refusal_kind=None,
        )

    return None


# --- verification --------------------------------------------------------------

# A probe set distinct from _GREETINGS_AR and _GREETINGS_EN above, so this
# checks the gate's actual behaviour rather than re-checking that a
# frozenset contains its own members. Carries a different alef seat
# (اهلا وسهلا بك against اهلا), trailing punctuation, English case
# variation, and the empty and whitespace-only cases _is_greeting_or_closer
# never sees directly.
_CHITCHAT_PROBES = (
    "مرحبا!", "أهلا وسهلا بك", "السلام عليكم", "صباح الخير.",
    "شكرا جزيلا", "تسلم ايدك", "مع السلامة",
    "Hi", "HELLO", "Thanks!", "thanks", "ok", "bye",
    "", "   ",
)


def verify_against_golden(golden_path: Path = GOLDEN_SET) -> list[str]:
    """What this gate has to be true for, against the two sets it exists to
    keep apart: a real chitchat probe, and every golden question. Empty
    when clean, the same contract golden.py's own gates and
    store.qdrant.verify use.
    """
    failures = []
    for probe in _CHITCHAT_PROBES:
        if check(probe) is None:
            failures.append(f"chitchat probe not caught: {probe!r}")

    questions, _ = load_golden(golden_path)
    for question in questions:
        decision = check(question.question)
        if decision is not None:
            failures.append(
                f"{question.id} wrongly caught by the pre-gate: {decision.reason}"
            )
    return failures


if __name__ == "__main__":
    # A failure message can quote Arabic question text or a folded Arabic
    # greeting, and a Windows console is cp1252: printing that directly
    # crashes for a reason that has nothing to do with the gate, the same
    # trap llm.md's probe.py hit first. Failures go to a file; only an
    # ASCII summary reaches the console.
    failures = verify_against_golden()
    out_path = PROCESSED_DIR / "06_gate_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(failures) if failures else "ok, no failures",
        encoding="utf-8",
    )
    if failures:
        print(f"FAIL  {len(failures)} failure(s), see {out_path}")
    else:
        print(f"ok  {len(_CHITCHAT_PROBES)} chitchat probes caught, "
              f"0 golden questions caught")
