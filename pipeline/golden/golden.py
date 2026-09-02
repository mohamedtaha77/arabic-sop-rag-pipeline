"""The gates on the golden set, run and verify in one module.

The five files before this one in the stage build the mechanism: a schema, a
renderer, a worksheet. None of them can tell a true answer from a false one, a
real chunk id from a mistyped one, or a quote that was actually copied from the
chunk it names. That is this module's whole job, following chunker.py's own
split of run() and verify() in one place rather than a separate script nobody
remembers to run.

Nothing here reads a page or writes an answer. This only checks that what was
written down is internally consistent and consistent with the rest of the
corpus, which is the only thing left after the reading pass that code can
still verify.
"""

from __future__ import annotations

import re
import statistics
import time
from pathlib import Path

from ..chunking.chunk import Chunk, load_chunks
from ..config import CHUNKS_OUTPUT, CONTEXT_OUTPUTS, GOLDEN_SET
from ..ingestion.quality import MIN_ARABIC_RATIO, assess_quality
from .question import (
    EXPECT_VALUES,
    EXPECTED_ROUTES,
    REFUSAL_KINDS,
    Question,
    corpus_fingerprint,
    load_golden,
    save_golden,
)

EXPECTED_QUESTION_COUNT = 20

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse whitespace so a quote copied across a line break still matches.

    A chunk's text carries its own line breaks, worksheet.py prints them one
    per source line, and a quote typed back from the worksheet can easily lose
    or gain one. The gate cares whether the words match, not the layout.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _chunk_map(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.metadata["chunk_id"]: c for c in chunks}


# --- gate 1: shape and required fields ---------------------------------------

def _gate1_missing_fields(question: Question) -> list[str]:
    """Field names required for this question's expect that are empty.

    Required fields differ by expect and, for a refusal, by refusal_kind:
    an out_of_domain refusal like Q9 has no page to read and no chunk to
    point at, so pages_read, evidence and distractor_chunk_ids stay empty by
    design rather than by omission. A non_answering_retrieval refusal like
    Q10 is the opposite: its distractor_chunk_ids and pages_read are the
    entire point, since the claim is that retrieval found something
    plausible and wrong, not that it found nothing.
    """
    missing = []

    if not question.answer:
        missing.append("answer")

    if question.expect == "answerable":
        if not question.answer_en:
            missing.append("answer_en")
        if not question.evidence:
            missing.append("evidence")
        if not question.pages_read:
            missing.append("pages_read")

    elif question.expect == "refuse":
        if question.refusal_kind == "non_answering_retrieval":
            if not question.distractor_chunk_ids:
                missing.append("distractor_chunk_ids")
            if not question.pages_read:
                missing.append("pages_read")

    return missing


def _gate1(questions: list[Question]) -> list[str]:
    failures = []

    if len(questions) != EXPECTED_QUESTION_COUNT:
        failures.append(
            f"gate 1: expected {EXPECTED_QUESTION_COUNT} questions, "
            f"found {len(questions)}"
        )

    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        failures.append(f"gate 1: duplicate question ids {duplicates}")

    for question in questions:
        if not question.question:
            failures.append(f"gate 1: {question.id} has no question text")
        if question.expect not in EXPECT_VALUES:
            failures.append(
                f"gate 1: {question.id} expect {question.expect!r} not in "
                f"{EXPECT_VALUES}"
            )
        if question.expected_route not in EXPECTED_ROUTES:
            failures.append(
                f"gate 1: {question.id} expected_route "
                f"{question.expected_route!r} not in {EXPECTED_ROUTES}"
            )
        if not question.tests:
            failures.append(f"gate 1: {question.id} has no tests description")

        if question.expect == "refuse":
            if question.refusal_kind not in REFUSAL_KINDS:
                failures.append(
                    f"gate 1: {question.id} is refuse but refusal_kind "
                    f"{question.refusal_kind!r} not in {REFUSAL_KINDS}"
                )
        elif question.refusal_kind is not None:
            failures.append(
                f"gate 1: {question.id} is answerable but carries a "
                f"refusal_kind ({question.refusal_kind!r})"
            )

        missing = _gate1_missing_fields(question)
        if missing:
            failures.append(f"gate 1: {question.id} missing {missing}")

    return failures


# --- gate 2: chunk ids resolve everywhere ------------------------------------

def _gate2(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    failures = []

    corpus_ids = {c.metadata["chunk_id"] for c in chunks}
    variant_ids: dict[str, set[str]] = {}
    for name, path in CONTEXT_OUTPUTS.items():
        if not path.exists():
            failures.append(f"gate 2: {path} missing, run `python cli.py context`")
            continue
        variant_ids[name] = {c.metadata["chunk_id"] for c in load_chunks(path)}

    for question in questions:
        referenced = (
            [("gold_chunk_ids", i) for i in question.gold_chunk_ids]
            + [("distractor_chunk_ids", i) for i in question.distractor_chunk_ids]
            + [("evidence", e.chunk_id) for e in question.evidence]
        )
        for field_name, chunk_id in referenced:
            if chunk_id not in corpus_ids:
                failures.append(
                    f"gate 2: {question.id} {field_name} {chunk_id!r} not in "
                    f"{CHUNKS_OUTPUT.name}"
                )
                continue
            for variant_name, ids in variant_ids.items():
                if chunk_id not in ids:
                    failures.append(
                        f"gate 2: {question.id} {field_name} {chunk_id!r} "
                        f"missing from the {variant_name} variant"
                    )

    return failures


# --- gate 3: gold chunks match expect ----------------------------------------

def _gate3(questions: list[Question]) -> list[str]:
    failures = []
    for question in questions:
        if question.expect == "answerable" and not question.gold_chunk_ids:
            failures.append(f"gate 3: {question.id} is answerable with no gold chunk")
        if question.expect == "refuse" and question.gold_chunk_ids:
            failures.append(
                f"gate 3: {question.id} is refuse but carries "
                f"gold_chunk_ids {question.gold_chunk_ids}"
            )
    return failures


# --- gate 4: evidence is real ------------------------------------------------

def _gate4(questions: list[Question], chunk_map: dict[str, Chunk]) -> list[str]:
    failures = []
    for question in questions:
        if question.expect == "answerable" and not question.evidence:
            failures.append(
                f"gate 4: {question.id} is answerable with no evidence, so its "
                f"gold chunk claim cannot be checked against anything"
            )
        for evidence in question.evidence:
            chunk = chunk_map.get(evidence.chunk_id)
            if chunk is None:
                # Already reported by gate 2; skip rather than double-report.
                continue
            if _normalise(evidence.quote) not in _normalise(chunk.text):
                # The quote itself is not included here: it is Arabic text
                # from the corpus, and a Windows console is cp1252 and cannot
                # print it, the same reason probe.py and context.py write
                # Arabic output to a file instead. Open golden_set.json in an
                # editor to see what the quote actually says.
                failures.append(
                    f"gate 4: {question.id} evidence quote "
                    f"({len(evidence.quote)} chars) not found in "
                    f"{evidence.chunk_id}"
                )
    return failures


# --- gate 5: script quality ---------------------------------------------------

def _gate5(questions: list[Question]) -> list[str]:
    failures = []
    for question in questions:
        for label, text in (("question", question.question), ("answer", question.answer)):
            if not text:
                continue
            quality = assess_quality(text)
            if quality["verdict"] == "degraded":
                failures.append(
                    f"gate 5: {question.id} {label} degraded, "
                    f"off_script_ratio={quality['off_script_ratio']}"
                )
            if quality["arabic_ratio"] < MIN_ARABIC_RATIO:
                failures.append(
                    f"gate 5: {question.id} {label} arabic_ratio "
                    f"{quality['arabic_ratio']} below {MIN_ARABIC_RATIO}"
                )
    return failures


# --- gate 6: depends_on is ordered -------------------------------------------

def _gate6(questions: list[Question]) -> list[str]:
    failures = []
    seen: list[str] = []
    for question in questions:
        if question.depends_on is not None and question.depends_on not in seen:
            failures.append(
                f"gate 6: {question.id} depends_on {question.depends_on!r}, "
                f"which is not an earlier question"
            )
        seen.append(question.id)
    return failures


# --- gate 7: source coverage --------------------------------------------------

def _gate7(questions: list[Question], chunk_map: dict[str, Chunk]) -> list[str]:
    corpus_sources = {c.metadata["source"] for c in chunk_map.values()}

    gold_sources: set[str] = set()
    for question in questions:
        for chunk_id in question.gold_chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                gold_sources.add(chunk.metadata["source"])

    missing = corpus_sources - gold_sources
    if missing:
        return [f"gate 7: no gold chunk from source(s) {sorted(missing)}"]
    return []


# --- gate 8: bound to the chunk build -----------------------------------------

def _gate8(fingerprint: dict, chunks_path: Path) -> list[str]:
    current = corpus_fingerprint(chunks_path)
    if fingerprint != current:
        return [
            f"gate 8: golden set was read against a different chunk build. "
            f"stored {fingerprint}, current {current}. Either the golden set "
            f"is stale or {chunks_path.name} changed underneath it."
        ]
    return []


# --- refingerprinting, done only when proven safe -----------------------------

# Every field a re-chunk could plausibly touch on a referenced chunk, checked
# and reported by name rather than folded into a single pass/fail. A person
# reading this list is what turns a rewrite into a decision instead of a
# silent stamp.
_REFERENCED_METADATA_FIELDS = (
    "source", "page", "section_path", "chunk_type", "actor", "unit",
    "table_id", "doc_version", "issue_date", "review_date",
    "extraction_quality", "char_count",
)


def _referenced_chunk_ids(questions: list[Question]) -> set[str]:
    """Every chunk id this golden set actually depends on.

    Gold and distractor ids are the retrieval claims; evidence ids are where
    an answer quote was copied from. All three have to survive a re-chunk
    unchanged in text, or the reading pass no longer means what it says.
    """
    ids: set[str] = set()
    for question in questions:
        ids.update(question.gold_chunk_ids)
        ids.update(question.distractor_chunk_ids)
        ids.update(e.chunk_id for e in question.evidence)
    return ids


def _diff_against_previous(
    questions: list[Question],
    chunk_map: dict[str, Chunk],
    previous_chunk_map: dict[str, Chunk],
) -> list[str]:
    """Compare every referenced chunk text, old build to new. Metadata is
    reported separately by _report_metadata_moves, never gated here: a
    metadata-only change is exactly what step 0's extraction_quality fix is
    meant to produce, and only a text change invalidates the reading pass.
    """
    failures = []
    for chunk_id in sorted(_referenced_chunk_ids(questions)):
        old_chunk = previous_chunk_map.get(chunk_id)
        new_chunk = chunk_map.get(chunk_id)
        if old_chunk is None:
            failures.append(
                f"refingerprint: {chunk_id!r} is referenced by the golden set "
                f"but is missing from the previous build; cannot confirm its "
                f"text held still"
            )
            continue
        if new_chunk is None:
            # Already reported by gate 2 against the current build; skip
            # rather than double-report the same missing id.
            continue
        if old_chunk.text != new_chunk.text:
            failures.append(
                f"refingerprint: {chunk_id!r} text changed between builds "
                f"({len(old_chunk.text)} chars to {len(new_chunk.text)}); "
                f"the reading pass was done against words that no longer "
                f"exist there"
            )
    return failures


def _report_metadata_moves(
    questions: list[Question],
    chunk_map: dict[str, Chunk],
    previous_chunk_map: dict[str, Chunk],
) -> None:
    """Print every metadata field that moved on a referenced chunk.

    Informational, never a failure: this is where step 0's 134
    empty-to-ok transitions are meant to show up, read by a person before
    the fingerprint is rewritten underneath them.
    """
    moves: dict[str, list[tuple[str, object, object]]] = {}
    for chunk_id in sorted(_referenced_chunk_ids(questions)):
        old_chunk = previous_chunk_map.get(chunk_id)
        new_chunk = chunk_map.get(chunk_id)
        if old_chunk is None or new_chunk is None:
            continue
        for field in _REFERENCED_METADATA_FIELDS:
            old_value = old_chunk.metadata.get(field)
            new_value = new_chunk.metadata.get(field)
            if old_value != new_value:
                moves.setdefault(field, []).append((chunk_id, old_value, new_value))

    if not moves:
        print("  no metadata field differs on any chunk the golden set references")
        return
    for field, changes in sorted(moves.items()):
        print(f"  {field}: {len(changes)} referenced chunk(s) moved")
        for chunk_id, old_value, new_value in changes[:5]:
            print(f"    {chunk_id}: {old_value!r} -> {new_value!r}")
        if len(changes) > 5:
            print(f"    ... and {len(changes) - 5} more")


def refingerprint(
    path: Path = GOLDEN_SET,
    chunks_path: Path = CHUNKS_OUTPUT,
    previous_chunks_path: Path | None = None,
    force: bool = False,
) -> bool:
    """Rewrite corpus_fingerprint against the current chunk build, once, and
    only once every other thing that could have silently broken is checked.

    Gate 8 exists to make a stale golden set loud; this is the one place
    allowed to clear it, and it earns that by re-running gates 1 through 7
    against the current build (gate 2's chunk-id resolution and gate 4's
    evidence-containment check both bear directly on whether a re-chunk broke
    something) and, when previous_chunks_path is given, by proving every
    chunk the golden set actually depends on carries identical text to the
    build the reading pass was done against. chunker.py's id scheme makes
    that provable rather than merely likely: chunk ids are scoped to
    (source, page, chunk_type, index), so a text change shows up as a text
    change on the same id rather than as a silently renumbered one.

    Without previous_chunks_path there is nothing on disk to prove text
    held still against, since data/processed/ is entirely gitignored and
    carries no history of its own. That case still requires force=True,
    printed as what it is: gates 1 through 7 passing, not independent proof.
    """
    if not path.exists():
        print(f"{path} not found. Write the skeleton first.")
        return False
    if not chunks_path.exists():
        print(f"{chunks_path} not found. Run `python cli.py chunk` first.")
        return False

    questions, old_fingerprint = load_golden(path)
    chunks = load_chunks(chunks_path)
    chunk_map = _chunk_map(chunks)

    print(f"re-fingerprinting {path.name} against {chunks_path.name} "
          f"({len(chunks)} chunks)")

    failures = []
    failures += _gate1(questions)
    failures += _gate2(questions, chunks)
    failures += _gate3(questions)
    failures += _gate4(questions, chunk_map)
    failures += _gate5(questions)
    failures += _gate6(questions)
    failures += _gate7(questions, chunk_map)
    if failures:
        print("\nRefusing: gates 1-7 must pass before the fingerprint moves")
        for failure in failures:
            print(f"  FAIL  {failure}")
        return False

    if previous_chunks_path is not None:
        if not previous_chunks_path.exists():
            print(f"{previous_chunks_path} not found")
            return False
        previous_chunk_map = _chunk_map(load_chunks(previous_chunks_path))
        text_failures = _diff_against_previous(questions, chunk_map, previous_chunk_map)
        print(f"\nText check against {previous_chunks_path.name}, "
              f"{len(_referenced_chunk_ids(questions))} referenced chunks")
        if text_failures:
            for failure in text_failures:
                print(f"  FAIL  {failure}")
            print("\nRefusing: the reading pass no longer matches the corpus")
            return False
        print("  ok  every referenced chunk's text is unchanged")

        print("\nMetadata moves on referenced chunks:")
        _report_metadata_moves(questions, chunk_map, previous_chunk_map)
    elif not force:
        print("\nNo previous_chunks_path given: gates 1-7 passed, but text "
              "identity on referenced chunks was not independently confirmed. "
              "Pass --previous <snapshot> to prove it, or force=True to "
              "proceed on gates 1-7 alone.")
        return False
    else:
        print("\nProceeding on gates 1-7 alone; text identity was not "
              "independently confirmed (no previous_chunks_path given).")

    new_fingerprint = corpus_fingerprint(chunks_path)
    print(f"\nfingerprint: {old_fingerprint['sha256'][:12]}... to "
          f"{new_fingerprint['sha256'][:12]}...")
    save_golden(questions, new_fingerprint, path)
    print(f"written to {path}")
    return True


# --- verification -------------------------------------------------------------

def verify(
    questions: list[Question], fingerprint: dict, chunks: list[Chunk]
) -> list[str]:
    """Run all eight gates. Returns failures, empty when the set is usable."""
    chunk_map = _chunk_map(chunks)
    failures: list[str] = []
    failures += _gate1(questions)
    failures += _gate2(questions, chunks)
    failures += _gate3(questions)
    failures += _gate4(questions, chunk_map)
    failures += _gate5(questions)
    failures += _gate6(questions)
    failures += _gate7(questions, chunk_map)
    failures += _gate8(fingerprint, CHUNKS_OUTPUT)
    return failures


# --- coverage report, not gated -----------------------------------------------

def _report_coverage(questions: list[Question], chunk_map: dict[str, Chunk]) -> None:
    """Print what the set reaches, without turning any of it into a gate.

    Following fragment_ratio in quality.py and oversized_prefixes in
    context.py: these numbers are worth knowing and none of them has a
    principled cutoff, so none is enforced.
    """
    gold_chunks = [
        chunk_map[i]
        for q in questions
        for i in q.gold_chunk_ids
        if i in chunk_map
    ]
    types_reached = sorted({c.metadata["chunk_type"] for c in gold_chunks})
    paths_reached = {c.metadata["section_path"] for c in gold_chunks}

    print(f"\nCoverage")
    print(f"  chunk types reached by gold chunks: {len(types_reached)} "
          f"({', '.join(types_reached)})")
    print(f"  distinct section paths reached: {len(paths_reached)}")

    gold_counts = [len(q.gold_chunk_ids) for q in questions if q.expect == "answerable"]
    if gold_counts:
        print(f"  gold chunks per answerable question: "
              f"median {int(statistics.median(gold_counts))}, "
              f"min {min(gold_counts)}, max {max(gold_counts)}")

    disagreements = sum(1 for q in questions if q.page_vs_chunk not in ("", "none"))
    print(f"  questions with a page-vs-chunk disagreement: {disagreements}")


# --- entry point ---------------------------------------------------------------

def run(path: Path = GOLDEN_SET, chunks_path: Path = CHUNKS_OUTPUT) -> bool:
    """Verify the golden set and report coverage. True when all gates pass."""
    if not path.exists():
        print(f"{path} not found. Write the skeleton first.")
        return False
    if not chunks_path.exists():
        print(f"{chunks_path} not found. Run `python cli.py chunk` first.")
        return False

    started = time.time()
    questions, fingerprint = load_golden(path)
    chunks = load_chunks(chunks_path)

    print(f"{len(questions)} questions from {path.name}, "
          f"checked against {len(chunks)} chunks")

    failures = verify(questions, fingerprint, chunks)

    print("\nGates")
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
    else:
        print("  ok  all 8 gates pass")

    _report_coverage(questions, _chunk_map(chunks))
    print(f"\nelapsed {time.time() - started:.1f}s")

    return not failures


if __name__ == "__main__":
    run()
