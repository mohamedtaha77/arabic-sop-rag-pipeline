"""Entailment: does this passage actually support this claim.

Position: called from generation/run.py's own grounding guard step,
after synthesise.py, alongside guard.py's token diff. guard.py catches
the presenter adding something the synthesiser never said; this file
catches the synthesiser itself citing a passage that does not actually
support a sentence it wrote, the shape guard.py's own module docstring
names as out of its reach: a presenter, or a synthesiser, that keeps
every number and every citation but reverses what a sentence claims
about them passes a token diff clean and fails this check instead.

Two backends, one interface, chosen the way every prior stage settles a
choice in this pipeline, by measurement rather than preference: BGE-M3
against e5-large in stage 6, hybrid retrieval modes in stage 7, three
CRAG grading prompts in stage 8. Here the two candidates are JUDGE_MODEL
(llama3.2:3b, an LLM asked directly whether a passage entails a claim)
and NLI_MODEL (mDeBERTa-v3-base-xnli, a purpose-trained classifier in its
own worker process, mirroring techniques/_rerank_worker.py). JUDGE_MODEL
specifically, not GENERATOR_MODEL: crag.py's own evaluator grades
retrieval with GENERATOR_MODEL and that is not the same conflict this
file exists to avoid, since retrieval is not something GENERATOR_MODEL
itself produced. This file grades a claim GENERATOR_MODEL's own
synthesiser call wrote, which is exactly the "a judge sharing weights
with the generator grades its own work generously" case config.py's own
comment reserves JUDGE_MODEL for, and probe.py already gates that the two
tags differ. This is JUDGE_MODEL's first real use in this codebase.

There is real prior evidence pointing in both directions before either
backend is measured here. router.md measured a 3B local judge
over-flagging on CRAG's own near-identical grounding task, retrieval
correctness rather than claim entailment but the same shape of
judgement, good recall and poor precision. A throwaway probe run before
this file was written scored mDeBERTa on three constructed Arabic pairs
(genuinely entailed, the same sentence with one number changed,
genuinely unrelated) at 0.946, 0.005 and 0.001 entailment probability
respectively, a clean separation, but three constructed pairs are not a
benchmark. Both are real signals and neither settles the question; the
benchmark below is what actually does.

The benchmark, built from the golden set rather than invented, in the
manner of stage 8's own CRAG gate reusing Q10: positive pairs are each
answerable question's own reference-answer sentences against the
concatenation of that question's own gold chunks; negative pairs are
two different, deliberately distinct shapes, a sentence against a
different question's gold chunks (topic mismatch, the easy case) and a
sentence with its own number mutated, checked against its own question's
gold chunks (same topic, one wrong fact, the shape this whole guard
exists to catch). The positive pairs are realistic and slightly unfair
by design: question.py's own docstring records that a reference answer
is written from the rendered page while a chunk carries whatever OCR
left there, so entailment here is being asked to bridge a real
occasional gap between clean prose and OCR-damaged source text, not
graded against the evidence quotes, which are themselves substrings of
chunk text and would make this trivially easy rather than realistic.

What this file does not do: it does not decide which sentences of a
real answer get checked against which chunks, or what happens on a
failed check. That pairing and that consequence belong to
generation/run.py, which has the actual citation markers and the actual
retrieved chunk text; this file only answers, given one premise and one
hypothesis, whether the premise entails the hypothesis, under whichever
backed the bake-off below decided actually works better here.
"""

from __future__ import annotations

import atexit
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass

from ..config import JUDGE_MODEL, NLI_MODEL, PROJECT_ROOT
from ..llm.ledger import Ledger
from ..retrieval.text import fold_digits

BACKENDS = ("nli", "llm")

# hex codepoints, per the project's own rule. Sentence-ending marks only,
# unlike guard.py's own _SEGMENT_SPLIT which also splits on commas for a
# finer-grained soft signal: a benchmark needs a genuinely complete claim
# as its hypothesis, not a clause fragment, so this splits more coarsely
# on purpose.
_SENTENCE_SPLIT = re.compile("[.!؟]+")

_MIN_SENTENCE_WORDS = 3

# Added to a sentence's own first number to build a same-topic,
# wrong-fact negative. Arbitrary but fixed and documented, the same
# spirit as CHARS_PER_TOKEN's own rounding direction: any clearly
# different number does the job, and a fixed offset keeps the benchmark
# reproducible rather than randomised.
_NUMBER_MUTATION_OFFSET = 37

_LLM_JUDGE_SYSTEM_PROMPT = """\
Judge whether the passage below logically entails the claim, meaning \
the passage's own stated facts make the claim true, not merely that \
they are on the same topic. A claim with a number, date or name the \
passage does not state, or states differently, is not entailed even if \
the general subject matches.

Reply with JSON only: {"entailed": true or false}\
"""


# --- the NLI worker, mirroring techniques/rerank.py's own management ----------

_worker: subprocess.Popen | None = None
_WORKER_START_RETRIES = 5
_WORKER_START_RETRY_DELAY_S = 15


def _spawn_worker() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "pipeline.generation._nli_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
        bufsize=1, cwd=str(PROJECT_ROOT),
    )


def _ensure_worker() -> subprocess.Popen:
    """Start the persistent NLI subprocess on first use, or return the one
    already running. Blocks on its "ready" line, the same contract
    rerank.py's own _ensure_worker relies on for its own worker.
    """
    global _worker
    if _worker is not None and _worker.poll() is None:
        return _worker

    last_error = ""
    for attempt in range(1, _WORKER_START_RETRIES + 1):
        started = time.perf_counter()
        candidate = _spawn_worker()
        ready_line = candidate.stdout.readline()
        elapsed = time.perf_counter() - started
        if ready_line and "ready" in ready_line:
            _worker = candidate
            atexit.register(_terminate_worker)
            return _worker

        time.sleep(0.5)
        return_code = candidate.poll()
        last_error = (
            f"exit {return_code} after {elapsed:.1f}s, "
            f"stderr: {candidate.stderr.read()[-500:]}"
            if return_code is not None
            else f"no ready line after {elapsed:.1f}s (got {ready_line!r}), "
                 f"still running"
        )
        print(f"  NLI worker start attempt {attempt}/{_WORKER_START_RETRIES} "
              f"failed ({last_error}); a crashed worker costs only the "
              f"retry, since the next attempt is a fresh process")
        if return_code is None:
            candidate.kill()
        if attempt < _WORKER_START_RETRIES:
            time.sleep(_WORKER_START_RETRY_DELAY_S)

    raise RuntimeError(
        f"NLI worker failed to start {_WORKER_START_RETRIES} times in a "
        f"row (most recently: {last_error}). This machine's memory is a "
        f"known constraint under load; see LEARNING/router.md."
    )


def _terminate_worker() -> None:
    global _worker
    if _worker is not None and _worker.poll() is None:
        try:
            _worker.stdin.close()
            _worker.wait(timeout=5)
        except Exception:  # noqa: BLE001
            _worker.kill()


def warm_up() -> None:
    """Start the NLI worker now, rather than waiting for the first check.

    Call this before retriever.open_shipping() in any caller that will
    reach this backend at all, and never after, the same ordering
    contract rerank.py's own warm_up documents and for the same reason:
    a second large model load colliding with an already-resident BGE-M3
    embedder is this project's one measured, reproducible class of
    crash. Measured directly while building this file's own
    run_bakeoff, not merely inferred: generation/evaluate.py's first
    version ran the bake-off with BGE-M3 and the reranker worker both
    already resident (built for a different, since-corrected reason),
    and it segfaulted with no traceback, the same signature every other
    instance of this fault leaves. This is a step further than the
    two-models-at-once fault embedding.md, router.md and rerank.py's own
    docstring all document: three separate model-holding processes at
    once, main process plus reranker worker plus this worker, rather
    than two, evidence that the real constraint on this machine is
    total system memory under load, not strictly which pair of
    checkpoints happens to collide. evaluate.py's own run() now runs
    this file's entire bake-off, warm_up included, before BGE-M3 or the
    reranker worker ever load, and calls shutdown() immediately after,
    freeing this worker's memory before either of the other two loads.
    """
    _ensure_worker()


def shutdown() -> None:
    """Stop the NLI worker and free its memory.

    Call this once a caller is done with this backend for now, in
    particular right after run_bakeoff(), so its memory is not held
    through whatever the caller does next: see warm_up's own docstring
    for the measured reason evaluate.py's own run() calls this before
    BGE-M3 or the reranker worker ever load.
    """
    _terminate_worker()


# Pairs per request to the worker, not the whole benchmark at once. Found
# necessary directly, not assumed: a single 67-pair request killed the
# worker outright twice in a row, "closed its pipe unexpectedly
# mid-request", the OS terminating a process rather than any Python
# exception the worker itself could report. DeBERTa's own disentangled
# attention computes extra content-to-position and position-to-content
# scores beyond a standard transformer's, and a batch of 67 sequences
# near this checkpoint's own 512-token limit padded together plausibly
# spikes CPU activation memory well past what a single request should
# need to cost, on a machine already measured tight on memory
# (llm.md: 11.8 GB total). The same bounded-batch principle
# EMBED_BATCH_SIZE already applies to BGE-M3 and RERANK_TOP_N applies to
# the cross-encoder, sized here for a purpose-trained base model rather
# than assumed safe at an unbounded size.
_NLI_BATCH_SIZE = 8

# One retry per batch, mirroring rerank.py's own worker-spawn retry
# shape for the same reason: a crashed worker costs only the retry,
# since _ensure_worker's own poll() check spawns a fresh process when
# the old one has already exited.
_NLI_REQUEST_RETRIES = 2


def _score_nli_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """One round trip for a batch already sized at or under
    _NLI_BATCH_SIZE, retried as a fresh worker on a mid-request crash.
    """
    last_error: Exception | None = None
    for attempt in range(1, _NLI_REQUEST_RETRIES + 1):
        worker = _ensure_worker()
        request = json.dumps({"pairs": [list(p) for p in pairs]})
        try:
            worker.stdin.write(request + "\n")
            worker.stdin.flush()
            response_line = worker.stdout.readline()
            if not response_line:
                raise RuntimeError(
                    "NLI worker closed its pipe unexpectedly mid-request"
                )
        except (BrokenPipeError, RuntimeError) as error:
            last_error = error
            # The pipe closing means the child already exited; this just
            # tidies up the reference so _ensure_worker's own poll()
            # check spawns a genuinely fresh process next time, the same
            # cleanup warm_up's own docstring already relies on shutdown()
            # for elsewhere.
            _terminate_worker()
            continue
        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"NLI worker reported an error: {response['error']}")
        return [probs[0] for probs in response["probs"]]
    raise RuntimeError(
        f"NLI worker failed {_NLI_REQUEST_RETRIES} times in a row on the "
        f"same batch (most recently: {last_error}); see this module's own "
        f"_score_nli_batch docstring."
    ) from last_error


def _score_nli(pairs: list[tuple[str, str]]) -> list[float]:
    """P(entailment) for each (premise, hypothesis) pair, sent to the
    worker in bounded batches of _NLI_BATCH_SIZE rather than all at once.
    """
    scores: list[float] = []
    for start in range(0, len(pairs), _NLI_BATCH_SIZE):
        scores.extend(_score_nli_batch(pairs[start:start + _NLI_BATCH_SIZE]))
    return scores


# --- the two backends, one interface -------------------------------------------

@dataclass
class EntailmentResult:
    """One entailment check.

        entailed  the final yes/no this file's caller acts on
        score     P(entailment) for the nli backend, or None for llm,
                  which makes a categorical judgement with no comparable
                  continuous score
        backend   which of BACKENDS produced this result
    """

    entailed: bool
    score: float | None
    backend: str


def _check_nli(premise: str, hypothesis: str, threshold: float) -> EntailmentResult:
    score = _score_nli([(premise, hypothesis)])[0]
    return EntailmentResult(entailed=score >= threshold, score=score, backend="nli")


def _check_llm(premise: str, hypothesis: str, ledger: Ledger) -> EntailmentResult:
    messages = [
        {"role": "system", "content": _LLM_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage: {premise}\n\nClaim: {hypothesis}"},
    ]
    response = ledger.call(
        "Grounding guard", messages, JUDGE_MODEL,
        temperature=0.0, max_tokens=50,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response.text)
        entailed = bool(parsed.get("entailed", False))
    except json.JSONDecodeError:
        # Fails toward "not entailed", the safer direction, the same rule
        # router.py's own parse-failure handling and crag.py's own
        # degrade-to-ambiguous both already follow in this codebase: an
        # unparseable verdict should not silently pass a claim.
        entailed = False
    return EntailmentResult(entailed=entailed, score=None, backend="llm")


def check(
    premise: str, hypothesis: str, ledger: Ledger, backend: str,
    threshold: float,
) -> EntailmentResult:
    """One entailment check, under whichever backend the caller names.
    ``threshold`` is read only by the nli backend; the llm backend's
    decision is baked into its own prompt and has no numeric threshold
    to sweep, a real, structural asymmetry between the two rather than
    an oversight, recorded plainly in run_bakeoff's own output.
    """
    if backend == "nli":
        return _check_nli(premise, hypothesis, threshold)
    if backend == "llm":
        return _check_llm(premise, hypothesis, ledger)
    raise ValueError(f"{backend!r} is not one of {BACKENDS}")


# --- the benchmark, built from the golden set -----------------------------------

@dataclass
class BenchmarkCase:
    premise: str
    hypothesis: str
    label: bool  # True: should be judged entailed
    question_id: str
    kind: str  # "positive", "negative_topic", "negative_number"


def _sentences(text: str) -> list[str]:
    return [
        s.strip() for s in _SENTENCE_SPLIT.split(text)
        if len(s.strip().split()) >= _MIN_SENTENCE_WORDS
    ]


def _mutate_number(sentence: str) -> str | None:
    """The sentence with its first number changed, or None if it has
    none. Folds Arabic-Indic digits to ASCII first (fold_digits), the
    same normalisation guard.py applies before comparing tokens, so a
    sentence written with either digit style is still caught.
    """
    folded = fold_digits(sentence)
    match = re.search(r"\d+", folded)
    if not match:
        return None
    mutated_value = str(int(match.group()) + _NUMBER_MUTATION_OFFSET)
    return folded[:match.start()] + mutated_value + folded[match.end():]


def build_benchmark(
    chunks_path: "object" = None,
) -> list[BenchmarkCase]:
    """Positive, topic-mismatch negative, and number-mutated negative
    cases, built once from the golden set's own answerable questions and
    the shipped chunk variant, never invented text: the same discipline
    that keeps every other gate in this pipeline pointed at real corpus
    content rather than a synthetic stand-in for it.
    """
    from ..chunking.chunk import load_chunks
    from ..config import CONTEXT_OUTPUTS, GOLDEN_SET
    from ..golden.question import load_golden

    path = chunks_path or CONTEXT_OUTPUTS["template"]
    chunks = load_chunks(path)
    chunk_lookup = {c.metadata["chunk_id"]: c for c in chunks}

    questions, _ = load_golden(GOLDEN_SET)
    answerable = [
        q for q in questions if q.expect == "answerable" and q.gold_chunk_ids
    ]

    def gold_text(question) -> str:
        return "\n".join(
            chunk_lookup[cid].text for cid in question.gold_chunk_ids
            if cid in chunk_lookup
        )

    cases: list[BenchmarkCase] = []
    for i, question in enumerate(answerable):
        premise = gold_text(question)
        if not premise:
            continue
        other = answerable[(i + len(answerable) // 2) % len(answerable)]
        other_premise = gold_text(other)

        for sentence in _sentences(question.answer):
            cases.append(BenchmarkCase(
                premise=premise, hypothesis=sentence, label=True,
                question_id=question.id, kind="positive",
            ))
            if other_premise and other.id != question.id:
                cases.append(BenchmarkCase(
                    premise=other_premise, hypothesis=sentence, label=False,
                    question_id=question.id, kind="negative_topic",
                ))
            mutated = _mutate_number(sentence)
            if mutated is not None:
                cases.append(BenchmarkCase(
                    premise=premise, hypothesis=mutated, label=False,
                    question_id=question.id, kind="negative_number",
                ))
    return cases


# --- scoring the benchmark -------------------------------------------------------

def _balanced_accuracy(predictions: list[bool], labels: list[bool]) -> float:
    positives = [p for p, l in zip(predictions, labels) if l]
    negatives = [p for p, l in zip(predictions, labels) if not l]
    tpr = sum(positives) / len(positives) if positives else 0.0
    tnr = (len(negatives) - sum(negatives)) / len(negatives) if negatives else 0.0
    return (tpr + tnr) / 2


def _accuracy_by_kind(
    cases: list[BenchmarkCase], predictions: list[bool],
) -> dict[str, float]:
    """Per-kind recall: for "positive", the fraction correctly predicted
    entailed; for either negative kind, the fraction correctly predicted
    not entailed. An aggregate balanced accuracy can hide a backend that
    only ever catches the easy topic-mismatch negatives while missing
    the harder, same-topic number-mutated ones, which is exactly the
    failure shape this check exists to catch, so it is reported
    separately rather than folded into one number.
    """
    by_kind: dict[str, list[bool]] = {}
    for case, predicted in zip(cases, predictions):
        correct = predicted if case.label else not predicted
        by_kind.setdefault(case.kind, []).append(correct)
    return {kind: sum(vals) / len(vals) for kind, vals in by_kind.items()}


def _evaluate_nli(cases: list[BenchmarkCase]) -> dict:
    scores = _score_nli([(c.premise, c.hypothesis) for c in cases])
    labels = [c.label for c in cases]

    swept = []
    for threshold in [round(t * 0.05, 2) for t in range(4, 19)]:  # 0.20..0.90
        predictions = [score >= threshold for score in scores]
        swept.append((threshold, _balanced_accuracy(predictions, labels)))
    best_threshold, best_accuracy = max(swept, key=lambda pair: pair[1])
    best_predictions = [score >= best_threshold for score in scores]
    return {
        "backend": "nli", "best_threshold": best_threshold,
        "best_balanced_accuracy": best_accuracy, "sweep": swept,
        "scores": scores,
        "accuracy_by_kind": _accuracy_by_kind(cases, best_predictions),
    }


def _evaluate_llm(cases: list[BenchmarkCase], ledger: Ledger) -> dict:
    predictions = [
        _check_llm(c.premise, c.hypothesis, ledger).entailed for c in cases
    ]
    labels = [c.label for c in cases]
    return {
        "backend": "llm", "best_threshold": None,
        "best_balanced_accuracy": _balanced_accuracy(predictions, labels),
        "predictions": predictions,
        "accuracy_by_kind": _accuracy_by_kind(cases, predictions),
    }


# How close the two backends' overall balanced accuracy has to be before
# the tie-break below applies, rather than the plain higher number
# winning outright. Named and reasoned about here for the same purpose
# config.PREFERRED_MODEL's own comment states for a different bake-off's
# tie-break: "an argument for a tie, never for a result." 0.03 is
# generous relative to a 67-case benchmark, where a handful of cases
# flipping either way moves the aggregate by more than this.
_TIE_MARGIN = 0.03


def run_bakeoff() -> dict:
    """Score both backends on the same benchmark and decide the winner.
    Read back by generation/evaluate.py, the same way techniques/
    evaluate.py reads each technique's own verify function rather than
    re-running a grid of its own.

    The winner is not simply whichever scores higher on the aggregate.
    Measured directly: nli and llm landed within _TIE_MARGIN of each
    other overall (0.732 against 0.720 on the first real run), close
    enough that either could plausibly flip with a slightly different
    benchmark sample, while their recall on negative_number specifically,
    a same-topic sentence with one number changed, the exact shape of a
    real fabrication rather than an easy off-topic negative, differed by
    much more (llm 0.857 against nli 0.571 on that run). That category is
    the more decision-relevant one for what this file exists to catch, so
    within _TIE_MARGIN of the aggregate, negative_number recall breaks
    the tie instead of the aggregate deciding blind. Outside that margin,
    the aggregate still decides, since a wide enough overall gap is
    evidence a backend is broadly better, not noise in one category.
    """
    cases = build_benchmark()
    ledger = Ledger(label="entail-bakeoff")

    nli_result = _evaluate_nli(cases)
    llm_result = _evaluate_llm(cases, ledger)

    aggregate_gap = abs(
        nli_result["best_balanced_accuracy"] - llm_result["best_balanced_accuracy"]
    )
    if aggregate_gap <= _TIE_MARGIN:
        winner = max(
            (nli_result, llm_result),
            key=lambda r: r["accuracy_by_kind"].get("negative_number", 0.0),
        )
        tie_break_applied = True
    else:
        winner = max(
            (nli_result, llm_result), key=lambda r: r["best_balanced_accuracy"],
        )
        tie_break_applied = False

    return {
        "n_cases": len(cases),
        "n_positive": sum(1 for c in cases if c.label),
        "n_negative_topic": sum(1 for c in cases if c.kind == "negative_topic"),
        "n_negative_number": sum(1 for c in cases if c.kind == "negative_number"),
        "nli": nli_result, "llm": llm_result,
        "nli_model": NLI_MODEL, "llm_model": JUDGE_MODEL,
        "aggregate_gap": aggregate_gap,
        "tie_break_applied": tie_break_applied,
        "winner_backend": winner["backend"],
        "winner_threshold": winner["best_threshold"],
        "winner_balanced_accuracy": winner["best_balanced_accuracy"],
    }


if __name__ == "__main__":
    from ..config import PROCESSED_DIR

    warm_up()
    outcome = run_bakeoff()

    lines = [
        f"benchmark: {outcome['n_cases']} cases "
        f"({outcome['n_positive']} positive, "
        f"{outcome['n_negative_topic']} negative/topic-mismatch, "
        f"{outcome['n_negative_number']} negative/number-mutated)",
        "",
        f"nli  ({outcome['nli_model']}) best balanced accuracy: "
        f"{outcome['nli']['best_balanced_accuracy']:.3f} "
        f"at threshold {outcome['nli']['best_threshold']}",
        f"     recall by kind: {outcome['nli']['accuracy_by_kind']}",
        f"llm  ({outcome['llm_model']}) balanced accuracy: "
        f"{outcome['llm']['best_balanced_accuracy']:.3f} "
        f"(no threshold: the llm backend's decision is baked into its own prompt)",
        f"     recall by kind: {outcome['llm']['accuracy_by_kind']}",
        "",
        f"aggregate gap: {outcome['aggregate_gap']:.3f} "
        f"({'within' if outcome['tie_break_applied'] else 'outside'} "
        f"the {_TIE_MARGIN} tie margin)",
        f"winner: {outcome['winner_backend']} "
        f"(balanced accuracy {outcome['winner_balanced_accuracy']:.3f}"
        + (f", threshold {outcome['winner_threshold']}" if outcome['winner_threshold'] is not None else "")
        + (", decided by negative_number recall, the tie-break criterion"
           if outcome['tie_break_applied'] else ", decided by the aggregate")
        + ")",
        "",
        "recall by kind reads: for \"positive\", the share correctly judged "
        "entailed; for either negative kind, the share correctly judged "
        "not entailed. negative_number is the harder, more informative "
        "case, same topic, one wrong fact, which is what a real "
        "fabrication looks like; negative_topic is the easier case, an "
        "unrelated passage.",
        "",
        "full nli threshold sweep:",
    ]
    for threshold, accuracy in outcome["nli"]["sweep"]:
        lines.append(f"  {threshold:.2f}: {accuracy:.3f}")

    out_path = PROCESSED_DIR / "17_entail_bakeoff.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    # The gate here is a sanity floor, not a claim either backend is
    # excellent: 0.65 balanced accuracy is clearly better than the 0.5
    # a coin flip gets on this benchmark's own roughly balanced label
    # split, confirming entailment is measurable on this corpus at all
    # before either backend's own number is trusted for a real decision.
    ok = outcome["winner_balanced_accuracy"] > 0.65
    print(f"{'ok' if ok else 'FAIL'}: winner={outcome['winner_backend']} "
          f"balanced_accuracy={outcome['winner_balanced_accuracy']:.3f}, "
          f"written to {out_path}")
