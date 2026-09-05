"""The three arms, once each, over the golden set: an ArmRun per question.

Position: every earlier stage already built and gated one mechanism.
generation.run.answer is the one sanctioned call path for turning a
question into an Answer, per that file's own module docstring, and this
file is what calls it three ways rather than once, so that section 14's
required comparison and section 16's "Basic RAG versus Advanced RAG"
question are answered by actually running both, not by reasoning about
what each would probably do.

The three arms, and why each exists:

    basic       section 2's own flow, no router at all: Question ->
                Embedding -> Vector Search -> Top-K -> Prompt -> LLM ->
                Answer. Built with route_override="basic_rag" and
                technique_set=TechniqueSet.none(), over all 20 golden
                questions. generation.run.answer's own route_override
                parameter exists because of this arm: forcing
                TechniqueSet.none() through the ordinary router would
                still let it refuse Q9 at the pre-gate or send Q10
                wherever a real classification lands, handing the
                baseline two behaviours section 2's own flow has no
                mechanism to produce.
    adaptive    the real shipped system, over all 20 golden questions:
                router, gate, every runtime trigger, exactly what
                `python cli.py ask "..."` already runs for one question.
    forced      each question's own expected technique forced on, over
                the 8-question population generation.evaluate's own
                measure_crag_threshold already uses (Q3-Q7, Q19, Q20,
                Q10). Exists because the router currently sends only 5 of
                20 questions to advanced_rag (06_router.md); since
                run.py's own _resolve_technique_set returns
                TechniqueSet.none() for any other route, basic and
                adaptive are identical code paths on the other 15, and a
                comparison resting on five questions would understate
                what the eight techniques actually do.

What this file does not do: it does not score, judge or compare anything.
score.py reads chunk ids back against gold_chunk_ids, ragas_judge.py
grades an answer's text, compare.py reads both against each other. This
file's own job ends the moment every (arm, question) pair has produced
one ArmRun and it has been written to disk once, so that every file after
this one is a read of RUNS_OUTPUT rather than a re-run of local
generation.
"""

from __future__ import annotations

from ..config import GOLDEN_SET, RUNS_OUTPUT
from ..golden.question import Question, load_golden
from ..llm import client
from ..llm.ledger import Ledger
from ..retrieval import embed_client
from ..retrieval.retriever import ShippingHandle, open_shipping
from ..router.schema import TechniqueSet
from ..generation.answer import Answer
from ..generation.run import answer as generation_answer
from .record import ARMS, ArmRun, index_by_arm, load_runs, save_runs, to_jsonable

# The forced arm's own eight-question population and each one's own named
# technique. Not invented for this file: Q3 to Q7, Q19 and Q20 name the
# technique they were built to exercise in their own golden_set.json
# `tests` field, confirmed against 06_router.md's own recorded requests
# for the four (Q4 Multi-Query, Q5/Q19 Decomposition, Q6 HyDE, Q7/Q20
# Self-Query); Q10 matches generation.run.verify_four_paths's own
# precedent exactly, TechniqueSet(crag=True) alone.
#
# Reranking is added alongside every query-transformation technique
# because that is the real shipped default (06_technique_decision.json:
# reranking_default=true) a genuine advanced_rag classification would
# have applied on top of whichever technique the router named; Q10 stays
# bare, matching the one precedent already in this codebase for forcing
# a technique_set rather than inventing a second convention for it.
_FORCED_TECHNIQUES: dict[str, TechniqueSet] = {
    "Q3": TechniqueSet(rewriting=True, reranking=True),
    "Q4": TechniqueSet(multi_query=True, reranking=True),
    "Q5": TechniqueSet(decomposition=True, reranking=True),
    "Q6": TechniqueSet(hyde=True, reranking=True),
    "Q7": TechniqueSet(self_query=True, reranking=True),
    "Q19": TechniqueSet(decomposition=True, reranking=True),
    "Q20": TechniqueSet(self_query=True, reranking=True),
    "Q10": TechniqueSet(crag=True),
}


def _sorted_questions(questions: list[Question]) -> list[Question]:
    """Q1 to Q20 in numeric order, not file order: Q3's own history
    threading depends on Q2 having already run in the same arm, which
    only holds if this file processes ids in the order their own numbers
    suggest rather than trusting golden_set.json's own row order to never
    drift.
    """
    return sorted(questions, key=lambda q: int(q.id[1:]))


def _arm_config(arm: str, question_id: str) -> tuple[str | None, TechniqueSet | None]:
    """route_override and technique_set for one (arm, question) pair.
    None, None for the adaptive arm means exactly what
    generation.run.answer's own defaults mean: run the real router and
    the real runtime triggers, nothing forced.
    """
    if arm == "basic":
        return "basic_rag", TechniqueSet.none()
    if arm == "adaptive":
        return None, None
    if arm == "forced":
        return "advanced_rag", _FORCED_TECHNIQUES[question_id]
    raise ValueError(f"{arm!r} is not one of {ARMS}")


def _build_history(
    question: Question, by_id: dict[str, Question], arm_answers: dict[str, str],
) -> tuple[list[tuple[str, str]] | None, str]:
    """(history, history_source) for one question in one arm.

    Only Q3 carries a depends_on in this golden set. arm_answers holds
    every answer this same arm has already generated, in id order; a
    dependency answered earlier in this arm is what "using preceding
    context" means per generation.run.answer's own docstring. The forced
    arm's own population does not include Q2, so Q3's dependency there
    always falls back to Q2's golden reference answer, the identical
    choice generation.evaluate.measure_crag_threshold already makes for
    the same reason: a measurement running its own narrower population
    has no generated answer to thread.
    """
    if not question.depends_on:
        return None, "none"
    dep_id = question.depends_on
    if dep_id in arm_answers:
        return [(by_id[dep_id].question, arm_answers[dep_id])], "same_arm"
    return [(by_id[dep_id].question, by_id[dep_id].answer)], "golden_reference"


def _to_arm_run(
    arm: str, question_id: str, question_text: str, result: Answer, history_source: str,
) -> ArmRun:
    run_result = result.run
    contexts = tuple(item.chunk.text for item in run_result.retrieved)
    return ArmRun(
        id=question_id, arm=arm, question=question_text,
        route=run_result.decision.route, gate_matched=run_result.gate_matched,
        refusal_kind=result.refusal_kind, executed=tuple(run_result.executed),
        chunk_ids=tuple(run_result.chunk_ids), contexts=contexts,
        context_text=run_result.context_text, history_source=history_source,
        kind=result.kind, text=result.text, synthesised=result.synthesised,
        presented=result.presented, presenter_rejected=result.presenter_rejected,
        citations=tuple(result.citations),
        generation_traces=to_jsonable(result.traces),
        technique_traces=to_jsonable(run_result.traces),
        ledger=result.ledger.to_dict(),
    )


def _error_arm_run(
    arm: str, question_id: str, question_text: str, ledger: Ledger,
    history_source: str, error: Exception,
) -> ArmRun:
    """What one question becomes when generation.run.answer raised before
    producing an Answer at all. See record.py's own module docstring for
    why "error" is a legal kind: client.py raises rather than silently
    truncating on a max_tokens overflow, the right call for one question
    and the wrong one for a 48-question unattended batch, and Q4 and Q19
    are already known (LEARNING/generation.md) to overflow
    SYNTHESIS_MAX_TOKENS specifically when no query-transformation
    technique has smoothed the context first, which is exactly the basic
    arm's own shape for every question by construction.

    ledger is still recorded, not discarded: client.py's own docstring
    notes a raised call already spent whatever tokens it spent before
    failing, and this is what lets the report say "this cost nothing
    because it never got a response" apart from "this cost real tokens
    before failing", two different findings a bare error string cannot
    tell apart on its own.
    """
    return ArmRun(
        id=question_id, arm=arm, question=question_text,
        route="error", gate_matched=False, refusal_kind=None, executed=(),
        chunk_ids=(), contexts=(), context_text="", history_source=history_source,
        kind="error", text="", synthesised="", presented="",
        presenter_rejected=False, citations=(), generation_traces={},
        technique_traces={}, ledger=ledger.to_dict(), error=str(error),
    )


# Substrings actually seen, verbatim, in this machine's own repeated
# CUDA allocation failures across two full harness runs and fifty-plus
# retry attempts: the same 13 questions failed identically every time,
# ruling out transient contention and confirming these specific calls'
# own larger contexts (Reranking widens to RERANK_TOP_N, CRAG's own
# grading and re-query add calls on top) structurally exceed what this
# 4 GB card has free, not something a same-request retry can ever clear.
# client.set_force_cpu is the one thing that does: Ollama's own native
# endpoint honours num_gpu=0, confirmed directly (a live call showed
# size_vram: 0 in /api/ps afterward), where the OpenAI-compatible one
# silently ignores an "options" field and loads onto the card anyway.
_GPU_MEMORY_ERROR_MARKERS = (
    "cudaMalloc failed",
    "CUDA error",
    "CUDA0 buffer",
    "CUDA_Host buffer",
    "llama-server process has terminated",
    "llama-server reported out-of-memory",
)


def _looks_like_gpu_memory_error(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _GPU_MEMORY_ERROR_MARKERS)


## Reset Ollama's own resident model every this many questions, not only
# between arms. Found necessary by running this exact file against this
# exact machine, with faulthandler enabled to trace a raw access
# violation that reproduced at the adaptive arm's own Q3, its first
# genuine BGE-M3 cache miss: the reranker worker subprocess
# rerank.warm_up() started before this loop ever runs is already resident
# by then, and router.md already documents this exact mirror direction as
# a real crash, "BGE-M3's own load failing when the reranker worker is
# already warm in a sibling process." First set to 2, then measured to
# have a real cost of its own: a full run with it at 2 later produced 13
# of 48 "error" records in one pass, every one of them Ollama's own
# llama-server failing "cudaMalloc failed: out of memory" during startup,
# not the reranker/embedder crash this constant was built for. Resetting
# this often means Ollama cold-starts its own CUDA context far more
# often, and a cold start is itself the moment that failed; forcing more
# of them under the same tight memory this constant exists because of
# was making a different failure more likely while fixing the first one.
# 5 is a middle ground, not a re-measured optimum: still meaningfully more
# frequent than resetting only between arms, without forcing a cold start
# on nearly every question. A raw access violation, and this new OOM
# shape, are both errors the try/except below cannot always prevent
# outright; run()'s own per-question resume is what actually recovers
# from whichever one still gets through.
_OLLAMA_RESET_EVERY = 5


def run_arm(
    arm: str, population: list[Question], by_id: dict[str, Question],
    handle: ShippingHandle, already_done: dict[str, ArmRun] | None = None,
    on_progress=None,
) -> tuple[list[ArmRun], dict[str, list[str]]]:
    """Every question in ``population``, once, through this arm's own
    route_override and technique_set. Returns the records and, keyed by
    question id, any ledger.verify() failure for that question: checked
    here, against the live Ledger object, rather than after
    ArmRun.ledger has already flattened it to a plain dict.

    already_done resumes a previously interrupted run of this exact arm:
    a question already present is not re-answered, but its own text
    still seeds arm_answers, since Q3's own history threading needs Q2's
    answer whether Q2 ran in this process or a now-dead earlier one.
    on_progress, if given, is called after every question, done or
    skipped, with the runs list so far, so a caller can persist to disk
    per question rather than per arm; see run()'s own docstring for why
    that granularity is what this machine's own crash history calls for.

    A question whose own call raises is caught here, not left to end the
    whole run: see _error_arm_run's own docstring. Caught broadly rather
    than narrowed to LLMError alone, because the point of catching at all
    is this loop surviving to the next question no matter which of
    several real, local-machine failure modes this project has already
    documented (a segfaulted worker, an Ollama OOM, a genuine content
    overflow) happens to fire on a given question; a narrower except
    would still let this run die on the first one it did not anticipate.
    Catching cannot cover every failure mode, though: _OLLAMA_RESET_EVERY
    above exists for one this loop cannot catch at all, a raw access
    violation, which ends the whole process regardless of any try/except
    written here, and already_done plus on_progress exist for the same
    fault at the level this loop actually can do something about: not
    preventing the crash, but never re-paying for work already done
    before the next one happens.
    """
    already_done = already_done or {}
    arm_answers: dict[str, str] = {
        qid: run.text for qid, run in already_done.items()
    }
    ledger_failures: dict[str, list[str]] = {}
    runs: list[ArmRun] = list(already_done.values())

    for index, question in enumerate(population):
        if question.id in already_done:
            print(f"{arm}/{question.id}: already done, skipping", flush=True)
            continue
        if index and index % _OLLAMA_RESET_EVERY == 0:
            client.unload_models()
        history, history_source = _build_history(question, by_id, arm_answers)
        route_override, technique_set = _arm_config(arm, question.id)
        ledger = Ledger(label=f"{arm}-{question.id}")

        try:
            result = generation_answer(
                question.question, ledger, handle, history=history,
                technique_set=technique_set, route_override=route_override,
            )
        except Exception as error:  # noqa: BLE001
            if _looks_like_gpu_memory_error(error):
                print(f"{arm}/{question.id}: GPU memory error ({error}); "
                      f"retrying this question on CPU", flush=True)
                ledger = Ledger(label=f"{arm}-{question.id}-cpu-retry")
                client.set_force_cpu(True)
                try:
                    result = generation_answer(
                        question.question, ledger, handle, history=history,
                        technique_set=technique_set, route_override=route_override,
                    )
                except Exception as retry_error:  # noqa: BLE001
                    error = retry_error
                    result = None
                finally:
                    client.set_force_cpu(False)
                if result is not None:
                    arm_answers[question.id] = result.text
                    failures = ledger.verify()
                    if failures:
                        ledger_failures[question.id] = failures
                    runs.append(_to_arm_run(
                        arm, question.id, question.question, result, history_source,
                    ))
                    print(f"{arm}/{question.id}: CPU retry succeeded: "
                          f"kind={result.kind}", flush=True)
                    if on_progress:
                        on_progress(runs)
                    continue
            arm_answers[question.id] = ""
            runs.append(_error_arm_run(
                arm, question.id, question.question, ledger, history_source, error,
            ))
            print(f"{arm}/{question.id}: ERROR {error}", flush=True)
            if on_progress:
                on_progress(runs)
            continue

        arm_answers[question.id] = result.text
        failures = ledger.verify()
        if failures:
            ledger_failures[question.id] = failures
        runs.append(_to_arm_run(arm, question.id, question.question, result, history_source))

        print(
            f"{arm}/{question.id}: route={result.run.decision.route} "
            f"kind={result.kind} executed={list(result.run.executed)} "
            f"chunks={len(result.run.retrieved)}"
            + (f" LEDGER FAILURES={failures}" if failures else ""),
            flush=True,
        )
        if on_progress:
            on_progress(runs)

    return runs, ledger_failures


def verify_harness(
    runs: list[ArmRun], questions: list[Question],
    ledger_failures: dict[str, dict[str, list[str]]],
) -> dict:
    """The gate this file exists to pass: every (arm, question) this
    stage promised produced a record, every one of those records' own
    ledger was clean, and the basic arm, which has no router to refuse
    anything, actually retrieved something for every answerable question,
    section 2's own flow having no other way to fail silently.

    errors is reported but does not gate ok: a genuine call failure
    (kind="error") is, on current evidence, an expected property of the
    basic arm specifically (see _error_arm_run's own docstring), not a
    wiring bug this gate exists to catch. basic_empty_retrieval excludes
    an errored question for the same reason: its own empty chunk_ids
    describe a lost record, not a retriever that returned nothing, and
    conflating the two would misreport a known content-length finding as
    a retrieval defect.
    """
    by_arm = index_by_arm(runs)
    missing = [
        f"{arm}/{q.id}" for arm in ("basic", "adaptive") for q in questions
        if q.id not in by_arm[arm]
    ] + [f"forced/{qid}" for qid in _FORCED_TECHNIQUES if qid not in by_arm["forced"]]

    errors = {
        arm: sorted(qid for qid, run in by_id_runs.items() if run.kind == "error")
        for arm, by_id_runs in by_arm.items()
    }
    errored = {(arm, qid) for arm, qids in errors.items() for qid in qids}

    basic_empty_retrieval = [
        q.id for q in questions
        if q.expect == "answerable"
        and ("basic", q.id) not in errored
        and not by_arm["basic"][q.id].chunk_ids
    ]

    any_ledger_failures = any(failures for failures in ledger_failures.values())

    return {
        "ok": not missing and not any_ledger_failures and not basic_empty_retrieval,
        "missing": missing,
        "ledger_failures": {k: v for k, v in ledger_failures.items() if v},
        "basic_empty_retrieval": basic_empty_retrieval,
        "errors": {arm: qids for arm, qids in errors.items() if qids},
    }


def run(output_path=RUNS_OUTPUT) -> bool:
    """Every arm, in ARMS order, one open ShippingHandle for the whole
    batch. client.unload_models() runs between arms rather than only at
    the very start, the same in-process reset generation.evaluate.py's
    own long multi-phase run already needed: dozens of sequential calls
    across three arms is a longer sustained load than any single earlier
    stage put on this machine's own Ollama server.

    output_path is written after every question, not only once per arm:
    three arms over the golden set is a longer sustained run than any
    earlier stage's own gate, on a machine LEARNING/router.md and
    LEARNING/generation.md both document real, recurring crashes under
    exactly this kind of sustained load, and this run reproduced a raw
    access violation (a segfault, not a catchable Python exception)
    repeatedly while being built, always inside a fresh BGE-M3 load
    colliding with the already-resident reranker worker, diagnosed with
    faulthandler. Pre-populating the disk vector cache for every
    query-transformation technique's own text (Rewriting, Multi-Query,
    Decomposition, HyDE, Self-Query) cleared most of it, but CRAG's own
    "incorrect -> rewrite -> re-query" retry path generates a genuinely
    new query text from an LLM call, one no amount of pre-warming can
    fully anticipate in advance, so the crash can still recur on whichever
    question happens to need a fresh embed under this machine's own
    currently tight memory (measured as low as 2.6 of 11.79 GB free
    during this build). That is router.md's own accepted operational
    reality: "a command-level retry stays the honest fallback." What
    changed here is the unit that retry has to repay for: run() now
    reads output_path back on startup, skips any (arm, question) pair
    already recorded, and writes again after every single question
    rather than every arm, so a fresh process restart after a crash
    replays nothing already answered and only pays, at most, for the one
    question that was mid-flight when it died.
    """
    questions, _ = load_golden(GOLDEN_SET)
    questions = _sorted_questions(questions)
    by_id = {q.id: q for q in questions}

    # kind="error" records are dropped from what counts as "already
    # done": an error is a failed call, not an answer, and Ollama's own
    # GPU allocator failing to start under transient memory pressure
    # (measured directly: a run of this file produced 13 of these in a
    # row, all "cudaMalloc failed: out of memory" from llama-server
    # itself, not the reranker/embedder crash this file's own earlier
    # docstring describes) is exactly the kind of failure a later retry,
    # with the machine's memory in a different state, can plausibly
    # clear. Treating an old error as done would make it permanent.
    existing = load_runs(output_path) if output_path.exists() else []
    succeeded = [r for r in existing if r.kind != "error"]
    done_by_arm = index_by_arm(succeeded)
    all_runs: list[ArmRun] = list(succeeded)
    all_ledger_failures: dict[str, dict[str, list[str]]] = {}

    # rerank.warm_up() is deliberately not called here any more.
    # rerank.py's own apply() now shuts the embedder worker down before
    # spawning its own, then tears itself down again immediately after
    # scoring (see embed_client.shutdown's own docstring for the
    # measured reason): pre-warming it here would just have it resident,
    # doing nothing, at the exact moment embed_client.warm_up() below
    # tries to bring the embedder up too, reproducing the very
    # two-workers-at-once memory peak this design exists to avoid.
    # embed_client.warm_up() alone is correct: the embedder is needed on
    # every question, reranking on only some, and rerank.py's own apply()
    # is what negotiates the handoff between the two from here on.
    embed_client.warm_up()

    with open_shipping() as handle:
        for arm in ARMS:
            population = (
                [by_id[qid] for qid in _FORCED_TECHNIQUES] if arm == "forced"
                else questions
            )
            already_done = done_by_arm.get(arm, {})
            print(
                f"--- arm: {arm} ({len(population)} questions, "
                f"{len(already_done)} already done) ---", flush=True,
            )
            other_arms_runs = [r for r in all_runs if r.arm != arm]

            def _save_this_arm(runs_so_far: list[ArmRun]) -> None:
                nonlocal all_runs
                all_runs = other_arms_runs + runs_so_far
                save_runs(all_runs, output_path)

            runs, ledger_failures = run_arm(
                arm, population, by_id, handle,
                already_done=already_done, on_progress=_save_this_arm,
            )
            all_runs = other_arms_runs + runs
            all_ledger_failures[arm] = ledger_failures
            save_runs(all_runs, output_path)
            print(f"written to {output_path} ({len(all_runs)} runs so far)", flush=True)
            client.unload_models()

    outcome = verify_harness(all_runs, questions, all_ledger_failures)
    print(f"{'ok' if outcome['ok'] else 'FAIL'}: {outcome}", flush=True)
    return outcome["ok"]


if __name__ == "__main__":
    run()
