"""The generation report, and the two real decisions this stage makes:
the CRAG confidence threshold, and which entailment backend ships.

Position: every file before this one built and gated one mechanism. This
is the first that writes a report a person reads, in the manner of
stage 8's own techniques/evaluate.py, and the only file that writes
GENERATION_DECISION, which generation/run.py reads back rather than
assuming.

Two decisions, not one, because two constants in this codebase were
always stated placeholders waiting on exactly this stage. entail.py's
own bake-off already measures the entailment backend; this file is what
reads that measurement back and folds it into one combined decision file
alongside the second placeholder techniques/run.py's own
PROVISIONAL_CRAG_THRESHOLD names directly: "crag.py (file 14) and
evaluate.py (file 15) are what get to set this for real." That comment
was written before stage 9's own file numbering existed; this is that
file, for this stage.

The CRAG threshold cannot do what a first reading of "measure a
threshold" suggests, and router.md's own docstring already proved why
before this file was written: Q10's own top retrieval score sits inside
the normal range of genuinely answerable questions, "the score
genuinely carries no separating signal here." Repeating that search at
the reranked score scale this file actually measures against would only
rediscover the same negative result with different numbers. What this
file measures instead is the minimum sensible threshold: the smallest
value that guarantees Q10's own reranked top score counts as low
confidence, so CRAG's own semantic grading, the mechanism actually shown
to catch Q10 reliably, gets invoked for it, while staying as low as
possible otherwise, since routing an answerable question through CRAG's
own imperfect grading is a real, measured risk (crag.py's own 8 of 18
false positives), not a free safety net.

What this module does not do: it does not decide whether any technique
or backend should exist. Both entailment backends stay built and
independently reachable through entail.check regardless of which one
this file measures as the default, the same discipline stage 7's and
stage 8's own evaluate.py files hold hybrid retrieval and Reranking to.
"""

from __future__ import annotations

import json
import urllib.request

from ..config import (
    GENERATION_DECISION,
    GENERATION_OUTPUT,
    GENERATOR_MODEL,
    GOLDEN_SET,
    JUDGE_MODEL,
    LLM_BASE_URL,
)
from ..golden.question import load_golden
from ..llm.ledger import Ledger
from ..retrieval.retriever import ShippingHandle, open_shipping
from ..techniques import rerank
# run aliased to generation_run: this file's own entry point below is
# also named run(), a plain top-level function definition that executes
# at module-load time and would otherwise silently rebind the name run
# away from the sibling module this import means, from the point that
# definition runs onward. Measured directly: _generation_report's own
# call to run.verify_four_paths(handle) resolved to this file's own
# run() function object instead, "'function' object has no attribute
# 'verify_four_paths'", before this alias was added.
from . import entail, guard, present, synthesise
from . import run as generation_run

# Technique-independent golden questions, the same eight synthesise.py's
# and present.py's own gates settled on and for the same reason: Q4 and
# Q19 need a query-transformation technique a standalone gate does not
# run, and would test that instability again rather than the mechanism
# each gate actually exists to check.
_STANDALONE_QUESTION_IDS = ("Q1", "Q2", "Q8", "Q11", "Q13", "Q16", "Q18", "Q20")

# The threshold itself is the midpoint between Q10's own score and the
# nearest score above it among the other seven questions measured, which
# needs no fitted constant. This is the fallback only: if some future
# re-measurement ever finds every other question scoring below Q10 (no
# "nearest score above" to take a midpoint with at all), this margin is
# what still guarantees the threshold clears Q10's own score rather than
# the function raising on a `min()` of an empty sequence.
_CRAG_THRESHOLD_MARGIN = 0.01


def _reset_ollama() -> None:
    """Force-unload whichever model Ollama is currently serving.

    Added after this file's own full run crashed three times in a row
    at different points, one a raw segfault with no output at all, one
    a genuine `cudaMalloc failed: out of memory` from inside Ollama's
    own llama-server process partway through the CRAG threshold
    measurement's own eight sequential questions. LEARNING/router.md
    already documents this machine's own established mitigation for a
    crash under sustained model-loading load, stop the resident model
    and retry as a fresh process; a full Python process restart is not
    available mid-function, so this calls the same unload Ollama's own
    API exposes (`keep_alive: 0`) directly, giving the server a clean
    slate between this file's own heaviest phases rather than letting
    state accumulate across dozens of sequential calls in one run.
    Best-effort: a failed reset is not this function's problem to raise
    on, since the worst case is simply that the next call pays a full
    reload instead of reusing a warm model, not a correctness issue.
    """
    # LLM_BASE_URL is the OpenAI-compatible path, ".../v1"; Ollama's own
    # native unload endpoint sits at the root, ".../api/generate", not
    # under "/v1" at all, so the "/v1" suffix is stripped rather than
    # navigated past with a literal ".." segment, which an HTTP request
    # does not resolve the way a filesystem path would.
    root_url = LLM_BASE_URL.removesuffix("/v1")
    for model in (GENERATOR_MODEL, JUDGE_MODEL):
        try:
            request = urllib.request.Request(
                f"{root_url}/api/generate",
                data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:  # noqa: BLE001
            pass


def measure_crag_threshold(handle: ShippingHandle) -> dict:
    """The reranked top score CRAG's own runtime trigger would actually
    see, for every question the trigger can ever fire on, and only
    those: `techniques/run.py`'s own `triggers_active` requires
    `decision.route == "advanced_rag"`, and `_resolve_technique_set`
    returns `TechniqueSet.none()` immediately for any other route,
    before Reranking is ever added. A `basic_rag` question never reaches
    Reranking or CRAG at all, so measuring a threshold against one would
    be scoring a code path that will never run for it; a first version
    of this function did exactly that, across all 18 answerable
    questions, and was corrected here before this file shipped. The
    real population is the golden set's own seven advanced_rag
    answerable questions (Q3 to Q7, Q19, Q20) plus Q10.

    Each question runs through the real `techniques.answer.answer`, not
    a direct retrieve-then-rerank call, and for the same reason: Q3 needs
    Rewriting and Q19 needs Decomposition to retrieve anything like what
    the real pipeline would score them on, and a raw, untransformed
    retrieval call measured 0.06 and 0.01 for them respectively in this
    function's own first, since-corrected version, an artifact of
    skipping the technique they are the golden set's own named case for,
    not a genuine confidence reading. Q3's own dependency on Q2 is
    threaded as history using Q2's golden reference answer, the only
    reproducible choice available to a measurement that runs on its own
    rather than as part of an ordered harness.

    `crag_threshold=0.0` is passed to force CRAG's own runtime trigger
    to never fire during this measurement: `_low_confidence` requires
    `top.score < threshold`, impossible once reranked scores are
    non-negative, so `QuestionRun.retrieved` always survives with the
    real, technique-applied, reranked ranking this function reads
    `[0].score` from, rather than being emptied by CRAG refusing before
    this function ever sees the score that would have triggered it.
    """
    from ..techniques.run import answer as techniques_answer

    questions, _ = load_golden(GOLDEN_SET)
    by_id = {q.id: q for q in questions}
    targets = [
        by_id[qid] for qid in ("Q3", "Q4", "Q5", "Q6", "Q7", "Q19", "Q20", "Q10")
    ]

    scores: dict[str, float] = {}
    for question in targets:
        history = (
            [(by_id[question.depends_on].question, by_id[question.depends_on].answer)]
            if question.depends_on else None
        )
        ledger = Ledger(label=f"crag-threshold-{question.id}")
        run_result = techniques_answer(
            question.question, ledger, handle, history=history, crag_threshold=0.0,
        )
        scores[question.id] = run_result.retrieved[0].score if run_result.retrieved else 0.0

    q10_score = scores["Q10"]
    next_above = min(
        (score for qid, score in scores.items() if qid != "Q10" and score > q10_score),
        default=q10_score + 2 * _CRAG_THRESHOLD_MARGIN,
    )
    threshold = (q10_score + next_above) / 2
    also_triggered = sorted(
        qid for qid, score in scores.items()
        if qid != "Q10" and score < threshold
    )
    return {
        "scores": scores, "q10_score": q10_score, "threshold": threshold,
        "also_triggered": also_triggered,
    }


def _entailment_bakeoff_section() -> tuple[list[str], dict]:
    """The entailment bake-off's own report section, run and returned
    before anything in this file ever opens a ShippingHandle. entail.py's
    own build_benchmark reads chunk JSON directly and needs no store, no
    embedder and no reranker at all, so this has no reason to run inside
    the handle-scoped section below, and a real, measured reason not to:
    see entail.warm_up's own docstring for the segfault this exact
    ordering was built to avoid, three model-holding processes alive at
    once rather than the two every earlier crash in this project was
    about.
    """
    lines = ["## Grounding guard: entailment bake-off", ""]
    bakeoff = entail.run_bakeoff()
    lines.append(
        f"benchmark: {bakeoff['n_cases']} cases ({bakeoff['n_positive']} positive, "
        f"{bakeoff['n_negative_topic']} negative/topic, "
        f"{bakeoff['n_negative_number']} negative/number)"
    )
    lines.append(
        f"nli ({bakeoff['nli_model']}): {bakeoff['nli']['best_balanced_accuracy']:.3f} "
        f"balanced accuracy at threshold {bakeoff['nli']['best_threshold']}, "
        f"recall by kind {bakeoff['nli']['accuracy_by_kind']}"
    )
    lines.append(
        f"llm ({bakeoff['llm_model']}): {bakeoff['llm']['best_balanced_accuracy']:.3f} "
        f"balanced accuracy, recall by kind {bakeoff['llm']['accuracy_by_kind']}"
    )
    lines.append(
        f"winner: {bakeoff['winner_backend']}"
        + (f" (tie-break on negative_number recall, aggregate gap "
           f"{bakeoff['aggregate_gap']:.3f} within margin)"
           if bakeoff["tie_break_applied"] else " (higher aggregate)")
    )
    lines.append(
        "Recorded honestly rather than smoothed over: the winning llm "
        "backend's own recall on genuine positives is only "
        f"{bakeoff['llm']['accuracy_by_kind'].get('positive', 0):.3f}, which is "
        "exactly why generation/run.py treats entailment as a reported "
        "signal and never as a whole-answer refusal trigger; see that "
        "file's own module docstring for the real question this finding "
        "changed."
    )
    lines.append("")
    return lines, bakeoff


def _generation_report(
    handle: ShippingHandle, bakeoff_lines: list[str], bakeoff: dict,
) -> tuple[list[str], dict]:
    lines = ["# Two-stage generation", ""]
    raw: dict = {"entail": bakeoff}

    lines += ["## CRAG confidence threshold", ""]
    print("measuring the CRAG threshold...", flush=True)
    crag_outcome = measure_crag_threshold(handle)
    raw["crag_threshold"] = crag_outcome["threshold"]
    lines.append(f"scores, only over the population the trigger can ever "
                 f"fire on (advanced_rag answerable questions plus Q10), "
                 f"each through the real technique-applying pipeline: "
                 f"{crag_outcome['scores']}")
    lines.append(f"Q10's own reranked top score: {crag_outcome['q10_score']:.4f}")
    lines.append(f"threshold (midpoint between Q10's score and the nearest "
                 f"score above it): {crag_outcome['threshold']:.4f}")
    lines.append(
        f"other questions in this population also falling below this "
        f"threshold, and so also routed through CRAG's own grading: "
        f"{crag_outcome['also_triggered']}"
    )
    lines.append(
        "No threshold separates Q10 from every other question with a "
        "real margin on both sides; this is the midpoint of whatever gap "
        "actually exists between Q10's own score and the nearest "
        "genuinely answerable question's score above it, not a claim "
        "that a clean separation exists across the whole set."
    )
    lines.append("")

    # A reset here, not just at process start: this file's own run
    # crashed twice inside the CRAG threshold measurement above,
    # ``cudaMalloc failed: out of memory`` once, partway through its
    # eight sequential questions, Ollama's own state degrading under
    # sustained sequential load rather than anything this process holds.
    # See _reset_ollama's own docstring for the measured reason.
    _reset_ollama()

    lines += ["## Synthesiser", ""]
    print("running the synthesiser gate...", flush=True)
    synth_results = synthesise.verify_grounded_and_cited(handle, _STANDALONE_QUESTION_IDS)
    all_cited = all(r["citations"] and not r["invalid_markers"] for r in synth_results)
    all_substantive = all(
        r["content_words"] >= synthesise.MIN_CONTENT_WORDS for r in synth_results
    )
    no_leaks = all(not r["chunk_id_leak"] for r in synth_results)
    retries_used = sum(1 for r in synth_results if r["repair_retry_used"])
    still_broken = [
        r["id"] for r in synth_results
        if r["repair_retry_used"] and not r["repair_retry_recovered"]
    ]
    raw["synthesise"] = {
        "all_cited": all_cited, "all_substantive": all_substantive,
        "no_leaks": no_leaks, "still_broken": still_broken,
    }
    lines.append(f"every answer cites at least one real chunk: {all_cited}")
    lines.append(f"every answer has real content, not a stub: {all_substantive}")
    lines.append(f"no answer leaks a raw chunk id: {no_leaks}")
    lines.append(f"repair retry fired on {retries_used} of {len(synth_results)} questions")
    if still_broken:
        lines.append(
            f"still broken after the retry, a measured residual rather "
            f"than a hidden one: {still_broken}"
        )
    lines.append("")

    lines += ["## Grounding guard: token diff", ""]
    print("running the guard gate...", flush=True)
    guard_outcome = guard.verify_rejects_fabrication_not_genuine(
        synth_results[0]["text"] if synth_results else "",
    )
    raw["guard"] = guard_outcome
    lines.append(f"fabricated content rejected: {guard_outcome['fabricated_rejected']}")
    lines.append(f"genuine reformatting accepted: {guard_outcome['genuine_accepted']}")
    lines.append("")

    lines += bakeoff_lines

    lines += ["## Presenter block rate", ""]
    print("running the presenter gate...", flush=True)
    present_outcome = present.verify_block_rate(handle, _STANDALONE_QUESTION_IDS)
    raw["present"] = {"block_rate": present_outcome["block_rate"]}
    lines.append(
        f"block rate: {present_outcome['block_rate']:.3f} "
        f"({sum(1 for r in present_outcome['results'] if not r['passed'])} "
        f"of {len(present_outcome['results'])} blocked)"
    )
    lines.append(
        "Not a defect count to minimise: a presenter that never tries to "
        "add anything would report zero, and would be indistinguishable "
        "from one this guard was never actually tested against. This "
        "number is the evidence the guard does something, the number "
        "this whole two-stage design exists to produce."
    )
    lines.append("")

    lines += ["## End-to-end wiring", ""]
    print("running the end-to-end wiring gate...", flush=True)
    _reset_ollama()
    run_outcome = generation_run.verify_four_paths(handle)
    raw["run"] = {"ok": run_outcome["ok"]}
    lines.append(f"all four paths (direct, both refusal kinds, grounded) correct: "
                 f"{run_outcome['ok']}")
    lines.append(f"ledger.verify() clean on all four: {run_outcome['all_ledgers_clean']}")
    lines.append("")

    return lines, raw


def _generation_decision(raw: dict) -> dict:
    return {
        "crag_threshold": raw["crag_threshold"],
        "guard_backend": raw["entail"]["winner_backend"],
        "entailment_threshold": raw["entail"]["winner_threshold"],
        "presenter_block_rate": raw["present"]["block_rate"],
        "basis": (
            f"crag_threshold from the midpoint between Q10's own reranked "
            f"top score and the nearest advanced_rag answerable question's "
            f"score above it; guard_backend from entail.py's own bake-off "
            f"({raw['entail']['n_cases']} cases, aggregate gap "
            f"{raw['entail']['aggregate_gap']:.3f}); presenter_block_rate "
            f"measured over {len(_STANDALONE_QUESTION_IDS)} standalone questions"
        ),
    }


def run() -> bool:
    # The entailment bake-off runs, and its own NLI worker shuts down,
    # entirely before BGE-M3 or the reranker worker ever load: see
    # entail.warm_up's own docstring for the segfault measured when this
    # file's first version ran the bake-off with both already resident,
    # three model-holding processes alive at once. print() here so a
    # long run shows real progress rather than going silent until the
    # very end.
    entail.warm_up()
    print("running the entailment bake-off...", flush=True)
    bakeoff_lines, bakeoff = _entailment_bakeoff_section()
    entail.shutdown()

    # warm_up() before open_shipping(): the CRAG threshold measurement
    # and the presenter/run gates below all reach Reranking, and
    # rerank.warm_up's own docstring has the measured reason this order
    # is load-bearing rather than a style preference.
    rerank.warm_up()
    with open_shipping() as handle:
        lines, raw = _generation_report(handle, bakeoff_lines, bakeoff)

    GENERATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GENERATION_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written to {GENERATION_OUTPUT}", flush=True)

    decision = _generation_decision(raw)
    GENERATION_DECISION.parent.mkdir(parents=True, exist_ok=True)
    GENERATION_DECISION.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"written to {GENERATION_DECISION}: {decision}", flush=True)

    return True


if __name__ == "__main__":
    run()
