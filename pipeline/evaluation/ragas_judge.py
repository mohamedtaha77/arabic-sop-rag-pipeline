"""The four required metrics (section 6), scored by a direct LLM rubric
against JUDGE_MODEL. Despite the module's own name, this file no longer
calls ragas at all; see "Why ragas was dropped" below for the measured
reason, kept in this file rather than deleted, since the attempt and its
failure are as real a finding as the fallback that replaced it.

Position: harness.py has already written every (arm, question) answer to
RUNS_OUTPUT. This file reads that back and produces the one thing it
does not carry, a judged score, over Context Relevance, Faithfulness,
Answer Relevance and Correctness, the same rubric applied identically to
every arm per section 6's own requirement.

Why ragas was dropped, measured rather than assumed. A first version of
this file built the four metrics from ragas 0.4.3's own classic metric
classes (Faithfulness, LLMContextPrecisionWithoutReference,
FactualCorrectness, SimpleCriteriaScore), each requiring the same
BaseRagasLLM adapter this file still uses. Run for real against
JUDGE_MODEL (llama3.2:3b-instruct-q4_K_M), the calibration probe alone
(a small, 37-case benchmark) took over an hour and produced constant
"OUTPUT_PARSING_FAILURE" errors from langchain's own structured-output
machinery, each one triggering langchain's own internal retry-and-fix
loop (a second, then a third LLM call trying to coerce the model's own
malformed JSON into the exact Pydantic schema ragas's claim-decomposition
step demands), and the real judging phase that follows it, 33 grounded
answers across three arms, never produced a single scored question in
that same hour. Two, not one, structural mismatches between this
project's judge model and ragas's own claim-decomposition design: this
model already measured, twice elsewhere in this project (crag.py's own
grading, entail.py's own bake-off), as unreliable at multi-step,
strictly-schema-typed reasoning, and ragas's own Faithfulness and
FactualCorrectness both first decompose an answer into a JSON array of
atomic statements before ever scoring anything, a harder structured-
output task than either of this project's own two existing 3B-model
judgement calls (crag.py's flat {"verdict", "reason"}, entail.py's flat
{"entailed": bool}) ever asks of it. This is exactly the fallback this
stage's own plan named in advance for this scenario: "If ragas's
structured calls fail or degenerate against JUDGE_MODEL, the fallback is
the same four metrics as a fixed local rubric through ledger.call, and
that failure is itself a finding worth reporting, not a setback to
hide." What follows is that fallback, built on the one structured-output
shape this project has already twice measured a 3B model can actually
produce reliably: one flat JSON object, a handful of named keys, no
nested decomposition step.

The four metrics, one call, one flat JSON object. All four scored 0 or 1
rather than a continuous scale: section 6 does not mandate a continuous
scale, and a binary judgement is the shape crag.py's own grading call
and entail.py's own bake-off both already measured this exact model
handling reliably, where a continuous, self-consistent score is not.
One call per (arm, question) rather than four, for the same reason
CRAG's own single grading call was chosen over one call per candidate
chunk: this project already measured, in generation/evaluate.py's own
_reset_ollama and this stage's own harness.py, that this machine's own
memory ceiling makes call count itself a real cost, and grading all four
metrics from one shared view of the question, context and answer is one
honest read of "did this answer do its job", not four independent ones
that could disagree with each other over the same evidence for no
principled reason.

What this file does not do: it does not decide what the four metrics
should be graded on, section 6 already does that; it does not run a
question, harness.py already has; and it does not compare arms against
each other, score.py does that from this file's own JUDGE_OUTPUT.
"""

from __future__ import annotations

import json

from ..config import GOLDEN_SET, JUDGE_LLM_MAX_TOKENS, JUDGE_MODEL, JUDGE_OUTPUT, JUDGE_TOP_K
from ..golden.question import Question
from ..llm.ledger import Ledger
from .record import ArmRun

_RUBRIC_SYSTEM_PROMPT = """\
You are grading one answer from a Retrieval-Augmented Generation system \
against four criteria, using only the material given below. Score each \
criterion 0 or 1, never a fraction.

context_relevance   1 if the retrieved passages are relevant to the \
question and contain evidence useful for answering it; 0 if they are \
off-topic or contain nothing useful for this question.
faithfulness         1 if every claim in the answer is supported by the \
retrieved passages, with no invented fact, number, date or name; 0 if \
the answer states something the passages do not support.
answer_relevance     1 if the answer directly and substantively \
addresses the user's actual question, without irrelevant content; 0 if \
it is off-topic, evasive, or answers a different question. An answer \
that correctly declines because the question is out of scope, or \
because the passages do not contain the answer, still scores 1 here: \
declining is exactly what that situation calls for.
correctness          1 if the answer's own factual content matches the \
reference answer where one is given; 0 if it contradicts the reference \
or is otherwise wrong. When no reference is given, score 1 if the \
answer is internally consistent with the retrieved passages, 0 \
otherwise.

Reply with JSON only, exactly these four keys, each an integer 0 or 1: \
{"context_relevance": 0 | 1, "faithfulness": 0 | 1, \
"answer_relevance": 0 | 1, "correctness": 0 | 1}\
"""


def _render_context(contexts: tuple[str, ...]) -> str:
    if not contexts:
        return "(no context was retrieved for this question)"
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))


def judge_run(
    run: ArmRun, question: Question, ledger: Ledger,
) -> dict[str, float | None]:
    """Score one grounded run on all four metrics from one call. A
    response this file cannot parse into the four expected keys
    degrades every metric to None for this question rather than a
    guessed 0 or 1: a missing score reads honestly in section 14's own
    table as "N/A", the same convention build_row already uses for a
    non-grounded answer, where a guessed number would silently misstate
    what the judge actually said.
    """
    user_content = (
        f"Question: {run.question}\n\n"
        f"Retrieved passages:\n{_render_context(run.contexts[:JUDGE_TOP_K])}\n\n"
        f"Answer given:\n{run.text}\n\n"
        f"Reference answer (may be empty if none exists): {question.answer}"
    )
    messages = [
        {"role": "system", "content": _RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        response = ledger.call(
            "Evaluation judge", messages, JUDGE_MODEL,
            temperature=0.0, max_tokens=JUDGE_LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.text)
    except Exception as error:  # noqa: BLE001
        print(f"judge/{run.arm}/{run.id}: FAILED {error}", flush=True)
        return {key: None for key in
                ("context_relevance", "faithfulness", "answer_relevance", "correctness")}

    scores: dict[str, float | None] = {}
    for key in ("context_relevance", "faithfulness", "answer_relevance", "correctness"):
        value = parsed.get(key)
        scores[key] = float(value) if isinstance(value, (int, float)) else None
    return scores


# --- calibration: is this judge worth reading? ---------------------------------

def calibration_probe(ledger: Ledger | None = None) -> dict:
    """Faithfulness alone, scored against entail.py's own labelled
    benchmark cases (17_entail_bakeoff.md), the same positive and
    negative_number pairs the entailment bake-off already built. Not
    the full 67 cases, and not routed through judge_run's own four-key
    prompt: this probe exists to sanity-check whether this judge's own
    faithfulness judgement, in isolation, can tell a real fabrication
    apart from the truth, before section 14's own faithfulness column
    is read as meaning anything, the same purpose the ragas-based
    version of this function had before the rubric it graded through
    changed underneath it.
    """
    from ..generation import entail  # local: entail.py's own heavy imports
    # (torch, transformers) only cost something when this probe actually runs

    cases = [
        case for case in entail.build_benchmark()
        if case.kind in ("positive", "negative_number")
    ]
    ledger = ledger or Ledger(label="judge-calibration")

    system_prompt = (
        "Judge whether the passage below supports the claim, using only "
        "what the passage itself says. Reply with JSON only: "
        '{"faithfulness": 0 | 1}, where 1 means the claim is fully '
        "supported by the passage and 0 means it is not (contradicted, "
        "unsupported, or the passage is about something else)."
    )

    predictions: list[bool] = []
    labels: list[bool] = []
    by_kind: dict[str, list[tuple[bool, bool]]] = {}
    for case in cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Passage: {case.premise}\n\nClaim: {case.hypothesis}"},
        ]
        try:
            response = ledger.call(
                "Evaluation judge", messages, JUDGE_MODEL,
                temperature=0.0, max_tokens=50,
                response_format={"type": "json_object"},
            )
            predicted = bool(json.loads(response.text).get("faithfulness"))
        except Exception as error:  # noqa: BLE001
            print(f"calibration/{case.question_id}: FAILED {error}", flush=True)
            continue
        predictions.append(predicted)
        labels.append(case.label)
        by_kind.setdefault(case.kind, []).append((predicted, case.label))

    def _recall(pairs: list[tuple[bool, bool]]) -> float | None:
        positives = [p for p, label in pairs if label]
        if not positives:
            return None
        return sum(positives) / len(positives)

    return {
        "n_cases": len(predictions),
        "recall_by_kind": {kind: _recall(pairs) for kind, pairs in by_kind.items()},
        "ledger": ledger.to_dict(),
    }


# --- entry point -----------------------------------------------------------------

def run(runs_path=None, output_path=JUDGE_OUTPUT) -> bool:
    from ..config import RUNS_OUTPUT
    from ..golden.question import load_golden
    from .record import load_runs

    runs_path = runs_path or RUNS_OUTPUT
    runs = load_runs(runs_path)
    questions, _ = load_golden(GOLDEN_SET)
    by_id = {q.id: q for q in questions}

    print("running the faithfulness calibration probe...", flush=True)
    calibration = calibration_probe()
    print(f"calibration: {calibration['recall_by_kind']}", flush=True)

    judged: list[dict] = []
    for run_record in runs:
        if run_record.kind != "grounded":
            judged.append({
                "arm": run_record.arm, "id": run_record.id,
                "scores": {
                    name: None for name in
                    ("context_relevance", "faithfulness", "answer_relevance", "correctness")
                },
                "skipped_reason": f"kind={run_record.kind!r}, nothing retrieved to grade",
                "ledger": None,
            })
            continue
        ledger = Ledger(label=f"judge-{run_record.arm}-{run_record.id}")
        scores = judge_run(run_record, by_id[run_record.id], ledger)
        judged.append({
            "arm": run_record.arm, "id": run_record.id, "scores": scores,
            "skipped_reason": None, "ledger": ledger.to_dict(),
        })
        print(f"judge/{run_record.arm}/{run_record.id}: {scores}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"calibration": calibration, "judged": judged}, indent=2),
        encoding="utf-8",
    )
    print(f"written to {output_path}", flush=True)

    graded = [j for j in judged if j["skipped_reason"] is None]
    all_scored = all(
        all(v is not None for v in j["scores"].values()) for j in graded
    )
    return bool(graded) and all_scored


if __name__ == "__main__":
    run()
