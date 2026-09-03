"""CRAG: grade retrieval, re-query once on a miss, refuse if that misses
too.

Position: called from techniques/run.py's answer() last, after
Reranking and Compression, on whatever RETRIEVAL_K chunks survived them,
either because the router selected CRAG or because run.py's own runtime
trigger fired, low retrieval confidence. This is the mechanism that
makes closed-domain refusal something enforced rather than something
merely instructed: Q10 is the corpus's own case for it, "ما هي إجراءات
إصدار بطاقات الائتمان للعملاء؟", where retrieval surfaces plausible,
on-topic, non-answering chunks about credit-file archiving rather than
credit-card issuance, and a retriever behaving correctly here still has
to be caught by something downstream, which is this file.

The paper's own CRAG falls back to a web search when it judges retrieval
incorrect. That option does not exist in a closed domain and would break
the whole scope rule anyway if it did, so the fallback here is
deliberately narrower: one corpus re-query with a rewritten query
(CRAG_MAX_REQUERIES, config.py), and refusal if that also fails to grade
as correct. This is the adaptation advanced-rag-plan.md names explicitly
and asks to be reported as a deliberate choice, not a missing feature.

What this module does not do: it does not decide whether CRAG should run
at all, and it does not write the refusal message a user sees; it
returns an empty context and refused=True, and stage 9 is what turns
that into an actual answer.

A measured limitation, recorded rather than hidden. This file's own
gate wants Q10 refused and none of the 18 answerable questions refused.
Three grading prompts were tried: the plain three-way rubric below, a
loosened version explicitly telling the model not to withhold "correct"
for partial or reworded matches, and a third version adding two
few-shot examples matching the real failure shapes. Q10 is caught by
all three, every time. The loosened and few-shot versions each made the
false-positive count worse, not better (9 and 16 of 18, against the
plain version's 8), the same overfitting risk retrieval.md's own
weighted-RRF note warns about, just in prompt space instead of weight
space. A parallel check ruled out a score-based alternative: Q10's own
top retrieval score, 0.628, sits inside the normal range of genuinely
answerable questions (Q7 scores 0.411, lower, and Q7 is answerable), so
a numeric threshold cannot separate them either, confirming
retrieval.md's own point that a retriever surfacing Q10's chunks is
behaving correctly and only semantic judgement can catch it. The plain
prompt below is what ships: it is the version that was actually
measured best, not the first one tried. A 3B local model's grading, on
this evidence, is well suited to catching a genuine miss (recall) and
poorly calibrated at confirming a genuine hit (precision), and that
asymmetry is a real property of this architecture's judge model choice,
worth the report saying plainly rather than smoothing over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import CRAG_MAX_REQUERIES, GENERATOR_MODEL, RETRIEVAL_K
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk, ShippingHandle, retrieve_scored

_GRADE_SYSTEM_PROMPT = """\
Judge whether the retrieved passages below actually answer the user's \
question, using only what the passages themselves say.

correct     at least one passage directly and substantially answers \
the question
ambiguous   the passages are on-topic and related but do not clearly \
answer the question, or only partially answer it
incorrect   the passages are off-topic, or the question needs \
information none of the passages contain

Reply with JSON only: {"verdict": "correct" | "ambiguous" | \
"incorrect", "reason": "short reason"}\
"""

_REQUERY_SYSTEM_PROMPT = """\
The first search for this question did not find a good answer in the \
documents. Rephrase the question with different wording, or a \
different angle, that might match how the source documents actually \
describe this, while still asking for exactly the same information. \
Reply with the rephrased question only, no explanation, no quotation \
marks.\
"""


@dataclass
class CragTrace:
    """What analysis question 9 asks for: a retrieval failure CRAG
    caught, and what decision the evaluator made.

        query               the original question graded
        verdict              the original grading: correct, ambiguous
                             or incorrect
        reason               the grader's own stated reason for that
                             verdict
        requeried            whether the one-shot fallback re-query ran
        requery_text         the rewritten query used for it, or "" if
                             it never ran
        requery_verdict       the re-query's own verdict, or "" if it
                             never ran
        refused               whether this ended in refusal: neither
                             the original nor the re-query (if it ran)
                             graded correct
        top_score             the best candidate's own score at the
                             moment CRAG first graded it, whatever it
                             was scored with upstream (dense cosine or
                             the reranker's own sigmoid score). Recorded
                             so the report can say whether a free
                             deterministic threshold on this number
                             would have made the same call the LLM
                             grader did. None when there were no
                             candidates to score at all
    """

    query: str
    verdict: str
    reason: str
    requeried: bool
    requery_text: str
    requery_verdict: str
    refused: bool
    top_score: float | None


def _grade(
    question: str, scored: list[ScoredChunk], ledger: Ledger,
) -> tuple[str, str]:
    """One verdict, from the model or from an empty candidate list
    directly. A verdict this function cannot parse degrades to
    "ambiguous", not "correct": that is the direction that still
    triggers the fallback rather than silently skipping CRAG's own job,
    the same "fail toward the safer behaviour" rule router.py's own
    parse-failure handling already follows.
    """
    if not scored:
        return "incorrect", "no candidates were retrieved at all"

    context_text = "\n\n".join(
        f"[{i}] {item.chunk.text}"
        for i, item in enumerate(scored[:RETRIEVAL_K], start=1)
    )
    messages = [
        {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nRetrieved passages:\n{context_text}"},
    ]
    response = ledger.call(
        "CRAG evaluator", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=500,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response.text)
        verdict = parsed.get("verdict")
        if verdict not in ("correct", "ambiguous", "incorrect"):
            verdict = "ambiguous"
        reason = str(parsed.get("reason", ""))
    except json.JSONDecodeError:
        verdict, reason = "ambiguous", "grader response unparseable"
    return verdict, reason


def _generate_requery(question: str, ledger: Ledger) -> str:
    messages = [
        {"role": "system", "content": _REQUERY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = ledger.call(
        "CRAG evaluator", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=150,
    )
    rephrased = response.text.strip().strip('"').strip("“”").strip()
    return rephrased or question


def apply(
    query_text: str,
    scored: list[ScoredChunk],
    handle: ShippingHandle,
    ledger: Ledger,
) -> tuple[list[ScoredChunk], bool, CragTrace]:
    """Grade the retrieved context; on anything short of "correct", one
    corpus re-query with a rewritten query (CRAG_MAX_REQUERIES attempts,
    currently one), and refusal if that also falls short.
    """
    top_score = scored[0].score if scored else None

    current_query = query_text
    current_scored = scored
    verdict, reason = _grade(current_query, current_scored, ledger)
    first_verdict, first_reason = verdict, reason

    requeried = False
    requery_text = ""
    attempts = 0
    while verdict != "correct" and attempts < CRAG_MAX_REQUERIES:
        attempts += 1
        requeried = True
        current_query = _generate_requery(current_query, ledger)
        requery_text = current_query
        current_scored = retrieve_scored(
            handle.context, current_query, handle.decision["mode"],
            apply_caps=handle.decision.get("apply_caps", False),
        )
        verdict, reason = _grade(current_query, current_scored, ledger)

    refused = verdict != "correct"
    trace = CragTrace(
        query=query_text, verdict=first_verdict, reason=first_reason,
        requeried=requeried, requery_text=requery_text,
        requery_verdict=(verdict if requeried else ""),
        refused=refused, top_score=top_score,
    )

    if refused:
        return [], True, trace
    return current_scored, False, trace


# --- verification --------------------------------------------------------------

def verify_q10_incorrect_others_correct(handle: ShippingHandle) -> dict:
    """The build order's own gate: Q10 ends in refusal, none of the 18
    answerable questions do. Both halves matter equally: a CRAG that
    refuses everything would trivially pass the Q10 half while failing
    every real question, and this checks both rather than either alone.

    Retrieval here is the plain shipping configuration at RETRIEVAL_K,
    not the wider RERANK_TOP_N Reranking would use: this gate is about
    CRAG's own grading and fallback logic, not about reproducing the
    full technique stack, the same scope every other technique file's
    own verify function keeps to.
    """
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden

    questions, _ = load_golden(GOLDEN_SET)
    answerable = [q for q in questions if q.expect == "answerable"]
    q10 = next(q for q in questions if q.id == "Q10")

    results = {}
    for question in answerable + [q10]:
        scored = retrieve_scored(
            handle.context, question.question, handle.decision["mode"],
            apply_caps=handle.decision.get("apply_caps", False),
        )
        ledger = Ledger(label=f"verify-crag-{question.id}")
        _result_scored, refused, trace = apply(question.question, scored, handle, ledger)
        results[question.id] = {
            "refused": refused, "verdict": trace.verdict, "reason": trace.reason,
            "requeried": trace.requeried, "requery_verdict": trace.requery_verdict,
            "top_score": trace.top_score,
        }

    q10_correctly_flagged = results["Q10"]["refused"]
    others_all_pass = all(not results[q.id]["refused"] for q in answerable)
    return {
        "q10_correctly_flagged": q10_correctly_flagged,
        "others_all_pass": others_all_pass,
        "results": results,
    }


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    with open_shipping() as shipping_handle:
        outcome = verify_q10_incorrect_others_correct(shipping_handle)

    lines = [
        f"Q10 correctly refused: {outcome['q10_correctly_flagged']}",
        f"all 18 answerable questions pass (not refused): {outcome['others_all_pass']}",
        "",
    ]
    false_positives = [
        qid for qid, row in outcome["results"].items()
        if qid != "Q10" and row["refused"]
    ]
    if false_positives:
        lines.append(f"false positives (wrongly refused): {false_positives}")
        lines.append("")
    for qid, row in outcome["results"].items():
        lines.append(
            f"{qid}: refused={row['refused']} verdict={row['verdict']} "
            f"requeried={row['requeried']} requery_verdict={row['requery_verdict']!r} "
            f"reason={row['reason']}"
        )

    out_path = PROCESSED_DIR / "14_crag_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    ok = outcome["q10_correctly_flagged"] and outcome["others_all_pass"]
    print(f"{'ok' if ok else 'FAIL'}: written to {out_path}")
