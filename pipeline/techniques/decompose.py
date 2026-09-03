"""Decomposition: split a multi-part question into sub-questions, retrieve
each separately, fuse.

Position: called from techniques/run.py's answer() as one of the three
mutually exclusive retrieval strategies, taking priority over Multi-Query
and HyDE in run.py's own priority order when the router selects it.
Reuses retriever.retrieve_scored and retriever.fuse_scored exactly the
way multiquery.py does; the only real difference between the two files is
what the LLM is asked to produce, sub-questions covering distinct
information needs here, paraphrases of one need there.

Q5 is this technique's own case: two approval matrices on two different
pages, central_alarm p6 (9 rows) and p7 (5 rows), each with its own
header, re-grounded by advanced-rag-plan.md's finding C after the
page-span merger it was first written to test turned out not to apply
here. A single query embeds as one vector and tends to land near one
matrix's own wording; decomposing into "what approval does a branch
manager need for their own cameras" and "when is executive director
approval required" lets each sub-question's own retrieval find its own
page, which is exactly what the fused result then has to carry both of
into stage 9's context.

What this module does not do: it does not decide whether Decomposition
should run at all, and it does not check that the final generated answer
actually covers every sub-question. That coverage check belongs to stage
9, which reads this module's own trace to do it without re-deriving the
sub-questions from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import CANDIDATE_K, DECOMPOSE_MAX, GENERATOR_MODEL, RETRIEVAL_K
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk, ShippingHandle, fuse_scored, retrieve_scored

_SYSTEM_PROMPT = """\
Break the user's question into up to {n} distinct sub-questions, each \
covering one separate piece of information the original question needs. \
Only split when the question genuinely asks for more than one thing, a \
comparison between named things, or a case-by-case answer; if it already \
asks for exactly one thing, return that one thing as the only \
sub-question. Each sub-question must be a complete, self-contained \
Arabic question in its own right, answerable on its own without seeing \
the others. Do not answer any of them. Reply with JSON only: \
{{"sub_questions": ["...", "...", ...]}}, at most {n} entries.\
"""


@dataclass
class DecomposeTrace:
    """What analysis question 4 asks for: a complex question that
    benefited from decomposition, and what it split into.

        original                  the question as asked
        sub_questions              what it was split into, in order
        per_sub_question_top       each sub-question's own top 3 chunk
                                   ids, from its own retrieval alone,
                                   before fusion. This is the coverage
                                   record: stage 9 can check that the
                                   final fused, reranked context still
                                   carries at least one entry from each
                                   sub-question's own top few, without
                                   re-running any retrieval to find out
    """

    original: str
    sub_questions: tuple[str, ...]
    per_sub_question_top: dict[str, tuple[str, ...]]


def _generate_sub_questions(
    question: str, ledger: Ledger, n: int = DECOMPOSE_MAX,
) -> list[str]:
    """Up to n sub-questions, parsed defensively: a malformed or empty
    response degrades to treating the original question as its own only
    sub-question, so retrieval always has at least one query to run.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(n=n)},
        {"role": "user", "content": question},
    ]
    response = ledger.call(
        "Decomposition", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=400,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response.text)
        raw = parsed.get("sub_questions", [])
        if not isinstance(raw, list):
            raw = []
    except json.JSONDecodeError:
        raw = []

    sub_questions = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    return sub_questions[:n] or [question]


def apply(
    query_text: str,
    handle: ShippingHandle,
    ledger: Ledger,
    allowed_chunk_ids: set[str] | None = None,
    k: int = RETRIEVAL_K,
) -> tuple[list[ScoredChunk], DecomposeTrace]:
    """Retrieve under every sub-question separately, fused.

    k defaults to RETRIEVAL_K but run.py passes RERANK_TOP_N instead
    whenever Reranking will run afterward, the same reason
    multiquery.apply's own k parameter exists: reranking needs more than
    10 candidates to promote from.

    Every sub-question is retrieved with candidate_k candidates and
    apply_caps off, the same reasoning multiquery.apply gives: a
    diversity cap belongs on the final fused ranking, not on each
    sub-question's own ranking taken alone.
    """
    sub_questions = _generate_sub_questions(query_text, ledger)

    per_sub_question = [
        retrieve_scored(
            handle.context, sub_q, handle.decision["mode"],
            candidate_k=CANDIDATE_K, apply_caps=False,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        for sub_q in sub_questions
    ]

    fused = fuse_scored(per_sub_question, handle.context, k=k)
    if handle.decision.get("apply_caps", False):
        from ..retrieval.fusion import apply_diversity_caps
        kept = set(apply_diversity_caps(
            [item.chunk_id for item in fused], handle.context.chunk_lookup,
        ))
        fused = [item for item in fused if item.chunk_id in kept]

    trace = DecomposeTrace(
        original=query_text,
        sub_questions=tuple(sub_questions),
        per_sub_question_top={
            sub_q: tuple(item.chunk_id for item in scored[:3])
            for sub_q, scored in zip(sub_questions, per_sub_question)
        },
    )
    return fused, trace


# --- verification --------------------------------------------------------------

def verify_q5_spans_both_pages(handle: ShippingHandle) -> tuple[bool, DecomposeTrace]:
    """Q5's own gate, from the build order: the fused result cites both
    central_alarm p6 and p7, the two approval matrices golden.md's own
    re-grounding note (finding C) says this question actually exercises.
    """
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden

    questions, _ = load_golden(GOLDEN_SET)
    q5 = next(q for q in questions if q.id == "Q5")

    ledger = Ledger(label="verify-decompose-q5")
    scored, trace = apply(q5.question, handle, ledger)
    pages = {item.chunk.metadata.get("page") for item in scored
             if item.chunk.metadata.get("source", "").startswith("Central Alarm")}
    spans_both = 6 in pages and 7 in pages
    return spans_both, trace


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    with open_shipping() as shipping_handle:
        spans_both, trace = verify_q5_spans_both_pages(shipping_handle)

    lines = [
        f"Q5 fused result spans both central_alarm p6 and p7: {spans_both}",
        f"sub-questions: {list(trace.sub_questions)}",
        f"per-sub-question top 3: {trace.per_sub_question_top}",
    ]
    out_path = PROCESSED_DIR / "09_decompose_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{'ok' if spans_both else 'FAIL'}: written to {out_path}")
