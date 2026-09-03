"""HyDE: retrieve with a hypothetical formal passage instead of the
question itself.

Position: called from techniques/run.py's answer() as one of the three
mutually exclusive retrieval strategies, the fallback when neither
Decomposition nor Multi-Query outranks it in run.py's own priority
order. A single retrieval call, not a fusion across several: the
classic HyDE design generates one hypothetical document and embeds only
that, on the bet that a passage written in the corpus's own register sits
closer, in embedding space, to the corpus's real answer than a casually
worded question ever would.

Q6 is this technique's own case, "كيف يتأكد البنك أن كاميرات المراقبة
شغالة؟", colloquial-register wording asking the same thing the manual
states formally. Rewriting a vague follow-up (rewrite.py's own job) and
smoothing register (this one) are different problems: Q3 means nothing
without a prior turn, Q6 means something on its own, it is simply phrased
the way a person asks a question rather than the way a manual states a
fact.

What this module does not do: it does not decide whether HyDE should
run at all, and it does not fall back to the original question's own
retrieval if the hypothetical passage scores worse. A caller wanting
that comparison, evaluate.py's own ablation grid, runs both and compares
outside this function; conditionally discarding HyDE's own result here
would hide exactly the "when did HyDE add unnecessary cost" case
analysis question 5 asks the report to answer honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CANDIDATE_K, GENERATOR_MODEL, HYDE_MAX_TOKENS, RETRIEVAL_K
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk, ShippingHandle, retrieve_scored

_SYSTEM_PROMPT = """\
Write one short passage, in formal written Arabic, in the style of an \
internal bank operating-procedures manual, that would directly answer \
the user's question if it appeared verbatim in that manual. State it as \
a fact the manual documents, not as an answer to a question: do not \
mention the question, do not add a heading or a label, and do not \
invent any specific number, name, or detail beyond what the question \
itself already implies. Reply with the passage text only, no \
explanation, no quotation marks, no other text before or after it.\
"""


@dataclass
class HydeTrace:
    """What analysis question 5 asks for: when HyDE helped and when it
    added unnecessary cost. original and hypothetical side by side let a
    reader judge register and content match by eye; top_chunk_id is what
    the hypothetical passage actually found.
    """

    original: str
    hypothetical: str
    top_chunk_id: str


def _generate_hypothetical(question: str, ledger: Ledger) -> str:
    """One hypothetical passage, or the original question itself if the
    model returns nothing: an empty query string would retrieve
    meaninglessly, while the original question is always a safe
    fallback, the same discipline rewrite.apply uses.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = ledger.call(
        "HyDE", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=HYDE_MAX_TOKENS,
    )
    hypothetical = response.text.strip().strip('"').strip("“”").strip()
    return hypothetical or question


def apply(
    query_text: str,
    handle: ShippingHandle,
    ledger: Ledger,
    allowed_chunk_ids: set[str] | None = None,
    k: int = RETRIEVAL_K,
) -> tuple[list[ScoredChunk], HydeTrace]:
    """Generate a hypothetical passage and retrieve with it in place of
    the question. A single retrieve_scored call: HyDE replaces the query
    text, it does not add a second one to fuse against.

    k defaults to RETRIEVAL_K but run.py passes RERANK_TOP_N instead
    whenever Reranking will run afterward, the same reason
    multiquery.apply's own k parameter exists.
    """
    hypothetical = _generate_hypothetical(query_text, ledger)

    scored = retrieve_scored(
        handle.context, hypothetical, handle.decision["mode"],
        k=k, candidate_k=CANDIDATE_K,
        apply_caps=handle.decision.get("apply_caps", False),
        allowed_chunk_ids=allowed_chunk_ids,
    )

    trace = HydeTrace(
        original=query_text, hypothetical=hypothetical,
        top_chunk_id=scored[0].chunk_id if scored else "",
    )
    return scored, trace


# --- verification --------------------------------------------------------------

def verify_q6_improves(handle: ShippingHandle) -> tuple[int, int, HydeTrace]:
    """Q6's own gate, from the build order: its gold chunk's rank
    improves under HyDE relative to the unmodified question. Rank
    reported as position (1-indexed, 0 meaning not found in the
    candidate window) rather than Recall@10 alone, since Q6 carries a
    single gold chunk and a rank number is the more direct thing to read
    for a one-chunk question.
    """
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden

    questions, _ = load_golden(GOLDEN_SET)
    q6 = next(q for q in questions if q.id == "Q6")
    gold_id = q6.gold_chunk_ids[0]
    mode = handle.decision["mode"]
    caps = handle.decision.get("apply_caps", False)

    from ..retrieval.retriever import retrieve
    unmodified_ranked = retrieve(handle.context, q6.question, mode, apply_caps=caps)
    before_rank = (unmodified_ranked.index(gold_id) + 1) if gold_id in unmodified_ranked else 0

    ledger = Ledger(label="verify-hyde-q6")
    scored, trace = apply(q6.question, handle, ledger)
    scored_ids = [item.chunk_id for item in scored]
    after_rank = (scored_ids.index(gold_id) + 1) if gold_id in scored_ids else 0

    return before_rank, after_rank, trace


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    with open_shipping() as shipping_handle:
        before_rank, after_rank, trace = verify_q6_improves(shipping_handle)

    def _fmt(rank: int) -> str:
        return str(rank) if rank else "not found"

    lines = [
        f"Q6 gold chunk rank, unmodified: {_fmt(before_rank)}",
        f"Q6 gold chunk rank, HyDE:       {_fmt(after_rank)}",
        f"hypothetical passage: {trace.hypothetical}",
    ]
    out_path = PROCESSED_DIR / "10_hyde_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    improved = after_rank != 0 and (before_rank == 0 or after_rank < before_rank)
    print(f"{'ok, improved' if improved else 'FLAT/WORSE'}: "
          f"{_fmt(before_rank)} -> {_fmt(after_rank)}, written to {out_path}")
