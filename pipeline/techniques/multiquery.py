"""Multi-Query: paraphrase the question several ways, retrieve each, fuse.

Position: called from techniques/run.py's answer() as one of the three
mutually exclusive retrieval strategies, when the router selects
Multi-Query and neither Decomposition nor HyDE outranks it in run.py's
own priority order. Reuses retriever.retrieve_scored for every paraphrase
and retriever.fuse_scored to combine them; nothing here reopens a Qdrant
client, rebuilds a BM25 index, or reimplements RRF a second time.

The idea this technique bets on: a single query text can miss the exact
wording the corpus itself uses, even when the underlying information need
is unambiguous. Q4, "ما هي الأعطال والمشاكل التي قد تصيب أنظمة المراقبة
التلفزيونية؟", is this shape: "problems" or "faults" is a broad,
open-ended noun, and the manual may enumerate specific fault types under
different words than whichever ones a single embedding happens to land
near. Several honestly different phrasings, fused, is what covers that
without deciding in advance which wording will win.

What this module does not do: it does not decide whether Multi-Query
should run at all. That is the router's call, or run.py's technique_set
override for an evaluate.py grid cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..config import CANDIDATE_K, GENERATOR_MODEL, MULTIQUERY_N, RETRIEVAL_K
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk, ShippingHandle, fuse_scored, retrieve_scored

_SYSTEM_PROMPT = """\
Generate {n} different Arabic search queries that each try to find the \
same information as the user's question, using genuinely different \
wording, phrasing, or vocabulary a document might use to say the same \
thing. Each query must be a complete, well-formed Arabic question or \
search phrase, meaningfully different from the others in wording, not \
merely one word changed. Do not answer the question, and do not narrow \
or widen what it is asking for. Reply with JSON only: \
{{"queries": ["...", "...", ...]}}, with exactly {n} entries.\
"""


@dataclass
class MultiQueryTrace:
    """What analysis question 3 asks for: an example where Multi-Query
    beat a single query, and why. paraphrases is what actually got
    searched, in the order generated; per_query_top is each paraphrase's
    own top hit, so a reader can see which phrasing found what without
    re-running anything.
    """

    original: str
    paraphrases: tuple[str, ...]
    per_query_top: dict[str, str]


def _generate_paraphrases(
    question: str, ledger: Ledger, n: int = MULTIQUERY_N,
) -> list[str]:
    """Up to n paraphrases, parsed defensively: a malformed or short
    response degrades to fewer paraphrases rather than raising, since a
    Multi-Query that ran with one fewer variant than requested is still
    a Multi-Query, while one that crashed the whole question is not.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(n=n)},
        {"role": "user", "content": question},
    ]
    response = ledger.call(
        "Multi-Query", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=400,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response.text)
        raw = parsed.get("queries", [])
        if not isinstance(raw, list):
            raw = []
    except json.JSONDecodeError:
        raw = []

    paraphrases = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
    return paraphrases[:n]


def apply(
    query_text: str,
    handle: ShippingHandle,
    ledger: Ledger,
    allowed_chunk_ids: set[str] | None = None,
    k: int = RETRIEVAL_K,
) -> tuple[list[ScoredChunk], MultiQueryTrace]:
    """Retrieve under the original text and every paraphrase, fused.

    k defaults to RETRIEVAL_K but run.py passes RERANK_TOP_N instead
    whenever Reranking will run afterward: reranking needs more than 10
    candidates to promote from, or it could only ever reorder an
    already-narrow top 10.

    Every query, the original included, is retrieved with candidate_k
    candidates and apply_caps deliberately off; a diversity cap belongs
    on the final fused ranking, not on each paraphrase's own ranking
    individually, the same reason retrieve_scored applies caps after
    fusing its own legs rather than before.
    """
    paraphrases = _generate_paraphrases(query_text, ledger)
    queries = [query_text] + [p for p in paraphrases if p != query_text]

    per_query = [
        retrieve_scored(
            handle.context, q, handle.decision["mode"],
            candidate_k=CANDIDATE_K, apply_caps=False,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        for q in queries
    ]

    fused = fuse_scored(per_query, handle.context, k=k)
    if handle.decision.get("apply_caps", False):
        from ..retrieval.fusion import apply_diversity_caps
        kept = set(apply_diversity_caps(
            [item.chunk_id for item in fused], handle.context.chunk_lookup,
        ))
        fused = [item for item in fused if item.chunk_id in kept]

    trace = MultiQueryTrace(
        original=query_text,
        paraphrases=tuple(queries),
        per_query_top={
            q: (scored[0].chunk_id if scored else "")
            for q, scored in zip(queries, per_query)
        },
    )
    return fused, trace


# --- verification --------------------------------------------------------------

def verify_q4_covers_both_gold(handle: ShippingHandle) -> tuple[bool, MultiQueryTrace]:
    """Q4's own gate, from the build order: both gold chunks retrieved.
    Q4 has two gold chunks; a single dense query already found in stage
    6 and 7's own grids tends to favour whichever one wording happens to
    sit closer to, so recovering both is the concrete thing this
    technique has to show on its own named question.
    """
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden

    questions, _ = load_golden(GOLDEN_SET)
    q4 = next(q for q in questions if q.id == "Q4")
    gold = set(q4.gold_chunk_ids)

    ledger = Ledger(label="verify-multiquery-q4")
    scored, trace = apply(q4.question, handle, ledger)
    found = {item.chunk_id for item in scored}
    return gold <= found, trace


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    with open_shipping() as shipping_handle:
        covers_both, trace = verify_q4_covers_both_gold(shipping_handle)

    lines = [
        f"Q4 both gold chunks retrieved: {covers_both}",
        f"paraphrases: {list(trace.paraphrases)}",
        f"per-query top hit: {trace.per_query_top}",
    ]
    out_path = PROCESSED_DIR / "08_multiquery_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{'ok' if covers_both else 'FAIL'}: written to {out_path}")
