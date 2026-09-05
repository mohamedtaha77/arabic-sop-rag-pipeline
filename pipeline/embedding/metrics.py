"""Retrieval metrics, the bootstrap tie-break, and the report they feed.

Pure numpy, nothing that touches a model or a GPU. That separation is load
bearing, not tidiness: run.py, the orchestrator that spawns one subprocess
per model, imports this module rather than bakeoff.py precisely so the
parent process spawning those subprocesses never has torch or transformers
resident itself. Measured on this machine: a parent that had already
imported them left noticeably less headroom for the child it was about to
spawn, on a card and a host both close to their limits already.

What this module does not do: it does not call a model, and it does not
know how a chunk gets turned into a vector. bakeoff.py's evaluate() does
that and hands this module plain floats and chunk ids.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from ..config import PREFERRED_MODEL, RETRIEVAL_K

# --- retrieval metrics, defined here rather than assumed ---------------------
#
# Recall@k is ambiguous the moment a question can have more than one gold
# chunk, and 10 of this golden set's 18 answerable questions do. Two
# definitions answer different questions, so both are kept:
#
#   Recall@k         mean, over questions, of the *fraction* of a question's
#                     gold chunks that land in the top k. A question with 2
#                     of 3 gold chunks retrieved scores 0.67, not 0 and not 1.
#   StrictRecall@k    the share of questions where *every* gold chunk lands
#                     in the top k. Harsher, and the one a user actually
#                     feels: a partially retrieved answer can still read as
#                     confidently incomplete, which is exactly what golden.md
#                     flagged Q5 for.
#
# MRR@k uses the first gold chunk found, and nDCG@k gives partial credit for
# rank order among several gold chunks, binary relevance since nothing in
# this golden set grades one gold chunk as more central than another.
#
# Precision@k, added for stage 10 (advanced-rag-plan.md names it alongside
# Recall, MRR and nDCG as a required retrieval metric, and no earlier stage
# needed it: 6 and 7's own grids settled their decisions on Recall@10 and
# MRR@10 alone), divides by however many chunks actually came back rather
# than by k unconditionally: a question whose retriever only ever returns 3
# chunks should not be capped at 0.3 precision for a reason that has
# nothing to do with what those 3 chunks were.


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def precision_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return len(set(top) & gold_ids) / len(top)


def strict_recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = set(ranked_ids[:k])
    return 1.0 if gold_ids and gold_ids <= top else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = ranked_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top, start=1)
        if chunk_id in gold_ids
    )
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def rank_by_similarity(
    query_vector: np.ndarray, passage_vectors: np.ndarray, chunk_ids: list[str],
) -> list[str]:
    """Chunk ids in descending cosine order.

    Vectors from embedder.py are unit norm, so a dot product is a cosine.
    Exported for store.qdrant's gate 6 to import directly: the store's search
    is verified against exactly this function, not against a second
    implementation of the same idea that could drift from it.
    """
    scores = passage_vectors @ query_vector
    order = np.argsort(-scores)
    return [chunk_ids[i] for i in order]


METRIC_KEYS = ("recall@10", "strict_recall@10", "mrr@10", "ndcg@10", "recall@5")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: statistics.mean(r[key] for r in rows) for key in METRIC_KEYS}


# --- the tie-break, declared before any number from this run is read ---------

def paired_bootstrap_ci(
    a: list[float], b: list[float], n_resamples: int = 10_000,
    seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """95% CI on mean(a) - mean(b), paired by question.

    Standard IR practice for exactly this situation: 18 questions means one
    question is worth 0.056 of Recall@10, so a hand-picked margin is an
    invented threshold and a bootstrap is not. Resamples question indices
    with replacement, computes the mean paired difference each time, and
    reads the 2.5th and 97.5th percentiles off the resampled distribution.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    diffs = a_arr - b_arr
    n = len(diffs)
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = diffs[resample_idx].mean(axis=1)
    low, high = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high), float(diffs.mean())


def decide(bge_rows: list[dict], e5_rows: list[dict]) -> dict[str, Any]:
    """The decision rule, fixed before the first number from this run exists.

    The winner is the model whose 95% CI on the per-query difference excludes
    zero on BOTH Recall@10 and MRR@10, and whose lead points the same
    direction on both. Any other outcome, neither CI excludes zero, or the
    two metrics disagree about which model leads, is a tie, and BGE-M3 takes
    it on the architectural argument: one model producing dense and learned
    sparse vectors removes a whole component from stage 7. That argument
    never overrides a measured win; it only settles a result too narrow for
    18 questions to measure.
    """
    bge_recall = [r["recall@10"] for r in bge_rows]
    e5_recall = [r["recall@10"] for r in e5_rows]
    bge_mrr = [r["mrr@10"] for r in bge_rows]
    e5_mrr = [r["mrr@10"] for r in e5_rows]

    recall_lo, recall_hi, recall_diff = paired_bootstrap_ci(bge_recall, e5_recall)
    mrr_lo, mrr_hi, mrr_diff = paired_bootstrap_ci(bge_mrr, e5_mrr)

    recall_excludes_zero = recall_lo > 0 or recall_hi < 0
    mrr_excludes_zero = mrr_lo > 0 or mrr_hi < 0
    recall_winner = "bge-m3" if recall_diff > 0 else "e5-large"
    mrr_winner = "bge-m3" if mrr_diff > 0 else "e5-large"

    if recall_excludes_zero and mrr_excludes_zero and recall_winner == mrr_winner:
        winner = recall_winner
        basis = "measurement"
    elif recall_excludes_zero and mrr_excludes_zero:
        winner = PREFERRED_MODEL
        basis = (f"architecture (metrics disagreed: recall favoured "
                 f"{recall_winner}, mrr favoured {mrr_winner})")
    else:
        winner = PREFERRED_MODEL
        basis = "architecture (tie: neither CI excluded zero on both metrics)"

    return {
        "winner": winner,
        "basis": basis,
        "recall_diff_bge_minus_e5": recall_diff,
        "recall_ci_95": [recall_lo, recall_hi],
        "mrr_diff_bge_minus_e5": mrr_diff,
        "mrr_ci_95": [mrr_lo, mrr_hi],
    }


# --- reporting -----------------------------------------------------------------

def render_report(
    results: dict[tuple[str, str], dict[str, Any]],
    decision: dict[str, Any],
    truncation: dict[tuple[str, str], tuple[int, int]],
    answerable_count: int,
) -> str:
    lines = [
        "# Embedding bake-off",
        "",
        f"BAAI/bge-m3 against intfloat/multilingual-e5-large, scored against "
        f"{answerable_count} answerable golden questions. Q9 and Q10 are "
        f"excluded from every metric below: Q9 carries no gold chunks by "
        f"design, and Q10's three ids are distractors, not gold. Both are "
        f"measured elsewhere, the router and CRAG, not retrieval quality.",
        "",
        "Q3 is scored on its bare text, `depends_on` Q2 unresolved: stage 8's "
        "Rewriter, which is what turns Q3's ambiguous wording into a "
        "self-contained query, does not exist yet. Both models see the same "
        "unresolved text, so the comparison between them stays fair, but a "
        "low Q3 score for both is expected here and is not evidence about "
        "either model's retrieval quality.",
        "",
        "## The 2 x 3 grid",
        "",
        "| Model | Variant | Recall@10 | StrictRecall@10 | MRR@10 | nDCG@10 | Recall@5 | e5 cap truncates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (model_key, variant), result in results.items():
        agg = aggregate(result["rows"])
        over, total = truncation[(model_key, variant)]
        trunc_note = f"{over}/{total}" if model_key == "e5-large" else "n/a (8192 cap)"
        lines.append(
            f"| {model_key} | {variant} | {agg['recall@10']:.3f} | "
            f"{agg['strict_recall@10']:.3f} | {agg['mrr@10']:.3f} | "
            f"{agg['ndcg@10']:.3f} | {agg['recall@5']:.3f} | {trunc_note} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        f"Winner: **{decision['winner']}**, basis: {decision['basis']}.",
        "",
        f"Recall@10 difference (bge-m3 minus e5-large), none variant: "
        f"{decision['recall_diff_bge_minus_e5']:+.4f}, "
        f"95% CI [{decision['recall_ci_95'][0]:+.4f}, "
        f"{decision['recall_ci_95'][1]:+.4f}]",
        "",
        f"MRR@10 difference (bge-m3 minus e5-large), none variant: "
        f"{decision['mrr_diff_bge_minus_e5']:+.4f}, "
        f"95% CI [{decision['mrr_ci_95'][0]:+.4f}, "
        f"{decision['mrr_ci_95'][1]:+.4f}]",
        "",
        "The decision rule was fixed before this run: a win requires both "
        "CIs to exclude zero in the same direction. A tie is broken toward "
        "bge-m3 on the architectural argument, one model producing dense "
        "and learned sparse vectors, never on the numbers themselves.",
        "",
        "The golden set this decision rests on was read off rendered pages "
        "by one person in the same session that built this pipeline, not "
        "independently verified by a second reader. The bootstrap CI above "
        "is the honest response to that: a point estimate from 18 questions "
        "would overstate how settled a narrow result actually is.",
        "",
        "## Per-question, none variant",
        "",
        "| Q | gold | bge-m3 Recall@10 | bge-m3 MRR@10 | e5-large Recall@10 | e5-large MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    bge_rows = {r["id"]: r for r in results[("bge-m3", "none")]["rows"]}
    e5_rows = {r["id"]: r for r in results[("e5-large", "none")]["rows"]}
    for qid in bge_rows:
        b, e = bge_rows[qid], e5_rows[qid]
        lines.append(
            f"| {qid} | {b['n_gold']} | {b['recall@10']:.3f} | {b['mrr@10']:.3f} | "
            f"{e['recall@10']:.3f} | {e['mrr@10']:.3f} |"
        )

    lines += ["", "## Q10, negative check (non_answering_retrieval)", ""]
    for (model_key, variant), result in results.items():
        check = result["q10_check"]
        if check is None:
            continue
        lines.append(
            f"- {model_key}/{variant}: {len(check['found_in_top_k'])} of "
            f"{len(check['distractor_ids'])} distractors in the top "
            f"{RETRIEVAL_K}. A retriever surfacing these is behaving "
            f"correctly; CRAG is what has to catch it downstream."
        )

    return "\n".join(lines) + "\n"
