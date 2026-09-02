"""Combining rankings, and then diversifying the result.

Two separate jobs, kept in one file because both answer the same question,
"which candidates actually reach the top k", from two different angles:
fusion decides which chunks are relevant at all, diversity decides whether
too many of them say the same thing.

Reciprocal Rank Fusion, not a weighted sum of raw scores. Dense cosine
similarity, BM25's term-frequency score and BGE-M3's learned-sparse score
live on three different, incomparable scales, and RRF sidesteps that by
using only each leg's rank, never its score. Qdrant's own server-side
FusionQuery was considered and rejected: BM25 lives outside the store
entirely, so fusion has to happen in this process regardless, and running
two fusion implementations that could silently disagree is worse than
running one.

What this module does not do: it does not retrieve anything. retriever.py
calls each leg, then hands their rankings here.
"""

from __future__ import annotations

import collections

import numpy as np

from ..config import MAX_PER_PAGE, MAX_PER_SOURCE, MMR_LAMBDA, RRF_K


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> list[str]:
    """Fuse several chunk-id rankings into one, by rank rather than score.

    Each ranking contributes 1 / (k + rank) to every chunk id it contains,
    rank starting at 1; a chunk absent from a ranking contributes nothing
    from it rather than a penalty, which is what lets a chunk found by only
    one leg still surface. weights scale a leg's whole contribution, used
    by evaluate.py's grid to check whether any weighting recovers what
    unweighted fusion costs, never tuned against the golden set beyond the
    small values the plan names.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: dict[str, float] = collections.defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += weight / (k + rank)

    return [
        chunk_id for chunk_id, _ in
        sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


# --- diversity caps -------------------------------------------------------------

def apply_diversity_caps(
    ranked_ids: list[str],
    chunk_lookup: dict[str, "object"],
    max_per_source: int = MAX_PER_SOURCE,
    max_per_page: int = MAX_PER_PAGE,
) -> list[str]:
    """Walk a ranking once, dropping a candidate that would push its source
    or its (source, page) pair over its cap. Dropped candidates are not
    reinserted later; a cap that let them back in once the top of the list
    was exhausted would not be enforcing anything.

    chunk_lookup maps chunk_id to an object with .metadata, matching
    chunking.chunk.Chunk, so this never needs its own copy of what a
    chunk's source or page is.

    Deliberately measured rather than assumed to help: Q13's three gold
    chunks sit on two adjacent pages, where a page cap can cost recall,
    while Q19's three sit one per source manual, where a source cap is
    exactly what it is for. evaluate.py's grid runs both capped and
    uncapped, and reports which questions a cap actually helped or hurt
    rather than switching it on because the plan named it.
    """
    per_source: collections.Counter[str] = collections.Counter()
    per_page: collections.Counter[tuple[str, int]] = collections.Counter()
    kept: list[str] = []

    for chunk_id in ranked_ids:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        page_key = (source, page)

        if per_source[source] >= max_per_source:
            continue
        if per_page[page_key] >= max_per_page:
            continue

        kept.append(chunk_id)
        per_source[source] += 1
        per_page[page_key] += 1

    return kept


# --- MMR --------------------------------------------------------------------

def mmr_select(
    query_vector: np.ndarray,
    candidate_ids: list[str],
    candidate_vectors: np.ndarray,
    k: int,
    lambda_: float = MMR_LAMBDA,
) -> list[str]:
    """Re-order candidates by Maximal Marginal Relevance.

    Greedy: at each step, picks whichever remaining candidate maximises
    lambda_ * relevance-to-query minus (1 - lambda_) * similarity to
    whatever has already been picked. lambda_ = 1 degenerates to plain
    relevance ranking, the baseline evaluate.py's grid checks this against.

    Vectors are assumed unit-norm, the contract embedder.embed_texts
    already guarantees, so a dot product is a cosine both here and in
    metrics.rank_by_similarity; this never renormalises its own input.
    """
    if k <= 0 or not candidate_ids:
        return []

    relevance = candidate_vectors @ query_vector
    remaining = list(range(len(candidate_ids)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        if not selected:
            best_local = int(np.argmax(relevance[remaining]))
            selected.append(remaining.pop(best_local))
            continue

        selected_vectors = candidate_vectors[selected]
        best_index, best_value = None, None
        for local_index, candidate_index in enumerate(remaining):
            max_similarity = float(
                (selected_vectors @ candidate_vectors[candidate_index]).max()
            )
            mmr_value = (
                lambda_ * relevance[candidate_index]
                - (1 - lambda_) * max_similarity
            )
            if best_value is None or mmr_value > best_value:
                best_value, best_index = mmr_value, local_index

        selected.append(remaining.pop(best_index))

    return [candidate_ids[i] for i in selected]
