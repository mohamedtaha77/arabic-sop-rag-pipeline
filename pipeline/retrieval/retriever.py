"""The single query-time entry point: a question in, ranked chunk ids out.

Position: everything before this file built one leg or one mechanism.
This is the first module that answers an actual query, against the real
store store.qdrant already built and verified, not against a brute-force
recomputation the way stage 6's bake-off checked itself. Reads the winning
model from BAKEOFF_DECISION the same way store.qdrant does, and reads the
shipping retrieval configuration from RETRIEVAL_DECISION once evaluate.py
has written it, never from a constant edited by hand after reading a
table.

The query-time device policy. advanced-rag-plan.md's component table put
the embedder "GPU, index time only" and Ollama as "the only model resident
at query time"; hybrid retrieval needs a dense and a sparse query vector,
so that assumption had to be revisited rather than quietly broken.
device_probe.py measured both sides: CUDA is 6.4x the steady-state speed
of CPU (28ms against 178ms per query) but holds about 1.1 GB of VRAM for
as long as the embedder stays resident, competing directly with the
partial GPU offload Ollama's generation step needs on this same 4 GB card.
178ms is negligible next to a generation call the config itself times out
at 300 seconds, while 1.1 GB is not negligible against a 4 GB budget. So
QUERY_DEVICE defaults to CPU, keeping the architecture's original promise
that the card is Ollama's alone at query time, at a cost measured and
accepted rather than hidden.

What this module does not do: it does not decide which mode, which
variant, or which cap setting ships. That is evaluate.py's decision,
written to RETRIEVAL_DECISION and read back here by whichever caller asks
for the shipping configuration rather than naming one directly.

Extended for stage 8 with a second query-time surface, retrieve_scored,
returning ScoredChunk objects (chunk text and a score, not just an id)
and accepting an optional Self-Query filter. Every function that existed
before that addition, retrieve(), _dense_ranking() and its siblings, and
retrieve_shipping's own observable behaviour, is unchanged: stage 8 adds
to this file, it does not revise what stage 6 and 7 already gated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from ..chunking.chunk import Chunk, load_chunks
from ..config import (
    CANDIDATE_K,
    CONTEXT_OUTPUTS,
    MAX_PER_PAGE,
    MAX_PER_SOURCE,
    QDRANT_PATH,
    RETRIEVAL_DECISION,
    RETRIEVAL_K,
)
from ..embedding import embedder, sparse
from ..embedding.bakeoff import load_winning_model
from ..store.qdrant import collection_name
from .bm25 import BM25Index
from .fusion import (
    apply_diversity_caps,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_scored,
)
from .text import tokenize_clitics

# The device query-time embedding runs on by default. See the module
# docstring for the measurement behind this; retriever.py's own tests and
# evaluate.py's grid can still request "cuda" explicitly when the question
# under test is about the model itself rather than about the query path.
QUERY_DEVICE = "cpu"

RETRIEVAL_MODES = (
    "dense",
    "sparse",
    "bm25",
    "dense_sparse",
    "dense_bm25",
    "dense_sparse_bm25",
)


def _legs(mode: str) -> list[str]:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"{mode!r} is not one of {RETRIEVAL_MODES}")
    return mode.split("_")


@dataclass
class RetrievalContext:
    """Everything one variant needs to answer many queries, built once.

    evaluate.py's grid scores 18 questions across 3 variants, 6 modes and
    2 cap settings; rebuilding a BM25 index or reopening the Qdrant client
    for every one of those cells would be needless work repeated hundreds
    of times over 357 documents that never change within a variant. This
    is the object that gets built once per variant and threaded through
    every cell scored against it.
    """

    variant: str
    client: QdrantClient
    model_key: str
    chunks: list[Chunk]
    chunk_lookup: dict[str, Chunk]
    bm25_index: BM25Index

    @property
    def collection(self) -> str:
        return collection_name(self.variant)


def build_context(
    variant: str,
    client: QdrantClient,
    model_key: str,
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
) -> RetrievalContext:
    chunks = load_chunks(variant_paths[variant])
    chunk_lookup = {c.metadata["chunk_id"]: c for c in chunks}
    bm25_index = BM25Index(
        [c.metadata["chunk_id"] for c in chunks],
        [c.text for c in chunks],
    )
    return RetrievalContext(
        variant=variant, client=client, model_key=model_key,
        chunks=chunks, chunk_lookup=chunk_lookup, bm25_index=bm25_index,
    )


# --- one leg at a time ---------------------------------------------------------

def _dense_ranking(
    context: RetrievalContext, question: str, k: int, device: str,
) -> list[str]:
    vector = embedder.embed_queries(
        [question], context.model_key, device=device,
    )[0]
    result = context.client.query_points(
        collection_name=context.collection, query=vector.tolist(),
        using="dense", limit=k,
    )
    return [point.payload["chunk_id"] for point in result.points]


def _sparse_ranking(
    context: RetrievalContext, question: str, k: int, device: str,
) -> list[str]:
    # device passed through explicitly: sparse.py's own module docstring
    # now documents why an omitted device here would try to load bge-m3 a
    # second time on whatever embedder._load_model auto-selects, colliding
    # with _dense_ranking's already-resident copy in the same process.
    (indices, values), = sparse.embed_sparse_queries([question], device=device)
    result = context.client.query_points(
        collection_name=context.collection,
        query=models.SparseVector(indices=indices, values=values),
        using="sparse", limit=k,
    )
    return [point.payload["chunk_id"] for point in result.points]


def _bm25_ranking(context: RetrievalContext, question: str) -> list[str]:
    return context.bm25_index.rank(question)


_LEG_FUNCTIONS = {
    "dense": _dense_ranking,
    "sparse": _sparse_ranking,
    "bm25": lambda context, question, k, device: _bm25_ranking(context, question),
}


# --- the entry point -----------------------------------------------------------

def retrieve(
    context: RetrievalContext,
    question: str,
    mode: str,
    k: int = RETRIEVAL_K,
    candidate_k: int = CANDIDATE_K,
    device: str = QUERY_DEVICE,
    apply_caps: bool = False,
    max_per_source: int = MAX_PER_SOURCE,
    max_per_page: int = MAX_PER_PAGE,
) -> list[str]:
    """Rank this variant's chunks against one question, under one mode.

    Every leg is asked for candidate_k candidates, wider than k so a chunk
    ranked just outside the top RETRIEVAL_K on one leg alone can still be
    lifted back in by fusion; a single-leg mode still goes through this
    same candidate_k request even though fusion over one ranking is a
    no-op, so a single-leg mode and a multi-leg mode differ only in which
    legs run, never in how candidates are gathered.
    """
    legs = _legs(mode)
    rankings = [
        _LEG_FUNCTIONS[leg](context, question, candidate_k, device)
        for leg in legs
    ]
    fused = (
        rankings[0] if len(rankings) == 1
        else reciprocal_rank_fusion(rankings)
    )

    if apply_caps:
        fused = apply_diversity_caps(
            fused, context.chunk_lookup, max_per_source, max_per_page,
        )

    return fused[:k]


# --- scored retrieval, for stage 8 ----------------------------------------------
#
# retrieve() above returns chunk ids alone, which is all stage 6 and 7 ever
# needed: a ranking to score against gold ids. Stage 8's techniques need
# more. Compression and reranking need the chunk text, not just its id, and
# CRAG needs a confidence signal to grade. Rather than bolt this onto
# retrieve() and risk changing what stage 7's already-gated evaluate.py
# calls, this is new, additive surface: retrieve() and every _LEG_FUNCTIONS
# entry above are untouched, byte-identical to what they were before this
# section existed.


@dataclass
class ScoredChunk:
    """One ranked result, carrying enough for a technique to act on it.

        chunk_id           as retrieve() returns
        chunk               the Chunk itself, text and metadata, looked up
                             from context.chunk_lookup rather than
                             refetched, since it was already loaded into
                             memory when this variant's context was built
        score                the leg's own score for a single-leg mode,
                             the RRF-fused score otherwise
        score_is_absolute   True only for a single-leg mode. A single
                             dense leg's score is cosine similarity, a
                             fixed, comparable scale; a single sparse or
                             BM25 leg's score is real but has no such
                             fixed scale either, so this flag says
                             "not rank-derived", not "directly comparable
                             to a cosine". False whenever two or more legs
                             were fused: an RRF score's absolute size is
                             an artifact of RRF_K and how many legs
                             contributed, not a confidence measure on any
                             fixed scale. CRAG (stage 8's own evaluator)
                             is the reader this distinction exists for.
    """

    chunk_id: str
    chunk: Chunk
    score: float
    score_is_absolute: bool


def _build_qdrant_filter(
    allowed_chunk_ids: set[str] | None,
) -> models.Filter | None:
    """A Self-Query filter, translated into Qdrant's own filter shape.

    None means unfiltered, the same as never passing query_filter at all;
    callers never need to branch on whether a filter is active. chunk_id
    is a plain payload field (PAYLOAD_FIELDS in store.qdrant already
    carries it on every point), so this needs no schema index to work at
    this collection's size, 357 points.
    """
    if allowed_chunk_ids is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(
            key="chunk_id", match=models.MatchAny(any=sorted(allowed_chunk_ids)),
        )]
    )


def _dense_ranking_scored(
    context: RetrievalContext, question: str, k: int, device: str,
    qdrant_filter: models.Filter | None,
) -> list[tuple[str, float]]:
    vector = embedder.embed_queries(
        [question], context.model_key, device=device,
    )[0]
    result = context.client.query_points(
        collection_name=context.collection, query=vector.tolist(),
        using="dense", limit=k, query_filter=qdrant_filter,
    )
    return [(point.payload["chunk_id"], point.score) for point in result.points]


def _sparse_ranking_scored(
    context: RetrievalContext, question: str, k: int, device: str,
    qdrant_filter: models.Filter | None,
) -> list[tuple[str, float]]:
    (indices, values), = sparse.embed_sparse_queries([question], device=device)
    result = context.client.query_points(
        collection_name=context.collection,
        query=models.SparseVector(indices=indices, values=values),
        using="sparse", limit=k, query_filter=qdrant_filter,
    )
    return [(point.payload["chunk_id"], point.score) for point in result.points]


def _bm25_ranking_scored(
    context: RetrievalContext, question: str, k: int,
    allowed_chunk_ids: set[str] | None,
) -> list[tuple[str, float]]:
    """BM25 never touches Qdrant, so the filter is a plain Python
    membership check instead of a query_filter. Filtering the full
    357-row ranking before slicing to k, rather than slicing first and
    filtering after, is what keeps this "top k of the allowed subset"
    rather than "top k overall, some of which then get thrown away",
    which could under-fill a narrow filter for no real reason.
    """
    full = context.bm25_index.rank_scored(question)
    if allowed_chunk_ids is not None:
        full = [(cid, score) for cid, score in full if cid in allowed_chunk_ids]
    return full[:k]


def retrieve_scored(
    context: RetrievalContext,
    question: str,
    mode: str,
    k: int = RETRIEVAL_K,
    candidate_k: int = CANDIDATE_K,
    device: str = QUERY_DEVICE,
    apply_caps: bool = False,
    max_per_source: int = MAX_PER_SOURCE,
    max_per_page: int = MAX_PER_PAGE,
    allowed_chunk_ids: set[str] | None = None,
) -> list[ScoredChunk]:
    """retrieve(), but returning ScoredChunk objects instead of bare ids,
    and accepting an optional Self-Query filter.

    allowed_chunk_ids narrows every leg to that subset before ranking, not
    after: for the dense and sparse legs this is a Qdrant query_filter, so
    the store itself only ever ranks the allowed points; for BM25 it is
    applied to the full in-memory ranking before truncation, for the same
    reason. A filter that empties the candidate set is the caller's
    concern (selfquery.py's own trace records that fallback), not this
    function's: an empty allowed_chunk_ids here simply returns nothing.
    """
    legs = _legs(mode)
    qdrant_filter = _build_qdrant_filter(allowed_chunk_ids)

    per_leg: list[list[tuple[str, float]]] = []
    for leg in legs:
        if leg == "dense":
            per_leg.append(_dense_ranking_scored(
                context, question, candidate_k, device, qdrant_filter,
            ))
        elif leg == "sparse":
            per_leg.append(_sparse_ranking_scored(
                context, question, candidate_k, device, qdrant_filter,
            ))
        else:
            per_leg.append(_bm25_ranking_scored(
                context, question, candidate_k, allowed_chunk_ids,
            ))

    if len(per_leg) == 1:
        fused = per_leg[0]
        score_is_absolute = True
    else:
        rankings = [[chunk_id for chunk_id, _ in scored] for scored in per_leg]
        fused = reciprocal_rank_fusion_scored(rankings)
        score_is_absolute = False

    if apply_caps:
        ranked_ids = [chunk_id for chunk_id, _ in fused]
        kept = set(apply_diversity_caps(
            ranked_ids, context.chunk_lookup, max_per_source, max_per_page,
        ))
        fused = [(chunk_id, score) for chunk_id, score in fused if chunk_id in kept]

    return [
        ScoredChunk(
            chunk_id=chunk_id, chunk=context.chunk_lookup[chunk_id],
            score=score, score_is_absolute=score_is_absolute,
        )
        for chunk_id, score in fused[:k]
    ]


def fuse_scored(
    per_query: list[list[ScoredChunk]],
    context: RetrievalContext,
    k: int = RETRIEVAL_K,
) -> list[ScoredChunk]:
    """RRF-fuse several ScoredChunk rankings, one per query, into one.

    Added for stage 8: Multi-Query fuses one ranking per paraphrase,
    Decomposition fuses one ranking per sub-question, and both need
    exactly this, so it lives once here rather than being written twice.
    Fusion is always by rank, never by comparing the raw scores each
    per_query ranking carries: those scores came from different query
    texts entirely and were never on a shared scale to begin with, the
    same reason retrieve_scored's own multi-leg fusion never compares
    a dense cosine against a BM25 score directly. The result is always
    score_is_absolute=False, for that reason.
    """
    if len(per_query) == 1:
        return per_query[0][:k]

    rankings = [[item.chunk_id for item in scored] for scored in per_query]
    fused = reciprocal_rank_fusion_scored(rankings)
    return [
        ScoredChunk(
            chunk_id=chunk_id, chunk=context.chunk_lookup[chunk_id],
            score=score, score_is_absolute=False,
        )
        for chunk_id, score in fused[:k]
    ]


# --- reading the shipping decision ----------------------------------------------

def load_retrieval_decision(decision_path: Path = RETRIEVAL_DECISION) -> dict[str, Any]:
    """What evaluate.py decided, read back from disk rather than trusted to
    a constant. Raises the same way bakeoff.load_winning_model does: a
    stage that has not been measured yet should refuse to be built against.
    """
    if not decision_path.exists():
        raise FileNotFoundError(
            f"{decision_path} not found. Run `python cli.py retrieve "
            f"--evaluate` first; nothing downstream should build against a "
            f"retrieval configuration the measurement never actually chose."
        )
    return json.loads(decision_path.read_text(encoding="utf-8"))


@dataclass
class ShippingHandle:
    """One open Qdrant client plus the shipping RetrievalContext, built
    once and reused across many questions.

    Added for stage 8: techniques/run.py answers up to 20 golden
    questions in one run, and opening a fresh client and rebuilding a
    BM25 index 20 times over would be exactly the needless repetition
    RetrievalContext's own docstring already explains evaluate.py's grid
    exists to avoid. retrieve_shipping below is the single-question
    convenience path built on top of this, not a second implementation
    of it.
    """

    client: QdrantClient
    context: RetrievalContext
    decision: dict[str, Any]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ShippingHandle":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_shipping(
    qdrant_path: Path = QDRANT_PATH,
    decision_path: Path = RETRIEVAL_DECISION,
) -> ShippingHandle:
    """Open the store once and build the shipping RetrievalContext once.
    Use as a context manager (`with open_shipping() as handle:`) so the
    client closes even if a question raises partway through a batch.
    """
    decision = load_retrieval_decision(decision_path)
    model_key = load_winning_model()
    client = QdrantClient(path=str(qdrant_path))
    context = build_context(decision["variant"], client, model_key)
    return ShippingHandle(client=client, context=context, decision=decision)


def retrieve_shipping(
    question: str,
    qdrant_path: Path = QDRANT_PATH,
    decision_path: Path = RETRIEVAL_DECISION,
) -> list[str]:
    """The one-call convenience path: whatever evaluate.py decided, applied
    to one question. This is what `cli.py retrieve "..."` calls; nothing
    inside pipeline/generation/ or pipeline/techniques/ should reopen a
    Qdrant client or rebuild a BM25 index on its own once this exists.

    Opens and closes its own handle, which is correct for exactly one
    question and wasteful for many: a caller answering a whole batch
    should call open_shipping() once and reuse the handle instead, the
    way techniques/run.py does.
    """
    with open_shipping(qdrant_path, decision_path) as handle:
        return retrieve(
            handle.context, question, handle.decision["mode"],
            apply_caps=handle.decision.get("apply_caps", False),
        )
