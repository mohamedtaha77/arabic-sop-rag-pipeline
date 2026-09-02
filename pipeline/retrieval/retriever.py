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
from .fusion import apply_diversity_caps, reciprocal_rank_fusion
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


def retrieve_shipping(
    question: str,
    qdrant_path: Path = QDRANT_PATH,
    decision_path: Path = RETRIEVAL_DECISION,
) -> list[str]:
    """The one-call convenience path: whatever evaluate.py decided, applied
    to one question. This is what `cli.py retrieve "..."` calls; nothing
    inside pipeline/generation/ or pipeline/techniques/ should reopen a
    Qdrant client or rebuild a BM25 index on its own once this exists.
    """
    decision = load_retrieval_decision(decision_path)
    model_key = load_winning_model()
    client = QdrantClient(path=str(qdrant_path))
    try:
        context = build_context(decision["variant"], client, model_key)
        return retrieve(
            context, question, decision["mode"],
            apply_caps=decision.get("apply_caps", False),
        )
    finally:
        client.close()
