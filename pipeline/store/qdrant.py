"""Three collections, one per context variant, in Qdrant's local file mode.

Reads whichever model run.py's bake-off chose, from BAKEOFF_DECISION on
disk, never from a constant edited by hand: build() refuses to run until
that file exists, the same way golden.py refuses to verify a set whose
chunk file is missing. That is what makes it impossible to build the store
against a model the measurement never actually chose.

The payload schema is declared once, as PAYLOAD_FIELDS, the same move
ledger.STEPS and chunk.CHUNK_TYPES make: a dropped field raises rather than
silently missing from every point in the store.

What this module does not do: it does not decide which model to embed with,
that is run.py's bake-off, writing BAKEOFF_DECISION; and it does not compute
BGE-M3's sparse vectors itself, that is sparse.py, imported only when the
winning model has one and the store can hold it.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

# torch before qdrant_client and numpy: see embedder.py's import-order note.
import torch  # noqa: F401
from qdrant_client import QdrantClient, models

from ..chunking.chunk import Chunk, load_chunks
from ..config import (
    COLLECTION_PREFIX,
    CONTEXT_OUTPUTS,
    GOLDEN_SET,
    QDRANT_PATH,
    RETRIEVAL_K,
)
from ..embedding import embedder
from ..embedding.bakeoff import load_winning_model
from ..embedding.metrics import rank_by_similarity
from ..golden.question import Question, load_golden
from . import probe as store_probe

# Every metadata key that survives from a chunk into its Qdrant point, decided
# in the plan against a named consumer for each: unit and actor for stage 8's
# Self-Query filter and stage 7's literal role matching, table_id for the
# per-page diversity cap and grid-row sibling grouping, context_prefix and
# llm_source because context.md's own open question, whether the prefix
# should be stripped back out at generation time, is stage 9's to answer and
# it needs the prefix present to strip. row_range is deliberately absent: it
# exists so chunking's own gate can be checked, and nothing at query time
# reads a row index.
PAYLOAD_FIELDS = (
    "chunk_id", "source", "page", "end_page", "section_path", "chunk_type",
    "actor", "unit", "table_id", "doc_version", "issue_date", "review_date",
    "extraction_quality", "char_count", "context_prefix", "llm_source",
)


def collection_name(variant: str) -> str:
    return f"{COLLECTION_PREFIX}{variant}"


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID from a chunk id.

    Tested directly against this store rather than assumed: Qdrant rejects an
    arbitrary string like a chunk id outright, an unsigned integer or a UUID
    is all it accepts. uuid5 makes the mapping reproducible, so re-running
    the build overwrites the same points instead of duplicating them, and
    chunk_id stays in the payload as the real key anything downstream reads.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": chunk.text}
    for field in PAYLOAD_FIELDS:
        payload[field] = chunk.metadata.get(field)
    return payload


# --- build ---------------------------------------------------------------------

def build_collection(
    client: QdrantClient, variant: str, chunks: list[Chunk],
    model_key: str, use_sparse: bool,
) -> None:
    """(Re)create one collection and fill it with this variant's chunks.

    Dropping and recreating rather than diffing: a chunk build changing
    underneath a stale collection is exactly the failure golden.py's gate 8
    exists to catch on the golden set's side, and the store gets the same
    discipline by simply never carrying old points forward.
    """
    name = collection_name(variant)
    if client.collection_exists(name):
        client.delete_collection(name)

    texts = [c.text for c in chunks]
    dense_vectors = embedder.embed_passages(texts, model_key)
    dimension = dense_vectors.shape[1]

    vectors_config = {"dense": models.VectorParams(
        size=dimension, distance=models.Distance.COSINE,
    )}
    sparse_vectors_config = {"sparse": models.SparseVectorParams()} if use_sparse else None

    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    sparse_vectors = None
    if use_sparse:
        from ..embedding import sparse
        sparse_vectors = sparse.embed_sparse_passages(texts)

    points = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.metadata["chunk_id"]
        vector: dict[str, Any] = {"dense": dense_vectors[i].tolist()}
        if use_sparse and sparse_vectors is not None:
            indices, values = sparse_vectors[i]
            vector["sparse"] = models.SparseVector(indices=indices, values=values)
        points.append(models.PointStruct(
            id=_point_id(chunk_id), vector=vector, payload=_payload(chunk),
        ))

    client.upsert(collection_name=name, points=points)


# --- verify --------------------------------------------------------------------

def verify(
    client: QdrantClient, variant: str, chunks: list[Chunk],
    model_key: str, questions: list[Question],
) -> list[str]:
    """Gates 5, 6 and 7 for one collection. Empty when clean.

    Gate 6 is the one worth the most here: it is the only check that can
    tell "the retriever is bad" apart from "the store is wired wrong", and
    from stage 7 those look identical. It reruns metrics.py's own
    rank_by_similarity, the same function the bake-off's decision was
    computed against, on the same vectors this collection was built from,
    so the store is checked against the exact ranking that decided the
    model, not against a second implementation that could quietly drift
    from it.
    """
    failures = []
    name = collection_name(variant)

    info = client.get_collection(name)
    if info.points_count != len(chunks):
        failures.append(
            f"gate 5: {name} has {info.points_count} points, "
            f"expected {len(chunks)}"
        )

    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    dense_vectors = embedder.embed_passages([c.text for c in chunks], model_key)

    answerable = [q for q in questions if q.expect == "answerable"]
    query_vectors = embedder.embed_queries([q.question for q in answerable], model_key)

    mismatches = []
    for question, query_vector in zip(answerable, query_vectors):
        brute = rank_by_similarity(query_vector, dense_vectors, chunk_ids)[:RETRIEVAL_K]
        result = client.query_points(
            collection_name=name, query=query_vector.tolist(),
            using="dense", limit=RETRIEVAL_K,
        )
        store_ranked = [p.payload["chunk_id"] for p in result.points]
        if store_ranked != brute:
            mismatches.append(question.id)
    if mismatches:
        failures.append(
            f"gate 6: {name} store ranking disagreed with brute-force cosine "
            f"on {len(mismatches)}/{len(answerable)} golden queries: "
            f"{mismatches}"
        )

    point_ids = [_point_id(cid) for cid in chunk_ids]
    records = client.retrieve(collection_name=name, ids=point_ids, with_payload=True)
    by_chunk_id = {r.payload["chunk_id"]: r.payload for r in records}
    bad_payload = []
    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        payload = by_chunk_id.get(chunk_id)
        if payload is None:
            bad_payload.append(chunk_id)
            continue
        for field in PAYLOAD_FIELDS:
            if payload.get(field) != chunk.metadata.get(field):
                bad_payload.append(chunk_id)
                break
    if bad_payload:
        failures.append(
            f"gate 7: {name} has {len(bad_payload)} points with a payload "
            f"field mismatch against their source chunk, first few: "
            f"{bad_payload[:5]}"
        )

    return failures


# --- entry point -----------------------------------------------------------------

def run(
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
    golden_path: Path = GOLDEN_SET,
    qdrant_path: Path = QDRANT_PATH,
) -> bool:
    try:
        model_key = load_winning_model()
    except FileNotFoundError as error:
        print(error)
        return False
    print(f"winning model: {model_key}")

    if not golden_path.exists():
        print(f"{golden_path} not found; gate 6 needs it. "
              f"Run `python cli.py golden` first.")
        return False
    questions, _ = load_golden(golden_path)

    sparse_available = store_probe.probe_sparse_support(qdrant_path)
    use_sparse = sparse_available and model_key == "bge-m3"
    print(f"sparse vectors: probe={'yes' if sparse_available else 'no'}, "
          f"model={model_key}, writing sparse={'yes' if use_sparse else 'no'}")

    client = QdrantClient(path=str(qdrant_path))
    all_failures: list[str] = []
    try:
        for variant, path in variant_paths.items():
            if not path.exists():
                print(f"Missing {path}. Run `python cli.py context` first.")
                return False
            chunks = load_chunks(path)
            print(f"\n{variant}: building {len(chunks)} points into "
                  f"{collection_name(variant)}")
            build_collection(client, variant, chunks, model_key, use_sparse)
            failures = verify(client, variant, chunks, model_key, questions)
            if failures:
                for failure in failures:
                    print(f"  FAIL  {failure}")
                all_failures += failures
            else:
                print("  ok  gates 5, 6 and 7 pass")
    finally:
        client.close()

    print(f"\ngate 8: sparse probe recorded as "
          f"{'yes' if sparse_available else 'no'}, points carry sparse "
          f"vectors: {'yes' if use_sparse else 'no'}")

    return not all_failures


if __name__ == "__main__":
    run()
