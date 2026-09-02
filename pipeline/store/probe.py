"""Whether Qdrant's local file mode accepts a sparse vector, measured rather
than looked up in a changelog.

advanced-rag-plan.md names this as an open item: BGE-M3 can produce a learned
sparse vector alongside its dense one, and if the local store can hold it,
that removes a whole component from stage 7's hybrid retrieval. If it cannot,
BM25 carries the sparse signal alone, and the plan already says that goes in
the report as a stated constraint rather than a silent gap.

The only way to answer this honestly is to try it: create a throwaway local
collection with a sparse vector configured, upsert one point, query it back,
and delete the collection. A changelog can be stale by the time this runs;
the store on this machine cannot.

What this module does not do: it does not decide whether BGE-M3 wins the
bake-off. That is bakeoff.py. This only answers whether the store could hold
its sparse output if it does.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from ..config import QDRANT_PATH

_PROBE_COLLECTION = "_probe_sparse_support"


def probe_sparse_support(qdrant_path=QDRANT_PATH) -> bool:
    """True if a local-mode collection accepts and returns a sparse vector.

    A fresh client against the real QDRANT_PATH, not a temp directory:
    local file mode holds a lock on its directory while open, and the honest
    test is whether the collection this project will actually use supports
    it, not a stand-in that could behave differently.
    """
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    try:
        if client.collection_exists(_PROBE_COLLECTION):
            client.delete_collection(_PROBE_COLLECTION)

        client.create_collection(
            collection_name=_PROBE_COLLECTION,
            vectors_config={"dense": models.VectorParams(
                size=4, distance=models.Distance.COSINE,
            )},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        client.upsert(
            collection_name=_PROBE_COLLECTION,
            points=[models.PointStruct(
                id=1,
                vector={
                    "dense": [0.1, 0.2, 0.3, 0.4],
                    "sparse": models.SparseVector(
                        indices=[3, 17, 42], values=[0.5, 0.25, 0.9],
                    ),
                },
                payload={"probe": True},
            )],
        )
        result = client.query_points(
            collection_name=_PROBE_COLLECTION,
            query=models.SparseVector(indices=[3, 17], values=[1.0, 1.0]),
            using="sparse",
            limit=1,
        )
        return len(result.points) == 1
    finally:
        if client.collection_exists(_PROBE_COLLECTION):
            client.delete_collection(_PROBE_COLLECTION)
        client.close()


def run(qdrant_path=QDRANT_PATH) -> bool:
    print("Qdrant local file mode: sparse vector support")
    print(f"path {qdrant_path}\n")
    try:
        supported = probe_sparse_support(qdrant_path)
    except Exception as error:  # noqa: BLE001 - reporting, not handling
        print(f"  FAIL  probe raised: {error}")
        print("  Treat this as unsupported: BM25 carries the sparse signal "
              "alone in stage 7, and that is a stated constraint, not a "
              "silent gap.")
        return False

    if supported:
        print("  ok    a sparse vector was upserted and queried back")
        print("  BGE-M3's sparse head can be written alongside the dense "
              "vector in every collection this stage builds. See sparse.py.")
    else:
        print("  no    the collection accepted the config but did not "
              "return the expected point")
        print("  BM25 carries the sparse signal alone in stage 7; recorded "
              "as a constraint in the report.")
    return supported


if __name__ == "__main__":
    run()
