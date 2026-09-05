"""The query-time embedder's own process: BGE-M3 loaded once, on CPU,
never in the same process as the reranker worker or whichever question
loop is asking for vectors.

Position: retriever.py's own _dense_ranking and _sparse_ranking are what
launch and talk to this, lazily, once, keeping it alive for the life of
the parent process, the same relationship rerank.py has with
_rerank_worker.py. Not meant to be imported for its functions; run only
as `python -m pipeline.retrieval._embed_worker`, reading one JSON request
per line from stdin and writing one JSON response per line to stdout.

Why this exists at all, found directly while building stage 10's
harness.py: the same fault _rerank_worker.py's own docstring already
names for the reranker, now hitting the embedder's own side instead.
retriever.py's own dense and sparse legs load BGE-M3 in-process, on
CPU, lazily, on whichever query is the first genuine cache miss; when
that first load happens after rerank.warm_up()'s own worker is already
resident, it segfaults, confirmed with faulthandler pointing at a raw
access violation inside embedder._load_model's own AutoModel.from_pretrained
call. Pre-warming every text a run would ever need helped but could not
fully close this: a query-transformation technique (Rewriting, CRAG's
own re-query) generates genuinely new text from an LLM call, and which
exact text needs a fresh embed on any given run cannot be known in
advance. Reordering the two warm_ups relative to each other does not
help either, since embedding.md's own original finding already showed
the fault reproduces in both directions. Process isolation is the fix
that already worked for the reranker; this is the same fix applied to
the other model sharing the collision.

BGE-M3 only, CPU only, matching retriever.py's own QUERY_DEVICE: this
worker exists for the query-time path alone. Index-time embedding
(the bake-off, the store build) stays exactly as it was, in its own
short-lived process per stage 6 and 7's own already-gated design,
never routed through this worker.

The protocol, one line in and one line out, both JSON, the same shape
_rerank_worker.py already uses and for the same reason (ensure_ascii
keeps the wire format pure ASCII regardless of the pipe's own default
encoding on this platform): {"op": "dense" | "sparse", "texts": [...],
"kind": "query" | "passage"} in, {"vectors": [[...], ...]} out for
"dense" (one unit-norm vector per text), {"results": [[[idx, ...],
[val, ...]], ...]} out for "sparse" (one (indices, values) pair per
text, mirroring sparse.py's own return shape exactly since retriever.py
reads it directly).
"""

from __future__ import annotations

import json
import sys

_MODEL_KEY = "bge-m3"
_DEVICE = "cpu"


def main() -> None:
    # Imported here, not at module level: this process's only job is to
    # hold these two heavy imports and the model they load.
    from ..embedding import embedder, sparse

    # embedder._load_model prints its own diagnostic line
    # ("  loaded bge-m3 on cpu (...)") straight to stdout, which
    # _rerank_worker.py's own _load never does; found directly here when
    # the parent's own readline() picked up that line instead of the
    # "ready" line below and treated the worker as failed to start,
    # five retries in a row, while the worker itself was actually fine.
    # Redirected to stderr for the duration of the load only, so this
    # process's own stdout stays pure JSON, one line per response,
    # exactly like _rerank_worker.py's own protocol.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Forces the load now, in this process, rather than on the first
        # request: the parent blocks on the "ready" line below exactly
        # the way rerank.py's own _ensure_worker does, so the first real
        # request is never sent before the model has actually finished
        # loading.
        embedder._load_model(_MODEL_KEY, _DEVICE)
    finally:
        sys.stdout = real_stdout
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            op = request["op"]
            texts = request["texts"]
            kind = request.get("kind", "query")
            if op == "dense":
                vectors = embedder._encode_uncached(
                    texts, _MODEL_KEY, kind, len(texts), _DEVICE,
                )
                print(json.dumps({"vectors": vectors.tolist()}), flush=True)
            elif op == "sparse":
                results = sparse._sparse_batch(texts, _DEVICE)
                print(json.dumps({"results": results}), flush=True)
            else:
                print(json.dumps({"error": f"unknown op {op!r}"}), flush=True)
        except Exception as error:  # noqa: BLE001
            # A malformed request or an encoding failure is reported back
            # rather than crashing the worker: the model load already
            # paid for is worth keeping alive for the next request, the
            # same reasoning _rerank_worker.py's own main loop already
            # follows.
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
