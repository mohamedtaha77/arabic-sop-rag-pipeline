"""The reranker's own process: loaded once, scored many times, never in
the same process as the query-time embedder.

Position: rerank.py's own apply() is what launches and talks to this,
lazily, once, keeping it alive for the life of the parent process. Not
meant to be imported for its functions; run only as
`python -m pipeline.techniques._rerank_worker`, reading one JSON request
per line from stdin and writing one JSON response per line to stdout.

Why this exists at all, found directly while building rerank.py: loading
bge-reranker-v2-m3 in the same process as an already-resident BGE-M3
dense embedder segfaulted, confirmed with faulthandler pointing at a raw
access violation inside the reranker's own from_pretrained call, model
construction, not the embedder's. This is embedding.md's own documented
class of fault, "a second distinct model checkpoint in one process
segfaults", playing out again on the query-time side rather than the
bake-off side that first found it. The bake-off's own fix was releasing
one model before loading the next; that is not available here, because
retriever.py's own embedder cache deliberately keeps BGE-M3 resident
across many questions, and releasing it before every rerank call would
defeat the entire reason that cache exists. Process isolation is the fix
that remains, and a persistent worker rather than one subprocess per call
is what keeps the 2.2 GB load a one-time cost rather than a per-question
one.

The protocol is deliberately the smallest thing that works: one line in,
one line out, both JSON. {"query": "...", "texts": ["...", ...],
"max_length": 1024} in, {"scores": [0.1, 0.9, ...]} out, one score per
text, in the same order given. json.dumps's default ensure_ascii=True is
kept deliberately: it escapes every non-ASCII character (this corpus is
Arabic) to \\uXXXX, so the wire format is pure ASCII regardless of
whatever the pipe's own default text encoding happens to resolve to on
this platform, the same class of surprise ingestion.py's own
ensure_ascii notes already warn about for a Windows console.
"""

from __future__ import annotations

import json
import sys

from ..config import RERANK_MODEL
from ..embedding.tokens import load_with_retry


def _load() -> tuple:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = load_with_retry(
        lambda: AutoTokenizer.from_pretrained(RERANK_MODEL),
        "reranker tokenizer",
    )
    # low_cpu_mem_usage=True: the default loading path builds a full
    # random-weight model first, then overlays the checkpoint on top,
    # transiently needing roughly twice the model's final memory before
    # the random buffer is freed. Kept as a reasonable precaution, safe
    # here because this checkpoint ships model.safetensors; it did not,
    # on its own, fix the crash rerank.py's own warm_up() docstring
    # describes. Launch order did. Both stay: this genuinely lowers peak
    # memory during the load regardless.
    model = load_with_retry(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, low_cpu_mem_usage=True,
        ),
        "reranker model weights",
    )
    model.eval()
    return tokenizer, model


def _score(tokenizer, model, query: str, texts: list[str], max_length: int) -> list[float]:
    import torch

    pairs = [(query, text) for text in texts]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=max_length,
        )
        logits = model(**inputs).logits.view(-1).float()
        scores = torch.sigmoid(logits)
    return scores.tolist()


def main() -> None:
    tokenizer, model = _load()
    # The parent blocks on this exact line before sending its first
    # request, so the model is fully loaded before anything is asked of
    # it; see rerank.py's own _ensure_worker.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            scores = _score(
                tokenizer, model, request["query"], request["texts"],
                request.get("max_length", 1024),
            )
            print(json.dumps({"scores": scores}), flush=True)
        except Exception as error:  # noqa: BLE001
            # A malformed request or a scoring failure is reported back
            # rather than crashing the worker: the 2.2 GB load already
            # paid for is worth keeping alive for the next request.
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
