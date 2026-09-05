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

# What GENERATOR_MODEL occupies on this card, measured from Ollama's own
# /api/ps: 2.16 GB for qwen2.5:3b-instruct-q4_K_M at 4096 context. The
# reranker refuses the GPU unless this much would still be left, because
# a reranker that wins the card and pushes generation onto the CPU trades
# 28 seconds of reranking for considerably more in the synthesiser.
_OLLAMA_VRAM_RESERVE = 2.2e9


def _pick_device() -> str:
    """The GPU if there is one with room, else the CPU path that shipped.

    This card sat completely idle while the reranker fought the embedder
    and Ollama for scarce system memory, which was the whole problem.
    Measured here, same 20 pairs: 0.28 s on the GPU against roughly 29 s
    of load-score-unload churn on the CPU, and in fp16 the weights take
    1.14 GB of VRAM while system memory actually goes *up*, because
    nothing large stays resident on the host side.

    A card too small to hold the weights is left alone rather than
    half-used: Ollama needs about 2.16 GB of the same 4 GB for
    GENERATOR_MODEL, so this claims the GPU only when both still fit.

    The test is against the card's *total* memory, deliberately, not
    what happens to be free right now. Free VRAM swings by more than
    2 GB depending on whether Ollama is holding GENERATOR_MODEL at this
    instant, and rerank.apply() may have just evicted it, so choosing on
    the live figure would put the reranker on the GPU or the CPU
    according to timing rather than according to the hardware. A real
    out-of-memory on the load is caught by the caller, which falls back.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        # 1.2 GB for these weights, 1.9 GB for GENERATOR_MODEL's own
        # weights, 0.85 GB for the display. Measured on this 4.29 GB
        # card: the reranker leaves 2.05 GB and Ollama wants 2.16 GB at
        # 4096 context, so its KV cache spills about 0.1 GB to the host.
        # That is a deliberate trade and not an oversight: a fraction of
        # a second on the generation call against the 28 seconds of
        # load-score-unload the CPU path costs on every single question.
        needed = 1.2e9 + 1.9e9 + 0.85e9
        return "cuda" if total_bytes > needed else "cpu"
    except Exception:
        return "cpu"


def _load() -> tuple:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = load_with_retry(
        lambda: AutoTokenizer.from_pretrained(RERANK_MODEL),
        "reranker tokenizer",
    )

    device = _pick_device()
    if device == "cuda":
        # device_map streams the shards straight into VRAM instead of
        # materialising the whole model in system RAM first, which is
        # what segfaulted here when host memory was down to ~2 GB.
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                RERANK_MODEL, torch_dtype=torch.float16, device_map={"": 0},
            )
            model.eval()
            return tokenizer, model, device
        except Exception as error:  # noqa: BLE001
            # Not retried on the GPU: if the card is genuinely full,
            # asking again changes nothing, and the CPU path below is a
            # working answer rather than a failed question. Said out
            # loud on stderr because a silent demotion here looks like
            # an unexplained thirty seconds later on.
            print(f"  reranker could not take the GPU ({error}); "
                  f"falling back to CPU", file=sys.stderr, flush=True)
            device = "cpu"
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
    return tokenizer, model, device


def _score(tokenizer, model, device: str, query: str, texts: list[str],
           max_length: int) -> list[float]:
    import torch

    pairs = [(query, text) for text in texts]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=max_length,
        )
        if device == "cuda":
            inputs = inputs.to("cuda")
        # .float() before sigmoid matters on the GPU path: the logits come
        # back in fp16 there, and the scores are compared against each
        # other and rounded into a trace, so they are widened first.
        logits = model(**inputs).logits.view(-1).float()
        scores = torch.sigmoid(logits)
    return scores.tolist()


def main() -> None:
    tokenizer, model, device = _load()
    # The parent blocks on this exact line before sending its first
    # request, so the model is fully loaded before anything is asked of
    # it; see rerank.py's own _ensure_worker. The device travels with the
    # ready line because it decides something the parent cannot see for
    # itself: a worker holding its weights in VRAM is not competing for
    # host memory, so rerank.apply() can leave it resident.
    print(json.dumps({"ready": True, "device": device}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            scores = _score(
                tokenizer, model, device, request["query"], request["texts"],
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
