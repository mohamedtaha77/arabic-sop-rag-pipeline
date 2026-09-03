"""The NLI model's own process: loaded once, scored many times.

Position: entail.py's own apply() launches and talks to this, lazily,
once, keeping it alive for the life of the parent process. Mirrors
techniques/_rerank_worker.py deliberately, including its spawn-retry
pattern in entail.py: this project has one documented class of fault,
two distinct large model checkpoints crashing when loaded in the same
process (embedding.md, router.md, rerank.py's own module docstring all
describe it independently), and mDeBERTa-v3-base-xnli is a second
checkpoint that could in principle collide with an already-resident
BGE-M3 embedder the same way bge-reranker-v2-m3 measured to. mDeBERTa is
roughly a quarter of the reranker's own footprint (557 MB of weights
against 2.2 GB), which lowers the odds without ever having been
confirmed safe in-process on this machine; process isolation costs one
subprocess and closes the question rather than assuming an answer that
was never actually measured.

Not meant to be imported for its functions; run only as
`python -m pipeline.generation._nli_worker`, reading one JSON request
per line from stdin and writing one JSON response per line to stdout.

The protocol: {"pairs": [["premise", "hypothesis"], ...]} in,
{"probs": [[p_entailment, p_neutral, p_contradiction], ...]} out, one
row per pair, in the same order given, in the model's own id2label
order (0 entailment, 1 neutral, 2 contradiction), confirmed directly
against this checkpoint's config.json rather than assumed from the
class name. json.dumps's default ensure_ascii=True is kept for the same
reason _rerank_worker.py keeps it: pure ASCII on the wire regardless of
the pipe's own platform text encoding.

truncation="only_first" is the one detail worth stating plainly: this
checkpoint's own max_position_embeddings is 512, well under a
retrieved chunk's own measured size (llm.md: up to roughly 1,040 tokens
against a different tokenizer, the right order of magnitude), so a long
premise is truncated while the hypothesis, always a single synthesised
sentence and always much shorter, is kept whole. Measured directly
before this file was written: a premise repeated to 3,280 characters
still left a short hypothesis intact after truncation to 512 tokens.
"""

from __future__ import annotations

import json
import sys

from ..config import NLI_MODEL
from ..embedding.tokens import load_with_retry

_MAX_LENGTH = 512


def _load() -> tuple:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = load_with_retry(
        lambda: AutoTokenizer.from_pretrained(NLI_MODEL),
        "NLI tokenizer",
    )
    model = load_with_retry(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL, low_cpu_mem_usage=True,
        ),
        "NLI model weights",
    )
    model.eval()
    return tokenizer, model


def _score(tokenizer, model, pairs: list[list[str]]) -> list[list[float]]:
    import torch

    with torch.no_grad():
        inputs = tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            padding=True, truncation="only_first",
            return_tensors="pt", max_length=_MAX_LENGTH,
        )
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    return probs.tolist()


def main() -> None:
    tokenizer, model = _load()
    # The parent blocks on this exact line before sending its first
    # request, the same contract rerank.py's own _ensure_worker relies
    # on for its own worker.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            probs = _score(tokenizer, model, request["pairs"])
            print(json.dumps({"probs": probs}), flush=True)
        except Exception as error:  # noqa: BLE001
            # A malformed request or a scoring failure is reported back
            # rather than crashing the worker: the load already paid for
            # is worth keeping alive for the next request.
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
