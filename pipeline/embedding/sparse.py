"""BGE-M3's sparse head: relu(W*h), max over repeated tokens, specials zeroed.

BGE-M3 ships its sparse (lexical) head as a small separate weight file,
sparse_linear.pt, alongside the dense encoder's pytorch_model.bin: a plain
nn.Linear(1024, 1), confirmed by downloading and inspecting the actual file
rather than assumed. The formula matches FlagEmbedding's own
_sparse_embedding method, checked against its source before writing this:
a ReLU-activated linear layer over the same hidden states the dense head
pools, CLS/EOS/PAD/UNK token weights zeroed, and a max taken over positions
sharing a token id so a repeated token contributes once at its strongest
weight rather than once per occurrence.

Built only because both conditions the plan named were met: Qdrant's local
store accepts a sparse vector (store/probe.py answered yes), and BGE-M3 won
the bake-off. e5-large has no sparse head; nothing here ever runs for it.

Reuses embedder._load_model("bge-m3") rather than loading the backbone a
second time: a second forward pass through an already-resident model is
cheap and safe. What crashed repeatedly while building this stage was a
second weight *transfer* to the GPU, never a second inference call, and
this module only ever does the latter.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

from ..config import EMBED_BATCH_SIZE, EMBED_MODELS
from . import embedder
from .tokens import get_tokenizer, max_content_length

_MODEL_KEY = "bge-m3"
_sparse_linear_cache: dict[str, torch.nn.Linear] = {}


def _load_sparse_linear() -> torch.nn.Linear:
    """The sparse head's own weight file, downloaded once, on the same
    device and dtype as the already-loaded dense backbone.
    """
    if _MODEL_KEY not in _sparse_linear_cache:
        path = hf_hub_download(EMBED_MODELS[_MODEL_KEY], "sparse_linear.pt")
        model, device, dtype = embedder._load_model(_MODEL_KEY)
        state = torch.load(path, map_location=device)
        layer = torch.nn.Linear(model.config.hidden_size, 1)
        layer.load_state_dict(state)
        layer = layer.to(device=device, dtype=dtype)
        layer.eval()
        _sparse_linear_cache[_MODEL_KEY] = layer
    return _sparse_linear_cache[_MODEL_KEY]


def _sparse_batch(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """One batch through the model, dense forward pass reused for sparse."""
    model, device, dtype = embedder._load_model(_MODEL_KEY)
    tokenizer = get_tokenizer(_MODEL_KEY)
    sparse_linear = _load_sparse_linear()
    full_length = max_content_length(_MODEL_KEY)

    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=full_length, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
        token_weights = torch.relu(sparse_linear(hidden)).squeeze(-1)

    input_ids = encoded["input_ids"]
    vocab_size = model.config.vocab_size
    sparse_embedding = torch.zeros(
        input_ids.shape[0], vocab_size, dtype=token_weights.dtype, device=device,
    )
    sparse_embedding.scatter_reduce_(
        dim=-1, index=input_ids, src=token_weights, reduce="amax", include_self=True,
    )

    # Special tokens carry no lexical meaning; a repeated pad id would
    # otherwise dominate every short sequence's sparse vector.
    unused_ids = {
        tokenizer.cls_token_id, tokenizer.eos_token_id,
        tokenizer.pad_token_id, tokenizer.unk_token_id,
    }
    for token_id in unused_ids:
        if token_id is not None:
            sparse_embedding[:, token_id] = 0.0

    results = []
    for row in sparse_embedding:
        nonzero = torch.nonzero(row, as_tuple=True)[0]
        indices = nonzero.cpu().tolist()
        values = row[nonzero].float().cpu().tolist()
        results.append((indices, values))
    return results


def embed_sparse_passages(
    texts: list[str], batch_size: int = EMBED_BATCH_SIZE,
) -> list[tuple[list[int], list[float]]]:
    """Sparse (indices, values) pairs for every text, in input order.

    No disk cache of its own: the expensive part, loading and tokenizing,
    is already paid for by the dense embedding this always runs alongside,
    and BGE-M3's sparse head only ever computes at store-build time, once
    per variant, never repeated the way the bake-off's scoring calls were.
    """
    results: list[tuple[list[int], list[float]]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        results.extend(_sparse_batch(batch))
    return results
