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

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ..config import EMBED_BATCH_SIZE, EMBED_MODELS, SPARSE_EMBEDDING_CACHE_DIR
from . import embedder
from .tokens import get_tokenizer, max_content_length

_MODEL_KEY = "bge-m3"
_sparse_linear_cache: dict[str, torch.nn.Linear] = {}

Kind = str  # "query" or "passage"


def _load_sparse_linear(device: str | None = None) -> torch.nn.Linear:
    """The sparse head's own weight file, downloaded once per device, on
    the same device and dtype as the already-loaded dense backbone.

    Cached by resolved device, not by _MODEL_KEY alone: a bare-string key
    was correct while this only ever ran once per process at store-build
    time, always on whatever embedder._load_model auto-selected. Stage 7's
    retriever calls this from the same process as a dense leg that may
    already hold bge-m3 on a specific device, and returning a CUDA-resident
    layer for a CPU-resident model's hidden states would raise a device
    mismatch the moment the two met in one matmul.
    """
    model, resolved_device, dtype = embedder._load_model(_MODEL_KEY, device)
    if resolved_device not in _sparse_linear_cache:
        path = hf_hub_download(EMBED_MODELS[_MODEL_KEY], "sparse_linear.pt")
        state = torch.load(path, map_location=resolved_device)
        layer = torch.nn.Linear(model.config.hidden_size, 1)
        layer.load_state_dict(state)
        layer = layer.to(device=resolved_device, dtype=dtype)
        layer.eval()
        _sparse_linear_cache[resolved_device] = layer
    return _sparse_linear_cache[resolved_device]


def _sparse_batch(
    texts: list[str], device: str | None = None,
) -> list[tuple[list[int], list[float]]]:
    """One batch through the model, dense forward pass reused for sparse.

    device is not optional in spirit even though it defaults to None here:
    a caller sharing a process with embedder.embed_queries's own dense leg
    has to pass the same device that leg used, or this loads bge-m3 a
    second time on a different device and segfaults, the exact failure
    embedder.py's _load_model docstring now documents. Confirmed directly:
    retriever.py's first smoke test crashed exactly this way before this
    function took a device argument at all.
    """
    model, resolved_device, dtype = embedder._load_model(_MODEL_KEY, device)
    tokenizer = get_tokenizer(_MODEL_KEY)
    sparse_linear = _load_sparse_linear(device)
    full_length = max_content_length(_MODEL_KEY)

    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=full_length, return_tensors="pt",
    ).to(resolved_device)

    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
        token_weights = torch.relu(sparse_linear(hidden)).squeeze(-1)

    input_ids = encoded["input_ids"]
    vocab_size = model.config.vocab_size
    sparse_embedding = torch.zeros(
        input_ids.shape[0], vocab_size, dtype=token_weights.dtype, device=resolved_device,
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


# --- disk cache, content-addressed like embedder._cache_key ------------------
#
# Absent until now on purpose: the expensive part, loading and tokenizing,
# was already paid for by the dense embedding every store-build call ran
# alongside, and that call happened once per variant. Stage 7's evaluate.py
# calls embed_sparse_queries once per golden question for every grid cell,
# the same per-query cost embedder.py's own cache exists to avoid paying
# twice, so this gets the same treatment now that a caller actually repeats
# it.

_CACHE_FORMULA_VERSION = 1  # bump if the sparse formula in _sparse_batch changes


def _cache_key(text: str, kind: Kind, device: str | None = None) -> str:
    """device resolves through embedder._device_and_dtype before hashing,
    the same reason embedder._cache_key does it: a CPU-fp32 sparse vector
    and a GPU-fp16 one for the same text are not guaranteed identical, and
    a cache that could not tell them apart would silently erase the
    distinction retriever.py's device policy depends on being real.
    """
    resolved_device, dtype = embedder._device_and_dtype(device)
    material = json.dumps(
        {
            "model": EMBED_MODELS[_MODEL_KEY],
            "kind": kind,
            "formula_version": _CACHE_FORMULA_VERSION,
            "max_length": max_content_length(_MODEL_KEY),
            "device": resolved_device,
            "dtype": str(dtype),
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return SPARSE_EMBEDDING_CACHE_DIR / f"{key}.npz"


def _load_cached(
    text: str, kind: Kind, device: str | None = None,
) -> tuple[list[int], list[float]] | None:
    path = _cache_path(_cache_key(text, kind, device))
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["indices"].tolist(), data["values"].tolist()


def _store_cached(
    text: str, kind: Kind, result: tuple[list[int], list[float]],
    device: str | None = None,
) -> None:
    """One (indices, values) pair to disk. .npz rather than JSON, the same
    reason embedder._store_cached gives for .npy: nothing here is meant to
    be read by eye, so it gets the smaller, faster format.
    """
    SPARSE_EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    indices, values = result
    np.savez(
        _cache_path(_cache_key(text, kind, device)),
        indices=np.asarray(indices, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
    )


def _embed_sparse(
    texts: list[str], kind: Kind, batch_size: int, use_cache: bool,
    device: str | None = None,
) -> list[tuple[list[int], list[float]]]:
    results: list[tuple[list[int], list[float]] | None] = [None] * len(texts)
    to_compute: list[int] = []
    for i, text in enumerate(texts):
        cached = _load_cached(text, kind, device) if use_cache else None
        if cached is not None:
            results[i] = cached
        else:
            to_compute.append(i)

    for start in range(0, len(to_compute), batch_size):
        batch_indices = to_compute[start:start + batch_size]
        batch_texts = [texts[i] for i in batch_indices]
        for i, result in zip(batch_indices, _sparse_batch(batch_texts, device)):
            results[i] = result
            if use_cache:
                _store_cached(texts[i], kind, result, device)

    return results


def embed_sparse_passages(
    texts: list[str], batch_size: int = EMBED_BATCH_SIZE, use_cache: bool = True,
    device: str | None = None,
) -> list[tuple[list[int], list[float]]]:
    """Sparse (indices, values) pairs for every passage text, input order.

    device is None for every existing caller, store.qdrant's build at
    index time, which keeps its auto-detect (GPU) behaviour unchanged.
    """
    return _embed_sparse(texts, "passage", batch_size, use_cache, device)


def embed_sparse_queries(
    texts: list[str], batch_size: int = EMBED_BATCH_SIZE, use_cache: bool = True,
    device: str | None = None,
) -> list[tuple[list[int], list[float]]]:
    """Sparse (indices, values) pairs for every query text, input order.

    BGE-M3's card says it needs no query/passage distinction for its
    sparse head, so this computes identically to embed_sparse_passages.
    A separate name rather than one function both sides call keeps that
    fact visible at the call site, following embedder.embed_queries's own
    reasoning: if that ever stops being true, the place that assumed
    otherwise is the query side, not a shared function nobody thought to
    check.

    device matters here in a way it does not for embed_sparse_passages:
    retriever.py calls this from the same process as a dense query leg
    that may already hold bge-m3 on a specific device, and the two have to
    agree or this loads a second checkpoint copy and segfaults.
    """
    return _embed_sparse(texts, "query", batch_size, use_cache, device)
