"""The model contract, and the only place a model is loaded.

Two candidates, one XLM-RoBERTa-large backbone each, so the library carries no
Arabic-specific behaviour of its own: the weights decide that, not this code.
What the library does decide is whether the one difference between the two
that could silently swing a bake-off stays visible. BGE-M3 pools the CLS
token and takes a query verbatim; e5-large pools a masked mean and *requires*
"query: " and "passage: " prefixes on every input, a training-time convention
rather than a formatting nicety. Loading both through sentence-transformers
would hide that behind one .encode() call and let a missing prefix look
exactly like e5 being worse at Arabic. Written out here, it is reviewable, and
BGE-M3's sparse head becomes fifteen lines in sparse.py instead of a second
heavy dependency.

Only one model is ever loaded in this process. embed_texts loads on first
use; release() frees it before a second one is asked for. That is stronger
than a 4 GB memory limit: loading a second distinct checkpoint here after
the first was loaded and released reproducibly crashes this process,
confirmed symmetric in either order and confirmed unrelated to available
memory (PyTorch's own accounting showed the card mostly free when it
happened). A fresh OS process is the isolation boundary that actually
holds, so anything that needs both models runs each in its own subprocess;
see bakeoff.py's worker mode.

What this module does not do: it does not decide which model wins, and it
does not orchestrate more than one model at a time. That is bakeoff.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

# torch first, numpy second: on Windows the two bundle different native BLAS
# builds, and importing numpy before torch touches CUDA is a known source of
# native-library conflicts that surface as a hard crash rather than a Python
# exception. Order matters here in a way it does not for any other pair of
# imports in this project.
import torch
import numpy as np

from ..config import EMBED_BATCH_SIZE, EMBED_MODELS, EMBEDDING_CACHE_DIR
from .tokens import get_tokenizer, load_with_retry, max_content_length

Kind = str  # "query" or "passage"


# --- pooling, one function per model, chosen by _POOLING below ---------------

def _pool_cls(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """BGE-M3's dense pooling: the CLS token's own hidden state, position 0.

    Confirmed against FlagEmbedding's own reference implementation rather than
    assumed: BGE-M3's default sentence_pooling_method is "cls", and its
    _dense_embedding method is exactly last_hidden_state[:, 0].
    attention_mask is accepted unused, only so this function has the same
    signature as _pool_mean and a caller can select between them by name
    rather than by a branch of its own.
    """
    return hidden_state[:, 0]


def _pool_mean(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """e5-large's dense pooling: masked mean over real tokens only.

    Copied from the model card's own average_pool function rather than a
    generic mean, because a mean taken over the padding too would shrink
    every embedding by however much padding a batch happened to carry, in
    proportion to batch composition rather than to the text itself, and it
    would still return a vector that looks like a working embedding.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    masked = hidden_state.masked_fill(mask == 0, 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).to(hidden_state.dtype)


_POOLING: dict[str, Callable] = {
    "bge-m3": _pool_cls,
    "e5-large": _pool_mean,
}

# What each model's card asks to be prepended before pooling. BGE-M3's card
# states it no longer requires adding instructions to the queries, so both
# are empty; e5-large's card supplies these as part of the input text itself,
# not as a formatting nicety, but as tokens the model was contrastively
# trained to expect. Omitting them is a known, quiet recall loss that would
# be indistinguishable from e5 simply being weaker at Arabic, which is exactly
# the asymmetry this module exists to keep visible.
_PREFIXES: dict[str, dict[str, str]] = {
    "bge-m3": {"query": "", "passage": ""},
    "e5-large": {"query": "query: ", "passage": "passage: "},
}


# --- model lifecycle, one resident at a time ----------------------------------

_model_cache: dict[tuple[str, str], tuple] = {}


def _device_and_dtype(device: str | None = None) -> tuple[str, torch.dtype]:
    """Resolve a device request to (device, dtype).

    device=None is the original, sole policy every index-time call site
    still uses: auto-detect, cuda if it is there. An explicit device is
    what stage 7's retriever asks for at query time, when Ollama may
    already be resident on the card and the architecture plan's own
    component table says the embedder should stay off it during a query;
    see LEARNING/retrieval.md for what that choice actually cost, measured
    rather than assumed.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def _load_model(model_key: str, device: str | None = None):
    """Load and cache one model on one device. Loading a second distinct
    checkpoint, or the same checkpoint a second time on any device, while
    one is already resident is a caller error on this machine: bakeoff.py
    calls release() between candidates, each in its own subprocess.

    That last clause is not a hedge. Confirmed directly while building
    stage 7's device policy: a second `AutoModel.from_pretrained` call in
    a process that has already moved a model to CUDA once segfaults even
    when the second load's own destination is CPU and never calls
    `.to("cuda")` at all. embedding.md's original finding described the
    fault as "a second CUDA transfer"; this is narrower and worse, a
    second *load* full stop, so a CPU-versus-GPU comparison for the same
    model has to run as two separate subprocesses, the same isolation
    bakeoff.py already needed for two different checkpoints, never as one
    load-release-reload sequence in this process.

    The cache key is (model_key, device), not model_key alone, so a
    CPU-resident load and a GPU-resident load of the same checkpoint
    cannot collide and silently shadow one another if a caller ever did
    hold both, however briefly.

    Casting to fp16 and moving to CUDA happen as two separate calls, not
    one. Measured directly on this 4 GB card, on both candidates: a combined
    `.to(device="cuda", dtype=torch.float16)` on a freshly loaded fp32 model
    reproducibly threw a CUDA OOM citing free memory well above what it
    asked to allocate, the signature of allocator fragmentation rather than
    a real shortage, since `nvidia-smi` showed the card fully idle
    immediately after. Casting to fp16 while still on CPU first means only
    the already-halved model, not the fp32 original, ever has to fit on the
    card, and that path has not reproduced the failure.
    """
    resolved_device, dtype = _device_and_dtype(device)
    cache_key = (model_key, resolved_device)
    if cache_key not in _model_cache:
        from transformers import AutoModel

        # load_with_retry: see tokens.py's own docstring for what this
        # guards against, found during stage 8 against this exact call.
        model = load_with_retry(
            lambda: AutoModel.from_pretrained(EMBED_MODELS[model_key]),
            f"model weights for {model_key}",
        )
        if dtype is not torch.float32:
            model = model.to(dtype=dtype)
        model = model.to(resolved_device)
        model.eval()
        _model_cache[cache_key] = (model, resolved_device, dtype)
        print(f"  loaded {model_key} on {resolved_device} ({dtype})")
    return _model_cache[cache_key]


def release(model_key: str | None = None) -> None:
    """Free a resident model's GPU memory. None releases whichever is loaded,
    on every device it happens to be cached on.

    Two XLM-R-large models do not share a 4 GB card. Calling this between
    candidates is what makes the bake-off possible on this machine rather
    than a design that assumes a bigger one.

    gc.collect() runs before empty_cache(), not after: popping the model
    from the cache drops its refcount, but a module holding an internal
    reference cycle, real in PyTorch, is not necessarily reclaimed by
    refcounting alone, and empty_cache() only returns memory PyTorch's own
    allocator already knows is free. Skipping the collect once produced a
    reproducible segfault immediately after loading a second model, on a
    card nvidia-smi reported fully idle: memory the allocator had not yet
    been told to give back.
    """
    keys = (
        [key for key in _model_cache if key[0] == model_key] if model_key
        else list(_model_cache)
    )
    for key in keys:
        _model_cache.pop(key, None)
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# --- disk cache, content-addressed like ocr.py and llm/cache.py --------------

def _cache_key(text: str, model_key: str, kind: Kind, device: str | None = None) -> str:
    """Hash of everything that determines the vector.

    Content-addressed on the text itself, not on a chunk id: the none,
    template and llm variants share no text once a prefix is added, so this
    never needs to know which variant a chunk came from, and a chunk edited
    upstream simply gets a new key rather than a stale hit.

    device resolves through _device_and_dtype before hashing, not stored as
    whatever the caller literally passed, so a None (auto) request and an
    explicit request that happen to resolve the same way still share a
    cache entry, while a genuine CPU-fp32 vector and a GPU-fp16 vector for
    the same text never can: stage 7's device-policy measurement depends on
    comparing exactly those two, and a shared cache entry would silently
    erase the difference it exists to measure.
    """
    resolved_device, dtype = _device_and_dtype(device)
    material = json.dumps(
        {
            "model": EMBED_MODELS[model_key],
            "pooling": _POOLING[model_key].__name__,
            "prefix": _PREFIXES[model_key][kind],
            "max_length": max_content_length(model_key),
            "device": resolved_device,
            "dtype": str(dtype),
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return EMBEDDING_CACHE_DIR / f"{key}.npy"


def _load_cached(
    text: str, model_key: str, kind: Kind, device: str | None = None
) -> np.ndarray | None:
    path = _cache_path(_cache_key(text, model_key, kind, device))
    if not path.exists():
        return None
    return np.load(path)


def _store_cached(
    text: str, model_key: str, kind: Kind, vector: np.ndarray,
    device: str | None = None,
) -> None:
    """Write one vector to disk, .npy rather than JSON.

    llm/cache.py stores its prompt beside the answer because Arabic prompts
    are only checkable by opening the file in an editor. A vector has nothing
    to read by eye either way, so it gets the format that is smaller and
    faster to load back, not the one that is inspectable.
    """
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_cache_path(_cache_key(text, model_key, kind, device)), vector)


# --- encoding ------------------------------------------------------------------

def _encode_uncached(
    texts: list[str], model_key: str, kind: Kind, batch_size: int,
    device: str | None = None,
) -> np.ndarray:
    """Run the model over texts with no cache lookups at all: pure compute.

    Batch size only ever shrinks within one call, on an OOM, and never grows
    back, so a card that had to fall back once is not asked to try the
    original size again on the very next batch.
    """
    model, device, dtype = _load_model(model_key, device)
    tokenizer = get_tokenizer(model_key)
    pool = _POOLING[model_key]
    prefix = _PREFIXES[model_key][kind]
    # max_content_length is already the safe *total* sequence length, CLS
    # and SEP included; see its docstring for why adding SPECIAL_TOKEN_OVERHEAD
    # here overflows the position embedding table on a RoBERTa-family model
    # rather than merely wasting two tokens of budget.
    full_length = max_content_length(model_key)

    prefixed = [prefix + t for t in texts]
    vectors: list[np.ndarray] = []
    i = 0
    current_batch = batch_size
    while i < len(prefixed):
        batch = prefixed[i:i + current_batch]
        try:
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=full_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output = model(**encoded)
            pooled = pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
            i += current_batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            print(f"  OOM for {model_key} at batch {batch_size}, "
                  f"retrying at {current_batch}")

    return np.concatenate(vectors, axis=0)


def embed_texts(
    texts: list[str], model_key: str, kind: Kind,
    batch_size: int = EMBED_BATCH_SIZE, use_cache: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Unit-norm vectors for every text, cache-aware, in the input order.

    kind is "query" or "passage": e5-large needs a different prefix for each
    and BGE-M3's card says it needs neither, so the distinction is carried
    through the call rather than left for a caller to guess which applies.

    device is None everywhere except retriever.py's query-time call sites,
    which is what keeps every existing caller, the bake-off, run_embed, the
    store build, on the original auto-detect behaviour with no change here.
    """
    vectors: list[np.ndarray | None] = [None] * len(texts)
    to_compute: list[int] = []
    for i, text in enumerate(texts):
        cached = _load_cached(text, model_key, kind, device) if use_cache else None
        if cached is not None:
            vectors[i] = cached
        else:
            to_compute.append(i)

    if to_compute:
        fresh = _encode_uncached(
            [texts[i] for i in to_compute], model_key, kind, batch_size, device
        )
        for idx, vector in zip(to_compute, fresh):
            vectors[idx] = vector
            if use_cache:
                _store_cached(texts[idx], model_key, kind, vector, device)

    return np.stack(vectors)


def embed_passages(texts: list[str], model_key: str, **kwargs) -> np.ndarray:
    return embed_texts(texts, model_key, "passage", **kwargs)


def embed_queries(texts: list[str], model_key: str, **kwargs) -> np.ndarray:
    return embed_texts(texts, model_key, "query", **kwargs)


# --- gate 1: pooling verified before it decides anything ---------------------

# The plan called for reproducing each model card's published example
# similarities to 1e-3. Checked directly against both cards while building
# this module: neither publishes one. BGE-M3's card runs a four-sentence
# example through sentence-transformers and shows only the [4, 4] output
# shape, no floats. e5-large's usage snippet computes
# scores = (embeddings[:2] @ embeddings[2:].T) * 100 and prints
# scores.tolist(), but the README does not show what that call actually
# printed. There is no number here to reproduce.
#
# What both examples do carry is a relationship they were plainly chosen to
# demonstrate: a near-duplicate sentence should score above a related one,
# which should score above an unrelated one, and a query should score its
# true passage above a mismatched one. That relationship is checkable without
# inventing a threshold, and Arabic sentences exercise it directly rather
# than through an English or Chinese proxy, since Arabic is what this
# pipeline actually retrieves. Text below is original, written for this
# check, not quoted from either card.
_BGE_ORDINAL_AR = (
    "ذلك شخص سعيد",       # that is a happy person, the anchor
    "ذلك شخص سعيد جدا",    # that is a very happy person, near-duplicate
    "ذلك كلب سعيد",       # that is a happy dog, related
    "اليوم يوم مشمس",      # today is a sunny day, unrelated
)

_E5_QUERY_AR = "كم غراما من البروتين يجب أن تتناول المرأة يوميا"
_E5_PASSAGE_MATCH_AR = (
    "يوصى عموما بأن تتناول المرأة البالغة حوالي 46 غراما من "
    "البروتين يوميا حسب الإرشادات الغذائية"
)
_E5_PASSAGE_MISMATCH_AR = (
    "الطماطم من الخضروات الشائعة في السلطات والصلصات"
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _verify_bge_ordinal() -> list[str]:
    """Anchor, near-duplicate, related, unrelated: the ordering must hold."""
    failures = []
    vectors = embed_passages(list(_BGE_ORDINAL_AR), "bge-m3", use_cache=False)
    anchor, near_dup, related, unrelated = vectors
    sim_dup = _cosine(anchor, near_dup)
    sim_rel = _cosine(anchor, related)
    sim_unrel = _cosine(anchor, unrelated)
    if not (sim_dup > sim_rel > sim_unrel):
        failures.append(
            f"bge-m3 ordinal check failed: near-duplicate {sim_dup:.4f}, "
            f"related {sim_rel:.4f}, unrelated {sim_unrel:.4f}, expected "
            f"strictly decreasing"
        )
    return failures


def _verify_e5_prefix_matters() -> list[str]:
    """The true passage must outscore the mismatched one, prefixed as the
    card specifies; and the margin has to shrink once the prefixes are
    stripped, which is the direct evidence that the prefix is doing real
    work rather than being a no-op this gate would otherwise miss.
    """
    failures = []

    q = embed_queries([_E5_QUERY_AR], "e5-large", use_cache=False)[0]
    p_match, p_mismatch = embed_passages(
        [_E5_PASSAGE_MATCH_AR, _E5_PASSAGE_MISMATCH_AR], "e5-large",
        use_cache=False,
    )
    margin_prefixed = _cosine(q, p_match) - _cosine(q, p_mismatch)
    if margin_prefixed <= 0:
        failures.append(
            f"e5-large retrieval check failed: true passage did not "
            f"outscore the mismatched one (margin {margin_prefixed:.4f}) "
            f"with query:/passage: prefixes applied"
        )

    q_bare = embed_texts([_E5_QUERY_AR], "e5-large", "query", use_cache=False)
    # Re-embed with an empty prefix table entry to isolate the prefix's
    # effect, rather than editing global state mid-check.
    original_prefixes = _PREFIXES["e5-large"]
    _PREFIXES["e5-large"] = {"query": "", "passage": ""}
    try:
        q_bare = embed_queries([_E5_QUERY_AR], "e5-large", use_cache=False)[0]
        p_match_bare, p_mismatch_bare = embed_passages(
            [_E5_PASSAGE_MATCH_AR, _E5_PASSAGE_MISMATCH_AR], "e5-large",
            use_cache=False,
        )
    finally:
        _PREFIXES["e5-large"] = original_prefixes
    margin_bare = _cosine(q_bare, p_match_bare) - _cosine(q_bare, p_mismatch_bare)

    if margin_bare >= margin_prefixed:
        failures.append(
            f"e5-large prefix ablation failed: stripping query:/passage: "
            f"did not shrink the retrieval margin (prefixed {margin_prefixed:.4f}, "
            f"bare {margin_bare:.4f}). Either the prefix is not being applied, "
            f"or this check no longer isolates its effect."
        )
    return failures


def verify_against_model_card(model_key: str) -> list[str]:
    """One model's pooling, checked before it embeds a corpus chunk. Empty
    when clean. See the module comment above the check functions for why
    this checks a relationship rather than a published number.

    Takes one model rather than both, and that is load-bearing, not a
    convenience. Loading a second distinct checkpoint in the same process
    after the first was loaded and released reproducibly segfaults on this
    machine, confirmed symmetric in either order and confirmed unrelated to
    available memory: PyTorch's own accounting showed the card at 3.4 of
    4.3 GB free, gc.collect() plus empty_cache() run, immediately before the
    second load crashed anyway. A fresh OS process is the only isolation
    boundary that actually held. bakeoff.py's worker calls this once per
    model, each in its own subprocess, rather than calling it twice here.
    """
    if model_key == "bge-m3":
        return _verify_bge_ordinal()
    if model_key == "e5-large":
        return _verify_e5_prefix_matters()
    raise ValueError(f"no model-card check defined for {model_key!r}")
