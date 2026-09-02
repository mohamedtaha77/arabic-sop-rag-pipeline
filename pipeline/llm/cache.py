"""How a call is not made twice.

Stage 4 generates a context prefix for every chunk in the corpus, which is
roughly 350 calls on the current count and closer to 800 once the three-way
comparison is built. At local speeds that is the difference between a re-run
costing seconds and costing an hour, and stage 4 is a stage that gets re-run.
So the cache is built here, before the stage that needs it, for the same reason
the ledger is.

The mechanism copies ocr.py deliberately, so the project has one caching idea
rather than two: a short sha256 over the inputs, keying a file on disk. There,
the key includes the render DPI because output differs by resolution. Here it
includes every parameter that changes the answer, for exactly that reason.

Caching a sampled generation would normally be unsound, since the same prompt
can legitimately return different text. It is sound here only because the seed
is part of the key: a given key names one reproducible answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import LLM_CACHE_DIR
from .client import Response, chat


def cache_key(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    response_format: dict[str, str] | None,
) -> str:
    """Short hash of everything that determines the answer.

    Deliberately not the endpoint URL or the timeout. Neither changes what the
    model says, and folding them in would invalidate the whole cache the first
    time the server moved to a different port.
    """
    material = json.dumps(
        {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def cache_path(key: str) -> Path:
    """Location of one cached call."""
    return LLM_CACHE_DIR / f"{key}.json"


def load(key: str) -> Response | None:
    """Return a cached call, or None if it was never made.

    ``cached`` is set on the way out rather than stored, so the flag always
    describes this run rather than the run that wrote the file.
    """
    path = cache_path(key)
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    return Response(**stored["response"], cached=True)


def store(key: str, request: dict[str, Any], response: Response) -> None:
    """Write one call to disk, prompt included.

    The prompt is stored beside the answer even though nothing reads it back.
    Eight hundred files named by hash are otherwise undebuggable, and the
    prompts here are Arabic, which means the only practical way to check one is
    to open the file in an editor, the same reason ingestion writes its output
    with ensure_ascii off.
    """
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"request": request, "response": asdict(response)}
    payload["response"].pop("cached")
    with open(cache_path(key), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def cached_chat(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 0,
    response_format: dict[str, str] | None = None,
    use_cache: bool = True,
) -> Response:
    """client.chat, served from disk when the same call was made before.

    ``use_cache=False`` exists for probe.py. Tokens per second cannot be
    measured from a cache hit, and a benchmark that silently measured one would
    report this machine as faster than any machine has ever been.
    """
    key = cache_key(messages, model, temperature, max_tokens, seed, response_format)

    if use_cache:
        hit = load(key)
        if hit is not None:
            return hit

    response = chat(
        messages,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        response_format=response_format,
    )
    store(
        key,
        {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
        },
        response,
    )
    return response
