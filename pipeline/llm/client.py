"""One call to the local model endpoint, and back.

Every stage from here on spends model calls, and each needs the same four
things: the text, what it cost in tokens, how long it took, and a loud failure
when something is wrong. That is this module and nothing else.

The endpoint is OpenAI-compatible, served by Ollama on localhost. Talking to it
over stdlib urllib rather than a client library is a deliberate choice: the
whole transport is one function, there is no flaky network between here and
localhost to ride out, and requirements.txt keeps its three justified entries.
Stage 10 installs the openai SDK for RAGAS alone, pointed at the same base URL.

What this module does not do: no streaming, no step semantics, no cost, no
cache, no prompts. A wrapper that already knew what a synthesiser prompt looked
like would be guessing at a stage that has not been measured yet.

It also does not retry. On localhost there is no transient failure to ride out,
and a retry that quietly rides over a dead server only makes the failure appear
later and somewhere less informative.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..config import GENERATOR_MODEL, GROQ_API_KEYS, JUDGE_MODEL, LLM_BASE_URL, LLM_TIMEOUT


# Stage 10's own fallback for a real, repeated finding: this machine's
# 4 GB card reliably fails to allocate CUDA memory for the larger-context
# calls Reranking, Multi-Query, Decomposition and CRAG's own grading
# produce (measured directly: the same 13 questions failed identically,
# same exact CUDA allocation errors, across more than fifty attempts
# spread over two full harness runs, which rules out transient
# contention and means retrying with the same request can never
# succeed). Ollama's own OpenAI-compatible endpoint silently ignores an
# "options" field (confirmed directly: a live options.num_gpu=0 call
# through it still showed the full model resident in VRAM afterward, per
# /api/ps); only its native /api/chat endpoint honours num_gpu, so this
# is a genuinely separate code path, not a parameter on the existing one.
# Module-level and off by default: every caller that never sets this
# keeps using the endpoint every earlier stage already measured and
# shipped against, and a caller that does set it accepts a real,
# measured cost, CPU inference is markedly slower than partial GPU
# offload, in exchange for a call this machine's card cannot otherwise
# complete at all.
_FORCE_CPU = False

# Ollama does not care, but Groq sits behind Cloudflare and Cloudflare
# blocks urllib's own default User-Agent ("Python-urllib/3.x") outright:
# measured directly, a request with that default UA came back HTTP 403,
# Cloudflare's own error code 1010, before the request ever reached
# Groq's API (the same key, same payload, worked immediately over curl,
# whose UA looks nothing like a scraper's). One header fixes it for both
# endpoints, so it is sent unconditionally rather than branched on
# LLM_PROVIDER.
_HEADERS = {"Content-Type": "application/json", "User-Agent": "gbg-rag-pipeline/1.0"}

# Which of GROQ_API_KEYS the next call starts from, sticky across calls
# rather than reset to key 0 every time. Verified directly this was the
# real bug behind an apparent shared rate limit: starting every call at
# key 0 concentrates a burst of many questions onto that one key first,
# draining its own free-tier budget (measured, 980/1000 requests left on
# key 0 against 999/1000 on the other three, after the same short test
# run) before the others are ever touched, and a fast enough burst can
# cascade through all four in sequence. This looks identical to one
# shared organisation-level limit from the outside, which is what it was
# first mistaken for, but checking each key's own remaining-requests
# header independently (not just the org id quoted in whichever key's
# error happened to be logged last) showed four different counts, not
# one shared one. Spreading the starting point fixes the actual cause.
_groq_key_index = 0


def _ordered_groq_keys() -> list[str | None]:
    """This call's key order, starting from wherever rotation last left
    off. [None] when Groq is not configured, so the loop in chat() runs
    its one Ollama-shaped iteration unchanged either way.
    """
    if not GROQ_API_KEYS:
        return [None]
    n = len(GROQ_API_KEYS)
    start = _groq_key_index % n
    return [GROQ_API_KEYS[(start + i) % n] for i in range(n)]


def set_force_cpu(value: bool) -> None:
    """Toggle every subsequent chat() call onto Ollama's native /api/chat
    endpoint with num_gpu forced to 0, or back to the normal
    OpenAI-compatible path. A caller that turns this on is responsible
    for turning it back off once its own retry is done: leaving it on
    would silently slow down every later, otherwise-fine call for a
    problem specific to the one that needed it.
    """
    global _FORCE_CPU
    _FORCE_CPU = value


class LLMError(RuntimeError):
    """A call that did not produce trustworthy text.

    One exception type rather than a hierarchy, because no caller branches on
    the kind of failure. What matters is that the message names the fix, so
    the reader is not left diagnosing a stack trace.
    """


@dataclass
class Response:
    """One completed call.

    ``latency_s`` is the wall clock of the HTTP request. When this response is
    replayed from cache it keeps the latency the original call measured, and
    ``cached`` says so; see cache.py for why it is not reported as zero.
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str
    cached: bool = False


def chat(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 0,
    response_format: dict[str, str] | None = None,
) -> Response:
    """Send one chat completion and return it with its token counts.

    Temperature and seed default to a deterministic pair. Stage 9 requires
    temperature 0 for the synthesiser, and every other step benefits from a
    re-run that reproduces rather than drifts, which is what makes the
    technique ablation in the report comparable between runs.
    """
    if _FORCE_CPU:
        # Ollama's own native endpoint, not the OpenAI-compatible one:
        # see set_force_cpu's own docstring for why num_gpu has no effect
        # through /v1/chat/completions.
        url = f"{LLM_BASE_URL.removesuffix('/v1')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "seed": seed, "num_gpu": 0},
        }
        if response_format is not None:
            # Ollama's native "format": "json" is the equivalent of the
            # OpenAI-compatible {"type": "json_object"} this project's
            # own callers already pass; no caller here uses a schema-typed
            # response_format, so this one mapping covers every real use.
            payload["format"] = "json"
    else:
        url = f"{LLM_BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if model.startswith("openai/gpt-oss"):
            # Groq's gpt-oss family reasons before answering, and that
            # reasoning is billed against the same max_tokens budget as
            # the visible answer, invisibly: measured directly, a
            # trivial one-word reply cost 70 completion tokens at the
            # default reasoning effort. Every one of this pipeline's own
            # per-step budgets (router.py's own short JSON, CRAG's
            # grading call, entail.py's single verdict) was sized for a
            # local 3B model that goes straight to the answer, and one
            # of those was measured hitting exactly this ceiling: cut
            # off after 150 tokens with no answer text at all. "low" cut
            # the same trivial case to 22 tokens, the least reasoning
            # this model will accept — "none" is rejected outright by
            # Groq's own API ("must be one of low, medium, or high").
            # Silently absent for every other model, including the judge
            # (allam-2-7b): Groq rejects this field outright for a model
            # that does not support it, so it is only ever sent to the
            # one model family measured to need it.
            payload["reasoning_effort"] = "low"

    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.perf_counter()

    # Ollama takes no key at all (GROQ_API_KEYS is empty for the local
    # provider), so this loop runs its one iteration with a plain header
    # and behaves exactly as before. For Groq, a 429 means this specific
    # key hit its free-tier rate limit, not that the question is
    # unanswerable, so the same request is retried on the next configured
    # key rather than surfacing a failure the caller cannot do anything
    # about. Any other error still raises immediately: a 4xx that is not
    # a rate limit, or a network failure, means the next key would not
    # help either.
    global _groq_key_index
    key_attempts = _ordered_groq_keys()
    for attempt, key in enumerate(key_attempts):
        headers = dict(_HEADERS)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            url, data=body_bytes, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as handle:
                body = json.loads(handle.read().decode("utf-8"))
            if GROQ_API_KEYS and key is not None:
                _groq_key_index = GROQ_API_KEYS.index(key)
            break
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < len(key_attempts) - 1:
                if GROQ_API_KEYS and key is not None:
                    _groq_key_index = (GROQ_API_KEYS.index(key) + 1) % len(GROQ_API_KEYS)
                continue
            if error.code == 429:
                raise LLMError(
                    f"all {len(key_attempts)} configured Groq keys are "
                    f"rate-limited (most recently: {_http_message(error, model)})"
                ) from error
            raise LLMError(_http_message(error, model)) from error
        except urllib.error.URLError as error:
            raise LLMError(_url_message(error)) from error
        except TimeoutError as error:
            raise LLMError(
                f"{model} did not answer within {LLM_TIMEOUT}s. On a 4 GB card a "
                f"long prompt runs mostly on CPU; raise LLM_TIMEOUT in config.py "
                f"or shorten the context."
            ) from error
    latency = time.perf_counter() - started

    return _parse_native(body, model, latency) if _FORCE_CPU else _parse(body, model, latency)


def list_models() -> list[str]:
    """Every model tag the server can serve right now.

    Here rather than in probe.py so that urllib stays in one file. Used to tell
    "the server is down" apart from "the server is up and the tag is wrong",
    which are the two setup failures worth separating.
    """
    headers = dict(_HEADERS)
    if GROQ_API_KEYS:
        headers["Authorization"] = f"Bearer {GROQ_API_KEYS[0]}"
    request = urllib.request.Request(f"{LLM_BASE_URL}/models", method="GET", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as handle:
            body = json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LLMError(_http_message(error, "models")) from error
    except urllib.error.URLError as error:
        raise LLMError(_url_message(error)) from error
    return sorted(entry["id"] for entry in body.get("data", []))


def _parse(body: dict[str, Any], model: str, latency: float) -> Response:
    """Turn the endpoint's JSON into a Response, refusing anything doubtful.

    Both refusals here are cases where returning something plausible is worse
    than raising.
    """
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "")
    except (KeyError, IndexError) as error:
        raise LLMError(f"{model} returned no choices: {body}") from error

    # A completion cut off at max_tokens reads like a finished answer and is
    # not one. Left to return, it would reach the grounding guard as a
    # truncated sentence and the report as a fact the corpus never stated.
    #
    # Raising here does lose the row: the prompt was evaluated, so tokens were
    # spent, and ledger.call never reaches its record. That is accepted rather
    # than worked around, because a raise aborts the answer it belonged to, and
    # a cost table for an answer that was never produced is not a table anyone
    # reads. A caller that ever chooses to catch this and retry has to record
    # the failed attempt itself, or its TOTAL will understate what ran.
    if finish_reason == "length":
        raise LLMError(
            f"{model} was cut off at max_tokens after "
            f"{usage_tokens(body)} completion tokens ({len(text)} characters). "
            f"Raise max_tokens for this call rather than accepting half an "
            f"answer."
        )

    # No usage block means no ledger row. Recording zeros instead would put a
    # free-looking call in the section 7 table, which is worse than no table:
    # it is wrong in the direction that flatters the result.
    usage = body.get("usage")
    if not usage or "prompt_tokens" not in usage:
        raise LLMError(
            f"{model} returned no usage block, so the call cannot be costed. "
            f"Ollama reports one on /v1/chat/completions; a server that does "
            f"not is the wrong server for this pipeline."
        )

    return Response(
        text=text,
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage.get("completion_tokens", 0),
        latency_s=latency,
        finish_reason=finish_reason,
    )


def _parse_native(body: dict[str, Any], model: str, latency: float) -> Response:
    """The same refusals _parse makes, read off Ollama's own native
    /api/chat response shape instead of the OpenAI-compatible one:
    "message.content" not "choices[0].message.content", "done_reason"
    not "finish_reason", "prompt_eval_count"/"eval_count" not a nested
    "usage" object. Kept as a second function rather than branching
    inside _parse, since the two response shapes share no field names to
    branch on cleanly and a shared function would just be two
    implementations glued together by an if.
    """
    try:
        text = body["message"]["content"]
        finish_reason = body.get("done_reason", "")
    except KeyError as error:
        raise LLMError(f"{model} returned no message: {body}") from error

    if finish_reason == "length":
        raise LLMError(
            f"{model} was cut off at max_tokens after "
            f"{body.get('eval_count', 0)} completion tokens ({len(text)} "
            f"characters). Raise max_tokens for this call rather than "
            f"accepting half an answer."
        )

    if "prompt_eval_count" not in body:
        raise LLMError(
            f"{model} returned no prompt_eval_count, so the call cannot be "
            f"costed. Ollama reports one on /api/chat; a server that does "
            f"not is the wrong server for this pipeline."
        )

    return Response(
        text=text,
        model=model,
        prompt_tokens=body["prompt_eval_count"],
        completion_tokens=body.get("eval_count", 0),
        latency_s=latency,
        finish_reason=finish_reason,
    )


def usage_tokens(body: dict[str, Any]) -> int:
    """Completion tokens the server reported, for a message that needs them."""
    return (body.get("usage") or {}).get("completion_tokens", 0)


def unload_models(models: tuple[str, ...] = (GENERATOR_MODEL, JUDGE_MODEL)) -> None:
    """Force-unload whichever of these models Ollama is currently serving.

    Promoted from generation/evaluate.py's own ``_reset_ollama``, written
    there first after that file's own run crashed three times in a row at
    different points, one a genuine ``cudaMalloc failed: out of memory``
    from inside Ollama's own llama-server process partway through a long
    sequence of sequential calls. LEARNING/router.md already documents
    this machine's established mitigation for a crash under sustained
    model-loading load: stop the resident model and retry as a fresh
    process. A full process restart is not available mid-function, so
    this calls the same unload Ollama's own API exposes (``keep_alive:
    0``) directly, a within-process version of that mitigation.

    Stage 10's own harness needs the identical reset between its three
    arms, which is what moved this here rather than leaving it private to
    one file: a second caller duplicating the endpoint knowledge (the
    ``/v1`` suffix has to be stripped, Ollama's native unload sits at the
    root, not under ``/v1``) would be exactly the kind of drift this
    project's own single-transport rule exists to prevent.

    Best-effort: a failed reset is not this function's problem to raise
    on. The worst case is the next call pays a full reload instead of
    reusing a warm model, not a correctness issue.
    """
    root_url = LLM_BASE_URL.removesuffix("/v1")
    for model in models:
        try:
            request = urllib.request.Request(
                f"{root_url}/api/generate",
                data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:  # noqa: BLE001
            pass


def _http_message(error: urllib.error.HTTPError, model: str) -> str:
    """Explain an HTTP failure in terms of what to do about it."""
    detail = error.read().decode("utf-8", errors="replace")[:300]
    if error.code == 404:
        return (f"{model} is not on this machine. Run `ollama pull {model}`, "
                f"or correct the tag in config.py. ({detail})")
    return f"the endpoint returned HTTP {error.code}: {detail}"


def _url_message(error: urllib.error.URLError) -> str:
    """Explain a connection failure in terms of what to do about it."""
    if isinstance(error.reason, ConnectionRefusedError):
        return (f"nothing is listening on {LLM_BASE_URL}. Start the server "
                f"with `ollama serve`, or check that the Ollama service is "
                f"running.")
    return f"could not reach {LLM_BASE_URL}: {error.reason}"
