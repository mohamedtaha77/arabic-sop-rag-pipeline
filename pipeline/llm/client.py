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

from ..config import LLM_BASE_URL, LLM_TIMEOUT


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
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    request = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as handle:
            body = json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
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

    return _parse(body, model, latency)


def list_models() -> list[str]:
    """Every model tag the server can serve right now.

    Here rather than in probe.py so that urllib stays in one file. Used to tell
    "the server is down" apart from "the server is up and the tag is wrong",
    which are the two setup failures worth separating.
    """
    request = urllib.request.Request(f"{LLM_BASE_URL}/models", method="GET")
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


def usage_tokens(body: dict[str, Any]) -> int:
    """Completion tokens the server reported, for a message that needs them."""
    return (body.get("usage") or {}).get("completion_tokens", 0)


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
