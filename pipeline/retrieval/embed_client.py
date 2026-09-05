"""The parent side of _embed_worker.py: spawn once, ask many times.

Position: embedder.py's own _encode_uncached and sparse.py's own
_sparse_batch call into this, only when their caller asked for the
worker path (retriever.py's own query-time calls, device="cpu"), never
for index-time embedding. This module knows the wire protocol and the
subprocess lifecycle; it does not know how a vector is computed or
where a vector is cached, both of which stay exactly where they already
were, in embedder.py and sparse.py.

Mirrors rerank.py's own _ensure_worker, _spawn_worker, warm_up and
_terminate_worker almost line for line, on purpose: this project has one
proven pattern for "a model that cannot share a process with another
model," and a second, differently-shaped implementation of the same idea
would be a second thing to get right rather than a second confirmation
of the first.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import time

from ..config import PROJECT_ROOT

_worker: subprocess.Popen | None = None

# Same retry shape as rerank.py's own _WORKER_START_RETRIES, for the same
# reason: a crashed worker startup on this machine is not always
# reproducible from the same starting conditions, and a fresh spawn costs
# only the retry.
_WORKER_START_RETRIES = 5
_WORKER_START_RETRY_DELAY_S = 15


# Below this much free host memory, BGE-M3's own load is the thing that
# dies: measured repeatedly here as exit 3221225477 with "loaded bge-m3 on
# cpu" already printed, the worker segfaulting partway through its first
# real encode. The weights want about 2.2 GB in fp32 and the process needs
# working room on top.
_HOST_MEMORY_FLOOR_GB = 3.0


def _reclaim_host_memory() -> None:
    """Give the reranker's host footprint back before loading BGE-M3.

    rerank.apply() shuts this worker down before it scores; this is the
    same bargain in the other direction, and it became necessary the
    moment the reranker started living on the GPU. A GPU-resident worker
    is deliberately kept alive between questions, but "on the GPU" only
    describes its weights: the process still holds a Python interpreter,
    torch, and a CUDA context in host memory, roughly 0.7 GB that it used
    to hand back between calls. On a machine with ~2 GB free that is
    exactly the margin BGE-M3 needs, so the embedder reclaims it rather
    than crashing. Costs the reranker a 6.6 s reload on the GPU, which is
    the cheap side of this trade against a failed question.
    """
    try:
        import psutil
        if psutil.virtual_memory().available / 1e9 >= _HOST_MEMORY_FLOOR_GB:
            return
    except Exception:  # noqa: BLE001
        return
    try:
        from ..techniques import rerank
        if rerank._worker is not None and rerank._worker.poll() is None:
            print(f"  host memory below {_HOST_MEMORY_FLOOR_GB} GB; releasing the "
                  f"reranker worker so the embedder can load")
            rerank._terminate_worker()
    except Exception:  # noqa: BLE001
        # Reclaiming is an optimisation, never a precondition: if the
        # reranker module will not cooperate the embedder still tries.
        pass


def _spawn_worker() -> subprocess.Popen:
    _reclaim_host_memory()
    return subprocess.Popen(
        [sys.executable, "-m", "pipeline.retrieval._embed_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
        bufsize=1, cwd=str(PROJECT_ROOT),
    )


def _ensure_worker() -> subprocess.Popen:
    """Start the persistent embedder subprocess on first use, or return
    the one already running. Blocks on its "ready" line so the first
    real request is never sent before BGE-M3 has actually finished
    loading, the same contract rerank.py's own _ensure_worker holds.
    """
    global _worker
    if _worker is not None and _worker.poll() is None:
        return _worker

    last_error = ""
    for attempt in range(1, _WORKER_START_RETRIES + 1):
        started = time.perf_counter()
        candidate = _spawn_worker()
        ready_line = candidate.stdout.readline()
        elapsed = time.perf_counter() - started
        if ready_line and "ready" in ready_line:
            _worker = candidate
            atexit.register(_terminate_worker)
            return _worker

        time.sleep(0.5)
        return_code = candidate.poll()
        last_error = (
            f"exit {return_code} after {elapsed:.1f}s, "
            f"stderr: {candidate.stderr.read()[-500:]}"
            if return_code is not None
            else f"no ready line after {elapsed:.1f}s (got {ready_line!r}), "
                 f"still running"
        )
        print(f"  embedder worker start attempt {attempt}/{_WORKER_START_RETRIES} "
              f"failed ({last_error}); a crashed worker costs only the "
              f"retry, since the next attempt is a fresh process")
        if return_code is None:
            candidate.kill()
        if attempt < _WORKER_START_RETRIES:
            time.sleep(_WORKER_START_RETRY_DELAY_S)

    raise RuntimeError(
        f"embedder worker failed to start {_WORKER_START_RETRIES} times "
        f"in a row (most recently: {last_error}). This machine's memory "
        f"is a known constraint under load; see LEARNING/router.md."
    )


def _terminate_worker() -> None:
    global _worker
    if _worker is not None and _worker.poll() is None:
        try:
            _worker.stdin.close()
            _worker.wait(timeout=5)
        except Exception:  # noqa: BLE001
            _worker.kill()


def shutdown() -> None:
    """Release the embedder worker's own memory, ahead of a call that
    needs it back: rerank.py's own apply() calls this immediately before
    spawning its own worker, and never has to call warm_up() again
    itself, since the next embed_client.dense/sparse call after
    reranking finishes simply lazily respawns this one through
    _ensure_worker, the same way a first-ever call would.

    Why this exists at all, measured directly rather than assumed:
    process isolation alone (this file's own reason to exist) fixed the
    same-process crash rerank.py's own worker was already built against,
    but two isolated, healthy worker processes still segfaulted the
    moment a real embed request needed real activation memory on top of
    both workers' own resident weights, with free system memory measured
    at 1.95 GB right before it happened. Neither worker is memory-heavy
    enough to justify its own permanent residency competing with the
    other's; making them mutually exclusive, at most one of the two ever
    holding memory at a time, is what actually keeps the combined peak
    under whatever this machine has free at the moment, rather than
    merely moving where the same crash happens.
    """
    _terminate_worker()


def warm_up() -> None:
    """Start the embedder worker now, rather than waiting for the first
    query.

    Superseded as a standing "keep it warm for the whole run" call by
    shutdown()'s own docstring: measured directly, two isolated worker
    processes, this one and rerank.py's own, both already resident and
    both individually healthy, still segfaulted on this machine the
    moment either one needed real activation memory rather than just
    its own resident weights. Callers that also reach Reranking should
    not treat this call as a one-time setup step the way rerank.warm_up()
    still is; rerank.py's own apply() now calls shutdown() before
    spawning its own worker and lets the next dense/sparse call respawn
    this one lazily afterward, so at most one of the two is ever
    resident at once. warm_up() itself is unchanged and still correct
    for a caller that never reaches Reranking at all.
    """
    _ensure_worker()


def _request(op: str, texts: list[str], kind: str, _retrying: bool = False) -> dict:
    worker = _ensure_worker()
    payload = json.dumps({"op": op, "texts": texts, "kind": kind})
    worker.stdin.write(payload + "\n")
    worker.stdin.flush()

    response_line = worker.stdout.readline()
    if not response_line:
        # poll() can briefly still read None immediately after readline()
        # already returned empty, the exact race rerank.py's own
        # _ensure_worker already documents and waits out; without this,
        # a genuine crash reports "exit code None" and an empty stderr
        # read before the OS has finished tearing the process down.
        time.sleep(0.5)
        return_code = worker.poll()
        stderr_tail = worker.stderr.read()[-2000:] if worker.stderr else ""
        _terminate_worker()
        if not _retrying:
            # Almost always memory rather than a bad request: the weights
            # loaded, then the first real encode asked for activation room
            # the machine did not have. Retrying blind would just crash
            # again, so the retry happens only after _reclaim_host_memory
            # (via _spawn_worker) has had the chance to free the
            # reranker's host footprint. Once only: a second failure is a
            # real one and belongs in the answer as an error.
            print("  embedder worker died mid-request; reclaiming memory "
                  "and retrying once")
            return _request(op, texts, kind, _retrying=True)
        raise RuntimeError(
            f"embedder worker closed its pipe unexpectedly mid-request "
            f"(exit code {return_code}, stderr: {stderr_tail!r})"
        )
    response = json.loads(response_line)
    if "error" in response:
        raise RuntimeError(f"embedder worker reported an error: {response['error']}")
    return response


def dense(texts: list[str], kind: str) -> list[list[float]]:
    """One unit-norm vector per text, computed by the worker process."""
    return _request("dense", texts, kind)["vectors"]


def sparse(texts: list[str], kind: str) -> list[tuple[list[int], list[float]]]:
    """One (indices, values) pair per text, computed by the worker
    process, the same shape sparse.py's own _sparse_batch returns.
    """
    return [tuple(pair) for pair in _request("sparse", texts, kind)["results"]]
