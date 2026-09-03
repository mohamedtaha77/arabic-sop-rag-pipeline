"""Reranking: a cross-encoder re-scores the fused candidates directly.

Position: called from techniques/run.py's answer() after retrieval,
forced on for every advanced_rag question regardless of what the router
requested, since router.py's own prompt instructs the model never to
select it, matching the plan's own policy: "Reranking is always on for
advanced_rag."

The whole reason this technique earns its place at all is that a
bi-encoder (BGE-M3's own dense leg) and a cross-encoder answer different
questions. A bi-encoder embeds the query and each chunk separately, so
relevance is a dot product between two vectors computed with no
knowledge of each other; a cross-encoder reads the query and one chunk
together, in one forward pass, and can catch a relevance signal the
separate embeddings never had a chance to represent. run.py widens
retrieval to RERANK_TOP_N candidates specifically so this technique has
more than the final RETRIEVAL_K to promote from: reranking an
already-narrow top 10 could only ever reorder it, never recover a chunk
the bi-encoder ranked 11th to 20th.

BAAI/bge-reranker-v2-m3, CPU, per advanced-rag-plan.md's own device
table: "around 20 pairs per query is acceptable" on CPU latency, which is
exactly RERANK_TOP_N. Its raw output is a single logit per pair, not a
probability; sigmoid maps it into [0, 1], the standard way this specific
model family is used, and what makes its score directly comparable to a
dense leg's own cosine similarity for CRAG's confidence check, both now
living on the same bounded scale.

The model itself never loads in this process. _rerank_worker.py's own
docstring has the full story: loading bge-reranker-v2-m3 here, alongside
an already-resident BGE-M3 dense embedder, segfaulted, confirmed with
faulthandler pointing at a raw access violation inside the reranker's
own from_pretrained call. This is embedding.md's documented class of
fault, a second distinct model checkpoint in one process, and the
bake-off's own fix, release one model before loading the next, is not
available here because retriever.py's embedder cache deliberately keeps
BGE-M3 resident across many questions. So this module talks to a
persistent worker subprocess instead, lazily started once and kept
alive for the life of this process, rather than loading anything itself.

What this module does not do: it does not decide whether Reranking
should run, and it does not touch the underlying dense or sparse legs;
it only re-scores and re-orders whatever retrieval already found.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import time
from dataclasses import dataclass

from ..config import PROJECT_ROOT, RERANK_MODEL, RETRIEVAL_K
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk

# Comfortably above the corpus's own measured chunk sizes (llm.md: the
# largest chunk is roughly 1,040 tokens against Qwen2.5's tokenizer, a
# different tokenizer but the right order of magnitude) plus a query,
# without reaching for the model's own 8,192 position limit, which would
# cost CPU latency this corpus never needs.
_MAX_LENGTH = 1024

_worker: subprocess.Popen | None = None

# A crashed worker startup on this development machine is not always
# reproducible from the same starting conditions, the identical pattern
# embedding/run.py's own _SUBPROCESS_RETRIES already documents and
# already exists to ride out: measured directly while building this
# file, the worker's own child process exited with Windows code
# 3221225477 (0xC0000005, STATUS_ACCESS_VIOLATION) attempting to load
# bge-reranker-v2-m3 while the parent already held BGE-M3 resident, a
# genuinely separate OS process with its own address space, which rules
# out in-process object corruption as the sole cause and points at
# system-wide memory pressure at spawn time instead. Retrying a fresh
# spawn costs nothing but the retry, the same accepted trade
# embedding/run.py's own docstring states.
_WORKER_START_RETRIES = 5
_WORKER_START_RETRY_DELAY_S = 15


def _spawn_worker() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "pipeline.techniques._rerank_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
        bufsize=1, cwd=str(PROJECT_ROOT),
    )


def _ensure_worker() -> subprocess.Popen:
    """Start the persistent reranker subprocess on first use, or return
    the one already running. Blocks on its "ready" line so the first
    real request is never sent before the 2.2 GB load has finished.
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

        # readline() returning empty is supposed to mean the process
        # already closed its stdout, which normally means it already
        # exited; a short pause here closes the small window observed
        # directly while building this, where poll() briefly still read
        # None immediately after readline() had already returned empty.
        time.sleep(0.5)
        return_code = candidate.poll()
        last_error = (
            f"exit {return_code} after {elapsed:.1f}s, "
            f"stderr: {candidate.stderr.read()[-500:]}"
            if return_code is not None
            else f"no ready line after {elapsed:.1f}s (got {ready_line!r}), "
                 f"still running"
        )
        print(f"  reranker worker start attempt {attempt}/{_WORKER_START_RETRIES} "
              f"failed ({last_error}); a crashed worker costs only the "
              f"retry, since the next attempt is a fresh process")
        if return_code is None:
            candidate.kill()
        if attempt < _WORKER_START_RETRIES:
            time.sleep(_WORKER_START_RETRY_DELAY_S)

    raise RuntimeError(
        f"reranker worker failed to start {_WORKER_START_RETRIES} times "
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


def warm_up() -> None:
    """Start the reranker worker now, rather than waiting for the first
    apply() call.

    Call this before retriever.open_shipping() in any caller that will
    reach Reranking at all, and never after. Measured directly, twice
    over: spawning the worker while a parent process already holds
    BGE-M3 resident crashed reproducibly, exit code 3221225477
    (STATUS_ACCESS_VIOLATION on Windows) inside the reranker's own
    AutoModelForSequenceClassification.from_pretrained, five spawn
    retries in a row, low_cpu_mem_usage=True made no difference either.
    The same worker, started before BGE-M3 ever loads, comes up clean
    every time and answers a real request correctly afterward, so the
    order embedder-then-reranker is the one thing that actually matters
    here, confirmed rather than assumed. techniques/run.py's own
    docstring repeats this contract for whoever calls answer() in a
    loop; apply() itself still lazily starts the worker if a caller
    skips this, which is correct for a standalone call but carries the
    ordering risk this function exists to avoid.
    """
    _ensure_worker()


def _score_pairs(query: str, texts: list[str]) -> list[float]:
    """One (query, chunk text) pair per candidate, scored in one round
    trip to the worker rather than one process call per pair.
    """
    worker = _ensure_worker()
    request = json.dumps({"query": query, "texts": texts, "max_length": _MAX_LENGTH})
    worker.stdin.write(request + "\n")
    worker.stdin.flush()

    response_line = worker.stdout.readline()
    if not response_line:
        raise RuntimeError(
            "reranker worker closed its pipe unexpectedly mid-request"
        )
    response = json.loads(response_line)
    if "error" in response:
        raise RuntimeError(f"reranker worker reported an error: {response['error']}")
    return response["scores"]


@dataclass
class RerankTrace:
    """What analysis question 7 asks for: how reranking changed the top
    results. before and after are chunk ids in their own resulting
    order, truncated to RETRIEVAL_K on both sides, so promotion and
    demotion read directly off the two tuples without re-deriving
    anything from separate score lists.
    """

    query: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    scores: dict[str, float]


def apply(
    query_text: str,
    scored: list[ScoredChunk],
    ledger: Ledger,
) -> tuple[list[ScoredChunk], RerankTrace]:
    """Re-score every candidate against the query with the cross-encoder,
    and return the top RETRIEVAL_K under the new order.
    """
    if not scored:
        return scored, RerankTrace(query=query_text, before=(), after=(), scores={})

    before_top = tuple(item.chunk_id for item in scored[:RETRIEVAL_K])
    texts = [item.chunk.text for item in scored]

    started = time.perf_counter()
    raw_scores = _score_pairs(query_text, texts)
    latency = time.perf_counter() - started
    ledger.record_local("Reranking", latency, model=RERANK_MODEL)

    reordered = sorted(zip(scored, raw_scores), key=lambda pair: -pair[1])
    top = reordered[:RETRIEVAL_K]

    result = [
        ScoredChunk(
            chunk_id=item.chunk_id, chunk=item.chunk,
            score=score, score_is_absolute=True,
        )
        for item, score in top
    ]

    trace = RerankTrace(
        query=query_text, before=before_top,
        after=tuple(item.chunk_id for item in result),
        scores={item.chunk_id: score for item, score in top},
    )
    return result, trace


# --- verification --------------------------------------------------------------

def verify_all_questions(handle: "object") -> list[dict]:
    """The build order's own gate: top 10 before and after recorded for
    every answerable question, not just one. Each question's retrieval
    is widened to RERANK_TOP_N first, the same width run.py gives
    Reranking in real use, so this exercises promotion from outside the
    unreranked top 10 rather than a mere reordering of it.
    """
    from ..config import GOLDEN_SET, RERANK_TOP_N
    from ..golden.question import load_golden
    from ..retrieval.retriever import retrieve_scored

    questions, _ = load_golden(GOLDEN_SET)
    answerable = [q for q in questions if q.expect == "answerable"]

    results = []
    for question in answerable:
        scored = retrieve_scored(
            handle.context, question.question, handle.decision["mode"],
            k=RERANK_TOP_N, apply_caps=False,
        )
        ledger = Ledger(label=f"verify-rerank-{question.id}")
        _reranked, trace = apply(question.question, scored, ledger)
        promoted = [
            cid for cid in trace.after if cid not in trace.before
        ]
        results.append({
            "id": question.id,
            "before_top10": list(trace.before),
            "after_top10": list(trace.after),
            "promoted_from_outside_top10": promoted,
        })
    return results


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    # warm_up() before open_shipping(), never after: see warm_up's own
    # docstring for the measured reason this order is load-bearing, not
    # a style preference.
    warm_up()
    with open_shipping() as shipping_handle:
        results = verify_all_questions(shipping_handle)

    lines = [f"{len(results)} questions reranked, top 10 before and after:", ""]
    total_promoted = 0
    for row in results:
        total_promoted += len(row["promoted_from_outside_top10"])
        lines.append(f"{row['id']}:")
        lines.append(f"  before: {row['before_top10']}")
        lines.append(f"  after:  {row['after_top10']}")
        lines.append(f"  promoted from outside top 10: {row['promoted_from_outside_top10']}")

    lines.insert(1, f"chunks promoted from outside the unreranked top 10: "
                     f"{total_promoted} across {len(results)} questions")
    lines.insert(2, "")

    out_path = PROCESSED_DIR / "12_rerank_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"ok: {total_promoted} promotions across {len(results)} questions, "
          f"written to {out_path}")
