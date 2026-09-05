"""The local API: one process, one open store, one question at a time.

Position: everything under pipeline/ up to stage 9 answers questions
from a command line; stage 10 measured what those answers are worth.
This file is the first thing that puts them in front of a person who
is not holding a terminal, and it does so without reopening any of it:
``generation.run.answer`` is the single sanctioned call path (that
file's own docstring says so) and this module calls it exactly the way
stage 10's harness already does.

Three constraints shape every decision here, and all three are
measured facts from this project rather than defensive habits:

1. **Qdrant's local file mode holds an exclusive lock.** One process,
   one open handle; techniques/run.py's own docstring recorded this
   when a second client refused to open the same path. So this server
   opens the store once, at startup, keeps it for its whole lifetime,
   and nothing else may run against the store while it is up.
2. **One question at a time.** Stage 10 spent a day on a machine that
   cannot hold the reranker, the embedder and Ollama at once
   (REPORT.md's own limitations section, 1.95 GB free measured at the
   moment of a crash). Two concurrent generations would put back
   exactly the memory peak that fix removed, so requests serialise
   behind a lock rather than racing.
3. **rerank.warm_up() must NOT run at startup.** Stage 10 measured
   that pre-warming the reranker collides with the embedder's own
   warm-up; rerank.apply() now shuts the embedder down, runs, and
   tears itself down again. Only embed_client.warm_up() belongs here.

What this module does not do: it does not decide how a question is
answered, and it does not compute a metric. It owns a store handle, a
queue, and the translation between an Answer object and JSON.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import DATA_DIR, LLM_PROVIDER, RAW_DIR
from ..chunking.chunk import source_slug
from ..evaluation.record import to_jsonable
from ..generation.run import answer as generation_answer
from ..llm.ledger import DEFAULT_CARD, Ledger
from ..retrieval import embed_client
from ..retrieval.retriever import open_shipping
from . import highlight

# Rendered pages are cached here rather than under data/golden/pages,
# which golden/worksheet.py owns for a different purpose (pages a
# person reads while building the golden set) and at a different
# resolution. Separate directory, same fingerprint-in-the-name rule.
PAGE_CACHE_DIR = DATA_DIR / "page_cache"

# Enough to read an Arabic scan on screen without making every request
# pay for 400 dpi. GOLDEN_RENDER_DPI is 400 because a person is reading
# the page to establish ground truth; a citation highlight only has to
# be legible next to the answer, and the browser scales it anyway.
SERVE_DPI = 150

# How long /api/stream waits for a run to appear before deciding nothing
# is coming. Generous, because a second question legitimately waits here
# for the whole of the one ahead of it (REPORT.md: 92 s worst case).
_STREAM_WAIT_S = 180.0

_STATIC_DIR = __import__("pathlib").Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    history: list[tuple[str, str]] | None = None


def _flatten(result: Any, elapsed: float) -> dict[str, Any]:
    """One Answer, as the JSON the browser reads.

    Deliberately the same shape evaluation/record.py already settled on
    for the same object, serialised through its own to_jsonable, rather
    than a second JSON vocabulary for the identical Answer: a field
    that means one thing in REPORT.md and another in the UI is a bug
    waiting for whoever reads both.
    """
    run = result.run
    return {
        "kind": result.kind,
        "text": result.text,
        "synthesised": result.synthesised,
        "presented": result.presented,
        "presenter_rejected": result.presenter_rejected,
        "refusal_kind": result.refusal_kind,
        "citations": list(result.citations),
        "route": run.decision.route,
        "reason": run.decision.reason,
        "gate_matched": run.gate_matched,
        "executed": list(run.executed),
        "chunk_ids": list(run.chunk_ids),
        "contexts": [item.chunk.text for item in run.retrieved],
        "ledger": result.ledger.to_dict(),
        "generation_traces": to_jsonable(result.traces),
        "technique_traces": to_jsonable(run.traces),
        "elapsed_s": round(elapsed, 2),
    }


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # embed_client.warm_up() before open_shipping(), and no
    # rerank.warm_up() at all: see this module's own docstring,
    # constraint 3, for the measured reason.
    embed_client.warm_up()
    handle_context = open_shipping()
    app.state.handle = handle_context.__enter__()
    app.state.lock = asyncio.Lock()
    app.state.live = {}
    try:
        yield
    finally:
        handle_context.__exit__(None, None, None)


app = FastAPI(title="Housing Bank RAG Assistant", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "busy": app.state.lock.locked()}


@app.post("/api/ask")
async def ask(request: AskRequest) -> JSONResponse:
    """Answer one question, start to finish.

    Runs the pipeline in a worker thread so the event loop stays free
    to serve the page, the source images and the progress stream while
    a 40-second answer is in flight (REPORT.md measured 39.7 s mean,
    92 s worst). The lock is what enforces constraint 2: a second
    question waits rather than running beside the first.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")

    async with app.state.lock:
        # The Groq copy is priced against Groq's own real rate, not the
        # notional Claude card every other entry point in this project
        # uses: pricing.py's own "groq" card, added specifically because
        # this path can genuinely be billed past its free tier, unlike
        # everything else here. Ledger's own DEFAULT_CARD is left alone,
        # since cli.py ask/evaluate and the stage 8-10 harnesses are all
        # calibrated against haiku-4.5 in REPORT.md.
        card = "groq" if LLM_PROVIDER == "groq" else DEFAULT_CARD
        ledger = Ledger(label="serve", card=card)
        # Published on app.state before the call starts, so the SSE
        # endpoint can watch this exact Ledger fill up while the thread
        # below is still inside generation.run.answer.
        app.state.live = {"ledger": ledger, "started": time.perf_counter()}
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                generation_answer, question, ledger, app.state.handle,
                request.history,
            )
        except Exception as error:  # noqa: BLE001
            # A genuine local-machine failure (an Ollama overflow, a
            # worker that could not start) is reported as data rather
            # than a 500 with a stack trace: the UI shows it in the
            # thread the same way stage 10's harness records an "error"
            # run, since on this machine that is a real, expected
            # outcome and not a bug in this endpoint.
            return JSONResponse({
                "kind": "error", "error": str(error),
                "elapsed_s": round(time.perf_counter() - started, 2),
                "ledger": ledger.to_dict(),
            })
        finally:
            app.state.live = {}

        return JSONResponse(_flatten(result, time.perf_counter() - started))


def _step_event(entry: Any, started: float) -> str:
    """One Ledger.Entry as an SSE frame."""
    return "event: step\ndata: " + json.dumps({
        "step": entry.step,
        "latency_s": round(entry.latency_s, 2),
        "cost_usd": entry.cost_usd,
        "prompt_tokens": entry.prompt_tokens,
        "completion_tokens": entry.completion_tokens,
        "cached": entry.cached,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }) + "\n\n"


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    """The steps of the run in flight, as they finish.

    This needs no change anywhere in stages 1-9, which is the whole
    reason it is shaped this way: ``Ledger.entries`` is a plain list
    that every step appends to through ``ledger.call``, so while
    /api/ask's worker thread is inside generation.run.answer, this
    endpoint watches that same Ledger object grow and reports each new
    entry. No callback to thread through the pipeline, no second
    accounting path that could disagree with the one REPORT.md used.

    There is no stream id because there cannot be two runs: constraint
    2 above means one question at a time, so "the live run" is
    unambiguous. The client opens this when it submits and closes it
    when the answer lands.
    """
    async def events():
        seen = 0
        started_watching = time.perf_counter()
        saw_run = False
        last: Any = None
        while True:
            live = app.state.live
            ledger = live.get("ledger")
            if ledger is not None:
                last = ledger
            if ledger is None:
                # The run clears app.state.live the moment its thread
                # returns, which can happen between two polls and after
                # the last steps were recorded. Drain the ledger that
                # was being watched before saying done, or the closing
                # steps (Presenter, Grounding guard) are lost every time.
                if saw_run and last is not None:
                    while seen < len(last.entries):
                        yield _step_event(last.entries[seen], started_watching)
                        seen += 1
                # The browser opens this at the same moment it POSTs, so
                # on the first ticks the run may not have taken the lock
                # yet, and a question queued behind another waits here
                # for as long as that one takes. Giving up immediately
                # would kill the stream before its own run began, so
                # this only ends once a run has actually been seen to
                # finish; before that it waits (capped, so a client that
                # never asks anything does not hold a socket for ever).
                if saw_run:
                    yield "event: done\ndata: {}\n\n"
                    return
                if time.perf_counter() - started_watching > _STREAM_WAIT_S:
                    yield "event: idle\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.2)
                continue
            saw_run = True
            entries = ledger.entries
            while seen < len(entries):
                yield _step_event(entries[seen], live["started"])
                seen += 1
            # A tick every poll, whether or not a step finished. Two
            # jobs: it keeps the connection warm through anything that
            # buffers idle sockets, and it is the client's proof that
            # the run is still moving during the long silent stretches
            # (Reranking alone can hold the floor for 25 s).
            yield ("event: tick\ndata: "
                   + json.dumps({"elapsed_s": round(time.perf_counter() - live["started"], 1),
                                 "steps": seen})
                   + "\n\n")
            await asyncio.sleep(0.4)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        # nginx buffers text/event-stream by default and would hold every
        # event until the run finished, which is exactly what this
        # endpoint exists to avoid.
        "X-Accel-Buffering": "no",
    })


@app.get("/api/highlight")
async def get_highlight(chunk_id: str = Query(...)) -> dict[str, Any]:
    """Where this chunk sits on its page, in pixels at SERVE_DPI.

    highlight.resolve is honest about the 41 prose chunks that carry no
    geometry: it returns covered=False with a reason the UI shows
    verbatim, rather than a plausible box over the wrong region.
    """
    return highlight.resolve(chunk_id, dpi=SERVE_DPI)


@app.get("/api/page")
async def get_page(source: str = Query(...), page: int = Query(..., ge=1)) -> FileResponse:
    """One rendered page, cached on disk after the first request."""
    from ..ingestion.ocr import file_fingerprint, render_page

    pdf_path = RAW_DIR / source
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"no such source: {source}")

    fingerprint = file_fingerprint(pdf_path)
    slug = source_slug(source)
    out_path = PAGE_CACHE_DIR / f"{slug}_{fingerprint}_p{page:02d}_{SERVE_DPI}dpi.png"

    if not out_path.exists():
        import pymupdf
        PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        document = pymupdf.open(pdf_path)
        try:
            if page > document.page_count:
                raise HTTPException(status_code=404, detail=f"page {page} out of range")
            out_path.write_bytes(render_page(document[page - 1], dpi=SERVE_DPI))
        finally:
            document.close()

    return FileResponse(out_path, media_type="image/png")


if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
