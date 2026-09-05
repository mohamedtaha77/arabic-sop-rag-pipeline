/* The browser half of the assistant.
 *
 * Vanilla, no framework and no build step: at this size React would add
 * a toolchain and a node_modules for nothing, and a bank-internal tool
 * is easier to hand over when the whole client is three files a person
 * can read.
 *
 * The one non-obvious mapping, and the reason the source pane works at
 * all: the answer text carries markers like [1], while Answer.citations
 * is "chunk ids in the order their markers first appear" (its own
 * contract in generation/answer.py). So marker N resolves to
 * citations[N - 1], and that chunk id is what /api/highlight turns into
 * a rectangle on a rendered page. */

const $ = (id) => document.getElementById(id);
const thread = $("thread");

const SESSION = { cost: 0, tokens: 0 };
let busy = false;
let lastCitations = [];

// The ledger declares fourteen steps; these are the twelve that can run
// while serving a question. "Contextualisation" belongs to chunking and
// "Evaluation judge" to stage 10, so both are always zero here and are
// left out rather than shown as two permanently dead rows.
const ALL_STEPS = [
  "Router", "Rewriter", "Multi-Query", "Decomposition", "HyDE", "Self-Query",
  "Reranking", "Compression", "CRAG evaluator", "Final generation",
  "Grounding guard", "Presenter",
];
const TECHNIQUE_PILLS = [
  "Rewriting", "Multi-Query", "Decomposition", "HyDE",
  "Self-Query", "Reranking", "Compression", "CRAG",
];
const ROUTE_LABEL = {
  simple: "Direct", basic_rag: "Basic RAG",
  advanced_rag: "Advanced RAG", error: "Error",
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clearEmpty() {
  const empty = thread.querySelector(".empty-thread");
  if (empty) empty.remove();
}

function addUser(text) {
  clearEmpty();
  const wrap = el("div", "msg-user");
  const bubble = el("div", "bubble-user ar", text);
  bubble.dir = "rtl";
  wrap.append(bubble);
  thread.append(wrap);
  thread.scrollTop = thread.scrollHeight;
}

function addThinking() {
  const wrap = el("div", "msg-bot");
  wrap.id = "thinking";
  const row = el("div", "thinking");
  row.append(el("div", "spinner"));
  const label = el("span", null, "Thinking… 0s");
  row.append(label);
  wrap.append(row);
  thread.append(wrap);
  thread.scrollTop = thread.scrollHeight;

  // A 40-second median (REPORT.md measured 39.7s mean, 92s worst) with
  // no feedback reads as a hung page, so the elapsed count runs from
  // the first second rather than after some grace period.
  const started = Date.now();
  let step = "";
  const paint = () => {
    const secs = Math.round((Date.now() - started) / 1000);
    label.textContent = step ? `${step}… ${secs}s` : `Thinking… ${secs}s`;
  };
  const timer = setInterval(paint, 1000);

  // The live step feed. /api/stream reports each ledger entry as it is
  // recorded, so the wait names what the pipeline is actually doing
  // instead of spinning anonymously for half a minute.
  const live = [];
  const source = new EventSource("/api/stream");
  source.addEventListener("step", (event) => {
    const entry = JSON.parse(event.data);
    live.push(entry);
    step = entry.step;
    paint();
    renderLiveSteps(live);
  });
  // The server's own clock for the run, which is the number the metrics
  // pane should show: the local timer above starts when the browser
  // submits, and a question queued behind another would otherwise count
  // its wait as though it were work.
  source.addEventListener("tick", (event) => {
    $("m-lat").textContent = `${JSON.parse(event.data).elapsed_s}s`;
  });
  const close = () => source.close();
  source.addEventListener("done", close);
  source.addEventListener("idle", close);
  source.onerror = close;

  return () => { clearInterval(timer); close(); wrap.remove(); };
}

/* Steps that have finished so far, in the panel's own layout, so the
 * metrics pane fills in during the run rather than all at once at the
 * end. The final /api/ask response replaces this with the authoritative
 * ledger rows; these are the same numbers arriving earlier. */
function renderLiveSteps(entries) {
  const totals = new Map();
  entries.forEach((entry) => {
    const prev = totals.get(entry.step) || { latency_s: 0, cost_usd: 0 };
    totals.set(entry.step, {
      latency_s: prev.latency_s + entry.latency_s,
      cost_usd: prev.cost_usd + entry.cost_usd,
    });
  });
  const box = $("steps");
  box.textContent = "";
  ALL_STEPS.forEach((name) => {
    const row = totals.get(name);
    const line = el("div", `step-row${row ? "" : " skipped"}`);
    line.append(el("span", "nm", name));
    line.append(el("span", "ran", row ? "✓" : "—"));
    line.append(el("span", "num", row ? `${row.latency_s.toFixed(1)}s` : "—"));
    line.append(el("span", "num", row ? `$${row.cost_usd.toFixed(4)}` : "$0"));
    box.append(line);
  });

  const cost = entries.reduce((sum, e) => sum + e.cost_usd, 0);
  const promptTokens = entries.reduce((sum, e) => sum + e.prompt_tokens, 0);
  const outTokens = entries.reduce((sum, e) => sum + e.completion_tokens, 0);
  $("m-cost").textContent = `$${cost.toFixed(4)}`;
  $("m-in").textContent = promptTokens.toLocaleString();
  $("m-out").textContent = outTokens.toLocaleString();
  if (entries.length) $("m-lat").textContent = `${entries[entries.length - 1].elapsed_s}s`;
}

/* Turn "[1]" markers into clickable chips. Everything that is not a
 * marker is appended as a text node, never as innerHTML: the answer is
 * model output over bank documents and must never be able to inject
 * markup into this page.
 *
 * markerChunkIds (generation_traces.Synthesis.marker_chunk_ids) is the
 * correct map from a literal "[n]" string to its chunk id, and citations
 * is that same chunk id set ordered by first appearance. A marker is
 * resolved through markerChunkIds, never by treating n as a 1-based
 * index into citations: the model writes whatever numbers it used for
 * its own 10-chunk context (measured directly, "[1]", "[5]", "[6]",
 * "[8]" in one real answer, never reset to a clean 1,2,3,4), so
 * citations[n-1] silently misses every marker whose number exceeds how
 * many distinct citations there are, rendering it as inert plain text
 * instead of a chip. The chip's own display number and colour still
 * come from the chunk's position in citations (pos below), which is
 * what stays in sync with the cite-tabs panel's own 1,2,3... numbering,
 * not the arbitrary literal number the model happened to write. */
function renderAnswerText(container, text, citations, markerChunkIds) {
  const pattern = /\[(\d+)\]/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      container.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const chunkId = markerChunkIds[match[0]];
    const pos = chunkId ? citations.indexOf(chunkId) + 1 : 0;
    if (chunkId && pos > 0) {
      const chip = el("button", `cite c${((pos - 1) % 4) + 1}`, String(pos));
      chip.title = chunkId;
      chip.addEventListener("click", () => showSource(chunkId, pos));
      container.append(chip);
    } else {
      container.append(document.createTextNode(match[0]));
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    container.append(document.createTextNode(text.slice(cursor)));
  }
}

function addAnswer(data) {
  const wrap = el("div", "msg-bot");

  const badges = el("div", "badges");
  badges.append(el("span", `route-badge route-${data.route || "error"}`,
                    ROUTE_LABEL[data.route] || "Error"));
  if (data.refusal_kind) {
    badges.append(el("span", "refusal-tag",
      data.refusal_kind === "out_of_domain" ? "Out of scope" : "Not found in documents"));
  }
  wrap.append(badges);

  if (data.route === "advanced_rag") {
    const pills = el("div", "pills");
    const ran = new Set(data.executed || []);
    // "CRAG evaluator" is the ledger's own name for the step the router
    // vocabulary calls "CRAG" (schema.TECHNIQUE_TO_STEP keeps the two
    // spellings deliberately); map it so the pill lights up.
    if (ran.has("CRAG evaluator")) ran.add("CRAG");
    TECHNIQUE_PILLS.forEach((name) => {
      pills.append(el("span", ran.has(name) ? "pill on" : "pill", name));
    });
    wrap.append(pills);
  }

  const isRefusal = data.kind === "refused";
  const isError = data.kind === "error";
  // synthesise.py's own retry can fail to recover a degenerate answer
  // (a bare citation marker with no claim attached) and, by design,
  // falls back to that same bad text rather than failing the call
  // outright: generation/synthesise.py's own docstring calls this a
  // deliberate trade, "its own failure should fall back to the original
  // response rather than take the whole call down". That is the right
  // call for a harness that logs and grades the text, but shown as a
  // normal answer bubble it reads as the app having crashed. The trace
  // already reports whether this happened, so it is shown honestly
  // instead of hidden behind a routine-looking citation chip.
  const synthesisTrace = (data.generation_traces || {}).Synthesis || {};
  const isDegenerate = data.kind === "grounded"
    && synthesisTrace.repair_retry_used && !synthesisTrace.repair_retry_recovered;
  // generation/run.py's own comment explains why this is never gated on:
  // the "llm" entailment backend only catches 46.7% of genuinely correct
  // answers (entail.py's own bake-off), so refusing whenever it disagrees
  // would reject more good answers than bad ones it catches. That is the
  // right call for what generation/run.py decides to return, but it does
  // not mean the disagreement should stay invisible to whoever reads the
  // answer: when the judge explicitly flagged a cited claim as
  // unsupported, that is worth a visible caution even though the answer
  // still stands. It only fires when the judge actively disagreed
  // (entailed < checked); a citation pointing at the wrong chunk that the
  // judge wrongly approved of leaves no trace here to act on.
  const entailmentTrace = (data.generation_traces || {}).Entailment || {};
  const hasUnverifiedClaim = data.kind === "grounded"
    && entailmentTrace.checked > 0 && entailmentTrace.entailed < entailmentTrace.checked;
  const answer = el("div", `answer ar${(isRefusal || isDegenerate) ? " refused" : ""}${isError ? " errored" : ""}`);
  if (!isError) answer.dir = "rtl";
  if (isError) {
    answer.textContent = `This question could not be answered on this machine: ${data.error}`;
  } else {
    if (isDegenerate) {
      const notice = el("div", "degenerate-notice",
        "The model could not produce a full answer for this question, even after retrying. "
        + "What it returned is shown below, but treat it as incomplete rather than an answer.");
      notice.dir = "ltr";
      answer.append(notice);
    } else if (hasUnverifiedClaim) {
      const n = entailmentTrace.checked - entailmentTrace.entailed;
      const notice = el("div", "degenerate-notice",
        `The automated check could not confirm ${n} of ${entailmentTrace.checked} claim(s) in this `
        + "answer against the source it cites. The answer is shown below, but verify it against the "
        + "source pane before relying on it.");
      notice.dir = "ltr";
      answer.append(notice);
    }
    const markerChunkIds = ((data.generation_traces || {}).Synthesis || {}).marker_chunk_ids || {};
    renderAnswerText(answer, data.text || "", data.citations || [], markerChunkIds);
  }
  wrap.append(answer);

  const meta = el("div", "meta-row");
  if (data.kind === "grounded") {
    const guard = el("span", data.presenter_rejected ? "guard-blocked" : "guard-ok");
    guard.textContent = data.presenter_rejected
      ? "Guard blocked the presenter — showing the grounded text"
      : "Guard passed";
    guard.title = data.presenter_rejected
      ? "The presenter tried to add something the synthesiser never wrote, so it was rejected and the synthesised answer is shown instead."
      : "The presenter added no fact, number, date or citation the synthesiser had not already written.";
    meta.append(guard);
  }
  meta.append(el("span", null, `${data.elapsed_s}s`));
  wrap.append(meta);

  thread.append(wrap);
  thread.scrollTop = thread.scrollHeight;
}

/* --- source pane --- */

function renderCiteTabs(citations, activeIndex) {
  const tabs = $("cite-tabs");
  tabs.textContent = "";
  citations.forEach((chunkId, i) => {
    const n = i + 1;
    const tab = el("button", `cite-tab c${((n - 1) % 4) + 1}${n === activeIndex ? " active" : ""}`);
    tab.append(el("span", "dot", String(n)));
    tab.append(el("span", null, chunkId.replace(/_/g, " ")));
    tab.addEventListener("click", () => showSource(chunkId, n));
    tabs.append(tab);
  });
}

async function showSource(chunkId, n) {
  const stage = $("page-stage");
  renderCiteTabs(lastCitations, n);
  stage.textContent = "";

  let info;
  try {
    info = await (await fetch(`/api/highlight?chunk_id=${encodeURIComponent(chunkId)}`)).json();
  } catch (err) {
    stage.append(el("div", "no-geometry", `Could not load the source: ${err}`));
    return;
  }

  $("breadcrumb").textContent = "";
  $("breadcrumb").append(document.createTextNode(info.source || "unknown source"));
  if (info.page) {
    $("breadcrumb").append(document.createTextNode(" → "));
    const b = el("b", null, `Page ${info.page}`);
    $("breadcrumb").append(b);
  }

  if (!info.source || !info.page) {
    stage.append(el("div", "no-geometry", info.reason || "No page for this citation."));
    return;
  }

  const wrap = el("div", "page-wrap");
  const img = document.createElement("img");
  img.src = `/api/page?source=${encodeURIComponent(info.source)}&page=${info.page}`;
  img.alt = `${info.source} page ${info.page}`;
  wrap.append(img);
  stage.append(wrap);

  // Rects come back in pixels at the server's own render dpi, so they
  // are placed as percentages of the natural image size: the browser
  // scales the page image to the pane, and a percentage survives that
  // where an absolute pixel offset would drift.
  img.addEventListener("load", () => {
    if (!info.covered) {
      stage.append(el("div", "no-geometry", info.reason));
      return;
    }
    (info.rects || []).forEach(([x, y, w, h]) => {
      const box = el("div", `hl c${((n - 1) % 4) + 1}`);
      box.style.left = `${(x / img.naturalWidth) * 100}%`;
      box.style.top = `${(y / img.naturalHeight) * 100}%`;
      box.style.width = `${(w / img.naturalWidth) * 100}%`;
      box.style.height = `${(h / img.naturalHeight) * 100}%`;
      wrap.append(box);
    });
  });
}

/* --- metrics --- */

function renderMetrics(ledger, elapsed) {
  const rows = (ledger && ledger.rows) || [];
  const total = rows.find((r) => r.step === "TOTAL");
  $("m-cost").textContent = total ? `$${total.cost_usd.toFixed(4)}` : "—";
  $("m-lat").textContent = elapsed !== undefined ? `${elapsed}s` : "—";
  $("m-in").textContent = total ? total.prompt_tokens.toLocaleString() : "—";
  $("m-out").textContent = total ? total.completion_tokens.toLocaleString() : "—";

  const box = $("steps");
  box.textContent = "";
  const byStep = new Map(rows.map((r) => [r.step, r]));
  ALL_STEPS.forEach((name) => {
    const row = byStep.get(name);
    const ran = row && row.executed;
    const line = el("div", `step-row${ran ? "" : " skipped"}`);
    line.append(el("span", "nm", name));
    line.append(el("span", "ran", ran ? "✓" : "—"));
    // Seconds, not prompt tokens, in this column. Reranking is a local
    // cross-encoder rather than an LLM call, so it books zero tokens and
    // zero cost while being the slowest step in the pipeline (REPORT.md
    // measured ~25 s per call). A cost-only row would read as free.
    line.append(el("span", "num", ran ? `${row.latency_s.toFixed(1)}s` : "—"));
    line.append(el("span", "num", ran ? `$${row.cost_usd.toFixed(4)}` : "$0"));
    box.append(line);
  });

  if (total) {
    SESSION.cost += total.cost_usd;
    SESSION.tokens += total.prompt_tokens + total.completion_tokens;
    $("session-total").textContent =
      `$${SESSION.cost.toFixed(6)} · ${SESSION.tokens.toLocaleString()} tok`;
  }
}

/* --- asking --- */

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  $("send").disabled = true;
  $("status-pill").textContent = "answering…";
  $("status-pill").classList.add("busy");

  addUser(question);
  const stopThinking = addThinking();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    stopThinking();
    addAnswer(data);
    renderMetrics(data.ledger, data.elapsed_s);

    lastCitations = data.citations || [];
    if (lastCitations.length) {
      showSource(lastCitations[0], 1);
    } else {
      renderCiteTabs([], 0);
      $("page-stage").textContent = "";
      const note = el("div", "no-geometry",
        data.kind === "refused"
          ? "This answer cited no passage: it was refused rather than grounded."
          : "This answer cited no passage.");
      $("page-stage").append(note);
      $("breadcrumb").textContent = "No source cited";
    }
  } catch (err) {
    stopThinking();
    addAnswer({ kind: "error", route: "error", error: String(err), elapsed_s: 0 });
  } finally {
    busy = false;
    $("send").disabled = false;
    $("status-pill").textContent = "ready";
    $("status-pill").classList.remove("busy");
  }
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("question");
  const question = input.value;
  input.value = "";
  ask(question);
});

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", () => ask(button.textContent.trim()));
});
