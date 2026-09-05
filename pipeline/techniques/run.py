"""The orchestration: a question in, a QuestionRun record out.

Position: gate.py and router.py have already decided whether a question
needs retrieval and, if so, which techniques it shows a signal for.
answer() is what turns that decision into an actual context, retrieved
under the shipping configuration stage 7 settled, transformed and
improved by whichever of the eight techniques schema.TechniqueSet says to
run. Stage 9's generation reads a QuestionRun's context_text; nothing in
this file writes an answer, and nothing in this file reopens a Qdrant
client or rebuilds a BM25 index on its own, per retrieval.md's own open
item: retriever.ShippingHandle, opened once by the caller, is the only
way this file ever touches the store.

A caller that opens a ShippingHandle for a run where Reranking might
ever fire has to call techniques.rerank.warm_up() first, before
retriever.open_shipping(), never after. answer() itself cannot enforce
this: by the time it receives a handle, open_shipping() has already
loaded BGE-M3, and rerank.warm_up's own docstring has the measured
reason that order is load-bearing rather than a style preference,
found while building this stage. rerank.py's own __main__ follows this
contract; any future caller doing the same (a real cli.py command,
evaluate.py's own batch runner) has to repeat it, since it lives at the
call site that decides when open_shipping() itself happens, outside
anything this file controls.

Every technique branch below does its own import locally, inside the
branch that needs it, rather than at module level. This is deliberate,
not an oversight: files 7 through 14 do not all exist yet, and importing
run.py has to keep working as each one lands rather than failing until
the last of the eight is written. A technique the router requested but
this package has not built yet raises NotImplementedError naming the
file that is missing, not an ImportError three frames down a stack trace
nobody asked to read.

Two orchestration choices are made here rather than left to the router,
and both are architectural judgement calls, not measurements, recorded
for that reason rather than with a number behind them:

  Multi-Query, Decomposition and HyDE all replace the same step, the
  first retrieval call, with a different way of building the query.
  Nothing about running two of them together composes cleanly; a router
  that requests more than one is resolved by priority, Decomposition
  first (it is the most structurally specific signal), then Multi-Query,
  then HyDE, and the other requested names are simply not applied. This
  matches "do not force every question through every technique" more
  than trying to compose three separate query strategies would.

  Reranking is forced on for every advanced_rag question regardless of
  whether the router named it, because router.py's own prompt instructs
  the model never to select it: the plan states Reranking is always on
  for advanced_rag, and this is where that policy is actually enforced.
  Read from TECHNIQUE_DECISION once evaluate.py (stage 8's own file 15)
  has measured whether that policy is worth keeping; defaults to the
  plan's own starting assumption, True, before that measurement exists,
  so this module is runnable before file 15 is.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import (
    CHARS_PER_TOKEN,
    GENERATION_DECISION,
    GENERATION_RESERVE_TOKENS,
    LLM_CONTEXT,
    RERANK_TOP_N,
    RETRIEVAL_K,
    TECHNIQUE_DECISION,
)
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk, ShippingHandle, retrieve_scored
from ..router.gate import check as pre_gate_check
from ..router.router import route as router_route
from ..router.schema import RouteDecision, TechniqueSet

# Priority order among the three query-transformation techniques that
# each replace the first retrieval call outright. See the module
# docstring for why this is a priority rather than a composition.
_RETRIEVAL_STRATEGY_PRIORITY = ("Decomposition", "Multi-Query", "HyDE")

# Provisional, not yet measured. crag.py (file 14) and evaluate.py
# (file 15) are what get to set this for real, from the separation
# between the 18 answerable questions' own top scores and Q10's; a fixed
# guess here only has to be good enough to make CRAG's runtime trigger
# testable before that measurement exists, the same reason RERANK_TOP_N
# in config.py is a stated budget rather than a fit. 0.5 is the midpoint
# of cosine similarity's own 0 to 1 range, not a corpus-specific number.
PROVISIONAL_CRAG_THRESHOLD = 0.5


@dataclass
class QuestionRun:
    """Everything section 2 of the task wants recorded for one question,
    plus what stage 9 and stage 10 need beyond that.

        question        the question as asked
        history          the (question, answer) pairs carried in as
                         prior turns, the same shape router.route takes
        gate_matched     True if the deterministic pre-gate caught this
                         before any model was ever called
        decision         the RouteDecision, from gate.py or router.py
        technique_set    what actually ran, after the priority and
                         always-on policies above and any explicit
                         override, not merely what the router requested
                         (that is decision.requested)
        executed         technique names, in the order they actually ran
        crag_refused     whether CRAG's own fallback path ended in
                         refusal. False whenever CRAG never ran
        retrieved        ScoredChunk list, empty for route "simple"
        context_text     retrieved, rendered as one prompt-ready string
                         with a citation marker per chunk. Section 2's
                         "Retrieved context" field
        traces           one entry per technique that actually ran,
                         keyed by its schema.TECHNIQUES name, holding
                         whatever that technique's own apply() returned
                         as its trace. What section 16's per-technique
                         analysis questions read
        ledger           this question's own cost and latency record
    """

    question: str
    history: list[tuple[str, str]] | None
    gate_matched: bool
    decision: RouteDecision
    technique_set: TechniqueSet
    executed: tuple[str, ...]
    retrieved: list[ScoredChunk]
    context_text: str
    traces: dict[str, Any]
    ledger: Ledger
    crag_refused: bool = False

    @property
    def chunk_ids(self) -> list[str]:
        """Section 2's "Retrieved document/chunk IDs" field."""
        return [item.chunk_id for item in self.retrieved]


def render_context(scored: list[ScoredChunk]) -> str:
    """The retrieved chunks, as one prompt-ready string, each carrying a
    citation marker a generation prompt and an answer's own citations can
    both point back at by the same number.
    """
    if not scored:
        return ""
    blocks = [
        f"[{i}] ({item.chunk_id})\n{item.chunk.text}"
        for i, item in enumerate(scored, start=1)
    ]
    return "\n\n".join(blocks)


# --- runtime triggers, invisible in the question text ---------------------------

def _context_over_budget(scored: list[ScoredChunk]) -> bool:
    """A cheap character-count estimate of whether the retrieved context
    would overflow the generation budget, using CHARS_PER_TOKEN's own
    conservative rounding. This only has to decide whether Compression's
    own, more careful accounting applies; it is not a replacement for it.
    """
    if not scored:
        return False
    total_chars = sum(len(item.chunk.text) for item in scored)
    budget_tokens = LLM_CONTEXT - GENERATION_RESERVE_TOKENS
    return (total_chars / CHARS_PER_TOKEN) > budget_tokens


def _low_confidence(scored: list[ScoredChunk], threshold: float) -> bool:
    """Whether retrieval looks weak enough that CRAG should grade it.

    An RRF-fused score has no fixed scale to threshold honestly against,
    per retriever.ScoredChunk's own docstring, so a rank-derived top
    score always counts as low confidence here rather than being read on
    a scale it was never on.
    """
    if not scored:
        return True
    top = scored[0]
    return (not top.score_is_absolute) or (top.score < threshold)


def _reranking_default(decision_path: Path = TECHNIQUE_DECISION) -> bool:
    if decision_path.exists():
        stored = json.loads(decision_path.read_text(encoding="utf-8"))
        return bool(stored.get("reranking_default", True))
    return True


def _crag_threshold_default(decision_path: Path = GENERATION_DECISION) -> float:
    """The same read-a-decision-file-or-fall-back-to-a-placeholder shape
    as _reranking_default above, for the other number this file's own
    module docstring flagged as provisional: PROVISIONAL_CRAG_THRESHOLD
    was always a stated placeholder, "just has to be good enough to make
    CRAG's runtime trigger testable before that measurement exists", and
    generation/evaluate.py (stage 9's own file 7) is what actually
    measures it, from the real separation between the 18 answerable
    questions' own top scores and Q10's. Read here, from config.py's
    GENERATION_DECISION, rather than the ledger PROVISIONAL_CRAG_THRESHOLD
    constant this file used to depend on directly, once that measurement
    exists; the constant stays as the honest fallback for a first build
    that has not run stage 9's own evaluate.py yet.
    """
    if decision_path.exists():
        stored = json.loads(decision_path.read_text(encoding="utf-8"))
        return float(stored.get("crag_threshold", PROVISIONAL_CRAG_THRESHOLD))
    return PROVISIONAL_CRAG_THRESHOLD


def _resolve_technique_set(
    decision: RouteDecision, override: TechniqueSet | None,
) -> TechniqueSet:
    if override is not None:
        return override
    if decision.route != "advanced_rag":
        return TechniqueSet.none()
    technique_set = TechniqueSet.from_names(decision.requested)
    if _reranking_default():
        technique_set = dataclasses.replace(technique_set, reranking=True)
    return technique_set


# --- technique dispatch, imported locally --------------------------------------

def _apply_or_not_built(module: str, label: str, *args: Any, **kwargs: Any) -> Any:
    """Import module.apply and call it, or fail with a message that names
    the missing file rather than an ImportError from three frames down.
    """
    try:
        imported = __import__(f"pipeline.techniques.{module}", fromlist=["apply"])
    except ImportError as error:
        raise NotImplementedError(
            f"{label} was selected but pipeline/techniques/{module}.py "
            f"does not exist yet. ({error})"
        ) from error
    return imported.apply(*args, **kwargs)


# --- the entry point -----------------------------------------------------------

def answer(
    question: str,
    ledger: Ledger,
    handle: ShippingHandle,
    history: list[tuple[str, str]] | None = None,
    technique_set: TechniqueSet | None = None,
    crag_threshold: float | None = None,
    route_override: str | None = None,
) -> QuestionRun:
    """Route, retrieve, and apply whichever techniques the route calls
    for, once, for one question.

    technique_set overrides everything the router and the policies above
    would otherwise choose, and also switches off both runtime triggers
    below: an explicit technique_set means exactly that set ran, nothing
    more, which is what makes evaluate.py's ablation grid cells
    well-defined ("Multi-Query alone" has to mean only Multi-Query ran,
    not Multi-Query plus whatever CRAG's confidence threshold happened to
    add on top). This exists for two callers: that ablation grid, which
    needs to force TechniqueSet.none() or a single technique on
    regardless of what the router picks, and this module's own gate
    below, which checks that a forced TechniqueSet.none() reproduces the
    shipping ranking exactly with nothing else touching it.

    route_override skips gate.check and router.route entirely and builds
    a RouteDecision locally with route=route_override, requesting no
    techniques of its own. Built for stage 10's own basic arm: the task's
    own section 2 specifies Basic RAG as "Question -> Embedding -> Vector
    Search -> Top-K Chunks -> Prompt -> LLM -> Answer", a flow with no
    routing step at all. Forcing TechniqueSet.none() through the ordinary
    router would still let it refuse Q9 at the pre-gate or send Q10
    wherever a real classification lands, handing the baseline two
    behaviours the spec's own flow has no mechanism to produce; this is
    the difference between "no techniques ran" and "no router ran",
    and the basic arm needs the second. Composes with technique_set
    rather than replacing it, since a caller forcing the route still
    needs to say what runs on it: stage 10's own harness passes both,
    route_override="basic_rag" and technique_set=TechniqueSet.none(),
    together. RouteDecision's own __post_init__ validates route_override
    against ROUTES, so this raises the same way an unrecognised route
    from the model would, rather than duplicating that check here.

    crag_threshold defaults to None rather than to
    PROVISIONAL_CRAG_THRESHOLD directly, the same reason
    _reranking_default is called from inside _resolve_technique_set's own
    body rather than baked into a default parameter value: a Python
    default is evaluated once, at import time, and would freeze whatever
    GENERATION_DECISION held (or did not yet hold) the moment this module
    was first imported, never re-reading it after stage 9's own
    evaluate.py writes the real, measured value. None here means "read
    the current decision file, or the honest placeholder if it does not
    exist yet", resolved fresh on every call.
    """
    if crag_threshold is None:
        crag_threshold = _crag_threshold_default()

    if route_override is not None:
        decision = RouteDecision(
            route=route_override,
            reason=f"route forced to {route_override!r} by the caller "
                   f"(route_override), the router never ran",
        )
    else:
        gate_decision = pre_gate_check(question)
        if gate_decision is not None:
            return QuestionRun(
                question=question, history=history, gate_matched=True,
                decision=gate_decision, technique_set=TechniqueSet.none(),
                executed=(), retrieved=[], context_text="", traces={},
                ledger=ledger,
            )
        decision = router_route(question, ledger, history=history)

    resolved = _resolve_technique_set(decision, technique_set)
    # Runtime triggers are themselves part of "what the route and the
    # router picked"; an explicit override switches them off along with
    # everything else they would otherwise add. They are also, per the
    # plan, an advanced_rag-only safety net: basic_rag has no CRAG and no
    # Compression, by design, so neither trigger is even evaluated
    # outside advanced_rag.
    triggers_active = technique_set is None and decision.route == "advanced_rag"

    if decision.route == "simple":
        return QuestionRun(
            question=question, history=history, gate_matched=False,
            decision=decision, technique_set=resolved, executed=(),
            retrieved=[], context_text="", traces={}, ledger=ledger,
        )

    query_text = question
    traces: dict[str, Any] = {}
    executed: list[str] = []

    if resolved.rewriting:
        query_text, trace = _apply_or_not_built(
            "rewrite", "Rewriting", question, history, ledger,
        )
        traces["Rewriting"] = trace
        executed.append("Rewriting")

    allowed_chunk_ids = None
    if resolved.self_query:
        query_text, allowed_chunk_ids, trace = _apply_or_not_built(
            "selfquery", "Self-Query", query_text, handle.context, ledger,
        )
        traces["Self-Query"] = trace
        executed.append("Self-Query")

    # Retrieval has to hand reranking more than RETRIEVAL_K candidates to
    # rerank, or reranking could only ever reorder an already-narrow top
    # 10, never promote a genuinely relevant chunk the bi-encoder ranked
    # 11th to 20th. RERANK_TOP_N is deliberately wider than RETRIEVAL_K
    # for exactly this reason; every retrieval path below reads
    # retrieval_k rather than assuming RETRIEVAL_K, so this is decided
    # once, in one place, rather than by each technique guessing whether
    # reranking comes after it.
    retrieval_k = RERANK_TOP_N if resolved.reranking else RETRIEVAL_K

    strategy = next(
        (
            name for name in _RETRIEVAL_STRATEGY_PRIORITY
            if getattr(resolved, {
                "Decomposition": "decomposition",
                "Multi-Query": "multi_query",
                "HyDE": "hyde",
            }[name])
        ),
        None,
    )
    if strategy == "Decomposition":
        scored, trace = _apply_or_not_built(
            "decompose", "Decomposition", query_text, handle, ledger,
            allowed_chunk_ids, retrieval_k,
        )
        traces["Decomposition"] = trace
        executed.append("Decomposition")
    elif strategy == "Multi-Query":
        scored, trace = _apply_or_not_built(
            "multiquery", "Multi-Query", query_text, handle, ledger,
            allowed_chunk_ids, retrieval_k,
        )
        traces["Multi-Query"] = trace
        executed.append("Multi-Query")
    elif strategy == "HyDE":
        scored, trace = _apply_or_not_built(
            "hyde", "HyDE", query_text, handle, ledger, allowed_chunk_ids,
            retrieval_k,
        )
        traces["HyDE"] = trace
        executed.append("HyDE")
    else:
        scored = retrieve_scored(
            handle.context, query_text, handle.decision["mode"],
            k=retrieval_k,
            apply_caps=handle.decision.get("apply_caps", False),
            allowed_chunk_ids=allowed_chunk_ids,
        )

    if resolved.reranking:
        scored, trace = _apply_or_not_built(
            "rerank", "Reranking", query_text, scored, ledger,
        )
        traces["Reranking"] = trace
        executed.append("Reranking")

    if resolved.compression or (triggers_active and _context_over_budget(scored)):
        scored, trace = _apply_or_not_built(
            "compress", "Compression", query_text, scored, ledger,
        )
        traces["Compression"] = trace
        executed.append("Compression")

    crag_refused = False
    if resolved.crag or (triggers_active and _low_confidence(scored, crag_threshold)):
        scored, crag_refused, trace = _apply_or_not_built(
            "crag", "CRAG", query_text, scored, handle, ledger,
        )
        traces["CRAG evaluator"] = trace
        executed.append("CRAG evaluator")

    return QuestionRun(
        question=question, history=history, gate_matched=False,
        decision=decision, technique_set=resolved,
        executed=tuple(executed), retrieved=scored,
        context_text=render_context(scored), traces=traces,
        ledger=ledger, crag_refused=crag_refused,
    )


# --- verification --------------------------------------------------------------

def verify_none_matches_shipping(question: str, handle: ShippingHandle) -> bool:
    """The gate this file exists to pass: with technique_set forced to
    TechniqueSet.none(), the retrieval call inside answer() reproduces
    the shipping ranking exactly, chunk id for chunk id. If this ever
    disagrees, stage 10's "advanced beat basic" numbers would be
    measuring an accidental difference in the plumbing rather than the
    techniques, which is what every technique file after this one
    depends on not being true.

    The override controls which techniques run and switches off both
    runtime triggers; it does not skip the router call itself, since
    routing decides whether retrieval happens at all. That matters here:
    a question the router correctly routes "simple" correctly retrieves
    nothing, and comparing that against a raw retrieve() call, which
    knows nothing about routing and always retrieves, would disagree for
    a reason that has nothing to do with the retrieval plumbing this gate
    exists to check. Q9's own shape is exactly this case. Such a question
    passes trivially rather than being compared against something it was
    never meant to match.

    Computed against handle's own already-open context and decision,
    the exact same call retrieve_shipping makes internally, rather than
    against retrieve_shipping itself: Qdrant's local file mode holds an
    exclusive lock on the storage directory per open client, and calling
    retrieve_shipping here would try to open a second client on the same
    path handle already has open, which is refused rather than queued.
    """
    from ..retrieval.retriever import retrieve

    ledger = Ledger(label="verify-none-matches-shipping")
    run_result = answer(
        question, ledger, handle, technique_set=TechniqueSet.none(),
    )
    if run_result.decision.route == "simple":
        return True

    shipping_ids = retrieve(
        handle.context, question, handle.decision["mode"],
        apply_caps=handle.decision.get("apply_caps", False),
    )
    return run_result.chunk_ids == shipping_ids


if __name__ == "__main__":
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden
    from ..retrieval.retriever import open_shipping

    questions, _ = load_golden(GOLDEN_SET)
    with open_shipping() as shipping_handle:
        mismatches = [
            q.id for q in questions
            if not verify_none_matches_shipping(q.question, shipping_handle)
        ]
    if mismatches:
        print(f"FAIL  TechniqueSet.none() disagreed with retrieve_shipping "
              f"on {len(mismatches)}: {mismatches}")
    else:
        print(f"ok  TechniqueSet.none() matches retrieve_shipping exactly "
              f"on all {len(questions)} golden questions")
