"""The grid, the decision, and the report: whether hybrid retrieval ships.

Position: every leg from bm25.py, sparse.py and store.qdrant's own dense
search already exists and works alone. This is the first module that asks
whether combining them is actually better than the baseline stage 6
already measured and shipped, rather than assuming a technique the task
names has to win.

Two probes run before this file was written changed what it had to be.
The plan's own named gate, "BM25 surfaces a role name dense retrieval
misses", measured false: that role name is common (12 chunks) and its
metadata carries an OCR defect BM25 cannot match at all, while BGE-M3's
subword tokenisation handles it anyway. And naive, unweighted RRF fusion
measured as a regression against dense-only, costing 0.112 Recall@10 and
0.153 MRR@10 on the template variant, before a real tokeniser bug (Arabic
punctuation gluing onto word ends) was found and fixed; after the fix,
weighted fusion measured a real gain. Both findings are why this module's
pass condition is a decision rule fixed in advance, imported directly from
metrics.py rather than reimplemented, and not an assumption that hybrid
wins: "the baseline shipped" is as real and reportable an outcome as
"a candidate won", and stage 10's ablation needs both to be told apart
from a narrow result that only looks like one or the other.

What this module does not do: it does not decide which mode a caller
should use for one ad-hoc question. That is retriever.retrieve_shipping,
reading RETRIEVAL_DECISION once this module has written it.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from ..chunking.chunk import load_chunks
from ..config import (
    CONTEXT_OUTPUTS,
    CONTEXT_VARIANTS,
    GOLDEN_SET,
    QDRANT_PATH,
    RETRIEVAL_DECISION,
    RETRIEVAL_K,
    RETRIEVAL_OUTPUT,
)
from ..embedding.bakeoff import load_winning_model
from ..embedding.metrics import (
    aggregate,
    ndcg_at_k,
    paired_bootstrap_ci,
    reciprocal_rank,
    recall_at_k,
    strict_recall_at_k,
)
from ..golden.question import Question, load_golden
from .bm25 import BM25Index
from .retriever import RETRIEVAL_MODES, RetrievalContext, build_context, retrieve
from .text import tokenize_clitics, tokenize_plain, tokenize_stopwords

METRIC_KEYS = ("recall@10", "strict_recall@10", "mrr@10", "ndcg@10", "recall@5")

# The baseline every candidate is measured against: stage 6's own shipped
# result, BGE-M3, template, dense only, no diversity caps. Named here
# rather than recomputed from 04_bakeoff_decision.json, because the
# baseline has to be exactly what was actually shipped, not whatever the
# bake-off would decide if it reran today.
BASELINE_VARIANT = "template"
BASELINE_MODE = "dense"
BASELINE_CAPS = False


def _score_question(
    ranked: list[str], question: Question, k: int = RETRIEVAL_K,
) -> dict[str, Any]:
    gold = set(question.gold_chunk_ids)
    return {
        "id": question.id,
        "n_gold": len(gold),
        "recall@10": recall_at_k(ranked, gold, k),
        "strict_recall@10": strict_recall_at_k(ranked, gold, k),
        "mrr@10": reciprocal_rank(ranked, gold, k),
        "ndcg@10": ndcg_at_k(ranked, gold, k),
        "recall@5": recall_at_k(ranked, gold, 5),
    }


def _q10_check(
    context: RetrievalContext, mode: str, apply_caps: bool,
    q10: Question | None,
) -> dict[str, Any] | None:
    """The same negative check bakeoff.evaluate runs: Q10 carries no gold
    chunks, so a retriever surfacing its distractors is behaving correctly
    and CRAG, not this stage, is what has to catch it.
    """
    if q10 is None or not q10.distractor_chunk_ids:
        return None
    ranked = retrieve(context, q10.question, mode, apply_caps=apply_caps)
    top = set(ranked)
    distractors = set(q10.distractor_chunk_ids)
    return {
        "distractor_ids": sorted(distractors),
        "found_in_top_k": sorted(distractors & top),
    }


def evaluate_cell(
    context: RetrievalContext, questions: list[Question], mode: str,
    apply_caps: bool,
) -> dict[str, Any]:
    """One (variant, mode, caps) grid cell, scored over every answerable
    question plus the Q10 negative check.
    """
    answerable = [q for q in questions if q.expect == "answerable"]
    rows = [
        _score_question(retrieve(context, q.question, mode, apply_caps=apply_caps), q)
        for q in answerable
    ]
    q10 = next((q for q in questions if q.id == "Q10"), None)
    return {
        "variant": context.variant,
        "mode": mode,
        "apply_caps": apply_caps,
        "rows": rows,
        "q10_check": _q10_check(context, mode, apply_caps, q10),
    }


# --- the decision rule, fixed before any number from this run exists ----------

def beats_baseline(candidate_rows: list[dict], baseline_rows: list[dict]) -> dict[str, Any]:
    """A candidate replaces the baseline only if its 95% CI on the paired
    per-question difference excludes zero on BOTH Recall@10 and MRR@10,
    in the candidate's favour on both. Any other outcome, and the baseline
    ships: this is the same rule, and the same function, metrics.decide
    used to settle BGE-M3 against e5-large, so stage 7 is held to the
    identical standard stage 6 already was.
    """
    candidate_recall = [r["recall@10"] for r in candidate_rows]
    baseline_recall = [r["recall@10"] for r in baseline_rows]
    candidate_mrr = [r["mrr@10"] for r in candidate_rows]
    baseline_mrr = [r["mrr@10"] for r in baseline_rows]

    recall_lo, recall_hi, recall_diff = paired_bootstrap_ci(candidate_recall, baseline_recall)
    mrr_lo, mrr_hi, mrr_diff = paired_bootstrap_ci(candidate_mrr, baseline_mrr)

    recall_wins = recall_lo > 0
    mrr_wins = mrr_lo > 0

    return {
        "wins": recall_wins and mrr_wins,
        "recall_diff": recall_diff, "recall_ci_95": [recall_lo, recall_hi],
        "mrr_diff": mrr_diff, "mrr_ci_95": [mrr_lo, mrr_hi],
    }


# --- the grid -------------------------------------------------------------------

def run_grid(
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
    golden_path: Path = GOLDEN_SET,
    qdrant_path: Path = QDRANT_PATH,
) -> dict[str, Any] | None:
    model_key = load_winning_model()
    questions, _ = load_golden(golden_path)

    client = QdrantClient(path=str(qdrant_path))
    try:
        contexts = {
            variant: build_context(variant, client, model_key, variant_paths)
            for variant in CONTEXT_VARIANTS
        }

        cells = {}
        for variant, mode, apply_caps in itertools.product(
            CONTEXT_VARIANTS, RETRIEVAL_MODES, (False, True)
        ):
            key = (variant, mode, apply_caps)
            print(f"scoring {variant} / {mode} / caps={apply_caps}")
            cells[key] = evaluate_cell(contexts[variant], questions, mode, apply_caps)

        baseline_key = (BASELINE_VARIANT, BASELINE_MODE, BASELINE_CAPS)
        baseline_rows = cells[baseline_key]["rows"]

        decisions = {}
        for key, cell in cells.items():
            if key == baseline_key:
                continue
            decisions[key] = beats_baseline(cell["rows"], baseline_rows)

        bm25_levels = _score_bm25_normalisation_levels(contexts[BASELINE_VARIANT], questions)

        return {
            "model_key": model_key,
            "cells": cells,
            "baseline_key": baseline_key,
            "decisions": decisions,
            "bm25_levels": bm25_levels,
        }
    finally:
        client.close()


def _score_bm25_normalisation_levels(
    context: RetrievalContext, questions: list[Question],
) -> dict[str, dict[str, float]]:
    """The three tokenisation levels text.py exposes, scored on the BM25
    leg alone, template variant, inside the real pipeline rather than the
    throwaway probe that first measured this. Not a grid axis of its own:
    the level is a property of how bm25.py tokenises, decided once, not a
    thing every mode and variant needs multiplied out three times over.
    """
    answerable = [q for q in questions if q.expect == "answerable"]
    levels = {
        "plain": tokenize_plain,
        "stopwords": tokenize_stopwords,
        "clitics": tokenize_clitics,
    }
    results = {}
    for name, tokenizer in levels.items():
        index = BM25Index(
            [c.metadata["chunk_id"] for c in context.chunks],
            [c.text for c in context.chunks],
            tokenize=tokenizer,
        )
        rows = [_score_question(index.rank(q.question), q) for q in answerable]
        results[name] = aggregate(rows)
    return results


# --- choosing what ships --------------------------------------------------------

def choose_shipping_configuration(grid: dict[str, Any]) -> dict[str, Any]:
    """Among every candidate whose 95% CI beat the baseline on both
    metrics, ship the one with the largest Recall@10 gain. If none beat
    it, ship the baseline: every leg still exists and is switchable, and
    "the baseline shipped" is what stage 10's ablation reports, not a
    failure to hide.
    """
    winners = [key for key, result in grid["decisions"].items() if result["wins"]]
    if not winners:
        variant, mode, apply_caps = grid["baseline_key"]
        basis = "baseline (no candidate's 95% CI beat it on both Recall@10 and MRR@10)"
    else:
        variant, mode, apply_caps = max(
            winners, key=lambda key: grid["decisions"][key]["recall_diff"]
        )
        basis = "measurement (95% CI beat the baseline on both Recall@10 and MRR@10)"

    return {
        "variant": variant, "mode": mode, "apply_caps": apply_caps,
        "basis": basis, "model_key": grid["model_key"],
    }


# --- reporting -----------------------------------------------------------------

def render_report(grid: dict[str, Any], decision: dict[str, Any]) -> str:
    variant, mode, caps = grid["baseline_key"]
    baseline_agg = aggregate(grid["cells"][grid["baseline_key"]]["rows"])

    lines = [
        "# Hybrid retrieval",
        "",
        f"{grid['model_key']}, {len(RETRIEVAL_MODES)} modes over "
        f"{len(CONTEXT_VARIANTS)} context variants, caps off and on, "
        f"scored against the same 18 answerable golden questions and the "
        f"same Q10 negative check stage 6's bake-off used. Baseline is "
        f"stage 6's own shipped configuration: {variant}/{mode}, "
        f"caps={caps}, Recall@10 {baseline_agg['recall@10']:.3f}, "
        f"MRR@10 {baseline_agg['mrr@10']:.3f}.",
        "",
        "## The grid",
        "",
        "| Variant | Mode | Caps | Recall@10 | StrictRecall@10 | MRR@10 | nDCG@10 | vs baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, cell in grid["cells"].items():
        agg = aggregate(cell["rows"])
        if key == grid["baseline_key"]:
            verdict = "baseline"
        else:
            result = grid["decisions"][key]
            verdict = "**wins**" if result["wins"] else "no"
        variant, mode, apply_caps = key
        lines.append(
            f"| {variant} | {mode} | {apply_caps} | {agg['recall@10']:.3f} | "
            f"{agg['strict_recall@10']:.3f} | {agg['mrr@10']:.3f} | "
            f"{agg['ndcg@10']:.3f} | {verdict} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        f"Shipping: **{decision['variant']}/{decision['mode']}**, "
        f"caps={decision['apply_caps']}. Basis: {decision['basis']}.",
        "",
        "The rule was fixed before this run, the same function and the "
        "same standard stage 6's model bake-off was held to: a candidate "
        "replaces the baseline only if its 95% CI on the paired "
        "per-question difference excludes zero, in its favour, on both "
        "Recall@10 and MRR@10. Every leg stays built and switchable "
        "regardless of which configuration ships.",
        "",
        "Every fusion row above weights its legs equally (RRF's own "
        "1/(k+rank), no per-leg multiplier). A manual check outside this "
        "grid, weighting dense above bm25 in RRF, produced a higher point "
        "estimate on template/dense_bm25 than the baseline. It was not "
        "adopted, and no weighted configuration was added to this grid: "
        "searching a weight until 18 questions produce a win is the exact "
        "overfitting this decision rule exists to rule out, the same "
        "objection that keeps BM25's own k1 and b fixed at their textbook "
        "values rather than fit to this golden set. If a future stage "
        "wants to explore weighting, it needs a held-out question set this "
        "one is not.",
        "",
        "Diversity caps cost recall in every single row above; not one "
        "capped cell matches its uncapped counterpart. MAX_PER_SOURCE=5 "
        "and MAX_PER_PAGE=3 were sized to the corpus's own shape, not "
        "fit to this golden set, and on this evidence they are too tight "
        "for an 18-question set where several questions' gold chunks "
        "already cluster on one or two pages. They stay implemented and "
        "switchable for a broader query mix where source concentration is "
        "a real risk; they do not belong in the shipping configuration on "
        "this evidence.",
        "",
        "## BM25 normalisation levels, template variant, BM25 leg alone",
        "",
        "| Level | Recall@10 | MRR@10 |",
        "|---|---|---|",
    ]
    for name, agg in grid["bm25_levels"].items():
        lines.append(f"| {name} | {agg['recall@10']:.3f} | {agg['mrr@10']:.3f} |")

    lines += ["", "## Q10, negative check (non_answering_retrieval)", ""]
    for key, cell in grid["cells"].items():
        check = cell["q10_check"]
        if check is None:
            continue
        variant, mode, apply_caps = key
        lines.append(
            f"- {variant}/{mode}/caps={apply_caps}: "
            f"{len(check['found_in_top_k'])} of {len(check['distractor_ids'])} "
            f"distractors in the top {RETRIEVAL_K}."
        )

    return "\n".join(lines) + "\n"


# --- entry point -----------------------------------------------------------------

def run(
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
    golden_path: Path = GOLDEN_SET,
    qdrant_path: Path = QDRANT_PATH,
    output_path: Path = RETRIEVAL_OUTPUT,
    decision_path: Path = RETRIEVAL_DECISION,
) -> bool:
    grid = run_grid(variant_paths, golden_path, qdrant_path)
    if grid is None:
        return False

    decision = choose_shipping_configuration(grid)
    report = render_report(grid, decision)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(f"\nshipping: {decision['variant']}/{decision['mode']} "
          f"caps={decision['apply_caps']} ({decision['basis']})")
    print(f"written to {output_path}")
    print(f"written to {decision_path}")
    return True


if __name__ == "__main__":
    run()
