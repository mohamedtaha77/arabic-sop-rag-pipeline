"""Retrieval metrics, the section 14 rows, and Basic against Advanced.

Position: harness.py has written every (arm, question) answer, and
ragas_judge.py has scored every grounded one on the four required
metrics. This file is where those two files' own output finally meet a
golden Question's own gold_chunk_ids: it computes Precision, Recall, MRR
and nDCG per (arm, question) the same way stage 6 and 7's own evaluate.py
files already did, reusing embedding.metrics rather than a second
implementation of the same formulas, then assembles section 14's own
required table and the Basic-versus-Advanced comparison section 17 asks
for.

compare.py never became its own file. The plan that opened this stage
named one, and this project's own build order already accepts folding a
planned file into a neighbour under real time pressure (stage 8's own
evaluate.py explicitly declines a properly powered grid for the same
reason); the comparison this file computes reads the exact same
(arm, question) records score.py already holds in memory, and a second
file would only add an import boundary between two things that are one
computation. report.py is what turns this file's own numbers into
REPORT.md's prose; this file only computes them.

What this file does not do: it does not run a question or call a model
of any kind, pure arithmetic over what harness.py and ragas_judge.py
already wrote to disk.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import GOLDEN_SET, JUDGE_OUTPUT, RETRIEVAL_K, RUNS_OUTPUT
from ..embedding.metrics import (
    ndcg_at_k,
    paired_bootstrap_ci,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from ..golden.question import Question, load_golden
from .record import ArmRun, load_runs

# The section 8 replacement set section 14's own table is required over,
# one row per id, in numeric order. Q11 to Q20 are this project's own
# corpus-coverage extension and get their own, clearly labelled
# supplementary table instead, per the plan this stage opened with.
REQUIRED_QUESTION_IDS = tuple(f"Q{n}" for n in range(1, 11))

RETRIEVAL_METRIC_KEYS = ("precision@10", "recall@10", "mrr@10", "ndcg@10")

# ragas_judge.py's own four keys, in section 14's own column order.
JUDGE_METRIC_KEYS = ("context_relevance", "faithfulness", "answer_relevance", "correctness")


def retrieval_metrics(run: ArmRun, question: Question) -> dict[str, float] | None:
    """Precision, Recall, MRR and nDCG at RETRIEVAL_K for one run, or
    None when the question carries no gold chunks to score against.

    Q9 (out_of_domain, no gold chunks by design) and Q10 (a CRAG case
    whose ids are distractors, not gold) are excluded here on exactly the
    same grounds stage 6 and 7's own evaluate.py already exclude them:
    "both are measured elsewhere, the router and CRAG, not retrieval
    quality" (04_bakeoff.md). A run with no gold chunks to check against
    is not a zero score, it is not a retrieval question at all.
    """
    if not question.gold_chunk_ids:
        return None
    gold = set(question.gold_chunk_ids)
    ranked = list(run.chunk_ids)
    return {
        "precision@10": precision_at_k(ranked, gold, RETRIEVAL_K),
        "recall@10": recall_at_k(ranked, gold, RETRIEVAL_K),
        "mrr@10": reciprocal_rank(ranked, gold, RETRIEVAL_K),
        "ndcg@10": ndcg_at_k(ranked, gold, RETRIEVAL_K),
    }


def _ledger_total(run: ArmRun) -> dict[str, float]:
    """This run's own TOTAL row, already computed once by
    Ledger.rows() and carried whole in ArmRun.ledger, not recomputed
    here: section 7 and section 14 both mean the same accumulated number,
    and the ledger is the one place it is already correct.
    """
    for row in run.ledger.get("rows", []):
        if row["step"] == "TOTAL":
            return row
    return {"cost_usd": 0.0, "latency_s": 0.0}


def _judge_index(judged: list[dict]) -> dict[tuple[str, str], dict]:
    """ragas_judge.py's own JUDGE_OUTPUT["judged"] list, keyed by
    (arm, id) the way every other lookup in this file already is.
    """
    return {(j["arm"], j["id"]): j for j in judged}


@dataclass
class Row:
    """One line of section 14's own required table, or its Q11-Q20
    supplementary counterpart: every field already a plain string or
    number, so report.py only has to format a Markdown table row, never
    decide what a missing value means.
    """

    id: str
    arm: str
    route: str
    techniques: str
    context_relevance: str
    faithfulness: str
    answer_relevance: str
    correctness: str
    total_cost: float
    latency_s: float


def _fmt_score(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def build_row(run: ArmRun, question: Question, judge_scores: dict | None) -> Row:
    """One Row, reading route and techniques off the ArmRun that already
    recorded them rather than re-deriving either: run.route is
    QuestionRun.decision.route, and run.executed is QuestionRun.executed,
    which is what actually ran, not merely what the router requested
    (RouteDecision.requested), the distinction schema.RouteDecision's own
    docstring draws.
    """
    scores = (judge_scores or {}).get("scores", {}) if judge_scores else {}
    skipped = judge_scores.get("skipped_reason") if judge_scores else (
        f"kind={run.kind!r}, nothing retrieved to grade" if run.kind != "grounded" else None
    )
    totals = _ledger_total(run)
    return Row(
        id=run.id, arm=run.arm, route=run.route,
        techniques=", ".join(run.executed) if run.executed else "(none)",
        context_relevance=_fmt_score(scores.get("context_relevance")) if not skipped else "N/A",
        faithfulness=_fmt_score(scores.get("faithfulness")) if not skipped else "N/A",
        answer_relevance=_fmt_score(scores.get("answer_relevance")) if not skipped else "N/A",
        correctness=_fmt_score(scores.get("correctness")) if not skipped else "N/A",
        total_cost=totals.get("cost_usd", 0.0), latency_s=totals.get("latency_s", 0.0),
    )


def build_tables(
    runs: list[ArmRun], questions: list[Question], judged: list[dict],
) -> dict[str, list[Row]]:
    """Rows for every arm, split into the required Q1-Q10 table and the
    Q11-Q20 supplementary one, both in numeric question order within
    each arm.
    """
    by_id = {q.id: q for q in questions}
    judge_index = _judge_index(judged)
    by_qid = sorted(
        ((int(r.id[1:]), r) for r in runs), key=lambda pair: pair[0],
    )

    required, supplementary = [], []
    for _, run in by_qid:
        row = build_row(run, by_id[run.id], judge_index.get((run.arm, run.id)))
        (required if run.id in REQUIRED_QUESTION_IDS else supplementary).append(row)
    return {"required": required, "supplementary": supplementary}


# --- Basic against Advanced ------------------------------------------------------

@dataclass
class ArmComparison:
    """One metric's own paired difference between two arms, the same
    95%-CI-excludes-zero standard stages 6, 7 and 8 already hold every
    other comparison in this project to, never a bare point estimate.
    """

    metric: str
    n_questions: int
    a_mean: float
    b_mean: float
    diff: float
    ci_95: tuple[float, float]
    a_wins: bool


def _paired_metric_values(
    metric: str, arm_a: str, arm_b: str, runs: list[ArmRun], questions: list[Question],
    judged: list[dict],
) -> tuple[list[float], list[float]] | None:
    """Values for one metric, paired by question id, over only the ids
    both arms actually answered with a real, scoreable number: an id
    missing a score on either side (an error record, a non-grounded
    answer, a failed judge call) is dropped from the pair rather than
    imputed, since a paired bootstrap needs a real value on both sides of
    every pair or the pairing itself is fiction.
    """
    by_id = {q.id: q for q in questions}
    judge_index = _judge_index(judged)
    a_runs = {r.id: r for r in runs if r.arm == arm_a}
    b_runs = {r.id: r for r in runs if r.arm == arm_b}

    a_values, b_values = [], []
    for qid in sorted(set(a_runs) & set(b_runs), key=lambda x: int(x[1:])):
        run_a, run_b = a_runs[qid], b_runs[qid]
        question = by_id[qid]
        if metric in RETRIEVAL_METRIC_KEYS:
            va = retrieval_metrics(run_a, question)
            vb = retrieval_metrics(run_b, question)
            if va is None or vb is None:
                continue
            a_values.append(va[metric])
            b_values.append(vb[metric])
        else:
            ja = judge_index.get((arm_a, qid))
            jb = judge_index.get((arm_b, qid))
            sa = (ja or {}).get("scores", {}).get(metric) if ja else None
            sb = (jb or {}).get("scores", {}).get(metric) if jb else None
            if sa is None or sb is None:
                continue
            a_values.append(sa)
            b_values.append(sb)
    return (a_values, b_values) if a_values else None


def compare_arms(
    arm_a: str, arm_b: str, runs: list[ArmRun], questions: list[Question], judged: list[dict],
) -> list[ArmComparison]:
    """arm_a minus arm_b, every metric this file and ragas_judge.py both
    produce, over whatever paired, scoreable subset of questions the two
    arms share. Called twice by report.py: basic against adaptive over
    the whole 20-question set (diluted by the 15 questions the router
    currently never sends anywhere advanced_rag, stated plainly rather
    than hidden), and basic against forced over the 8-question population
    that actually isolates what the eight techniques do.
    """
    results = []
    for metric in (*RETRIEVAL_METRIC_KEYS, *JUDGE_METRIC_KEYS):
        pair = _paired_metric_values(metric, arm_a, arm_b, runs, questions, judged)
        if pair is None:
            continue
        a_values, b_values = pair
        lo, hi, diff = paired_bootstrap_ci(a_values, b_values)
        results.append(ArmComparison(
            metric=metric, n_questions=len(a_values),
            a_mean=sum(a_values) / len(a_values), b_mean=sum(b_values) / len(b_values),
            diff=diff, ci_95=(lo, hi), a_wins=lo > 0,
        ))
    return results


# --- per-technique cost and latency, read off the ledgers already recorded -------

def technique_cost_latency(runs: list[ArmRun]) -> dict[str, dict[str, float]]:
    """Total cost and latency attributable to each named step, across
    every run handed in, read from each run's own already-computed
    ledger rows rather than re-deriving anything: ledger.STEPS already
    names every row, and "executed": True on a row is already the fact
    this needs.
    """
    totals: dict[str, dict[str, float]] = {}
    for run in runs:
        for row in run.ledger.get("rows", []):
            if row["step"] == "TOTAL" or not row["executed"]:
                continue
            bucket = totals.setdefault(row["step"], {"cost_usd": 0.0, "latency_s": 0.0, "n": 0})
            bucket["cost_usd"] += row["cost_usd"]
            bucket["latency_s"] += row["latency_s"]
            bucket["n"] += 1
    return totals


# --- loading ---------------------------------------------------------------------

def load_inputs(
    runs_path=RUNS_OUTPUT, judge_path=JUDGE_OUTPUT, golden_path=GOLDEN_SET,
) -> tuple[list[ArmRun], list[Question], list[dict]]:
    import json

    runs = load_runs(runs_path)
    questions, _ = load_golden(golden_path)
    judged = json.loads(judge_path.read_text(encoding="utf-8"))["judged"] if judge_path.exists() else []
    return runs, questions, judged
