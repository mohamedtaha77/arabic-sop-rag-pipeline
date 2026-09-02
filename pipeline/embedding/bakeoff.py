"""The worker: one model, fully evaluated, in its own process.

Scores one candidate, BAAI/bge-m3 or intfloat/multilingual-e5-large, against
the 18 answerable golden questions over all three context variants. Called
only through run.py, one subprocess per model: loading a second distinct
model checkpoint in the same process after the first was loaded and
released reproducibly crashes on this machine, confirmed symmetric in
either order and confirmed unrelated to available memory. See
embedder.py's module docstring for the measurements behind that.

That constraint is also why the metric functions, the bootstrap tie-break
and render_report live in metrics.py rather than here: run.py, the
orchestrator that spawns this module as a subprocess, imports metrics.py
so the parent process never has torch or transformers resident while a
child is trying to load one of them.

What this module does not do: it does not compare the two models, and it
does not write the bake-off's decision. That is run.py, reading this
module's JSON output back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# torch before numpy: on Windows the two bundle different native BLAS
# builds, and importing numpy before torch touches CUDA is a known source of
# native-library conflicts that surface as a hard crash rather than a Python
# exception. See embedder.py's own note; this module hits the same hazard
# since it imports both, directly and via embedder.
import torch  # noqa: F401
import numpy as np  # noqa: F401

from ..chunking.chunk import Chunk, load_chunks
from ..config import BAKEOFF_DECISION, CONTEXT_OUTPUTS, GOLDEN_SET
from ..golden.question import Question, load_golden
from . import embedder
from .metrics import aggregate, rank_by_similarity
from .tokens import count_tokens, max_pure_content_tokens


def evaluate(
    model_key: str, chunks: list[Chunk], questions: list[Question],
    k: int = 10,
) -> dict[str, Any]:
    """Score one model against every answerable question in one chunk set.

    Also runs Q10 as a negative check rather than skipping it: Q10 carries no
    gold_chunk_ids, so it never enters the metrics below, but its three
    distractor_chunk_ids are real and golden.md is explicit about what they
    mean. A retriever that surfaces them here is behaving correctly; stage 8's
    CRAG is what has to catch it from there, not this stage.
    """
    from .metrics import ndcg_at_k, reciprocal_rank, recall_at_k, strict_recall_at_k

    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    passage_vectors = embedder.embed_passages([c.text for c in chunks], model_key)

    answerable = [q for q in questions if q.expect == "answerable"]
    query_vectors = embedder.embed_queries([q.question for q in answerable], model_key)

    rows = []
    for question, query_vector in zip(answerable, query_vectors):
        ranked = rank_by_similarity(query_vector, passage_vectors, chunk_ids)
        gold = set(question.gold_chunk_ids)
        rows.append({
            "id": question.id,
            "n_gold": len(gold),
            "recall@10": recall_at_k(ranked, gold, k),
            "strict_recall@10": strict_recall_at_k(ranked, gold, k),
            "mrr@10": reciprocal_rank(ranked, gold, k),
            "ndcg@10": ndcg_at_k(ranked, gold, k),
            "recall@5": recall_at_k(ranked, gold, 5),
        })

    q10 = next((q for q in questions if q.id == "Q10"), None)
    q10_check = None
    if q10 is not None and q10.distractor_chunk_ids:
        query_vector = embedder.embed_queries([q10.question], model_key)[0]
        ranked = rank_by_similarity(query_vector, passage_vectors, chunk_ids)
        top = set(ranked[:k])
        distractors = set(q10.distractor_chunk_ids)
        q10_check = {
            "distractor_ids": sorted(distractors),
            "found_in_top_k": sorted(distractors & top),
        }

    return {"model": model_key, "rows": rows, "q10_check": q10_check}


def truncation_rate(chunks: list[Chunk], model_key: str) -> tuple[int, int]:
    """How many of these chunks e5-large's 512-token cap silently truncates.

    BGE-M3 never truncates on this corpus, tokens.py already measured that,
    so this is meaningful only for e5-large; called for either model, it
    reports whatever that model's real limit cuts off, honestly, rather than
    special-casing one of the two.
    """
    limit = max_pure_content_tokens(model_key)
    over = sum(1 for c in chunks if count_tokens(c.text, model_key) > limit)
    return over, len(chunks)


# --- worker: one model, one subprocess ----------------------------------------

def run_worker(
    model_key: str, output_path: Path,
    golden_path: Path = GOLDEN_SET,
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
) -> bool:
    """Gate 1 and evaluate() for one model, over all three variants, writing
    the results to output_path as JSON. Called only as a subprocess.
    """
    print(f"[{model_key}] gate 1: pooling against its own card")
    card_failures = embedder.verify_against_model_card(model_key)
    if card_failures:
        for failure in card_failures:
            print(f"  FAIL  {failure}")
        return False
    print(f"[{model_key}]   ok  reproduces its card's example relationship")

    questions, _ = load_golden(golden_path)
    variant_results = {}
    variant_truncation = {}
    for variant, path in variant_paths.items():
        chunks = load_chunks(path)
        result = evaluate(model_key, chunks, questions)
        variant_results[variant] = result
        variant_truncation[variant] = truncation_rate(chunks, model_key)
        agg = aggregate(result["rows"])
        print(f"[{model_key}] {variant}: recall@10={agg['recall@10']:.3f} "
              f"mrr@10={agg['mrr@10']:.3f}")

    payload = {
        "model": model_key,
        "results": variant_results,
        "truncation": variant_truncation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return True


def load_winning_model(decision_path: Path = BAKEOFF_DECISION) -> str:
    """The model run.py chose, read back from disk rather than trusted to a
    constant edited by hand. store.qdrant imports this rather than keeping
    a second copy: the module downstream of the decision reads it from the
    same place, so there is exactly one place that can get it wrong.
    """
    if not decision_path.exists():
        raise FileNotFoundError(
            f"{decision_path} not found. Run `python cli.py embed --bakeoff` "
            f"first; nothing downstream should build against a model the "
            f"bake-off never chose."
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    return decision["winner"]


def run_embed(variant_paths: dict[str, Path] = CONTEXT_OUTPUTS) -> bool:
    """Embed every chunk in every variant with the winning model.

    evaluate() already computes and caches these same vectors while scoring
    the bake-off, so a run right after `--bakeoff` is a cache hit reporting
    what already exists on disk. This exists as its own step for the case
    that matters later: re-embedding after a chunk file changes, without
    re-running the comparison that already produced a decision.

    Touches exactly one model, the winner, so it runs in-process rather
    than through run.py's subprocess protocol: the crash that protocol
    exists for only appears when a second distinct model is loaded in the
    same process, and this function never does that.
    """
    try:
        model_key = load_winning_model()
    except FileNotFoundError as error:
        print(error)
        return False
    print(f"embedding with {model_key}")

    missing = [p for p in variant_paths.values() if not p.exists()]
    if missing:
        print(f"Missing {[str(p) for p in missing]}. Run `python cli.py context` first.")
        return False

    for variant, path in variant_paths.items():
        chunks = load_chunks(path)
        vectors = embedder.embed_passages([c.text for c in chunks], model_key)
        print(f"  {variant}: {vectors.shape[0]} vectors, dimension {vectors.shape[1]}")
    embedder.release(model_key)
    return True


if __name__ == "__main__":
    # --worker <model_key> <output_path> is an internal protocol, not a
    # user-facing flag: run.py invokes exactly this via subprocess.run, one
    # model per process, and nothing else should call it directly.
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        ok = run_worker(sys.argv[2], Path(sys.argv[3]))
        sys.exit(0 if ok else 1)
    print("bakeoff.py is a worker module now; run `python cli.py embed --bakeoff` "
          "or `python -m pipeline.embedding.run`.")
    sys.exit(1)
