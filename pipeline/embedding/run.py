"""Runs the bake-off, one model at a time, each in its own process.

The orchestrator. Deliberately the lightest module in this package: it
imports metrics.py for the bootstrap and the report, and stdlib for
everything else, and it never imports torch, transformers or embedder.py
directly. That is not a style preference, it is the actual fix for a real
problem measured on this machine: bakeoff.py's --worker subprocess crashed
far more often when spawned from a parent that had already loaded torch and
transformers itself than when spawned from a bare interpreter. A parent that
never touches either leaves the child, which does the real work of loading
a 2+ GB model, more of this machine's tight memory to work with.

subprocess.run, not multiprocessing or a thread: each model needs a fresh
OS process with a fresh CUDA context, and multiprocessing's fork-like
start methods do not reliably give that on every platform, where a plain
subprocess always does.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..config import (
    BAKEOFF_DECISION,
    BAKEOFF_OUTPUT,
    CHUNKS_OUTPUT,
    CONTEXT_OUTPUTS,
    EMBED_MODELS,
    GOLDEN_SET,
    PROJECT_ROOT,
)
from ..golden.question import corpus_fingerprint, load_golden
from .metrics import decide, render_report

# A CUDA model load failing on this development machine is not always
# reproducible from the same starting conditions: the same load sometimes
# succeeds and sometimes segfaults, traced to system virtual memory running
# genuinely low (measured directly: FreeVirtualMemory swung between about
# 535 MB and 4.6 GB across this project's build, driven by what else was
# running at the time), not to anything this code controls. A crashed
# subprocess costs nothing but the retry, since each attempt starts a fresh
# process regardless, so retrying is the honest response to a resource
# ceiling that has not been raised yet rather than a way to paper over a
# logic bug. If every attempt fails, that is real and gets reported as one.
_SUBPROCESS_RETRIES = 8

# Seconds between attempts. A crash frees whatever it was holding, but not
# necessarily the instant the process exits; a longer pause gave the
# following attempt a better chance in practice than retrying immediately.
_SUBPROCESS_RETRY_DELAY_S = 45


def _run_model_in_subprocess(model_key: str, work_dir: Path) -> dict[str, Any]:
    """Launch bakeoff.py's worker for one model in a fresh process and read
    its JSON result back. sys.executable rather than a bare "python": the
    subprocess has to be this same virtualenv's interpreter, not whatever
    "python" resolves to on PATH.
    """
    out_path = work_dir / f"{model_key}.json"
    last_returncode = None
    for attempt in range(1, _SUBPROCESS_RETRIES + 1):
        if out_path.exists():
            out_path.unlink()
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.embedding.bakeoff",
             "--worker", model_key, str(out_path)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and out_path.exists():
            return json.loads(out_path.read_text(encoding="utf-8"))
        last_returncode = result.returncode
        print(f"  {model_key} subprocess attempt {attempt}/{_SUBPROCESS_RETRIES} "
              f"failed (exit {last_returncode}); a crashed subprocess costs "
              f"only the retry, since the next attempt is a fresh process")
        if attempt < _SUBPROCESS_RETRIES:
            time.sleep(_SUBPROCESS_RETRY_DELAY_S)

    raise RuntimeError(
        f"{model_key} subprocess failed {_SUBPROCESS_RETRIES} times in a row "
        f"(last exit {last_returncode}). This machine's virtual memory is a "
        f"known constraint; see LEARNING/embedding.md."
    )


def run(
    golden_path: Path = GOLDEN_SET,
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
    output_path: Path = BAKEOFF_OUTPUT,
    decision_path: Path = BAKEOFF_DECISION,
) -> bool:
    if not golden_path.exists():
        print(f"{golden_path} not found. Run `python cli.py golden` first.")
        return False
    missing = [p for p in variant_paths.values() if not p.exists()]
    if missing:
        print(f"Missing {[str(p) for p in missing]}. Run `python cli.py context` first.")
        return False

    questions, fingerprint = load_golden(golden_path)
    current_fingerprint = corpus_fingerprint(CHUNKS_OUTPUT)
    if fingerprint != current_fingerprint:
        print(
            f"golden set fingerprint does not match {CHUNKS_OUTPUT.name}. "
            f"Run `python cli.py golden` to check, or the chunk build "
            f"changed underneath the golden set."
        )
        return False

    answerable = [q for q in questions if q.expect == "answerable"]
    print(f"{len(answerable)} answerable questions, "
          f"{sum(len(q.gold_chunk_ids) for q in answerable)} gold chunks total")

    results: dict[tuple[str, str], dict[str, Any]] = {}
    truncation: dict[tuple[str, str], tuple[int, int]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for model_key in EMBED_MODELS:
            print(f"\n=== {model_key} (subprocess) ===")
            payload = _run_model_in_subprocess(model_key, work_dir)
            for variant, result in payload["results"].items():
                results[(model_key, variant)] = result
                over, total = payload["truncation"][variant]
                truncation[(model_key, variant)] = (over, total)

    decision = decide(
        results[("bge-m3", "none")]["rows"], results[("e5-large", "none")]["rows"]
    )
    print(f"\nDecision: {decision['winner']} ({decision['basis']})")

    report = render_report(results, decision, truncation, len(answerable))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(f"\nwritten to {output_path} and {decision_path}")
    return True


if __name__ == "__main__":
    run()
