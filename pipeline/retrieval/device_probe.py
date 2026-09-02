"""Whether the query-time embedder should run on CPU or GPU, measured.

advanced-rag-plan.md's component table put the embedder "GPU, index time
only" and Ollama as "the only model resident at query time". Hybrid
retrieval breaks that assumption: every query needs a dense vector and, for
BGE-M3, a sparse one too, so something has to decide where that computation
runs once a generation model may already be resident on the same 4 GB card.

This is not decided by argument. It is measured here, once, the same way
the embedding model itself was chosen by a bake-off rather than a
preference, and the answer retriever.py actually builds against is written
to disk rather than assumed from this file's own conclusion.

Two separate subprocesses, never one process asking for both devices in
turn. Confirmed directly while building this probe, in an isolated
throwaway script before this module existed: a second `AutoModel.
from_pretrained` call in a process that already moved a model to CUDA once
segfaults, even when the second load's destination is CPU and never
touches CUDA again. embedder.py's own module docstring documents the
original, narrower version of this finding (a second CUDA *transfer*); the
isolated test behind this module found the real boundary is a second
*load*, full stop, which is why this file's two device measurements are
run.py-style subprocesses rather than a single load, time, release, reload
sequence.

What this module does not do: it does not pick a device for you. It writes
a comparison; a person, or LEARNING/retrieval.md, reads it and retriever.py
is built to whichever answer that reading settles on.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import GOLDEN_SET, PROCESSED_DIR
from ..golden.question import load_golden

DEVICE_PROBE_OUTPUT = PROCESSED_DIR / "05_device_probe.md"

_SAMPLE_SIZE = 10  # answerable questions to time; the golden set has 18


def _sample_queries(golden_path: Path = GOLDEN_SET, n: int = _SAMPLE_SIZE) -> list[str]:
    questions, _ = load_golden(golden_path)
    answerable = [q.question for q in questions if q.expect == "answerable"]
    return answerable[:n]


# --- worker: one device, one subprocess ---------------------------------------

def run_worker(device: str, output_path: Path) -> bool:
    """Load bge-m3 on one device, embed the sample queries, time each call
    individually, write vectors and latencies to output_path. Called only
    as a subprocess; see the module docstring for why.
    """
    import psutil

    # torch before numpy: see embedder.py's own import-order note. This
    # module hits the same hazard, importing both directly and via embedder.
    import torch  # noqa: F401
    from .. import config  # noqa: F401  (import order sentinel, no-op)
    from ..embedding import embedder

    process = psutil.Process()
    rss_before = process.memory_info().rss

    queries = _sample_queries()
    latencies = []
    vectors = []
    for query in queries:
        started = time.perf_counter()
        vector = embedder.embed_queries([query], "bge-m3", device=device, use_cache=False)[0]
        latencies.append(time.perf_counter() - started)
        vectors.append(vector)

    rss_after = process.memory_info().rss

    gpu_memory_mib = None
    if device == "cuda" and torch.cuda.is_available():
        gpu_memory_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    payload = {
        "device": device,
        "queries": queries,
        "latencies_s": latencies,
        "rss_before_mib": rss_before / (1024 ** 2),
        "rss_after_mib": rss_after / (1024 ** 2),
        "gpu_max_allocated_mib": gpu_memory_mib,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path.with_suffix(".npz"),
        vectors=np.stack(vectors),
    )
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    embedder.release("bge-m3")
    return True


# --- orchestrator: two subprocesses, one comparison ----------------------------

def run(output_path: Path = DEVICE_PROBE_OUTPUT) -> bool:
    work_dir = output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    vectors: dict[str, np.ndarray] = {}

    for device in ("cuda", "cpu"):
        json_path = work_dir / f"_device_probe_{device}.json"
        npz_path = json_path.with_suffix(".npz")
        print(f"probing device={device} in its own subprocess...")
        completed = subprocess.run(
            [sys.executable, "-m", "pipeline.retrieval.device_probe",
             "--worker", device, str(json_path)],
        )
        if completed.returncode != 0:
            print(f"  FAIL  worker for device={device} exited "
                  f"{completed.returncode}")
            return False
        results[device] = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            vectors[device] = data["vectors"]
        json_path.unlink()
        npz_path.unlink()

    if results["cuda"]["queries"] != results["cpu"]["queries"]:
        print("  FAIL  the two workers did not embed the same queries")
        return False

    cosines = [
        float(np.dot(vectors["cuda"][i], vectors["cpu"][i]))
        for i in range(len(vectors["cuda"]))
    ]

    lines = [
        "# Query-time embedder device policy",
        "",
        f"BGE-M3, {len(results['cuda']['queries'])} golden-set queries, "
        f"each device in its own subprocess so the comparison never risks "
        f"the second-load segfault this module's docstring records.",
        "",
    ]

    # The first call on each device pays a one-time cost, cuDNN algorithm
    # selection on CUDA, that has nothing to do with steady-state per-query
    # latency: measured directly while building this probe, the first CUDA
    # call took 7.72s and every one after took 0.01-0.05s. retriever.py
    # holds the model resident across many queries, so the number that
    # matters for it is steady state, and reporting only a mean across all
    # calls would let that one outlier dominate a number meant to describe
    # the other nine.
    def _stats(latencies: list[float]) -> dict[str, float]:
        first, rest = latencies[0], latencies[1:]
        return {
            "first_call_s": first,
            "steady_mean_s": float(np.mean(rest)) if rest else first,
            "steady_median_s": float(np.median(rest)) if rest else first,
        }

    cuda_stats = _stats(results["cuda"]["latencies_s"])
    cpu_stats = _stats(results["cpu"]["latencies_s"])

    lines += [
        "| | CUDA (fp16) | CPU (fp32) |",
        "|---|---|---|",
        f"| first-call latency (s) | {cuda_stats['first_call_s']:.4f} | "
        f"{cpu_stats['first_call_s']:.4f} |",
        f"| steady-state mean (s), remaining {len(results['cuda']['latencies_s']) - 1} queries | "
        f"{cuda_stats['steady_mean_s']:.4f} | {cpu_stats['steady_mean_s']:.4f} |",
        f"| steady-state median (s) | {cuda_stats['steady_median_s']:.4f} | "
        f"{cpu_stats['steady_median_s']:.4f} |",
        f"| host RSS delta (MiB) | "
        f"{results['cuda']['rss_after_mib'] - results['cuda']['rss_before_mib']:.1f} | "
        f"{results['cpu']['rss_after_mib'] - results['cpu']['rss_before_mib']:.1f} |",
        f"| GPU peak allocated (MiB) | "
        f"{results['cuda']['gpu_max_allocated_mib']:.1f} | n/a |",
    ]

    speedup = (
        cpu_stats["steady_mean_s"] / cuda_stats["steady_mean_s"]
        if cuda_stats["steady_mean_s"] > 0 else float("inf")
    )
    lines += [
        "",
        f"Cosine agreement between the CPU-fp32 and GPU-fp16 vector for the "
        f"same query text, same {len(cosines)} queries: mean "
        f"{np.mean(cosines):.6f}, min {min(cosines):.6f}. Passages in the "
        f"store were embedded fp16 on the card; a query embedded fp32 on "
        f"CPU is compared against that fp16 passage space regardless of "
        f"which device policy wins, so this number is what that mismatch "
        f"actually costs in practice rather than a theoretical concern.",
        "",
        f"CUDA is {speedup:.1f}x the steady-state speed of CPU once both are "
        f"warm. CUDA's own first call costs {cuda_stats['first_call_s']:.2f}s "
        f"against CPU's {cpu_stats['first_call_s']:.2f}s, a one-time cost "
        f"paid once per process, not per query.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {output_path}")
    for line in lines:
        print(line)
    return True


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        ok = run_worker(sys.argv[2], Path(sys.argv[3]))
        sys.exit(0 if ok else 1)
    sys.exit(0 if run() else 1)
