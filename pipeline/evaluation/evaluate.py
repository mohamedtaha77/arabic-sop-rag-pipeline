"""The stage 10 entry point: harness, judge, report, in order.

Position: every other file in this package does one job (run the three
arms, grade them, compute rows and comparisons, write prose). This file
is what a cold start actually calls, in the manner of every earlier
stage's own evaluate.py, and the only file in this package that decides
an order to run the others in.

client.unload_models() runs between phases, not only within harness.py's
own loop: the judge phase loads JUDGE_MODEL fresh after the harness
phase leaves whichever model it last used resident, and generation
evaluate.py's own docstring already measured that a long, multi-phase
run benefits from a reset between its own heaviest phases, not only
between individual calls.

A cold start of this file is only as reliable as harness.run()'s own
first pass, and that pass is not guaranteed clean on this machine.
harness.py's own module docstring has the full account: a raw access
violation, not a catchable Python exception, can still occur on a
genuine cache miss under tight system memory, and a raw access
violation ends this whole process, judge and report phases included,
before they ever run. Confirmed directly: a cold `python cli.py
evaluate` segfaulted mid-harness on this machine while every phase
run separately, against data already on disk, completed cleanly.
harness.py's own resumability is what makes this recoverable rather
than fatal: re-running this same command (or harness.py alone) picks
up from whatever RUNS_OUTPUT already holds and answers only what is
still missing. On a machine measured to crash repeatedly at the same
point, the honest operational answer is the same one router.md
already gives for a different fault: a command-level retry, not a
fix inside this file, since there is no Python-level catch for a
segfault to defend against.

What this file does not do: it does not decide how any question gets
answered, judged, scored, compared, or written up. Every one of those
decisions lives in the file that makes it.
"""

from __future__ import annotations

from ..config import REPORT_OUTPUT
from ..llm import client
from ..techniques import rerank
from . import harness, ragas_judge, report


def run() -> bool:
    # rerank.warm_up() before harness.run()'s own open_shipping(): the
    # adaptive and forced arms both reach Reranking, and every earlier
    # stage's own docstring for this ordering states the measured reason
    # it is load-bearing rather than a style preference.
    rerank.warm_up()
    print("--- phase 1: harness, three arms over the golden set ---", flush=True)
    harness.run()
    client.unload_models()

    print("--- phase 2: judge, the four required metrics ---", flush=True)
    ragas_judge.run()
    client.unload_models()

    print("--- phase 3: report ---", flush=True)
    report.run()

    print(f"stage 10 complete. See {REPORT_OUTPUT}.", flush=True)
    return True


if __name__ == "__main__":
    run()
