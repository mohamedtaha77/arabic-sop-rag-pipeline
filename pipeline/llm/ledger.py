"""Which steps ran, what they cost, and the table the task asks for.

Section 7 of the task wants a per-step breakdown: every technique, whether it
executed, and what it spent. The tempting way to build that is to collect
whatever happened and print it at the end. That produces a transcript, not a
table. A transcript shows the four steps that ran and stays silent about the
six that did not, and that silence is exactly the information an ablation
needs.

So the row schema is declared first, in STEPS, and the table is rendered from
the schema rather than from the events. A technique that never fired gets a row
saying Executed? No against zeros, which is the difference between a report
showing a routing decision and one showing only its consequences.

The other half is that a step cannot be spent without being recorded. `call`
wraps the client, the cache and the record together, so no path reaches the
model without leaving a row. That is the same move stage 9 makes with the
grounding guard: enforce the property in the call path rather than trusting
every future caller to remember.

This module makes no calls of its own and holds no prompts.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
from typing import Any

from .cache import cached_chat
from .client import Response
from .pricing import DEFAULT_CARD, price

# Every row the table can hold, in the order the pipeline runs them. Declared
# rather than discovered, for the reason above.
#
# The task's own example names seven of these. Five are additions:
#
#   Multi-Query, Self-Query, Reranking   named in the plan as rows the example
#                                        leaves out, and each is a real
#                                        technique the ablation has to separate
#   Presenter, Grounding guard           stage 9 splits generation in two, and
#                                        the presenter is a second model call.
#                                        A call spending tokens without a row
#                                        makes TOTAL wrong, and TOTAL is the
#                                        number the table exists for
#
# SPEC_STEPS records which is which, so stage 10 can render the task's exact
# seven rows as well as the full set without either being rebuilt by hand.
STEPS = (
    "Router",
    "Rewriter",
    "Multi-Query",
    "Decomposition",
    "HyDE",
    "Self-Query",
    "Reranking",
    "Compression",
    "CRAG evaluator",
    "Final generation",
    "Grounding guard",
    "Presenter",
    "Contextualisation",
)

SPEC_STEPS = frozenset({
    "Router", "Rewriter", "Decomposition", "HyDE", "Compression",
    "CRAG evaluator", "Final generation",
})

# Steps that are CPU models rather than endpoint calls, recorded through
# record_local. They spend no tokens and real seconds. Running locally makes
# latency the scarce resource, so a cross-encoder costing two seconds a query
# belongs in the table rather than being invisible because it is free.
#
# "Grounding guard" was written here before stage 9 existed, on the plan's
# own original assumption that entailment would run on a local NLI model.
# Stage 9's own bake-off (generation/entail.py) measured two backends
# against a benchmark built from the golden set and found the LLM judge
# backend (JUDGE_MODEL, a real endpoint call) wins where it matters most:
# recall on a same-topic, wrong-fact fabrication, the exact shape this
# guard exists to catch, well ahead of the local NLI model's own recall
# on that specific case, even though the NLI model scored marginally
# higher in aggregate. The shipped mechanism spends real tokens, so it
# cannot stay classified as free CPU work; this is the same kind of
# correction techniques/run.py's own PROVISIONAL_CRAG_THRESHOLD comment
# already anticipates, an earlier stage's placeholder assumption
# superseded once a later stage actually measures the real answer. If a
# future re-run of that bake-off ever favours the local NLI backend
# instead, "Grounding guard" belongs back in this set, and
# generation/entail.py's own NLI path would need a record_local call
# added to match, which it does not have today because today's shipped
# path never reaches it.
LOCAL_STEPS = frozenset({"Reranking"})

# Steps paid once when the index is built, never per question. Stage 4's ~357
# contextualisation calls are the first of these: every per-question ledger
# carries the row, correctly reading Executed? No, because the cost belongs to
# the index rather than to any answer. verify() reads this set so a ledger that
# only ever does build work is not held to the "a question was answered" rule
# below, which is the rule for query-time ledgers, not index-time ones.
BUILD_STEPS = frozenset({"Contextualisation"})


@dataclass
class Entry:
    """One recorded call. A step may have several; Multi-Query always does."""

    step: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool


@dataclass
class Ledger:
    """The record for answering one question.

    One per question rather than one per run, because the task wants a
    per-question cost breakdown and a single global counter cannot be taken
    apart again afterwards.
    """

    label: str = ""
    card: str = DEFAULT_CARD
    entries: list[Entry] = field(default_factory=list)

    # --- recording ----------------------------------------------------------

    def call(self, step: str, messages: list[dict[str, str]], model: str,
             **kwargs: Any) -> Response:
        """Spend one model call against a step and record it.

        The only path in this pipeline from a prompt to the endpoint. Going
        around it by importing client.chat directly is possible and would leave
        the call out of the table, which is why nothing downstream should.
        """
        response = cached_chat(messages, model, **kwargs)
        self.record(step, response)
        return response

    def record(self, step: str, response: Response) -> None:
        """Record a call that has already been made."""
        self._check(step)
        self.entries.append(
            Entry(
                step=step,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_s=response.latency_s,
                cost_usd=price(response.prompt_tokens,
                               response.completion_tokens, self.card),
                cached=response.cached,
            )
        )

    def record_local(self, step: str, latency_s: float, model: str = "") -> None:
        """Record a CPU model step: real latency, no tokens, no cost."""
        self._check(step)
        self.entries.append(
            Entry(step=step, model=model, prompt_tokens=0, completion_tokens=0,
                  latency_s=latency_s, cost_usd=0.0, cached=False)
        )

    def _check(self, step: str) -> None:
        """Refuse a step outside the schema.

        chunk.py declares CHUNK_TYPES for the same reason, and notes that a
        typo there would return nothing rather than raise. Here it would drop a
        row out of the cost table in silence, so it raises.
        """
        if step not in STEPS:
            raise ValueError(
                f"{step!r} is not a declared step. Add it to STEPS in "
                f"ledger.py, so it gets a row whether or not it runs."
            )

    # --- reporting ----------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        """One row per declared step, in schema order, plus TOTAL."""
        by_step: dict[str, list[Entry]] = collections.defaultdict(list)
        for entry in self.entries:
            by_step[entry.step].append(entry)

        rows = [self._summarise(step, by_step.get(step, [])) for step in STEPS]
        rows.append(self._summarise("TOTAL", self.entries))
        return rows

    @staticmethod
    def _summarise(step: str, spent: list[Entry]) -> dict[str, Any]:
        """Aggregate the calls belonging to one row."""
        return {
            "step": step,
            "executed": bool(spent),
            "calls": len(spent),
            "prompt_tokens": sum(e.prompt_tokens for e in spent),
            "completion_tokens": sum(e.completion_tokens for e in spent),
            "latency_s": sum(e.latency_s for e in spent),
            "cost_usd": sum(e.cost_usd for e in spent),
        }

    def render(self) -> str:
        """The section 7 table, as Markdown for the report.

        Markdown because REPORT.md is where it ends up, and it still reads
        straight in a terminal, unlike anything right-to-left in this project.
        """
        lines = [
            "| Step | Executed? | Calls | Input | Output | Latency (s) | Cost (USD) |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in self.rows():
            lines.append(
                "| {step} | {executed} | {calls} | {prompt_tokens} | "
                "{completion_tokens} | {latency_s:.2f} | {cost_usd:.6f} |".format(
                    **{**row, "executed": "Yes" if row["executed"] else "No"}
                )
            )

        cached = sum(1 for e in self.entries if e.cached)
        note = (f"\nPriced against {self.card}; no money was spent. "
                f"{cached} of {len(self.entries)} calls served from cache, "
                f"reporting the tokens and latency the original call measured.")
        return "\n".join(lines) + "\n" + note

    def verify(self) -> list[str]:
        """Check what the table has to be true for. Empty when clean."""
        failures = []

        # The task states final generation always executes. A ledger without it
        # describes a question that was never answered, and a cost table for an
        # answer that does not exist is not a cheap run, it is a bug.
        #
        # Exempted when every entry is a build step: stage 4's ledger builds an
        # index and answers no question, so holding it to a rule written for
        # query-time ledgers would fail a run that never went wrong.
        query_time = any(e.step not in BUILD_STEPS for e in self.entries)
        if query_time and not any(e.step == "Final generation" for e in self.entries):
            failures.append("Final generation never ran")

        for entry in self.entries:
            if entry.step in LOCAL_STEPS and entry.prompt_tokens:
                failures.append(f"{entry.step} is a CPU step but spent tokens")
            if entry.step not in LOCAL_STEPS and not entry.prompt_tokens:
                failures.append(f"{entry.step} recorded a call with no input tokens")
        return failures

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for stage 10's per-question breakdown."""
        return {
            "label": self.label,
            "card": self.card,
            "entries": [asdict(e) for e in self.entries],
            "rows": self.rows(),
        }
