"""The ArmRun contract: what one question, answered by one arm, looks like.

Position: nothing here decides anything and nothing here calls a model, in
the manner of generation/answer.py and golden/question.py. harness.py is
what produces an ArmRun, running a question through one of three arms;
this file only says what shape one takes and which values are legal.

Why the shape is a record and not a re-derivation. Answer and QuestionRun
already carry everything a report needs, but neither is meant to survive
past the process that built it: QuestionRun holds a live ShippingHandle's
own ScoredChunk objects, and Answer's own traces mix dataclass instances
(SynthesisTrace, EntailmentSummary, PresentTrace, GuardReport) with plain
dicts depending on which path an answer took. ArmRun is the flattened,
JSON-safe copy of the fields stage 10 actually reads, taken once, right
after generation.run.answer returns, so that scoring, judging, comparing
and re-rendering REPORT.md are all reads of one file on disk rather than
re-runs of a model. That split is the difference between a report that
survives a re-render and one that pays for local generation every time
it changes a sentence, and on this machine, with the crash history
LEARNING/generation.md records, that is not a theoretical concern.

What this file does not do: it does not run a question, and it does not
decide which arm a question belongs to. harness.py does both.

One kind outside generation.answer.ANSWER_KINDS is legal here, "error",
found necessary rather than anticipated: the basic arm's own section 2
flow sends every golden question through raw, untransformed retrieval,
including Q4 and Q19, which LEARNING/generation.md already documents as
unstable under exactly that condition (both overflow SYNTHESIS_MAX_TOKENS
without a query-transformation technique smoothing the context first,
the reason both are excluded from every standalone gate elsewhere in this
project). client.py raises rather than silently truncating on that
overflow, which is the right call for a single question and the wrong
one for a 48-question unattended batch: one raise would otherwise end the
whole harness run. "error" is what a question becomes when the call
genuinely raised before an Answer could be produced, recorded as data
this stage's own report states plainly, on the same terms Q2's uncited
residual and CRAG's false-positive count already ship as measured facts
rather than smoothed away.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The three ways one question gets answered for this stage's own
# comparison, not to be confused with router.ROUTES, which names how one
# question moves through the shipped system regardless of which arm asked
# for it:
#
#   basic       section 2's own flow, no router: Question -> Embedding ->
#               Vector Search -> Top-K -> Prompt -> LLM -> Answer. Built
#               with route_override="basic_rag" and TechniqueSet.none(),
#               over all 20 golden questions
#   adaptive    the real shipped system, router and all: technique_set is
#               left as None so run.py's own routing and runtime triggers
#               decide everything, over all 20 golden questions
#   forced      each question's own expected technique set forced on,
#               over the 8-question population generation/evaluate.py's
#               own measure_crag_threshold already uses (Q3-Q7, Q19, Q20,
#               Q10). Exists because the router currently sends only 5 of
#               20 questions to advanced_rag, which makes basic and
#               adaptive identical code paths on the other 15: without a
#               third arm the headline comparison would rest on five
#               questions
ARMS = ("basic", "adaptive", "forced")


@dataclass
class ArmRun:
    """One question, answered once by one arm, flattened for storage.

        id                    the golden question id, Q1 to Q20
        arm                   one of ARMS
        question              the question as asked
        route                 the route this arm's call actually took,
                               "simple"/"basic_rag"/"advanced_rag" for
                               adaptive and forced, always "basic_rag" for
                               the basic arm by construction
        gate_matched          whether the deterministic pre-gate caught
                               this before any model ran. Always False for
                               the basic arm, since route_override skips
                               the pre-gate along with the router
        refusal_kind          one of question.REFUSAL_KINDS when this
                               answer refused, None otherwise
        executed              technique names that actually ran, in
                               execution order (QuestionRun.executed).
                               Empty for the basic arm by construction,
                               and for a "simple" route on any arm
        chunk_ids             retrieved chunk ids, in ranked order
                               (QuestionRun.chunk_ids). Empty for
                               "simple". section 2's own "Retrieved
                               document/chunk IDs" field, per arm
        contexts              the retrieved chunks' own text, same order
                               as chunk_ids, kept apart from context_text
                               so ragas_judge.py can hand a judge each
                               passage separately rather than re-parsing
                               a rendered prompt block back into pieces
        context_text          QuestionRun.context_text, the same rendered,
                               citation-numbered string the synthesiser
                               was actually shown. Kept whole for the
                               report and for a person reading by eye
        history_source        how this question's own prior-turn history
                               was built, for the one question in the
                               golden set that carries a depends_on (Q3).
                               "none" for every question without one,
                               "same_arm" when this arm's own earlier,
                               generated answer to the dependency was
                               used, "golden_reference" when it was not
                               available (the dependency falls outside
                               the forced arm's own eight-question
                               population) and Q2's own reference answer
                               was used instead, the same fallback
                               generation/evaluate.py's own
                               measure_crag_threshold already uses for
                               exactly this reason
        kind                  one of generation.answer.ANSWER_KINDS, or
                               "error" when generation.run.answer raised
                               before producing one (see the module
                               docstring). Every field below is empty or
                               False on an error record except ledger,
                               which still holds whatever was spent before
                               the call that failed
        error                 the exception's own str(), or None. The
                               single field every reader checks first:
                               non-None means every other content field
                               (text, citations, traces) is empty by
                               construction, not by coincidence
        text                  what a user sees, Answer.text
        synthesised           Answer.synthesised
        presented             Answer.presented, empty when the presenter
                               never ran
        presenter_rejected    Answer.presenter_rejected
        citations             chunk ids the answer actually cites, in the
                               order their markers first appear
        generation_traces     Answer.traces, serialised: every dataclass
                               value (SynthesisTrace, EntailmentSummary,
                               PresentTrace, GuardReport, or a refusal's
                               own GuardReport) turned into a plain dict
                               by _to_jsonable, keyed by the same step
                               name Answer already uses. Empty for "direct"
        technique_traces      QuestionRun.traces, serialised the same way,
                               keyed by schema.TECHNIQUES names. What
                               section 16's per-technique analysis
                               questions read (paraphrases, sub-questions,
                               extracted filters), per arm
        ledger                Answer.ledger.to_dict(), the one combined
                               record covering the router call, every
                               technique this arm ran, and both generation
                               calls: generation.run.answer hands the same
                               Ledger through techniques.run.answer,
                               synthesise.apply, entail.check and
                               present.apply in turn, so this single dict
                               is already section 7's full per-question
                               cost, not one piece of it
    """

    id: str
    arm: str
    question: str
    route: str
    gate_matched: bool
    refusal_kind: str | None
    executed: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    contexts: tuple[str, ...]
    context_text: str
    history_source: str
    kind: str
    text: str
    synthesised: str
    presented: str
    presenter_rejected: bool
    citations: tuple[str, ...]
    generation_traces: dict[str, Any]
    technique_traces: dict[str, Any]
    ledger: dict[str, Any]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"{self.arm!r} is not one of {ARMS}")


# --- serialising traces, once, in the one place that needs to -----------------

def to_jsonable(value: Any) -> Any:
    """A dataclass instance, a dict of them, or a plain value, all turned
    into something json.dump accepts.

    Answer.traces and QuestionRun.traces both hold whatever the step that
    produced them returned, most often a dataclass (SynthesisTrace,
    EntailmentSummary, PresentTrace, GuardReport, or one of the eight
    techniques' own trace types), sometimes already a plain dict (a
    refusal's own {"Refusal guard": GuardReport} still nests one). One
    recursive helper here, rather than each caller reaching for
    dataclasses.asdict and getting it wrong the first time a trace turns
    out to nest a tuple of tuples, which GuardReport.drift_segments
    already does.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


# --- storage --------------------------------------------------------------------

def save_runs(runs: list[ArmRun], path: Path) -> None:
    """Write every arm's runs to one JSON file.

    UTF-8 and ensure_ascii off, the same reason every other Arabic-bearing
    file in this project is written that way: the file has to stay
    readable in an editor, which is the only practical way to check an
    Arabic answer by eye.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(run) for run in runs]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_runs(path: Path) -> list[ArmRun]:
    """Read every arm's runs back, tuples restored where ArmRun declares
    them: json has no tuple type, so a list read back verbatim would fail
    ArmRun's own type contract silently on first use rather than loudly
    here.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    # __future__ annotations means every field's own .type is the
    # unevaluated string ("tuple[str, ...]"), not a real type object, so
    # this checks the string rather than an __origin__ that was never
    # going to be there.
    tuple_fields = {
        name for name, f in ArmRun.__dataclass_fields__.items()
        if "tuple" in str(f.type)
    }
    runs = []
    for entry in raw:
        data = dict(entry)
        for name in tuple_fields:
            if name in data and isinstance(data[name], list):
                data[name] = tuple(data[name])
        runs.append(ArmRun(**data))
    return runs


def index_by_arm(runs: list[ArmRun]) -> dict[str, dict[str, ArmRun]]:
    """runs grouped by arm, then by question id, the shape every reader
    after harness.py actually wants rather than a flat list each has to
    re-group on its own.
    """
    by_arm: dict[str, dict[str, ArmRun]] = {arm: {} for arm in ARMS}
    for run in runs:
        by_arm[run.arm][run.id] = run
    return by_arm
