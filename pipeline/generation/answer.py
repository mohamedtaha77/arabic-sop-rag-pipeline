"""The Answer contract: what one answered question looks like.

Position: nothing here decides anything and nothing here calls a model,
in the manner of router/schema.py and golden/question.py. synthesise.py,
guard.py, present.py and run.py are what produce an Answer; this file
only says what shape one takes and which values are legal, so a typo in
a kind raises here rather than travelling into stage 10's report as a
row nobody notices.

Why this stage exists at all, stated once here because every other file
in the package is a piece of it: a prompt telling a model not to add
facts is not enforcement. So generation is split in two, with a
programmatic guard between the halves. The synthesiser sees the
retrieved context and writes a grounded, cited answer. The presenter
sees only that answer and its citations, never the context and never a
retrieval tool, and is rejected outright if it introduces a number, a
date, a citation or a Latin-script name the synthesiser did not already
say. The difference between claiming that constraint and enforcing it is
this rejection, and how often it fires is the number stage 9 exists to
produce.

The trace for each step lives in the file that produces it, the same way
stage 8 keeps CragTrace in crag.py and RerankTrace in rerank.py rather
than gathering them somewhere central. Answer therefore carries them in
a dict keyed by step name, exactly as techniques.run.QuestionRun carries
its own, which also keeps this module free of an import from every file
that would otherwise import it back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..golden.question import REFUSAL_KINDS
from ..llm.ledger import Ledger
from ..techniques.run import QuestionRun

# How this question was answered. Three kinds rather than four, because
# the two refusals differ in why they refused, not in what happened
# afterwards: refusal_kind carries that distinction, reusing
# question.REFUSAL_KINDS rather than inventing a second vocabulary for
# the same two cases the golden set already names.
#
#   direct     the "simple" route. Answered by the model with no
#              retrieved context at all, either because the
#              deterministic pre-gate caught a greeting or because the
#              router judged the question needed no documents
#   grounded   basic_rag or advanced_rag. Synthesised from retrieved
#              context, cited, guarded, and presented
#   refused    the corpus does not answer this. Worded by the model from
#              a fixed refusal instruction, carrying no citations
ANSWER_KINDS = ("direct", "grounded", "refused")


@dataclass
class Answer:
    """One answered question, end to end.

        question            the question as asked
        kind                one of ANSWER_KINDS
        text                what a user sees. The presented text when
                             the presenter ran and passed the guard, the
                             synthesised text when it ran and was
                             rejected, and the direct or refusal text
                             otherwise. One field to read rather than a
                             rule to reapply at every call site
        synthesised         the synthesiser's own words, kept whether or
                             not the presenter improved on them, because
                             the rejection path returns these and the
                             report has to be able to show both
        presented           the presenter's own words, or empty when the
                             presenter never ran. Kept even when
                             rejected: an answer the guard threw away is
                             exactly the evidence that the guard did
                             something
        presenter_rejected  whether the guard blocked the presenter and
                             fell back to the synthesised text. False
                             whenever the presenter never ran, which is
                             not the same as passing, so kinds other
                             than "grounded" are read through kind
                             rather than through this flag alone
        citations           chunk ids the answer actually cites, in the
                             order their markers first appear. Empty for
                             "direct" and "refused", and section 2 of the
                             task reads this beside QuestionRun's own
                             chunk_ids: what was retrieved and what was
                             used are different questions
        refusal_kind        one of REFUSAL_KINDS when kind is "refused",
                             None otherwise. out_of_domain comes from the
                             router, before retrieval spends anything;
                             non_answering_retrieval comes from CRAG,
                             after retrieval has already run and returned
                             plausible chunks that do not answer
        run                 the QuestionRun this was generated from,
                             carried whole rather than copied field by
                             field, so stage 10 reads the route, the
                             executed techniques and the retrieved ids
                             from the one record that already holds them
        traces              one entry per generation step that ran, keyed
                             by that step's own ledger.STEPS name
        ledger              this question's cost and latency record, the
                             same one techniques.run.answer was handed,
                             now carrying the generation rows too

    Constructed only by run.py. Every other file in this package returns
    text and a trace, so there is one place where a kind, a refusal and a
    guard verdict are reconciled into a single record rather than several.
    """

    question: str
    kind: str
    text: str
    synthesised: str
    presented: str
    presenter_rejected: bool
    citations: tuple[str, ...]
    refusal_kind: str | None
    run: QuestionRun
    ledger: Ledger
    traces: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ANSWER_KINDS:
            raise ValueError(f"{self.kind!r} is not one of {ANSWER_KINDS}")
        if self.refusal_kind is not None and self.refusal_kind not in REFUSAL_KINDS:
            raise ValueError(
                f"{self.refusal_kind!r} is not one of {REFUSAL_KINDS}"
            )
        # A refusal that forgot to say why it refused, or a non-refusal
        # carrying a refusal reason, are both states that would read as
        # ordinary rows in stage 10's per-question table while meaning
        # something went wrong here. Neither can be recovered from
        # downstream, so both raise rather than being tidied up.
        if (self.kind == "refused") != (self.refusal_kind is not None):
            raise ValueError(
                f"kind={self.kind!r} and refusal_kind={self.refusal_kind!r} "
                f"disagree: a refusal needs a kind, and only a refusal has one"
            )
        if self.kind != "grounded" and self.citations:
            raise ValueError(
                f"kind={self.kind!r} cites {self.citations}, but only a "
                f"grounded answer reads retrieved context at all"
            )
