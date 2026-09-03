"""The router's contract: routes, techniques, and the decision they compose.

Position: nothing here decides anything. gate.py's deterministic pre-gate
and router.py's model call are what produce a RouteDecision; this file only
says what shape one takes and which values are legal, in the manner of
chunk.CHUNK_TYPES and ledger.STEPS. A typo in a route or technique name
raises here rather than silently returning nothing, or dropping a row out of
the cost table the way ledger.py warns a step name outside STEPS would.

What this module does not do: it does not call a model, and it does not
decide a route for any real question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..golden.question import EXPECTED_ROUTES, REFUSAL_KINDS

# The same tuple question.py already declares, imported rather than retyped:
# EXPECTED_ROUTES exists precisely so a golden question's expected_route can
# be compared against whatever the router actually returns, and a second,
# hand-copied tuple here could drift from it with nothing to catch that.
# advanced-rag-plan.md's own wording note still applies: the machine-readable
# value is "simple", the report's own prose calls the same thing "Direct".
ROUTES = EXPECTED_ROUTES

# The eight techniques sections 3 and 4 of the task name, as a closed
# vocabulary the router is given rather than invited to invent from. A
# technique name outside this tuple gets dropped by router.py and kept in
# RouteDecision.dropped, not silently accepted and not crashed on.
TECHNIQUES = (
    "Rewriting",
    "Multi-Query",
    "Decomposition",
    "HyDE",
    "Self-Query",
    "Reranking",
    "Compression",
    "CRAG",
)

# Every technique's own name onto its ledger.STEPS row, declared once because
# the ledger already spells two of these differently: "Rewriting" is
# ledger's "Rewriter", and "CRAG" is ledger's "CRAG evaluator". A rename on
# either side that forgot to update the other would drop a row out of the
# cost table in silence; this is what keeps the two vocabularies from
# drifting apart in the first place.
TECHNIQUE_TO_STEP = {
    "Rewriting": "Rewriter",
    "Multi-Query": "Multi-Query",
    "Decomposition": "Decomposition",
    "HyDE": "HyDE",
    "Self-Query": "Self-Query",
    "Reranking": "Reranking",
    "Compression": "Compression",
    "CRAG": "CRAG evaluator",
}


@dataclass
class RouteDecision:
    """What the router decided for one question.

        route          one of ROUTES
        reason         the router's own stated reason, kept verbatim for
                        the report and for analysis question 1
        refusal_kind   one of question.REFUSAL_KINDS when route is "simple"
                        and the question is refused before retrieval ever
                        runs, None otherwise. A "simple" route with
                        refusal_kind None is the other real case that value
                        has to cover: a question in scope, or close enough
                        to it, that the LLM answers directly with no
                        retrieval at all, the shape the task's own Q1,
                        "What is RAG?", is built around. Reused rather than
                        a second vocabulary invented here, since golden.py's
                        expect/refusal_kind pair already names this exact
                        distinction against the same questions the router
                        is scored on. CRAG's own downstream refusal (Q10,
                        non_answering_retrieval) is never set here: that
                        decision is made after retrieval has already run,
                        so it belongs on techniques.run's QuestionRun record,
                        not on the router's own decision
        requested      techniques the router asked for, straight from its
                        own output, already validated against TECHNIQUES
        executed       techniques that actually ran, which can differ from
                        requested once run.py's own runtime triggers fire
                        (Compression on context overflow, CRAG on low
                        retrieval confidence). Section 14's "Techniques
                        Used" column reads this one; analysis question 1
                        reads requested
        dropped        technique names the router returned that were not in
                        TECHNIQUES, kept here rather than discarded so a
                        hallucination shows up in the trace instead of
                        vanishing from it
        raw            the model's own JSON text, unparsed, so the router
                        report can quote a decision directly rather than
                        reconstruct it from the fields above

    Constructed by gate.py for a pre-gate match, and by router.py for a
    model call. run.py never builds one directly, so both sources are held
    to the same validation.
    """

    route: str
    reason: str
    refusal_kind: str | None = None
    requested: tuple[str, ...] = field(default_factory=tuple)
    executed: tuple[str, ...] = field(default_factory=tuple)
    dropped: tuple[str, ...] = field(default_factory=tuple)
    raw: str = ""

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"{self.route!r} is not one of {ROUTES}")
        if self.refusal_kind is not None and self.refusal_kind not in REFUSAL_KINDS:
            raise ValueError(f"{self.refusal_kind!r} is not one of {REFUSAL_KINDS}")
        for label, names in (("requested", self.requested), ("executed", self.executed)):
            unknown = set(names) - set(TECHNIQUES)
            if unknown:
                raise ValueError(f"{label} contains unknown techniques: {unknown}")


# --- the eight switches -------------------------------------------------------

# TechniqueSet's own field names onto the TECHNIQUES vocabulary, declared
# once and read in both directions: from_names builds a TechniqueSet from
# whatever the router or a caller names, names() reads one back out. A
# second, inverted copy of this mapping living beside it would be exactly
# the kind of duplication TECHNIQUE_TO_STEP above exists to avoid.
_FIELD_TO_TECHNIQUE = {
    "rewriting": "Rewriting",
    "multi_query": "Multi-Query",
    "decomposition": "Decomposition",
    "hyde": "HyDE",
    "self_query": "Self-Query",
    "reranking": "Reranking",
    "compression": "Compression",
    "crag": "CRAG",
}
_TECHNIQUE_TO_FIELD = {technique: name for name, technique in _FIELD_TO_TECHNIQUE.items()}


@dataclass(frozen=True)
class TechniqueSet:
    """The eight switches, one per technique, independently on or off.

    This is what makes stage 10's ablation real rather than claimed: every
    technique is reachable through this one object, never through a
    hand-written conditional buried in run.py. none() is the configuration
    techniques/run.py's own gate checks against retriever.retrieve_shipping,
    chunk id for chunk id: if the two disagree with every technique off, the
    "advanced" numbers downstream would be measuring an accidental
    difference in the plumbing rather than the techniques.
    """

    rewriting: bool = False
    multi_query: bool = False
    decomposition: bool = False
    hyde: bool = False
    self_query: bool = False
    reranking: bool = False
    compression: bool = False
    crag: bool = False

    @classmethod
    def none(cls) -> "TechniqueSet":
        return cls()

    @classmethod
    def from_names(cls, names: tuple[str, ...]) -> "TechniqueSet":
        """Build from the router's own vocabulary. Any name outside
        TECHNIQUES is ignored here, not raised on: router.py already
        separated a hallucinated name into RouteDecision.dropped before this
        is ever called, so a second rejection here would just repeat that
        check on a narrower signal.
        """
        kwargs = {
            _TECHNIQUE_TO_FIELD[name]: True
            for name in names
            if name in _TECHNIQUE_TO_FIELD
        }
        return cls(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Back to the TECHNIQUES vocabulary, in TECHNIQUES order, for
        whichever switches are on.
        """
        return tuple(
            technique for technique in TECHNIQUES
            if getattr(self, _TECHNIQUE_TO_FIELD[technique])
        )
