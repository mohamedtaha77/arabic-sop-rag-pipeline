"""One structured call: a question in, a RouteDecision out.

Position: gate.py has already had first refusal and returned None, meaning
this question is not an obvious greeting or fragment. router.route is what
decides the three things section 1 of the task asks for: whether the
corpus is even in scope, which of the three routes the question needs, and
which techniques it shows a real signal for. Everything downstream,
techniques/run.py's orchestration and evaluate.py's report, reads a
RouteDecision and never talks to the model directly for routing.

The suggested output in the task, {"route", "reason", "techniques"}, is
extended with one field, "in_scope", not in that suggestion. It exists
because "simple" genuinely covers two different outcomes, caught in
schema.RouteDecision's own docstring: a question this system should refuse
because it has nothing to do with the three manuals (Q9's shape), and a
question general enough to answer directly with no retrieval at all (the
task's own Q1, "What is RAG?", would be exactly this shape here). Both
route to "simple". Nothing in the suggested schema tells them apart, and
step 0's own probe already confirmed the endpoint tolerates an extra field
in JSON mode without complaint, so this is a cheap, honest extension rather
than a workaround.

What this module does not do: it does not run any technique, and it does
not retrieve anything. It returns a decision; techniques/run.py acts on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import GENERATOR_MODEL, GOLDEN_SET
from ..llm.client import LLMError
from ..llm.ledger import Ledger
from .schema import ROUTES, TECHNIQUES, RouteDecision

# Written once, read by every call. Names the corpus by its three actual
# manuals rather than describing the domain abstractly, because a router
# that only knows "bank procedures" in the abstract has nothing concrete to
# compare an out-of-scope question against; Q9 asking for investment advice
# has to fail a real comparison against these three subjects, not a vague
# one.
#
# The six-signal mapping is written out as a literal table rather than left
# implicit, the same move advanced-rag-plan.md's own architecture section
# makes: "the six signals are a technique-selection map rather than prose,
# and that mapping is what stops every question running every technique."
# Reranking, Compression and CRAG are deliberately excluded from what the
# model is asked to select: the plan already decided these three are
# always-on-for-advanced-rag or runtime-triggered, never a router judgement
# call, so asking the model to choose them would invite it to guess at a
# decision that was never its to make.
_SYSTEM_PROMPT = """\
You are the router for a retrieval-augmented question-answering system \
over exactly three Arabic internal procedure manuals of a bank (Housing \
Bank, marked Internal Use):

1. "Assets and Warehouse Operation Tasks and Procedures Manual": fixed \
asset inventory and depreciation, warehouse storage and FIFO issuance, \
quarterly account reconciliation, expense approval forms.
2. "Central Alarm Tasks and Procedures Manual": surveillance camera \
viewing approvals, recording retention and periodic inspection, fault \
handling, revision history.
3. "Central Mail and Files Unit Procedures Manual": incoming and outgoing \
mail via BPM and Jordan Post, registered mail, credit-file archiving and \
retention, lost or damaged mail handling, customer statement mailing.

Nothing else is in scope. No general banking advice, no products or \
services, nothing these three manuals do not themselves document as an \
internal operating procedure.

For every question, decide in_scope first, on its own, before anything \
else. Only then decide route and techniques.

in_scope: true or false. False whenever the question asks for advice, an \
opinion, a recommendation, or information about anything other than what \
these three manuals themselves document as their own internal \
procedures, no matter how short, plain, or innocent the question sounds. \
A short, simply worded question is not evidence of in_scope by itself: \
judge what the question is actually asking for, not how it is phrased. \
True for a question genuinely about these manuals' own subject matter, \
whether or not looking anything up will turn out to be necessary to \
answer it.

If in_scope is false, route is always "simple" and techniques is always \
an empty list; stop there. Otherwise, decide route:

route: one of "simple", "basic_rag", "advanced_rag".
  simple        in_scope, but general enough to answer without looking \
anything up in the manuals: an abbreviation or term the manuals' own \
subject matter touches on only in passing, never applying it as one of \
their own named procedures. The moment the question asks how these \
manuals themselves apply that term (FIFO issuance in warehouse storage, \
BPM in mail routing, both named above as this pipeline's own manuals' \
subject matter), it is basic_rag, not simple, even though the same term \
also has a well-known general meaning outside these manuals. A term's \
having an ordinary dictionary meaning never by itself makes a question \
about that term simple.
  basic_rag     the default for an in_scope question that asks for one \
fact, one number, one procedure, or one definition specific to these \
manuals, stated plainly and standing on its own. Most in_scope questions \
are this. Being about a specific procedure does not by itself make a \
question advanced: a plain "what is the maximum X" or "what are the \
steps for Y" is basic_rag even when X or Y is a precise procedural \
detail.
  advanced_rag  only when the question itself shows one of the signals \
listed under techniques below: it depends on an earlier turn to mean \
anything, it could be phrased several genuinely different ways, it asks \
to compare more than one thing or has more than one distinct part, it \
uses everyday wording for something phrased formally in the manuals, or \
it names a date or manual as a filter. A question with none of these \
signals is basic_rag, even if it sounds procedural or specific.

techniques: zero or more of exactly these five names, each chosen only \
when the question actually shows its own signal below. An empty list is \
correct far more often than a full one, and basic_rag always carries an \
empty list.
  Rewriting     the question only means something given an earlier turn: \
a bare "why", "and then?", "what about the other one?" with no subject \
of its own
  Multi-Query   the question's own idea could be searched several \
genuinely different ways (a general "problems" or "issues" question \
covering an unspecified range of causes)
  Decomposition the question itself explicitly compares two or more \
named things, or asks two clearly separate questions joined by "and". A \
question whose one, single answer happens to involve several steps is \
not this: the steps belong to the procedure, not to the question, and a \
question asking for a whole procedure end to end is basic_rag, not \
Decomposition, no matter how many steps that procedure turns out to have
  HyDE          the question's own wording is casual or colloquial \
(spoken-register phrasing, a dialect word) rather than the manuals' own \
formal register. A question written in plain formal Arabic asking "what \
are the steps for X" is not colloquial just because X is a multi-step \
procedure; select HyDE for how the question is worded, never for how \
long its answer will be
  Self-Query    the question itself names a specific date, year, or \
picks out one manual as a filter to narrow which documents count, not \
merely because a procedure it asks about happens to belong to one manual

Naming a manual, or a procedure that takes several steps, is normal for \
almost every in-scope question here and is not on its own a signal for \
anything above.

Reranking, Compression and CRAG are never selected here; they are applied \
automatically once route is advanced_rag or triggered by a condition \
outside the question's wording, so never include them in techniques.

Examples, none of them from the real manuals, showing the boundary. Note \
the field order: in_scope is decided and written first in every one.

Q: "ما هو الحد الأدنى للموافقة المطلوبة لصرف مبلغ 200 دينار؟"
{"in_scope": true, "route": "basic_rag", "reason": "single direct \
factual lookup", "techniques": []}

Q: "ما معنى اختصار FIFO؟"
{"in_scope": true, "route": "simple", "reason": "general definition, no \
manual lookup needed", "techniques": []}

Q: "ما هو أفضل سهم للاستثمار فيه هذا العام؟"
{"in_scope": false, "route": "simple", "reason": "asks for investment \
advice, not for anything these manuals document", "techniques": []}

Q: "ما هي أفضل طريقة لتوفير المال؟"
{"in_scope": false, "route": "simple", "reason": "asks for personal \
financial advice, not for anything these manuals document, even though \
the question itself is short and plainly worded", "techniques": []}

Prior turn Q: "متى يتم تدقيق سجل البريد الوارد؟" A: "أسبوعياً."
Q: "ولماذا بالتحديد؟"
{"in_scope": true, "route": "advanced_rag", "reason": "depends on the \
prior turn to mean anything", "techniques": ["Rewriting"]}

Q: "ما هي المشاكل التي قد تواجه أجهزة الطباعة في مركز البريد؟"
{"in_scope": true, "route": "advanced_rag", "reason": "open-ended \
problem question, several phrasings could each surface different \
causes", "techniques": ["Multi-Query"]}

Q: "قارن بين إجراءات فتح ملف ائتماني جديد وإغلاقه، ومتى تلزم موافقة \
المدير التنفيذي في كل حالة؟"
{"in_scope": true, "route": "advanced_rag", "reason": "explicit \
comparison plus a second distinct sub-question", "techniques": \
["Decomposition"]}

Q: "كيف بيتأكدوا انو الكاميرات شغالة منيح؟"
{"in_scope": true, "route": "advanced_rag", "reason": "colloquial \
phrasing of a question the manual answers formally", "techniques": \
["HyDE"]}

Q: "ما هي الأدلة التي تمت مراجعتها بعد سنة 2025؟"
{"in_scope": true, "route": "advanced_rag", "reason": "explicit date \
filter across manuals", "techniques": ["Self-Query"]}

Reply with JSON only, no text before or after it, in exactly this field \
order:
{"in_scope": true, "route": "...", "reason": "short reason", \
"techniques": ["..."]}\
"""


def _as_bool(value: object, default: bool = True) -> bool:
    """A tolerant boolean read. JSON mode returns real booleans in every
    call step 0's probe made, but this is one field, not the whole
    contract, and a stray string here should not raise the way a malformed
    route does.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "no", "0", "")
    if value is None:
        return default
    return bool(value)


def _build_messages(
    question: str, history: list[tuple[str, str]] | None,
) -> list[dict[str, str]]:
    """The system prompt, then the prior turn if this question depends on
    one, then the question itself. Only Q3 in the golden set uses history,
    but the parameter is not Q3-specific: any question the harness runs
    after another can carry its predecessor's question and answer forward,
    which is what the task means by using preceding context.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for prior_question, prior_answer in history or []:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})
    return messages


def route(
    question: str,
    ledger: Ledger,
    history: list[tuple[str, str]] | None = None,
) -> RouteDecision:
    """Ask the model to classify one question, and record the call.

    Two failure modes are handled deliberately rather than left to raise
    past this function, and both degrade toward advanced_rag, never toward
    simple: falling back to simple on a malformed response would refuse an
    answerable question because of a JSON error, which is the failure that
    looks exactly like a correct refusal in the report and is the harder
    of the two to catch by reading results afterwards.
    """
    messages = _build_messages(question, history)
    try:
        response = ledger.call(
            "Router", messages, GENERATOR_MODEL,
            temperature=0.0, max_tokens=300,
            response_format={"type": "json_object"},
        )
    except LLMError as error:
        return RouteDecision(
            route="advanced_rag",
            reason=f"router call failed, defaulting to advanced_rag: {error}",
        )

    try:
        parsed = json.loads(response.text)
        route_value = parsed["route"]
        if route_value not in ROUTES:
            raise ValueError(f"{route_value!r} is not one of {ROUTES}")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
        return RouteDecision(
            route="advanced_rag",
            reason=(f"router response unparseable, defaulting to "
                    f"advanced_rag: {error}"),
            raw=response.text,
        )

    requested_raw = parsed.get("techniques")
    if not isinstance(requested_raw, list):
        requested_raw = []
    requested = tuple(t for t in requested_raw if t in TECHNIQUES)
    dropped = tuple(t for t in requested_raw if t not in TECHNIQUES)

    refusal_kind = None
    if route_value == "simple" and not _as_bool(parsed.get("in_scope")):
        refusal_kind = "out_of_domain"

    return RouteDecision(
        route=route_value,
        reason=str(parsed.get("reason", "")),
        refusal_kind=refusal_kind,
        requested=requested,
        dropped=dropped,
        raw=response.text,
    )


# --- verification --------------------------------------------------------------

def verify_against_golden(golden_path: Path = GOLDEN_SET) -> list[str]:
    """Run every golden question through route() in order, Q3 carrying
    Q2's question and reference answer forward the way depends_on says to,
    and report agreement against each question's expected_route.

    Returns one line per question rather than a bare accuracy number,
    because expected_route is one person's judgement recorded while
    reading pages, and a disagreement can mean the router is wrong or the
    label is; a reader deciding which needs both named side by side.
    """
    from ..golden.question import load_golden  # local: avoids a cycle at import time

    questions, _ = load_golden(golden_path)
    ledger = Ledger(label="router-verify-against-golden")
    answered: dict[str, tuple[str, str]] = {}

    lines = []
    agree = 0
    for question in questions:
        history = None
        if question.depends_on and question.depends_on in answered:
            history = [answered[question.depends_on]]
        decision = route(question.question, ledger, history=history)
        answered[question.id] = (question.question, question.answer)

        matches = decision.route == question.expected_route
        agree += matches
        lines.append(
            f"{question.id}: expected={question.expected_route} "
            f"got={decision.route} {'OK' if matches else 'DISAGREE'} "
            f"requested={list(decision.requested)} "
            f"dropped={list(decision.dropped)} "
            f"refusal_kind={decision.refusal_kind} "
            f"reason={decision.reason}"
        )

    lines.insert(0, f"{agree}/{len(questions)} route agreement")
    lines.append("")
    lines.append(ledger.render())
    return lines


if __name__ == "__main__":
    from ..config import PROCESSED_DIR

    report_lines = verify_against_golden()
    out_path = PROCESSED_DIR / "06_router_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_lines[0])
    print(f"written to {out_path}")
