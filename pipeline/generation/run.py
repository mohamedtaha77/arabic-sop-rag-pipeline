"""The orchestration: a QuestionRun in, an Answer out.

Position: techniques/run.py's own answer() has already routed, retrieved
and applied whichever techniques the question calls for; this file is
what turns that into something a user reads, over three kinds, one of
them reachable two different ways.

    direct     route "simple", not refused. A greeting the pre-gate
               caught, or an in-scope general question the router judged
               needs no retrieval (the task's own Q1, "What is RAG?").
               One plain call, no context, no guard: there is no
               retrieved evidence for a guard to check faithfulness
               against, and section 9's own citation contract only ever
               applies to a grounded answer
    refused    route "simple" with a refusal_kind (out_of_domain, the
               router judged the question has nothing to do with the
               corpus), or crag_refused (non_answering_retrieval, CRAG
               judged retrieval plausible but non-answering). One
               guarded call either way: the model words the refusal,
               guard.apply_to_refusal checks it added no citation and no
               fact absent from the question itself, and a fixed Arabic
               string covers the rare case where it did
    grounded   basic_rag or advanced_rag, retrieval returned something
               and CRAG (if it ran) accepted it. synthesise, check
               entailment, present, guard; the four-part chain the plan's
               own two-stage generation section describes

Two mechanisms enforce the plan's own "the final model only presents and
never adds", at two different points, checking two different things.
entail.py checks the synthesiser's own sentences against the chunks they
actually cite, catching a claim that names a real source but does not
say what that source says; this runs once, right after synthesis, before
the presenter ever sees the text. guard.py checks the presenter's output
against the synthesiser's output it was given, catching anything added
that was not already there; this runs once, right after presentation.
Neither substitutes for the other, and both are exercised on every
grounded answer.

A caller that opens a ShippingHandle for a run where Reranking might
fire has to call techniques.rerank.warm_up() first, before
retriever.open_shipping(), never after; techniques/run.py's own
docstring has the measured reason. entail.py's own warm_up is not called
here: the bake-off's measured winner is the LLM judge backend, which
spends no worker process at all, so calling entail.warm_up() in the
current shipped configuration would start a process nothing reads from.
If a future re-run of that bake-off ever favours the NLI backend
instead, this file's own caller needs entail.warm_up() added to that
same pre-open sequence, and ledger.py's LOCAL_STEPS needs "Grounding
guard" added back, both stated in the files that would need to change
rather than solved for a hypothetical that is not today's reality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import (
    DIRECT_MAX_TOKENS,
    GENERATION_DECISION,
    GENERATOR_MODEL,
    REFUSAL_MAX_TOKENS,
)
from ..llm.ledger import Ledger
from ..retrieval.retriever import ShippingHandle
from ..router.schema import TechniqueSet
from ..techniques.run import answer as techniques_answer
from . import entail, guard, present, synthesise
from .answer import Answer
from .guard import GuardReport
from .synthesise import SynthesisTrace

_CITATION_MARKER = re.compile(r"\[(\d+)\]")

# hex codepoints, matching entail.py's own _SENTENCE_SPLIT exactly:
# period, exclamation, Arabic question mark. A genuinely complete claim
# is the right unit to check entailment against, the same reason
# entail.py's own benchmark splits sentences this way rather than
# guard.py's finer, comma-inclusive segmentation.
_SENTENCE_SPLIT = re.compile("[.!؟]+")

_DIRECT_SYSTEM_PROMPT = """\
أنت مساعد يجيب بإيجاز وبأسلوب طبيعي على تحية أو سؤال عام، بنفس لغة \
السؤال. لا تدّعِ معرفة تفاصيل داخلية خاصة بإجراءات بنك معين؛ إن سُئلت \
عن ذلك تحديداً، وضّح أن الإجابة التفصيلية تتطلب البحث في الأدلة الداخلية \
الخاصة بالبنك.\
"""

_REFUSAL_REASON = {
    "out_of_domain": "يقع خارج نطاق موضوعات هذه الأدلة الثلاثة تماماً",
    "non_answering_retrieval": (
        "تم البحث عنه في الأدلة الثلاثة، ولم يتم العثور على إجراء أو "
        "معلومة تجيب عليه بشكل مباشر"
    ),
}

_REFUSAL_SYSTEM_TEMPLATE = """\
أنت جزء من نظام أسئلة وأجوبة يعمل حصراً على ثلاثة أدلة إجراءات داخلية \
لبنك: الموجودات والمستودعات، الانذار المركزي، والبريد المركزي والملفات. \
السؤال التالي {reason}. اكتب رفضاً مهذباً وواضحاً بالعربية، جملة أو \
جملتين، يوضح أن هذا خارج نطاق هذه الأدلة الثلاثة. لا تذكر أي رقم مرجعي \
بين قوسين مربعين. لا تُضف أي رقم أو تاريخ أو اسم غير موجود في السؤال \
نفسه. أجب بنص الرفض فقط.\
"""

# A fixed fallback per refusal_kind, used only when the guarded call
# above adds a citation or a fact the question never named. Deterministic
# and unfabricatable by construction, the same reason CRAG's own
# module docstring gives for capping its own re-query at one attempt
# rather than looping: a second, cheap, certain answer beats chasing a
# model call that already showed it cannot be trusted this time.
_REFUSAL_FALLBACK = {
    "out_of_domain": (
        "هذا السؤال يقع خارج نطاق الأدلة الثلاثة التي يغطيها هذا النظام "
        "(الموجودات والمستودعات، الانذار المركزي، البريد المركزي "
        "والملفات)، ولا يمكنني الإجابة عليه."
    ),
    "non_answering_retrieval": (
        "لم أجد في الأدلة الثلاثة إجراءً أو معلومة تجيب على هذا السؤال "
        "بشكل مباشر."
    ),
}

# Read once per call, not baked into a default parameter, for the exact
# reason techniques/run.py's own _crag_threshold_default is a function
# rather than a frozen default: a stage 9 decision file written after
# this module is first imported still has to be picked up. "llm" and 0.5
# are the honest placeholders for a first build that has not run
# generation/evaluate.py yet; entail.py's own module docstring records
# why a threshold applies to the nli backend only.


def _entailment_config() -> tuple[str, float]:
    if GENERATION_DECISION.exists():
        import json
        stored = json.loads(GENERATION_DECISION.read_text(encoding="utf-8"))
        threshold = stored.get("entailment_threshold")
        return (
            stored.get("guard_backend", "llm"),
            0.5 if threshold is None else float(threshold),
        )
    return "llm", 0.5


@dataclass
class EntailmentSummary:
    """What generation/evaluate.py's own report reads back for how often
    a grounded answer's own cited claims actually held up against the
    chunk they cited.

        checked            sentences that carried a citation marker and
                            so were actually checked. A sentence with no
                            marker at all is not entailment's job, that
                            is guard.py's territory instead
        entailed           of those, how many the backend judged entailed
        failed_sentences   the sentences that did not, kept verbatim for
                            the report rather than only counted
        backend            which of entail.BACKENDS graded this answer
    """

    checked: int
    entailed: int
    failed_sentences: tuple[str, ...]
    backend: str


def _sentence_premise_pairs(
    text: str, marker_chunk_ids: dict[str, str], chunk_lookup: dict[str, str],
) -> list[tuple[str, str]]:
    """One (premise, hypothesis) pair per sentence that cites at least
    one chunk this run actually retrieved. A sentence citing more than
    one marker is checked against the concatenation of everything it
    cites, since the claim may draw on all of them together; a sentence
    with no marker at all is skipped, since there is nothing for
    entailment to check it against.
    """
    pairs = []
    for raw_sentence in _SENTENCE_SPLIT.split(text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        markers = _CITATION_MARKER.findall(sentence)
        cited_chunk_ids = [
            marker_chunk_ids[f"[{m}]"] for m in markers
            if f"[{m}]" in marker_chunk_ids
        ]
        if not cited_chunk_ids:
            continue
        premise = "\n".join(
            chunk_lookup[cid] for cid in dict.fromkeys(cited_chunk_ids)
            if cid in chunk_lookup
        )
        hypothesis = _CITATION_MARKER.sub("", sentence).strip()
        if premise and hypothesis:
            pairs.append((premise, hypothesis))
    return pairs


def _check_entailment(
    synthesised_text: str, synth_trace: SynthesisTrace,
    retrieved_lookup: dict[str, str], ledger: Ledger,
) -> EntailmentSummary:
    backend, threshold = _entailment_config()
    pairs = _sentence_premise_pairs(
        synthesised_text, synth_trace.marker_chunk_ids, retrieved_lookup,
    )
    failed = []
    entailed_count = 0
    for premise, hypothesis in pairs:
        result = entail.check(premise, hypothesis, ledger, backend, threshold)
        if result.entailed:
            entailed_count += 1
        else:
            failed.append(hypothesis)
    return EntailmentSummary(
        checked=len(pairs), entailed=entailed_count,
        failed_sentences=tuple(failed), backend=backend,
    )


def _refuse(question: str, refusal_kind: str, ledger: Ledger) -> tuple[str, GuardReport]:
    messages = [
        {"role": "system", "content": _REFUSAL_SYSTEM_TEMPLATE.format(
            reason=_REFUSAL_REASON[refusal_kind],
        )},
        {"role": "user", "content": question},
    ]
    response = ledger.call(
        "Final generation", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=REFUSAL_MAX_TOKENS,
    )
    report = guard.apply_to_refusal(question, response.text)
    text = response.text if report.passed else _REFUSAL_FALLBACK[refusal_kind]
    return text, report


def _direct(
    question: str, ledger: Ledger, history: list[tuple[str, str]] | None,
) -> str:
    messages = [{"role": "system", "content": _DIRECT_SYSTEM_PROMPT}]
    for prior_question, prior_answer in history or []:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})
    response = ledger.call(
        "Final generation", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=DIRECT_MAX_TOKENS,
    )
    return response.text


def answer(
    question: str,
    ledger: Ledger,
    handle: ShippingHandle,
    history: list[tuple[str, str]] | None = None,
    technique_set: TechniqueSet | None = None,
) -> Answer:
    """Route, retrieve, generate, and guard, once, for one question.

    ``history`` carries (question, answer) pairs of the actual, generated
    answer text, not the reference answer: Q3's own dependency on Q2 is
    what the task means by using preceding context, and the harness that
    runs the golden set in order is what has to thread this file's own
    Answer.text from one call into the next call's history, the same way
    techniques.run.answer's own history parameter already expects.
    """
    run_result = techniques_answer(
        question, ledger, handle, history=history, technique_set=technique_set,
    )

    if run_result.decision.route == "simple" and run_result.decision.refusal_kind is not None:
        text, report = _refuse(question, run_result.decision.refusal_kind, ledger)
        return Answer(
            question=question, kind="refused", text=text, synthesised=text,
            presented="", presenter_rejected=False, citations=(),
            refusal_kind=run_result.decision.refusal_kind, run=run_result,
            ledger=ledger, traces={"Refusal guard": report},
        )

    if run_result.decision.route == "simple":
        text = _direct(question, ledger, history)
        return Answer(
            question=question, kind="direct", text=text, synthesised=text,
            presented="", presenter_rejected=False, citations=(),
            refusal_kind=None, run=run_result, ledger=ledger, traces={},
        )

    if run_result.crag_refused:
        text, report = _refuse(question, "non_answering_retrieval", ledger)
        return Answer(
            question=question, kind="refused", text=text, synthesised=text,
            presented="", presenter_rejected=False, citations=(),
            refusal_kind="non_answering_retrieval", run=run_result,
            ledger=ledger, traces={"Refusal guard": report},
        )

    synthesised_text, synth_trace = synthesise.apply(
        question, run_result.retrieved, ledger, history=history,
    )
    retrieved_lookup = {
        item.chunk_id: item.chunk.text for item in run_result.retrieved
    }
    entailment_summary = _check_entailment(
        synthesised_text, synth_trace, retrieved_lookup, ledger,
    )

    # entailment_summary is reported, never gated on, and this was a
    # real design change made while building this file rather than the
    # original plan: a first version refused the whole answer outright
    # whenever every cited sentence failed entailment, on the reasoning
    # that a total failure is the same signal as a CRAG miss. Run
    # end to end against this file's own four-path gate, that version
    # wrongly refused Q11, an answer already read by eye and judged
    # genuinely correct and well cited. The cause was not a bug in the
    # wiring; it is entail.py's own bake-off number for the backend that
    # actually won, llm, recall on genuinely entailed positives of only
    # 0.467 (17_entail_bakeoff.md). A well-grounded two-sentence answer
    # then has roughly a (1-0.467)^2, about 28%, chance of every sentence
    # independently misjudged as unentailed by pure per-sentence noise,
    # and the chance only grows with more sentences. Gating a whole
    # answer's fate on that number would refuse a large share of
    # genuinely correct answers, the same over-trusting-an-imperfect-
    # judge mistake crag.py's own measured precision problem already
    # warns against elsewhere in this pipeline. So entailment here is
    # exactly what the plan's own wording says it is, "unentailed
    # sentences are flagged", reported in the trace for stage 10's
    # report to state honestly, and left there rather than turned into
    # an enforcement action neither backend's own measured accuracy
    # earns yet.
    presented_text, present_trace = present.apply(synthesised_text, ledger)
    guard_report = guard.apply(synthesised_text, presented_text)
    final_text = presented_text if guard_report.passed else synthesised_text

    return Answer(
        question=question, kind="grounded", text=final_text,
        synthesised=synthesised_text, presented=presented_text,
        presenter_rejected=not guard_report.passed,
        citations=synth_trace.citations, refusal_kind=None,
        run=run_result, ledger=ledger,
        traces={
            "Synthesis": synth_trace, "Entailment": entailment_summary,
            "Presenter": present_trace, "Grounding guard": guard_report,
        },
    )


# --- verification --------------------------------------------------------------

def verify_four_paths(handle: ShippingHandle) -> dict:
    """The build order's own gate, run end to end: one question per
    path, checked for the shape each path promises. Not a re-test of
    guard.py's or present.py's own fabrication-rejection gates, both
    already exercise that in isolation; this checks the wiring between
    every piece is correct for a real question actually reaching each
    branch, the same scope every other file's own verify function in
    this package keeps to.

    Q10's own CRAG path is forced with an explicit technique_set rather
    than left to the router's own classification, and this is a real
    finding, not a style choice: router.md already documents Q10 as one
    of three questions (with Q3 and Q19) whose route "swapped identity
    across small prompt perturbations... a 3B model's own ceiling on
    this signal, not a fixable prompt bug." Measured directly while
    building this gate, that instability has a sharper, previously
    untraced consequence than router.md's own accuracy count states:
    CRAG's own runtime trigger only ever evaluates when
    decision.route == "advanced_rag" (techniques/run.py's own
    triggers_active), so a single router miss classifying Q10 as
    basic_rag instead silently skips Reranking and CRAG both, for the
    one question the corpus was built to need CRAG's own safety net on.
    That is a real property of the shipped system worth stating in
    stage 10's own report, not something this stage can fix without
    reopening stage 8's own router prompt, already closed and already
    measured against the same overfitting risk. This gate exists to
    check generation/run.py's own wiring, not to re-litigate the
    router's own accuracy a second time, so it forces CRAG directly,
    the same override techniques.run.answer's own technique_set
    parameter exists for.
    """
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden

    questions, _ = load_golden(GOLDEN_SET)
    by_id = {q.id: q for q in questions}

    results = {}

    greeting = answer("مرحبا", Ledger(label="verify-run-direct"), handle)
    results["direct"] = {
        "kind": greeting.kind, "has_citations": bool(greeting.citations),
        "text": greeting.text, "ledger_failures": greeting.ledger.verify(),
    }

    q9 = by_id["Q9"]
    out_of_domain = answer(q9.question, Ledger(label="verify-run-refuse-domain"), handle)
    results["refused_out_of_domain"] = {
        "kind": out_of_domain.kind, "refusal_kind": out_of_domain.refusal_kind,
        "has_citations": bool(out_of_domain.citations), "text": out_of_domain.text,
        "ledger_failures": out_of_domain.ledger.verify(),
    }

    q10 = by_id["Q10"]
    non_answering = answer(
        q10.question, Ledger(label="verify-run-refuse-crag"), handle,
        technique_set=TechniqueSet(crag=True),
    )
    results["refused_non_answering"] = {
        "kind": non_answering.kind, "refusal_kind": non_answering.refusal_kind,
        "has_citations": bool(non_answering.citations), "text": non_answering.text,
        "ledger_failures": non_answering.ledger.verify(),
    }

    q11 = by_id["Q11"]
    grounded = answer(q11.question, Ledger(label="verify-run-grounded"), handle)
    results["grounded"] = {
        "kind": grounded.kind, "has_citations": bool(grounded.citations),
        "presenter_rejected": grounded.presenter_rejected, "text": grounded.text,
        "ledger_failures": grounded.ledger.verify(),
    }

    all_ledgers_clean = all(not row["ledger_failures"] for row in results.values())
    ok = (
        results["direct"]["kind"] == "direct"
        and not results["direct"]["has_citations"]
        and results["refused_out_of_domain"]["kind"] == "refused"
        and results["refused_out_of_domain"]["refusal_kind"] == "out_of_domain"
        and not results["refused_out_of_domain"]["has_citations"]
        and results["refused_non_answering"]["kind"] == "refused"
        and results["refused_non_answering"]["refusal_kind"] == "non_answering_retrieval"
        and not results["refused_non_answering"]["has_citations"]
        and results["grounded"]["kind"] == "grounded"
        and results["grounded"]["has_citations"]
        and all_ledgers_clean
    )
    return {"ok": ok, "results": results, "all_ledgers_clean": all_ledgers_clean}


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping
    from ..techniques import rerank

    # warm_up() before open_shipping(): the grounded path's own
    # advanced_rag question always reaches Reranking; see
    # rerank.warm_up's own docstring for the measured reason this order
    # is load-bearing.
    rerank.warm_up()
    with open_shipping() as shipping_handle:
        outcome = verify_four_paths(shipping_handle)

    lines = [
        f"all four paths correct: {outcome['ok']}",
        f"ledger.verify() clean on all four: {outcome['all_ledgers_clean']}",
        "",
    ]
    for name, row in outcome["results"].items():
        lines.append(f"{name}: {row}")

    out_path = PROCESSED_DIR / "19_generation_run_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{'ok' if outcome['ok'] else 'FAIL'}: written to {out_path}")
