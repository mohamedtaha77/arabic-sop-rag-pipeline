"""The presenter: format and polish, nothing else.

Position: called from generation/run.py after synthesise.py, on the
synthesiser's own text alone. This is the half of the split the plan's
own framing puts plainly: "the final model only presents and never
adds." The presenter never sees retrieved context, never sees a chunk,
and has no retrieval tool; every fact it could possibly write is
already sitting in the text it was handed, which is what makes
guard.py's own diff a real enforcement mechanism rather than a
courtesy. A model with no access to the corpus cannot ground a new
fact in it even if it tried; it can only invent one outright, and an
invented number, date, name or citation marker is exactly what
guard.apply catches.

GENERATOR_MODEL runs this call, not JUDGE_MODEL. The "a judge sharing
weights with the generator grades its own work generously" rule
(config.py's own comment, entail.py's own module docstring) is about
grading, and the presenter is not a judge, it does not decide whether
anything is correct. Enforcement of the no-new-facts constraint is
guard.py's diff, not the presenter's own good behaviour, so nothing is
lost by sharing GENERATOR_MODEL's tag; what is gained is avoiding an
Ollama unload-and-reload between the synthesiser call and this one, a
real latency cost on a 4 GB card this stage has no reason to pay per
question.

What this module does not do: it does not decide whether the presented
text is trustworthy. That is guard.py's job immediately afterward, and
run.py's own responsibility to fall back to the synthesised text
outright when guard.apply reports a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import GENERATOR_MODEL, PRESENTER_MAX_TOKENS
from ..llm.ledger import Ledger

_SYSTEM_PROMPT = """\
أنت منسّق عرض فقط. ستُعطى إجابة مكتملة ومُتحقق منها بالفعل. أعد صياغة \
تنسيقها فقط لتكون أوضح للقارئ، مثل تحويلها إلى نقاط أو خطوات مرقمة إذا \
كان ذلك أنسب، دون تغيير أي حقيقة أو رقم أو تاريخ أو اسم. يجب أن يبقى كل \
رقم مرجعي بين قوسين مربعين، مثل [1]، ملاصقاً لنفس الادعاء الذي كان \
يدعمه في النص الأصلي بالضبط، دون حذف أي رقم مرجعي أو إضافة رقم جديد. لا \
تُضف أي معلومة أو جملة جديدة غير موجودة في النص الأصلي. أجب بالنص \
المعاد تنسيقه فقط، دون أي شرح أو مقدمة.\
"""


@dataclass
class PresentTrace:
    """What analysis reads back for the presenter step.

        input_text   the synthesised text this call was given
        output_text  what the model returned, before guard.apply ever
                     runs; run.py's own record of whether this survived
                     the guard lives on Answer, not here, since this
                     trace is this call's own output, not the guard's
                     verdict on it
    """

    input_text: str
    output_text: str


def apply(synthesised_text: str, ledger: Ledger) -> tuple[str, PresentTrace]:
    """Reformat one synthesised answer. Always returns the model's raw
    output; guard.apply is what decides whether run.py is allowed to use
    it.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": synthesised_text},
    ]
    response = ledger.call(
        "Presenter", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=PRESENTER_MAX_TOKENS,
    )
    return response.text, PresentTrace(
        input_text=synthesised_text, output_text=response.text,
    )


# --- verification --------------------------------------------------------------

def verify_block_rate(handle: "object", question_ids: tuple[str, ...]) -> dict:
    """The gate this file ships against: run the real presenter on a
    real synthesised answer for each question and report how often
    guard.apply blocks it. Unlike synthesise.py's own gate, this is not
    a pass/fail on a fixed criterion; a low block rate and a high block
    rate are both real, informative numbers, and section 16's own
    analysis question asks for exactly this rate, not for it to be
    driven to zero.

    Takes an already-open ShippingHandle rather than opening its own,
    matching every other verify function in this package
    (synthesise.verify_grounded_and_cited, run.verify_four_paths): Qdrant
    local file mode holds an exclusive lock per open client, and
    generation/evaluate.py calls this from inside its own already-open
    handle, so a second open_shipping() call here would be refused
    rather than queued, measured directly as the exact RuntimeError
    Qdrant itself raises for this case before this signature was fixed.
    handle is typed loosely for the same reason compress.py's own verify
    function is: this is the only function here that would otherwise
    import qdrant_client, transitively, and only __main__ needs it.
    """
    from . import guard, synthesise
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden
    from ..retrieval.retriever import retrieve_scored

    questions, _ = load_golden(GOLDEN_SET)
    by_id = {q.id: q for q in questions}

    results = []
    for qid in question_ids:
        question = by_id[qid]
        scored = retrieve_scored(
            handle.context, question.question, handle.decision["mode"],
            apply_caps=handle.decision.get("apply_caps", False),
        )
        ledger = Ledger(label=f"verify-present-{qid}")
        synthesised_text, _synth_trace = synthesise.apply(
            question.question, scored, ledger,
        )
        presented_text, _present_trace = apply(synthesised_text, ledger)
        report = guard.apply(synthesised_text, presented_text)
        results.append({
            "id": qid,
            "synthesised": synthesised_text,
            "presented": presented_text,
            "passed": report.passed,
            "added_tokens": list(report.added_tokens),
            "added_citations": list(report.added_citations),
        })
    return {
        "results": results,
        "block_rate": sum(1 for r in results if not r["passed"]) / len(results),
    }


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    # The same eight technique-independent questions synthesise.py's own
    # gate settled on, for the same reason: Q4 and Q19 need a
    # query-transformation technique this standalone gate does not run,
    # and testing the presenter against a synthesised answer already
    # measured unstable there would test that instability twice rather
    # than the presenter itself.
    question_ids = ("Q1", "Q2", "Q8", "Q11", "Q13", "Q16", "Q18", "Q20")
    with open_shipping() as shipping_handle:
        outcome = verify_block_rate(shipping_handle, question_ids)

    lines = [f"block rate: {outcome['block_rate']:.3f} "
             f"({sum(1 for r in outcome['results'] if not r['passed'])} "
             f"of {len(outcome['results'])} blocked)", ""]
    for row in outcome["results"]:
        lines.append(f"{row['id']}: passed={row['passed']} "
                     f"added_tokens={row['added_tokens']} "
                     f"added_citations={row['added_citations']}")
        lines.append(f"  synthesised: {row['synthesised']}")
        lines.append(f"  presented:   {row['presented']}")
        lines.append("")

    out_path = PROCESSED_DIR / "18_present_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"ok: block rate {outcome['block_rate']:.3f}, written to {out_path}")
