"""Authors the golden set skeleton: the 20 questions before the reading pass.

Deliberately not wired into cli.py. Every other stage's CLI command is safe to
re-run, but running this one again after the reading pass would overwrite
answers a person spent time writing, so it stays a manual invocation,
``python -m pipeline.golden.skeleton``, guarded against exactly that.

Kept as a tracked module rather than a throwaway script for a reason specific
to this stage: data/golden/ is entirely gitignored, the corpus is Housing Bank
material and its questions and answers cannot be committed. If this file were
deleted after one run, every reason behind a question's wording or a page
choice, why Q3 was re-grounded, why Q5's pages_to_read includes both p6 and
p7, why Q7 also reads the revision log, would exist nowhere in git history.
This file is where that reasoning is preserved.

id, question, expect, expected_route, tests, depends_on, refusal_kind and a
candidate pages_to_read are set for all 20 questions here. Every answer field
starts empty, for the reading pass to fill by hand in an editor.
"""

from __future__ import annotations

from pathlib import Path

from .question import Question, load_golden, page_ref, save_golden, corpus_fingerprint
from ..config import CHUNKS_OUTPUT, GOLDEN_SET

AW = "assets_wearhouse"
CA = "central_alarm"
CM = "central_mail"


def pr(slug, *pages):
    return [page_ref(slug, p) for p in pages]


QUESTIONS = [
    Question(
        id="Q1",
        question="ما هو مبدأ الوارد أولاً صادر أولاً (FIFO) في تخزين المواد؟",
        expect="answerable", expected_route="basic_rag",
        tests="Simple, single hop",
        pages_to_read=pr(AW, 5),
    ),
    Question(
        id="Q2",
        question="كم المدة القصوى بين عمليات فحص تسجيلات المراقبة؟",
        expect="answerable", expected_route="basic_rag",
        tests="Document-grounded factual",
        pages_to_read=pr(CA, 7),
    ),
    Question(
        id="Q3",
        question="ولماذا يتم ذلك؟",
        expect="answerable", expected_route="advanced_rag",
        tests="Ambiguous, needs Rewriting, context is Q2. Re-grounded: the "
              "corpus states no rationale for the 3-month figure itself, but "
              "does state a rationale for the periodic check, that the "
              "recording retention period must match Central Bank and "
              "Ministry of Interior instructions, verified monthly. Confirm "
              "this against the page before writing the answer; if the page "
              "reading finds it does not hold up, fall back to recording the "
              "rationale as unstated rather than forcing a fit.",
        depends_on="Q2",
        pages_to_read=pr(CA, 7),
    ),
    Question(
        id="Q4",
        question="ما هي الأعطال والمشاكل التي قد تصيب أنظمة المراقبة التلفزيونية؟",
        expect="answerable", expected_route="advanced_rag",
        tests="Multi-Query",
        pages_to_read=pr(CA, 7),
    ),
    Question(
        id="Q5",
        question="قارن بين الموافقات المطلوبة لمشاهدة كاميرات الفرع وللتدقيق "
                 "الداخلي، واذكر الموافقة الداخلية في كل حالة، ومتى تُطلب "
                 "موافقة المدير التنفيذي",
        expect="answerable", expected_route="advanced_rag",
        tests="Decomposition. Two approval matrices, p6 (9 rows) and p7 (5 "
              "rows), each with its own header. Confirm while reading whether "
              "p6 row 4 (department requests) and p7 row 4 (department "
              "staff) are genuinely distinct rows on the page or a layout "
              "artifact, and whether p6 rows 6 and 8 (both reading "
              "\"other cases not listed in this table\") are really repeated "
              "or duplicated by layout.",
        pages_to_read=pr(CA, 6, 7),
    ),
    Question(
        id="Q6",
        question="كيف يتأكد البنك أن كاميرات المراقبة شغالة؟",
        expect="answerable", expected_route="advanced_rag",
        tests="HyDE, colloquial against formal wording",
        pages_to_read=pr(CA, 7),
    ),
    Question(
        id="Q7",
        question="ما هي الأدلة التي صدرت قبل عام 2025؟",
        expect="answerable", expected_route="advanced_rag",
        tests="Self-Query, metadata filter on issue_date. Central Alarm is "
              "the only manual with issue_date before 2025 (08/2024); the "
              "other two read 02/2026. Its revision log (p20) lists a "
              "version 2 dated 02/2026 while the page header reads "
              "doc_version 1: read both the header band and p20 before "
              "writing the answer, and note the contradiction in "
              "page_vs_chunk regardless of how it resolves.",
        pages_to_read=pr(CA, 3, 20),
    ),
    Question(
        id="Q8",
        question="ما هو الجرد الدوري للموجودات الثابتة ولماذا يتم؟",
        expect="answerable", expected_route="basic_rag",
        tests="Concept plus rationale",
        pages_to_read=pr(AW, 10),
    ),
    Question(
        id="Q9",
        question="ما هي أفضل طريقة لاستثمار مدخراتي؟",
        expect="refuse", expected_route="simple",
        tests="Scope refusal, outside the corpus",
        refusal_kind="out_of_domain",
    ),
    Question(
        id="Q10",
        question="ما هي إجراءات إصدار بطاقات الائتمان للعملاء؟",
        expect="refuse", expected_route="advanced_rag",
        refusal_kind="non_answering_retrieval",
        tests="CRAG, plausible but non-answering retrieval. "
              "\"بطاقة ائتمان\" has no hits; \"ائتمان\" has hits from "
              "credit-file mail routing (central_mail, credit file "
              "inventory and retention procedures), which is plausible, "
              "related and non-answering for a card-issuance question. Read "
              "the credit-file pages to pick real distractor_chunk_ids, not "
              "just cite the count.",
        pages_to_read=pr(CM, 20, 24),
    ),
    Question(
        id="Q11",
        question="ما هي آلية تبادل البريد من خلال نظام BPM لأتمتة العمليات؟",
        expect="answerable", expected_route="basic_rag",
        tests="A literal Latin system name inside Arabic text, a BM25 case",
        pages_to_read=pr(CM, 5),
    ),
    Question(
        id="Q12",
        question="ما هي إجراءات استلام البريد الوارد بالبريد المسجل الأردني؟",
        expect="answerable", expected_route="basic_rag",
        tests="أولا against ثانيا sub-procedure discrimination on one page",
        pages_to_read=pr(CM, 10),
    ),
    Question(
        id="Q13",
        question="من الجهة المسؤولة عن الجرد الشامل لملفات الائتمان؟",
        expect="answerable", expected_route="basic_rag",
        tests="Actor binding across a page-span merge",
        pages_to_read=pr(CM, 24, 25),
    ),
    Question(
        id="Q14",
        question="ما هي إجراءات التعامل مع حالات نقص أو تلف أو ضياع البريد؟",
        expect="answerable", expected_route="basic_rag",
        tests="The tail of the longest manual",
        pages_to_read=pr(CM, 29),
    ),
    Question(
        id="Q15",
        question="ما هي إجراءات استلام الكشوفات والرسائل وتغليفها؟",
        expect="answerable", expected_route="basic_rag",
        tests="The lead-in block chunking.md had to fix, where a block "
              "opens above the first actor row",
        pages_to_read=pr(CM, 26, 27),
    ),
    Question(
        id="Q16",
        question="ما هو قيد ترحيل الاستهلاك للموجودات الثابتة؟",
        expect="answerable", expected_route="basic_rag",
        tests="accounting_entry, the one chunk type with an inferred rather "
              "than document-drawn boundary",
        pages_to_read=pr(AW, 28),
    ),
    Question(
        id="Q17",
        question="ما هي إجراءات المطابقات الربعية لحسابات الموجودات الثابتة؟",
        expect="answerable", expected_route="basic_rag",
        tests="ثانيا picked out from أولا on a different page, plus a page "
              "span (p13 to p14)",
        pages_to_read=pr(AW, 13, 14),
    ),
    Question(
        id="Q18",
        question="ما هي النماذج والملاحق المذكورة في دليل الموجودات والمستودعات؟",
        expect="answerable", expected_route="basic_rag",
        tests="reference chunk_type, the two chunks reclassified away from "
              "procedure_block in stage 2",
        pages_to_read=pr(AW, 27),
    ),
    Question(
        id="Q19",
        question="من الجهات التي توافق على اعتماد أدلة الإجراءات في البنك؟",
        expect="answerable", expected_route="advanced_rag",
        tests="approval chunk_type, decomposition across all three sources "
              "since each manual's approval page differs slightly",
        pages_to_read=pr(AW, 3) + pr(CA, 3) + pr(CM, 3),
    ),
    Question(
        id="Q20",
        question="ما هو التعديل الذي طرأ على دليل الانذار المركزي منذ إصداره؟",
        expect="answerable", expected_route="advanced_rag",
        tests="revision chunk_type, pairs with Q7's metadata filter on the "
              "same manual's version history",
        pages_to_read=pr(CA, 20),
    ),
]


def _already_answered(path: Path) -> bool:
    """True if any question in the file at path carries a written answer.

    The only thing standing between a person's finished reading pass and this
    script quietly erasing it. Checked before every write.
    """
    if not path.exists():
        return False
    questions, _ = load_golden(path)
    return any(q.answer for q in questions)


def write_skeleton(path: Path = GOLDEN_SET, force: bool = False) -> None:
    assert len(QUESTIONS) == 20, len(QUESTIONS)
    assert len({q.id for q in QUESTIONS}) == 20

    if not force and _already_answered(path):
        raise RuntimeError(
            f"{path} already has at least one answered question. Refusing to "
            f"overwrite a finished reading pass. Pass force=True if this is "
            f"really what you want."
        )

    fingerprint = corpus_fingerprint(CHUNKS_OUTPUT)
    save_golden(QUESTIONS, fingerprint, path)
    print(f"wrote {path} with {len(QUESTIONS)} questions")


if __name__ == "__main__":
    write_skeleton()
