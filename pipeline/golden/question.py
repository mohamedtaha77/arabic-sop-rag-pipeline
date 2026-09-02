"""The Question contract for the golden set, stage 5's evaluation anchor.

Mirrors chunk.py: an envelope and its storage, nothing here decides what makes
a good question or a correct answer. Ground truth is established by a person
reading rendered pages, not by anything in this module.

Position in the pipeline: nothing produces a Question the way layout.py
produces a Document or chunker.py produces a Chunk. golden.py writes the
skeleton by hand, from the plan's ten questions plus ten aimed at corpus
coverage, and a person fills in every field that can only come from reading a
page. Everything from stage 6 onward reads data/golden/golden_set.json back
through load_golden and treats it as fixed, outside evidence, the only anchor
that is not itself one of the things being compared.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Whether a correctly functioning system should answer this question from the
# corpus, or refuse it. Two refusal mechanisms exist and gate on different
# things, which is why expect stays two-valued while refusal_kind splits
# "refuse" further:
#
#   out_of_domain            refused at the router, before retrieval spends
#                             anything. The question has nothing to do with
#                             the corpus (Q9).
#   non_answering_retrieval  in domain, passes the router, retrieval returns
#                             plausible chunks that do not answer it, and CRAG
#                             has to catch that downstream rather than the
#                             router (Q10).
#
# A system can pass one and fail the other, so both are measured rather than
# collapsed into one "refused" outcome.
EXPECT_VALUES = ("answerable", "refuse")
REFUSAL_KINDS = ("out_of_domain", "non_answering_retrieval")

# Mirrors the router schema from advanced-rag-plan.md's architecture section,
# {"route": "simple | basic_rag | advanced_rag", ...}, so a question's
# expected_route can be compared against the router's actual output with no
# translation step in between.
EXPECTED_ROUTES = ("simple", "basic_rag", "advanced_rag")


@dataclass
class Evidence:
    """One quote establishing that a chunk supports an answer.

    Quoted from the chunk text, not retyped, so golden.py's containment gate
    can check the quote really is a substring of the chunk rather than
    trusting whoever typed it.

    Kept separate from Question.answer on purpose. The answer is written from
    the rendered page, so it carries correct Arabic. A quote here is copied
    from the chunk, so it carries whatever OCR left there, misreads included.
    Collapsing the two would either bake OCR damage into the reference answer,
    or make the containment gate unfalsifiable by grading the quote against
    itself.
    """

    chunk_id: str
    quote: str


@dataclass
class Question:
    """One golden-set question with its reference answer and gold chunk ids.

        id                     Q1 to Q20
        question                the Arabic question as asked
        expect                  one of EXPECT_VALUES
        expected_route          one of EXPECTED_ROUTES
        tests                   what this question exists to exercise, e.g.
                                 "Decomposition" or "Simple, single hop"
        depends_on              an earlier id whose turn is carried as
                                 context, or None. Only Q3 uses this
        refusal_kind            one of REFUSAL_KINDS when expect is "refuse",
                                 otherwise None
        pages_to_read           page references, e.g. "central_alarm:p07",
                                 proposed before reading, so worksheet.py
                                 knows what to render
        pages_read              page references the answer was actually read
                                 off. Not required to equal pages_to_read: a
                                 page turning out to hold the answer, or not
                                 to, is exactly the kind of fact only the
                                 reading can settle
        answer                  the Arabic reference answer, written from the
                                 page rather than the chunk text
        answer_en                a short English gloss, for the report only
        evidence                list of Evidence, quoted from chunk text
        gold_chunk_ids           chunk ids a correct retriever must return
        distractor_chunk_ids     plausible, non-answering chunk ids. Q10's
                                  CRAG case turns on this: a retriever that
                                  returns exactly these and nothing else is
                                  behaving correctly, and CRAG is what has to
                                  catch it from there
        page_vs_chunk            where the rendered page and the chunk text
                                  disagree, free text, or "none"

    Every field beyond the first five defaults empty rather than being
    required at construction. golden.py writes the skeleton with those five
    set and everything else blank, and its own gates are what require the rest
    to be filled before the set is usable, not this dataclass.
    """

    id: str
    question: str
    expect: str
    expected_route: str
    tests: str
    depends_on: str | None = None
    refusal_kind: str | None = None
    pages_to_read: list[str] = field(default_factory=list)
    pages_read: list[str] = field(default_factory=list)
    answer: str = ""
    answer_en: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    distractor_chunk_ids: list[str] = field(default_factory=list)
    page_vs_chunk: str = "none"


# --- page references ----------------------------------------------------------

def page_ref(source_slug: str, page: int) -> str:
    """Build a page reference, e.g. "central_alarm:p07".

    Deliberately shaped as a prefix of the chunk ids that page produces, e.g.
    "central_alarm_p07_grid_table_00", so a page reference and the chunk ids on
    it visibly relate to each other rather than needing a lookup table to
    connect a page number to what chunk.make_chunk_id built from it.
    """
    return f"{source_slug}:p{page:02d}"


def parse_page_ref(ref: str) -> tuple[str, int]:
    """Split a page reference back into (source_slug, page)."""
    slug, _, page_part = ref.partition(":p")
    return slug, int(page_part)


# --- storage --------------------------------------------------------------------

def corpus_fingerprint(chunks_path: Path) -> dict[str, Any]:
    """Hash the chunk file a golden set is read against.

    make_chunk_id documents the worry this closes: a golden set silently
    invalidated by a re-chunk. Renumbering is contained by scoping chunk ids to
    one page and one type, but nothing stops the corpus itself changing under a
    stable id, a section path corrected, a row reclassified, a chunk merged
    differently. Hashed rather than only counted, because two builds can hold
    the same chunk count and differ in content.
    """
    raw = chunks_path.read_bytes()
    return {
        "chunk_count": len(json.loads(raw)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def save_golden(
    questions: list[Question], fingerprint: dict[str, Any], path: Path
) -> None:
    """Write the golden set to JSON.

    UTF-8 is required rather than stylistic, the same reason chunk.save_chunks
    gives: the Windows default cannot represent Arabic and raises on the first
    question. ensure_ascii stays off so the file remains readable in an editor,
    which is the only practical way to check Arabic output, and the only way
    the reading pass in this stage happens at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_fingerprint": fingerprint,
        "questions": [asdict(q) for q in questions],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_golden(path: Path) -> tuple[list[Question], dict[str, Any]]:
    """Read the golden set back, returning (questions, corpus_fingerprint)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = [_question_from_dict(q) for q in raw["questions"]]
    return questions, raw["corpus_fingerprint"]


def _question_from_dict(raw: dict[str, Any]) -> Question:
    """Reconstruct one Question, rewrapping its nested Evidence list.

    asdict flattens Evidence into plain dicts on the way out; this is the one
    place that has to undo that, since dataclasses does not do it for you on
    the way back in.
    """
    data = dict(raw)
    data["evidence"] = [Evidence(**e) for e in data.get("evidence", [])]
    return Question(**data)
