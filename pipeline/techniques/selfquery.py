"""Self-Query: split a question into a semantic part and a metadata
filter, and narrow the searchable set before anything embeds.

Position: called from techniques/run.py's answer() before retrieval,
when schema.TechniqueSet.self_query is set. Returns the semantic query
text retrieval should search with, and the allowed chunk id set every
downstream leg (retriever.retrieve_scored's own query_filter for dense
and sparse, a plain membership check for BM25) narrows to before ranking
anything, not after.

The filterable fields are only the ones with real variance on this
corpus, measured rather than assumed: source (3 manuals), issue_date (2
distinct dates, 08/2024 and 02/2026), chunk_type (8 values), and page.
extraction_quality, doc_version and review_date are deliberately absent
from the filter vocabulary below: retrieval.md's own chunking fix note
records that extraction_quality is now constant, "ok", across all 357
chunks, and doc_version and review_date are effectively constant too (see
04_bakeoff_decision.json's own corpus read). A filter on a constant would
be untestable code that looks like a feature, the exact thing
retrieval.md names as the reason nothing downstream reads
extraction_quality for ranking.

Q7 is this technique's own case, "ما هي الأدلة التي صدرت قبل عام
2025؟", a metadata filter on issue_date with no equivalent semantic
content: Central Alarm is the only one of the three manuals issued before
2025, and the semantic query alone, with no filter, would have no lexical
or semantic anchor pointing at that distinction at all.

What this module does not do: it does not retrieve anything itself, and
it does not decide whether Self-Query should run at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..chunking.chunk import CHUNK_TYPES, Chunk, source_slug
from ..config import GENERATOR_MODEL
from ..llm.ledger import Ledger
from ..retrieval.retriever import RetrievalContext

# The three real manuals, by the same slug chunk.make_chunk_id already
# gives every chunk id, so the LLM's own output is compared against the
# corpus using one shared vocabulary rather than a second spelling of
# the same three names that could drift from source_slug's own output.
_VALID_SOURCES = frozenset({"central_alarm", "assets_wearhouse", "central_mail"})

_SYSTEM_PROMPT = """\
Extract two things from the user's question, to search a store of \
Arabic internal bank procedure manuals with structured metadata \
filters.

semantic_query: the part of the question that should still be searched \
by meaning, with any explicit metadata constraint (which manual, a \
date, a kind of content) removed. If the whole question is itself \
semantic content with no such constraint, repeat it unchanged.

filters, every field optional and null when the question does not name \
that exact constraint. Never guess a filter from context; only set one \
the question explicitly states.
  source              one of "central_alarm", "assets_wearhouse", \
"central_mail": set only when the question names or clearly identifies \
one single manual
  issue_date_before    a four-digit year: set only when the question \
asks for manuals issued before a stated year
  issue_date_after     a four-digit year: set only when the question \
asks for manuals issued after a stated year
  chunk_type           one of procedure_block, reference, grid_row, \
grid_table, accounting_entry, approval, revision, prose: set only when \
the question explicitly asks for that kind of content, such as "the \
revision log" (revision) or "the approval page" (approval)
  page                 an exact page number: set only when the question \
names one

Reply with JSON only, no other text: {"semantic_query": "...", \
"filters": {"source": null, "issue_date_before": null, \
"issue_date_after": null, "chunk_type": null, "page": null}}\
"""


@dataclass
class SelfQueryTrace:
    """What analysis question 6 asks for: the semantic query and the
    metadata filters actually extracted, side by side.

        original        the question as asked
        semantic_query   what retrieval actually searches with
        filters          the filter dict, non-null entries only
        allowed_count    chunks the filter narrowed to, or None when no
                         filter was named at all (not the same as the
                         fallback case below, which also has no active
                         restriction but for a different reason)
        fell_back        True when a filter was named but matched no
                         chunk, and retrieval fell back to unfiltered
                         rather than searching an empty set
    """

    original: str
    semantic_query: str
    filters: dict[str, Any] = field(default_factory=dict)
    allowed_count: int | None = None
    fell_back: bool = False


def _as_year(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _issue_year(chunk: Chunk) -> int | None:
    """The year out of an issue_date field stored as "MM/YYYY", or None
    for a chunk with no such field to compare against.
    """
    raw = chunk.metadata.get("issue_date")
    if not raw or "/" not in raw:
        return None
    _, _, year_part = raw.rpartition("/")
    return int(year_part) if year_part.isdigit() else None


def _extract(question: str, ledger: Ledger) -> tuple[str, dict[str, Any]]:
    """The semantic query and a validated filters dict, every value
    checked against its own closed vocabulary or coerced to an int, so
    a hallucinated field value is dropped rather than reaching a filter
    that would silently never match anything.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = ledger.call(
        "Self-Query", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=300,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError:
        parsed = {}

    semantic_query = parsed.get("semantic_query")
    if not isinstance(semantic_query, str) or not semantic_query.strip():
        semantic_query = question

    raw_filters = parsed.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}

    source = raw_filters.get("source")
    chunk_type = raw_filters.get("chunk_type")
    filters = {
        "source": source if source in _VALID_SOURCES else None,
        "issue_date_before": _as_year(raw_filters.get("issue_date_before")),
        "issue_date_after": _as_year(raw_filters.get("issue_date_after")),
        "chunk_type": chunk_type if chunk_type in CHUNK_TYPES else None,
        "page": _as_year(raw_filters.get("page")),
    }
    return semantic_query.strip(), filters


def _matches(chunk: Chunk, filters: dict[str, Any]) -> bool:
    """Every non-null filter field is an AND condition; a chunk with no
    value for a field being filtered on (an unparseable issue_date, for
    instance) never matches that field rather than matching by default.
    """
    if filters.get("source") is not None:
        if source_slug(chunk.metadata.get("source", "")) != filters["source"]:
            return False
    if filters.get("issue_date_before") is not None:
        year = _issue_year(chunk)
        if year is None or not year < filters["issue_date_before"]:
            return False
    if filters.get("issue_date_after") is not None:
        year = _issue_year(chunk)
        if year is None or not year > filters["issue_date_after"]:
            return False
    if filters.get("chunk_type") is not None:
        if chunk.metadata.get("chunk_type") != filters["chunk_type"]:
            return False
    if filters.get("page") is not None:
        if chunk.metadata.get("page") != filters["page"]:
            return False
    return True


def _build_allowed_set(
    context: RetrievalContext, filters: dict[str, Any],
) -> tuple[set[str] | None, bool]:
    """The allowed chunk id set for these filters, or (None, False) when
    no filter field is active at all, or (None, True) when every field
    was active but matched nothing: an empty filter is a caller error in
    intent, not evidence the corpus has nothing to say, so retrieval
    falls back to searching everything rather than returning nothing.
    """
    if not any(value is not None for value in filters.values()):
        return None, False

    allowed = {
        chunk.metadata["chunk_id"]
        for chunk in context.chunks if _matches(chunk, filters)
    }
    if not allowed:
        return None, True
    return allowed, False


def apply(
    query_text: str,
    context: RetrievalContext,
    ledger: Ledger,
) -> tuple[str, set[str] | None, SelfQueryTrace]:
    """Extract a semantic query and a filter, and build the allowed
    chunk id set that filter narrows retrieval to.
    """
    semantic_query, filters = _extract(query_text, ledger)
    allowed, fell_back = _build_allowed_set(context, filters)

    trace = SelfQueryTrace(
        original=query_text, semantic_query=semantic_query, filters=filters,
        allowed_count=(len(allowed) if allowed is not None else None),
        fell_back=fell_back,
    )
    return semantic_query, allowed, trace


# --- verification --------------------------------------------------------------

def verify_q7_restricts_and_finds_gold(
    handle: "object",
) -> tuple[bool, bool, SelfQueryTrace]:
    """Q7's own gate, from the build order: the filter restricts to
    Central Alarm alone, and the gold chunk still returns once retrieval
    runs against that restricted set. handle is a
    retriever.ShippingHandle, typed loosely to avoid importing
    qdrant_client at module level for a function only __main__ calls.
    """
    from ..config import GOLDEN_SET
    from ..golden.question import load_golden
    from ..retrieval.retriever import retrieve_scored

    questions, _ = load_golden(GOLDEN_SET)
    q7 = next(q for q in questions if q.id == "Q7")
    gold_id = q7.gold_chunk_ids[0]

    ledger = Ledger(label="verify-selfquery-q7")
    semantic_query, allowed, trace = apply(q7.question, handle.context, ledger)

    restricts_to_central_alarm = bool(
        allowed is not None
        and 0 < len(allowed) < len(handle.context.chunks)
        and all(
            handle.context.chunk_lookup[cid].metadata["source"].startswith("Central Alarm")
            for cid in allowed
        )
    )

    scored = retrieve_scored(
        handle.context, semantic_query, handle.decision["mode"],
        apply_caps=handle.decision.get("apply_caps", False),
        allowed_chunk_ids=allowed,
    )
    gold_found = gold_id in {item.chunk_id for item in scored}

    return restricts_to_central_alarm, gold_found, trace


def verify_empty_filter_falls_back(context: RetrievalContext) -> bool:
    """A filter combination that matches nothing (no chunk is both a
    revision row and a grid_table on the same manual) falls back to
    unfiltered (None) rather than returning an empty, unsearchable set.
    Tested directly against the filter-building logic, bypassing the LLM
    call, since this is a property of _build_allowed_set alone.
    """
    impossible = {
        "source": "assets_wearhouse", "issue_date_before": None,
        "issue_date_after": None, "chunk_type": "revision", "page": 1,
    }
    allowed, fell_back = _build_allowed_set(context, impossible)
    return allowed is None and fell_back


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    with open_shipping() as shipping_handle:
        restricts, found, trace = verify_q7_restricts_and_finds_gold(shipping_handle)
        fallback_ok = verify_empty_filter_falls_back(shipping_handle.context)

    lines = [
        f"Q7 filter restricts to Central Alarm alone: {restricts}",
        f"Q7 gold chunk found after filtering: {found}",
        f"empty-filter fallback exercised and correct: {fallback_ok}",
        f"semantic_query: {trace.semantic_query}",
        f"filters: {trace.filters}",
        f"allowed_count: {trace.allowed_count}",
    ]
    out_path = PROCESSED_DIR / "11_selfquery_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    ok = restricts and found and fallback_ok
    print(f"{'ok' if ok else 'FAIL'}: written to {out_path}")
