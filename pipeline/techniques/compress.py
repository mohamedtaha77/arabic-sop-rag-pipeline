"""Compression: shrink each retrieved chunk to the sentences that
actually bear on the question, verified rather than trusted.

Position: called from techniques/run.py's answer() after Reranking, on
whatever RETRIEVAL_K chunks survived it, either because the router
selected Compression or because run.py's own runtime trigger fired,
context over LLM_CONTEXT's budget once rendered. Analysis question 8
asks how many tokens compression removed and whether answer quality
changed; the first half is this module's own trace, the second is
stage 9's to measure once generation exists.

The instruction given to the model is extractive: copy the relevant
sentences verbatim, drop the rest, change nothing. A 3B local model
does not reliably follow that instruction; asking nicely is not
enforcement, the same lesson stage 9's own grounding guard is built
around. So the extraction is checked, not trusted: what comes back has
to be a genuine substring of the original chunk text (whitespace
normalised, since a model reflowing line breaks is not the same failure
as a model inventing content) or the original chunk is kept whole,
uncompressed, rather than risk feeding a paraphrase forward as if it
were the corpus's own words. That is also why a chunk is never dropped
entirely here, even when the model reports nothing relevant: apply()
always returns the same chunk_ids it was given, in the same order,
each either genuinely shorter or exactly as it arrived. A model too
readily concluding "nothing relevant" would otherwise silently shrink
the very context CRAG or the synthesiser was going to reason over.

What this module does not do: it does not decide whether Compression
should run, and it does not rerank or refuse anything; it only shortens
text that something else already chose to keep.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..config import CHARS_PER_TOKEN, GENERATOR_MODEL
from ..llm.ledger import Ledger
from ..retrieval.retriever import ScoredChunk

# A compressed chunk should be shorter than the corpus's own largest
# measured chunk (llm.md: roughly 1,040 tokens against Qwen2.5's
# tokenizer); generous enough that a genuinely long relevant extract is
# never itself truncated into an LLMError.
_MAX_TOKENS = 1024

_SYSTEM_PROMPT = """\
Extract only the sentences from the passage that are directly relevant \
to answering the question, copied exactly as written, word for word, \
changing nothing at all: no paraphrasing, no summarising, no adding or \
removing a single word from what you keep. Drop irrelevant sentences \
entirely. If nothing in the passage is relevant, reply with the single \
word NONE. Reply with the extracted text only, no explanation, no \
quotation marks, no other text before or after it.\
"""


@dataclass
class CompressTrace:
    """What analysis question 8 asks for: how many tokens compression
    removed. chars_before and chars_after are measured directly rather
    than estimated from a token count that was never computed;
    tokens_removed_estimate applies CHARS_PER_TOKEN the same way
    run.py's own context-budget trigger does, so the two numbers stay
    on the same scale.

        rewritten_chunk_ids   chunk ids whose text was genuinely
                               shortened by a verified extraction
        kept_whole_chunk_ids   chunk ids returned unchanged, either
                               because the model found nothing to drop,
                               or because its own output failed the
                               substring check and was discarded in
                               favour of the original
    """

    chars_before: int
    chars_after: int
    tokens_removed_estimate: int
    rewritten_chunk_ids: tuple[str, ...]
    kept_whole_chunk_ids: tuple[str, ...]


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _compress_one(question: str, chunk_text: str, ledger: Ledger) -> str | None:
    """One chunk's own extraction, or None if the model's output did not
    pass the substring check and the original should be kept instead.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nPassage:\n{chunk_text}"},
    ]
    response = ledger.call(
        "Compression", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=_MAX_TOKENS,
    )
    extracted = response.text.strip().strip('"').strip("“”").strip()

    if not extracted or extracted.upper() == "NONE":
        return None

    normalised_original = _normalise(chunk_text)
    normalised_extracted = _normalise(extracted)
    if normalised_extracted and normalised_extracted in normalised_original:
        return extracted
    return None


def apply(
    query_text: str,
    scored: list[ScoredChunk],
    ledger: Ledger,
) -> tuple[list[ScoredChunk], CompressTrace]:
    """Compress every candidate's text against the question, one call
    each. Always returns exactly len(scored) items, same chunk_ids,
    same order: compression shortens what is kept, it never decides
    what to drop.
    """
    if not scored:
        return scored, CompressTrace(0, 0, 0, (), ())

    result: list[ScoredChunk] = []
    rewritten: list[str] = []
    kept_whole: list[str] = []
    chars_before = 0
    chars_after = 0

    for item in scored:
        original_text = item.chunk.text
        chars_before += len(original_text)
        extracted = _compress_one(query_text, original_text, ledger)

        if extracted is not None:
            chars_after += len(extracted)
            rewritten.append(item.chunk_id)
            new_chunk = dataclasses.replace(item.chunk, text=extracted)
            result.append(dataclasses.replace(item, chunk=new_chunk))
        else:
            chars_after += len(original_text)
            kept_whole.append(item.chunk_id)
            result.append(item)

    trace = CompressTrace(
        chars_before=chars_before, chars_after=chars_after,
        tokens_removed_estimate=int((chars_before - chars_after) / CHARS_PER_TOKEN),
        rewritten_chunk_ids=tuple(rewritten),
        kept_whole_chunk_ids=tuple(kept_whole),
    )
    return result, trace


# --- verification --------------------------------------------------------------

def verify_reduces_and_stays_faithful(handle: "object", question_ids: tuple[str, ...]) -> list[dict]:
    """The build order's own gate: tokens removed counted, and every
    surviving chunk's text is genuinely a substring of what retrieval
    originally returned. handle is a retriever.ShippingHandle, typed
    loosely to avoid importing qdrant_client at module level for a
    function only __main__ calls.

    Run on a chosen subset rather than the full 18: each question costs
    up to RERANK_TOP_N separate LLM calls here, and this gate exists to
    check the mechanism is sound, not to reproduce evaluate.py's own
    full-set grid.
    """
    from ..config import GOLDEN_SET, RERANK_TOP_N
    from ..golden.question import load_golden
    from ..retrieval.retriever import retrieve_scored
    from . import rerank

    questions, _ = load_golden(GOLDEN_SET)
    by_id = {q.id: q for q in questions}

    results = []
    for qid in question_ids:
        question = by_id[qid]
        candidates = retrieve_scored(
            handle.context, question.question, handle.decision["mode"],
            k=RERANK_TOP_N, apply_caps=False,
        )
        ledger = Ledger(label=f"verify-compress-{qid}")
        reranked, _rerank_trace = rerank.apply(question.question, candidates, ledger)
        compressed, trace = apply(question.question, reranked, ledger)

        faithful = all(
            _normalise(new.chunk.text) in _normalise(old.chunk.text)
            for old, new in zip(reranked, compressed)
        )
        results.append({
            "id": qid,
            "chars_before": trace.chars_before,
            "chars_after": trace.chars_after,
            "tokens_removed_estimate": trace.tokens_removed_estimate,
            "rewritten": list(trace.rewritten_chunk_ids),
            "kept_whole": list(trace.kept_whole_chunk_ids),
            "chunk_count_unchanged": len(compressed) == len(reranked),
            "faithful": faithful,
        })
    return results


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping
    from . import rerank

    # warm_up() before open_shipping(): this gate runs Reranking first,
    # see rerank.warm_up's own docstring for why the order matters.
    rerank.warm_up()
    with open_shipping() as shipping_handle:
        results = verify_reduces_and_stays_faithful(
            shipping_handle, ("Q5", "Q8", "Q13", "Q17", "Q19"),
        )

    lines = []
    total_before = total_after = 0
    all_faithful = True
    all_size_stable = True
    for row in results:
        total_before += row["chars_before"]
        total_after += row["chars_after"]
        all_faithful = all_faithful and row["faithful"]
        all_size_stable = all_size_stable and row["chunk_count_unchanged"]
        lines.append(
            f"{row['id']}: {row['chars_before']} -> {row['chars_after']} chars "
            f"(~{row['tokens_removed_estimate']} tokens removed), "
            f"rewritten={len(row['rewritten'])}, kept_whole={len(row['kept_whole'])}, "
            f"faithful={row['faithful']}, chunk_count_unchanged={row['chunk_count_unchanged']}"
        )

    lines.insert(0, f"total: {total_before} -> {total_after} chars, "
                     f"faithful across all questions: {all_faithful}, "
                     f"chunk count always preserved: {all_size_stable}")
    lines.insert(1, "")

    out_path = PROCESSED_DIR / "13_compress_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    ok = all_faithful and all_size_stable and total_after < total_before
    print(f"{'ok' if ok else 'FAIL'}: {total_before} -> {total_after} chars, "
          f"written to {out_path}")
