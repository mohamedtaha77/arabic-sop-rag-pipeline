"""Rewriting: turn an ambiguous or context-dependent question into one
that stands on its own before it ever reaches an embedder.

Position: called from techniques/run.py's answer(), before retrieval,
when schema.TechniqueSet.rewriting is set. What reaches retrieve_scored
afterwards is this function's own output, not the original question text
plain; every other technique downstream sees only the rewritten form.

Two shapes exercise this, and this module has to handle both without
being told which one it is looking at, since the router's own selection
of Rewriting is the same for either:

  Q3's shape       "ولماذا هذه المدة تحديداً؟" means nothing without Q2's
                   turn beside it. Resolving it is the whole job.
  A standalone,     the plan's own six-signal table names "vague or
  vague question    conversational wording" as a Rewriting signal on its
                   own, with no prior turn required. golden.md never
                   built a question exercising this shape in isolation,
                   so this path is real but only Q3's own before/after
                   Recall@10 is what this file's gate can actually check.

What this module does not do: it does not decide whether Rewriting
should run at all. That is the router's call, or run.py's technique_set
override for an evaluate.py grid cell; by the time apply() is called,
that decision is already made.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import GENERATOR_MODEL
from ..llm.ledger import Ledger

# A rewritten question is a search query, not an answer; long is a sign
# the model padded it with explanation rather than just rewriting it.
REWRITE_MAX_TOKENS = 150

_SYSTEM_PROMPT = """\
Rewrite the user's question into one clear, self-contained Arabic \
question, ready to search a document store with no other context \
available.

If a prior turn (a previous question and its answer) is given, resolve \
every pronoun, demonstrative, or implicit reference in the question \
against it explicitly, so the rewritten question would make sense to \
someone who never saw the prior turn at all.

If no prior turn is given, only smooth vague or conversational wording \
into a plain, direct question. Do not change what is being asked, and \
do not add any fact, number, or assumption the original question did \
not already contain or clearly imply.

Keep the same language and register the question was asked in. Reply \
with the rewritten question only: no explanation, no quotation marks, \
no other text before or after it.\
"""


@dataclass
class RewriteTrace:
    """What analysis question 2 asks for: an example of rewriting, and
    why. original and rewritten side by side are the "why"; a reader
    checking the report against this trace should be able to see the
    resolved reference without re-deriving it from the prior turn.
    """

    original: str
    rewritten: str
    used_history: bool


def apply(
    question: str,
    history: list[tuple[str, str]] | None,
    ledger: Ledger,
) -> tuple[str, RewriteTrace]:
    """Rewrite one question, returning the text retrieval should search
    with and a trace of what changed.

    Falls back to the original question, unmodified, on anything that
    would otherwise hand retrieval an empty or clearly broken string: a
    model returning nothing is a worse query than the one it started
    from, and retrieval on the original text is always a safe fallback
    since that is exactly what would have run had Rewriting never fired.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for prior_question, prior_answer in history or []:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    response = ledger.call(
        "Rewriter", messages, GENERATOR_MODEL,
        temperature=0.0, max_tokens=REWRITE_MAX_TOKENS,
    )
    rewritten = response.text.strip().strip('"').strip("“”").strip()

    if not rewritten:
        rewritten = question

    return rewritten, RewriteTrace(
        original=question, rewritten=rewritten, used_history=bool(history),
    )


# --- verification --------------------------------------------------------------

def verify_q3_improves(handle: "object") -> tuple[float, float, str]:
    """Q3's own Recall@10, with and without Rewriting, against the
    shipping retrieval configuration. handle is a
    retriever.ShippingHandle; typed loosely here to avoid importing
    qdrant_client at module level for a function only __main__ calls.

    A single question's before/after, not a bootstrap CI: that broader,
    properly powered claim about Rewriting in general is evaluate.py's
    job (file 15), held to the same paired-bootstrap rule stages 6 and 7
    were. This is the narrower thing the build order actually names as
    this file's own gate.
    """
    from ..config import GOLDEN_SET, RETRIEVAL_K
    from ..embedding.metrics import recall_at_k
    from ..golden.question import load_golden
    from ..retrieval.retriever import retrieve

    questions, _ = load_golden(GOLDEN_SET)
    by_id = {q.id: q for q in questions}
    q3, q2 = by_id["Q3"], by_id["Q2"]
    gold = set(q3.gold_chunk_ids)
    mode = handle.decision["mode"]
    caps = handle.decision.get("apply_caps", False)

    unrewritten_ranked = retrieve(handle.context, q3.question, mode, apply_caps=caps)
    before = recall_at_k(unrewritten_ranked, gold, RETRIEVAL_K)

    ledger = Ledger(label="verify-rewrite-q3")
    rewritten_text, _trace = apply(q3.question, [(q2.question, q2.answer)], ledger)
    rewritten_ranked = retrieve(handle.context, rewritten_text, mode, apply_caps=caps)
    after = recall_at_k(rewritten_ranked, gold, RETRIEVAL_K)

    return before, after, rewritten_text


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    from ..retrieval.retriever import open_shipping

    with open_shipping() as shipping_handle:
        before_score, after_score, rewritten = verify_q3_improves(shipping_handle)

    out_path = PROCESSED_DIR / "07_rewrite_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"Q3 Recall@10 unrewritten: {before_score:.3f}\n"
        f"Q3 Recall@10 rewritten:   {after_score:.3f}\n"
        f"rewritten text: {rewritten}\n",
        encoding="utf-8",
    )
    verdict = "ok, improved" if after_score > before_score else "FLAT/WORSE"
    print(f"{verdict}: {before_score:.3f} -> {after_score:.3f}, "
          f"written to {out_path}")
