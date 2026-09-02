"""What a free call would have cost.

Every call in this pipeline runs on this machine and spends no money. That is
the point of the local decision, and it is also a reporting problem: the task
asks for a per-step cost table, and a table of zeros distinguishes nothing. It
would show Basic RAG and Advanced RAG as equally free, which is true of the
bank balance and useless as a comparison.

So the token counts are real, measured by the server, and only the price is
notional: what these same tokens would have cost against a published hosted
rate card. That makes the Basic against Advanced comparison carry a scale
without pretending an invoice exists. Every figure this module produces is a
counterfactual and the report says so.

Latency is the number that is not notional here. Running locally makes it the
scarce resource, which is why ledger.py reports it beside the cost rather than
leaving it out.
"""

from __future__ import annotations

from dataclasses import dataclass

# Anthropic first-party API list prices, USD per million tokens, read
# 2026-06-24. Re-check before the report is published: rate cards move, and a
# stale card quoted as current is the kind of error nobody catches by reading
# the code.
#
# Three cards rather than one because they answer different questions. The
# report can state what this pipeline would cost at the tier its own 3B local
# model actually belongs to, and what it would cost on a frontier model, and
# those are different arguments.


@dataclass(frozen=True)
class RateCard:
    """Published prices for one hosted model, USD per million tokens."""

    name: str
    input_per_mtok: float
    output_per_mtok: float


RATE_CARDS = {
    "haiku-4.5": RateCard("Claude Haiku 4.5", 1.00, 5.00),
    "sonnet-5": RateCard("Claude Sonnet 5", 2.00, 10.00),
    "opus-5": RateCard("Claude Opus 5", 5.00, 25.00),
}

# The small-model tier, chosen as the default because it is the honest
# comparison: a 3B instruct model priced at frontier rates would inflate every
# figure in the report by five times and flatter the local decision. The
# frontier cards stay available for the upper bound the report also wants.
DEFAULT_CARD = "haiku-4.5"


def price(prompt_tokens: int, completion_tokens: int,
          card: str = DEFAULT_CARD) -> float:
    """Notional USD for one call. No money is spent.

    Output tokens cost several times input on every card here, which is why
    both are counted separately rather than summed. The techniques this
    pipeline compares differ mostly in how much context they send, so a single
    blended rate would hide the thing being measured.
    """
    rates = RATE_CARDS[card]
    return (prompt_tokens * rates.input_per_mtok
            + completion_tokens * rates.output_per_mtok) / 1_000_000
