"""Okapi BM25, written out rather than imported.

rank_bm25 is not a dependency of this project, and fifteen real lines of
formula follow sparse.py's own precedent: a library pulled in for a short,
well-known equation is a heavier and less reviewable choice than the
equation itself, and camel-kenlm-style packages carry a real Windows build
risk on Python 3.13 that this project has already spent enough of a
five-day budget dodging in embedder.py's dependency choices.

Position: bm25.py is one retrieval leg among several. fusion.py combines
its ranking with the dense and learned-sparse legs; nothing here decides
how those combine, and nothing here embeds anything or touches a model.

What this module does not do: it does not choose k1 or b by fitting them
to the golden set. Both stay at their standard textbook values, named and
explained below, because tuning two parameters against 18 questions would
fit the golden set rather than the corpus, the same objection the plan
raises against tuning fusion weights.
"""

from __future__ import annotations

import collections
import math
from typing import Callable

from ..config import BM25_B, BM25_K1
from .text import tokenize_clitics


class BM25Index:
    """One corpus's worth of documents, indexed once, ranked many times.

    Built in process rather than cached to disk: 357 documents tokenizes
    and indexes in well under a second, so a cache here would be machinery
    guarding nothing.
    """

    def __init__(
        self,
        chunk_ids: list[str],
        texts: list[str],
        k1: float = BM25_K1,
        b: float = BM25_B,
        tokenize: Callable[[str], list[str]] = tokenize_clitics,
    ) -> None:
        """tokenize defaults to the clitic-stripped level, the one text.py's
        own module docstring records as measuring 0.713-0.769 Recall@10
        against plain tokenisation's 0.630 on the golden set. evaluate.py's
        grid passes tokenize_plain and tokenize_stopwords too, so that
        comparison is reproduced inside the real pipeline rather than
        trusted to the probe that first found it.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self.chunk_ids = chunk_ids
        self.k1 = k1
        self.b = b
        self.tokenize = tokenize

        self._term_frequencies: list[collections.Counter[str]] = [
            collections.Counter(tokenize(text)) for text in texts
        ]
        self._doc_lengths = [sum(tf.values()) for tf in self._term_frequencies]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths else 0.0
        )

        document_frequency: collections.Counter[str] = collections.Counter()
        for tf in self._term_frequencies:
            document_frequency.update(tf.keys())

        # The non-negative Robertson-Sparck-Jones form, log(1 + (N-n+.5)/(n+.5)),
        # stated explicitly rather than left as the classic log((N-n+.5)/(n+.5)).
        # The classic form goes negative once a term sits in more than half the
        # corpus, and that is a real case here, not a textbook footnote: the
        # template context variant's prefix puts a document's own section and
        # title words into every one of its chunks, which is exactly the shape
        # that pushes a term past the 50% mark.
        n_docs = len(texts)
        self._inverse_document_frequency = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        """One BM25 score per document, in the index's own chunk order."""
        scores = [0.0] * len(self.chunk_ids)
        for i, (tf, length) in enumerate(zip(self._term_frequencies, self._doc_lengths)):
            total = 0.0
            length_norm = 1 - self.b + self.b * (
                length / self._avg_doc_length if self._avg_doc_length else 0.0
            )
            for term in query_tokens:
                frequency = tf.get(term)
                if not frequency:
                    continue
                idf = self._inverse_document_frequency.get(term, 0.0)
                total += idf * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * length_norm
                )
            scores[i] = total
        return scores

    def rank(self, question: str) -> list[str]:
        """Chunk ids in descending BM25 order for this question.

        Ties (an all-zero score when no query term appears anywhere) are
        broken by chunk_id's own sort order, which is deterministic and
        readable rather than by whatever order Python's sort happens to
        leave equal-scored items in; that matters for gate reproducibility
        the same way rank_by_similarity's own docstring cares about it.
        """
        query_tokens = self.tokenize(question)
        scores = self.score(query_tokens)
        order = sorted(
            range(len(self.chunk_ids)),
            key=lambda i: (-scores[i], self.chunk_ids[i]),
        )
        return [self.chunk_ids[i] for i in order]

    def rank_scored(self, question: str) -> list[tuple[str, float]]:
        """rank(), keeping each chunk id's own BM25 score.

        Added for stage 8: retriever.retrieve_scored's BM25 leg needs a
        score alongside the id, and a Self-Query filter needs the full
        357-row ranking to filter down from rather than a pre-truncated
        one, since filtering after truncation could drop a chunk that
        would have ranked inside the allowed subset's own top k. rank()
        above is unchanged and stays what evaluate.py's grid and
        retriever.retrieve both call.
        """
        query_tokens = self.tokenize(question)
        scores = self.score(query_tokens)
        order = sorted(
            range(len(self.chunk_ids)),
            key=lambda i: (-scores[i], self.chunk_ids[i]),
        )
        return [(self.chunk_ids[i], scores[i]) for i in order]
