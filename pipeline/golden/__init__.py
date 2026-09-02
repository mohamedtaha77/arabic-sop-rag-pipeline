"""Golden set stage: the evaluation anchor outside everything it will be
compared against.

Stages 1 to 4 built things that can only be compared against each other, a
table reconstruction against its own gates, three context variants against one
another. This stage is different: it is read off the rendered pages by hand,
the same ground truth method that produced ingestion's 44% against 97% figure,
and no model is called anywhere in it.

    question    the Question contract, page references, storage
    skeleton    the 20 authored questions, and why each one is shaped as it
                is. Not exported below and not wired into cli.py on purpose:
                run once by hand, `python -m pipeline.golden.skeleton`, and
                it refuses to overwrite a golden set that already has answers
    worksheet   renders pages and lays out the worksheet a person reads from
    golden      the eight gates that check what was written down

After the skeleton exists, run `python cli.py golden --pages` to render pages
and write the worksheet, fill in the answer fields by hand in an editor, then
run `python cli.py golden` to check the result. Both come after `python
cli.py chunk` and `python cli.py context`, since gate 2 checks gold and
distractor chunk ids against all three context variants.
"""

from .golden import run as run_golden
from .golden import verify
from .question import (
    EXPECT_VALUES,
    EXPECTED_ROUTES,
    REFUSAL_KINDS,
    Evidence,
    Question,
    corpus_fingerprint,
    load_golden,
    page_ref,
    parse_page_ref,
    save_golden,
)
from .worksheet import run as run_worksheet
from .worksheet import run_pages

__all__ = [
    "Question",
    "Evidence",
    "EXPECT_VALUES",
    "REFUSAL_KINDS",
    "EXPECTED_ROUTES",
    "page_ref",
    "parse_page_ref",
    "corpus_fingerprint",
    "save_golden",
    "load_golden",
    "run_worksheet",
    "run_pages",
    "verify",
    "run_golden",
]
