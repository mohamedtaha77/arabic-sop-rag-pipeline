"""Scores the two extraction routes against known correct spellings.

Character counts alone cannot rank two extractions. A higher count may mean
hallucinated text; a lower count may mean skipped content, or it may mean the
extractor stopped emitting repeated junk. Only the fidelity check is anchored
outside the two files being compared, so it is the part that decides.

Ground truth was established by reading rendered page images and recording
words the text layer had corrupted.
"""

from __future__ import annotations

from ..config import OCR_OUTPUT, TEXTLAYER_OUTPUT
from .document import Document
from .storage import load_documents

# (correct spelling, corrupted spelling, English gloss)
GROUND_TRUTH = [
    ("على", "عسى", "on"),
    ("مساعد", "م اعد", "assistant"),
    ("التأكد", "الت كد", "verification"),
    ("حسب", "ح ب", "according to"),
    ("الرقابة", "الرقارة", "control"),
    ("التدقيق", "التدقيع", "audit"),
    ("نظام", "نما", "system"),
    ("التسجيلات", "التسعيلات", "recordings"),
]


def joined(documents: list[Document]) -> str:
    return "\n".join(d.text for d in documents)


def run() -> None:
    for path in (TEXTLAYER_OUTPUT, OCR_OUTPUT):
        if not path.exists():
            print(f"Missing {path.name}. Run both extraction routes first.")
            return

    textlayer = load_documents(TEXTLAYER_OUTPUT)
    ocr = load_documents(OCR_OUTPUT)
    text_a, text_b = joined(textlayer), joined(ocr)

    print("Text layer against OCR\n")

    print("Volume")
    print(f"{'':<20}{'text layer':>14}{'OCR':>14}")
    print(f"{'documents':<20}{len(textlayer):>14,}{len(ocr):>14,}")
    print(f"{'characters':<20}{len(text_a):>14,}{len(text_b):>14,}")
    print(f"{'chars per page':<20}"
          f"{len(text_a) // max(len(textlayer), 1):>14,}"
          f"{len(text_b) // max(len(ocr), 1):>14,}")

    print("\nFidelity, correct spelling against corrupted spelling")
    print(f"{'':<16}{'text layer':>22}{'OCR':>22}")
    print(f"{'gloss':<16}{'correct':>11}{'corrupt':>11}"
          f"{'correct':>11}{'corrupt':>11}")

    totals = [0, 0, 0, 0]
    for correct, corrupt, gloss in GROUND_TRUTH:
        counts = [text_a.count(correct), text_a.count(corrupt),
                  text_b.count(correct), text_b.count(corrupt)]
        totals = [t + c for t, c in zip(totals, counts)]
        print(f"{gloss:<16}" + "".join(f"{c:>11}" for c in counts))

    print(f"{'total':<16}" + "".join(f"{t:>11}" for t in totals))

    def accuracy(good: int, bad: int) -> str:
        return f"{good / (good + bad):.0%}" if good + bad else "n/a"

    print(f"{'accuracy':<16}{accuracy(totals[0], totals[1]):>22}"
          f"{accuracy(totals[2], totals[3]):>22}")

    print("\nLargest per page character differences")
    by_key = {(d.metadata["source"], d.metadata["page"]):
              d.metadata["char_count"] for d in textlayer}
    deltas = []
    for d in ocr:
        before = by_key.get((d.metadata["source"], d.metadata["page"]), 0)
        after = d.metadata["char_count"]
        deltas.append((after - before, d.metadata["source"],
                       d.metadata["page"], before, after))
    deltas.sort(reverse=True)

    for label, rows in (("OCR recovered more", deltas[:5]),
                        ("OCR produced less", sorted(deltas)[:5])):
        print(f"\n  {label}")
        for delta, source, page, before, after in rows:
            print(f"    {source[:38]:<38} p{page:>3} "
                  f"{before:>5} to {after:>5} ({delta:+,})")

    print("\n  Pages where OCR produced fewer characters are not losses on "
          "this corpus.")
    print("  The text layer decodes Arabic kashida, a letter stretching "
          "character used")
    print("  for justification, as repeated literal letters, which inflates "
          "its count.")


if __name__ == "__main__":
    run()
