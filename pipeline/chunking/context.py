"""Three chunk sets for the Contextual Retrieval comparison.

02_chunks.json is what stage 2 produced, and this stage does not change a
single retrievable unit in it. What it decides is what, if anything, gets
prepended to each chunk's text before embedding, and it builds three parallel
corpora so stage 6 can compare them against the golden set: no prefix, a
prefix assembled from metadata already in hand, and a prefix a local model
writes after reading the chunk in its section's context.

The reason for three rather than one, or two: without the no-prefix baseline,
a result can show which prefix wins and not whether prefixing helps at all,
and that is the question this project set out to answer rather than assert.

What this module does not do: it does not retrieve anything, and it does not
decide whether a prefix helped. That is stage 6's evaluation, against a golden
set that does not exist yet. This stage's job ends at three files on disk,
each recoverable back to the original chunk text, plus a stated cost for the
one variant that spends anything.
"""

from __future__ import annotations

import collections
import copy
import re
import time
from pathlib import Path

from ..config import (
    CHUNKS_OUTPUT,
    CONTEXT_OUTPUTS,
    CONTEXT_SAMPLES_OUTPUT,
    CONTEXT_VARIANTS,
    GENERATOR_MODEL,
    LAYOUT_OUTPUT,
    LLM_CONTEXT,
)
from ..ingestion.cleaning import clean_text, correct_ocr_misreads
from ..ingestion.document import Document
from ..ingestion.quality import OFF_SCRIPT_LIMIT, assess_quality
from ..ingestion.storage import load_documents
from ..llm.client import LLMError
from ..llm.ledger import Ledger
from .chunk import CHUNK_TYPES, Chunk, load_chunks, save_chunks

# Joins a prefix to the chunk it describes: a blank line, the same paragraph
# break clean_text produces, so a prefix reads as its own line rather than
# running into the chunk's first word. Never applied when the prefix is
# empty, which is what keeps the none variant byte-identical to
# 02_chunks.json (gate 2).
#
# compose and decompose are written together, one undoing the other, so the
# join and the check that recovers the original text (gate 3) can never drift
# apart from each other. That is the same move cleaning.py makes compiling
# ENTRY_LINE from folded source: make the property true by construction
# rather than by two people remembering to agree.
PREFIX_SEPARATOR = "\n\n"


def compose(prefix: str, body: str) -> str:
    """Build the text a variant actually embeds."""
    return body if not prefix else f"{prefix}{PREFIX_SEPARATOR}{body}"


def decompose(prefix: str, text: str) -> str:
    """Undo compose(). Raises if text was not built from this exact prefix."""
    if not prefix:
        return text
    joined = f"{prefix}{PREFIX_SEPARATOR}"
    if not text.startswith(joined):
        raise ValueError("text does not start with its own recorded prefix")
    return text[len(joined):]


# --- the manual title, read off the cover -----------------------------------

def manual_title(cover: Document) -> str | None:
    """The Arabic title on one manual's page-1 cover, or None if the page
    does not have the expected shape.

    All three covers hold one table, one row, one cell, three or four lines:
    the bank's Arabic name, its English name, then the title on one or two
    lines. The title is derived by dropping the first two lines rather than
    matched by content, so a fourth manual needs no code change here. Assets'
    cover is OCR-damaged in exactly this text and is corrected by three
    entries added to cleaning.py's KNOWN_OCR_MISREADS for this reason, each
    confirmed by rendering the page and reading it.

    If a future cover does not match this shape, None is returned and the
    template prefix drops the title rather than emit a wrong one.
    """
    tables = cover.metadata.get("tables") or []
    if len(tables) != 1:
        return None
    rows = tables[0]["rows"]
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    lines = [line.strip() for line in rows[0][0].split("\n") if line.strip()]
    if len(lines) < 3 or "بنك" not in lines[0]:
        return None
    title = correct_ocr_misreads(clean_text(" ".join(lines[2:])))
    return title or None


def manual_titles(documents: list[Document]) -> dict[str, str | None]:
    """One title per source, keyed the same way chunk.metadata['source'] is."""
    covers: dict[str, Document] = {}
    for document in documents:
        if document.metadata["page"] == 1:
            covers[document.metadata["source"]] = document
    return {source: manual_title(cover) for source, cover in covers.items()}


# --- the template prefix -----------------------------------------------------

def _page_label(page: int, end_page: int | None) -> str:
    """صفحة 7, or صفحات 10-12 when layout.py merged a table across pages.

    148 of 357 chunks carry a different end_page, so the plural form is not
    an edge case worth collapsing into the singular one.
    """
    if end_page and end_page != page:
        return f"صفحات {page}-{end_page}"
    return f"صفحة {page}"


def template_prefix(chunk: Chunk, title: str | None) -> str:
    """Assemble a prefix from metadata already in the chunk. No model call.

    review_date is left out deliberately: every chunk in the corpus reads
    02/2026 for it, so it discriminates nothing and would cost 14 characters
    on every one of them. chunk_type is left out too: grid_row would add the
    Arabic word for "table" beside a section_path that already says so, a
    term no query will use.
    """
    meta = chunk.metadata
    parts: list[str] = []
    if title:
        parts.append(title)
    if meta.get("section_path"):
        parts.append(meta["section_path"])
    parts.append(_page_label(meta["page"], meta.get("end_page")))
    if meta.get("doc_version"):
        parts.append(f"نسخة {meta['doc_version']}")
    if meta.get("issue_date"):
        parts.append(f"إصدار {meta['issue_date']}")

    # Actor and unit only when the chunk's own text does not already open
    # with them. Most procedure blocks do, because rows.py builds the block
    # header from the same الجهة المنفذة / المنفذ rows. The blocks that do
    # not are chunking.md's "six actor-less blocks": ones whose actor
    # survived a page break on the section tracker rather than living in this
    # block's own row_range, and those are exactly the ones that need it
    # stated here rather than skipped. المنفذة contains المنفذ as a
    # substring, so one check covers both labels.
    if "المنفذ" not in chunk.text:
        if meta.get("actor"):
            parts.append(f"المنفذ: {meta['actor']}")
        if meta.get("unit"):
            parts.append(f"الجهة المنفذة: {meta['unit']}")

    return "، ".join(parts)


# --- none and template variants ----------------------------------------------

def _apply(chunks: list[Chunk], prefix_of) -> list[Chunk]:
    """Deep-copy the base chunks and attach one prefix function's output.

    Deep copy because none, template and llm are all built from the same
    loaded chunk list; without it, setting metadata on one variant's chunks
    would mutate the dict every other variant shares, since Chunk.metadata is
    a plain dict rather than something copy-on-write.
    """
    built = []
    for chunk in chunks:
        new_chunk = copy.deepcopy(chunk)
        prefix = prefix_of(chunk)
        new_chunk.metadata["context_prefix"] = prefix
        new_chunk.text = compose(prefix, chunk.text)
        built.append(new_chunk)
    return built


def build_none(chunks: list[Chunk]) -> list[Chunk]:
    """The baseline: 02_chunks.json with an explicit, empty context_prefix.

    Redundant as a file and worth having as one anyway: it makes the
    baseline an artifact stage 6 loads the same way as the other two rather
    than a special case in stage 6's code, and it lets gate 2 assert byte
    identity against 02_chunks.json instead of stage 6 assuming it.
    """
    return _apply(chunks, lambda c: "")


def build_template(chunks: list[Chunk],
                    titles: dict[str, str | None]) -> list[Chunk]:
    return _apply(chunks, lambda c: template_prefix(c, titles.get(c.metadata["source"])))


# --- llm variant: grouping and the document surrogate ------------------------

def _group_by_section(chunks: list[Chunk]) -> dict[tuple[str, str | None], list[Chunk]]:
    """Every chunk sharing a (source, section_path), in corpus order.

    Section rather than page: a procedure spans pages (layout.py's
    row_page_breaks exists for exactly that reason), and a page window would
    separate a block of steps from the actor row that governs them.
    """
    groups: dict[tuple[str, str | None], list[Chunk]] = collections.defaultdict(list)
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata.get("section_path"))
        groups[key].append(chunk)
    return groups


def manual_header(source: str, title: str | None,
                   all_chunks: list[Chunk]) -> str:
    """The whole-document surrogate's global half: title, version, issue
    date, and the manual's full outline in order.

    The whole document is 36,000 to 54,000 characters per manual, well over
    the 8,192 context llm.md measured the server actually enforcing, and
    llm.md also measured what happens past that line: the server keeps
    roughly half and answers fluently from it, silently. This header plus
    the target section, not the whole document, is what actually fits, and
    the outline gives the model the rest of the manual's shape without its
    text.
    """
    version_bits = []
    sample_meta = next(
        c.metadata for c in all_chunks if c.metadata["source"] == source
    )
    if sample_meta.get("doc_version"):
        version_bits.append(f"نسخة {sample_meta['doc_version']}")
    if sample_meta.get("issue_date"):
        version_bits.append(f"إصدار {sample_meta['issue_date']}")

    lines = []
    if title:
        lines.append(title)
    if version_bits:
        lines.append("، ".join(version_bits))

    paths: list[str] = []
    for chunk in all_chunks:
        if chunk.metadata["source"] == source:
            path = chunk.metadata.get("section_path")
            if path and path not in paths:
                paths.append(path)
    lines.append("أقسام الدليل: " + "؛ ".join(paths))
    return "\n".join(lines)


# Budget for header + section + target chunk + prompt scaffolding, in
# characters. 6,000 tokens at the 2.8 characters per token llm.md measured on
# this corpus with Qwen's tokenizer, leaving room under the 8,192 context for
# the completion. Measured worst case on this corpus is header 1,901 + section
# 9,614 + chunk 2,903 plus scaffolding, about 15,000 characters, under this
# budget, so the trimmer below is expected to fire zero times here. Written
# anyway, because a check that cannot fail is not a check, and llm.md's own
# lesson was paid for by trusting an untested one.
PROMPT_BUDGET_CHARS = 16_800


def _section_budget_text(header: str, members: list[Chunk],
                          scaffold_chars: int) -> tuple[str, bool]:
    """The section block sent for every chunk in this section, trimmed once
    if the section itself is too large, using the section's largest member
    so the trim covers every chunk in it.

    Computed once per section rather than once per chunk. If it were
    recomputed against each target chunk's own length, the header-plus-section
    text would differ slightly call to call within one section, which
    defeats the reason chunks are grouped by section in the first place: a
    shared prompt prefix a KV cache can reuse across consecutive calls.
    Whether that reuse actually happens is measured in LEARNING/context.md,
    not assumed here.
    """
    section_text = "\n\n".join(member.text for member in members)
    largest_chunk = max(len(member.text) for member in members)
    allowed = PROMPT_BUDGET_CHARS - len(header) - largest_chunk - scaffold_chars
    if len(section_text) <= allowed:
        return section_text, False
    return section_text[:max(allowed, 0)], True


# --- llm variant: the prompt and its validation -------------------------------

SYSTEM_PROMPT = (
    "أنت مساعد يكتب جملة سياق قصيرة جداً بالعربية، بحد أقصى 20 كلمة، لمقطع "
    "من دليل إجراءات بنكي، تُستخدم في فهرسة المقطع لغايات البحث. صف موضوع "
    "المقطع والإجراء والجهة المنفذة إن وجدت، بإيجاز شديد. لا تلخص أو تعد "
    "صياغة محتوى المقطع بالتفصيل، ولا تنسخ نص المقطع، ولا تكتب أي مقدمة أو "
    "علامات اقتباس، وأجب بالعربية فقط وبجملة واحدة قصيرة فقط."
)

SYSTEM_PROMPT_STRICT = SYSTEM_PROMPT + (
    " المحاولة السابقة لم تُقبل لأنها تجاوزت الطول المسموح أو أعادت صياغة "
    "محتوى المقطع بدل وصفه بإيجاز. اكتب هذه المرة عبارة قصيرة جداً، بحد "
    "أقصى 12 كلمة، تسمي الموضوع والجهة المنفذة فقط من غير أي تفاصيل إضافية."
)


def _build_prompt(chunk: Chunk, header: str, section_text: str,
                   *, strict: bool) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT_STRICT if strict else SYSTEM_PROMPT
    user = (
        f"معلومات الدليل:\n{header}\n\n"
        f"نص القسم:\n{section_text}\n\n"
        f"المقطع المطلوب وصفه بجملة سياق واحدة:\n{chunk.text}\n\n"
        "اكتب الجملة فقط، من غير علامات اقتباس ومن غير مقدمة."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


MIN_PREFIX_CHARS = 20
# 300 rather than the 220 first tried, and rather than the original 400.
# 400 let the model satisfy "one sentence" with a paragraph-length paraphrase
# (the first --sample 20 run). 220 fixed that but proved too tight once
# measured against the full 357-chunk run: procedure_block is the corpus's
# largest and most content-dense type, and a compliant ~20-word description
# naming an actor, a unit and several steps routinely lands a bit over 220
# without being a runaway paraphrase. The 54 length rejections in that run
# formed a clean two-part shape: 51 sat between 221 and 332 characters, close
# clustered, then a real gap to three outliers at 427-472, which is what a
# paragraph-length paraphrase actually looks like. 300 sits inside the first
# group, rescuing the compliant-but-slightly-long answers while still
# rejecting the three that were never going to be short.
MAX_PREFIX_CHARS = 300
MIN_ARABIC_RATIO = 0.5
CONTAINMENT_RATIO = 0.9
# A prefix counts as "barely shorter than what it describes" above this
# fraction of the chunk's own length. Paired with CONTAINMENT_RATIO below
# rather than replacing it: see validate_prefix's comment for why compression
# is what actually separates a lazy reformat from a genuine synthesis in a
# fixed-register administrative corpus, where token overlap alone cannot.
COMPRESSION_CEILING = 0.75

_QUOTE_CHARS = "\"'«»“”‘’"
_LABEL_PREFIX = re.compile(r"^(?:السياق|الجملة|جملة\s+السياق|الوصف)\s*[:：]\s*")
_PREAMBLE_MARKERS = (
    "بالتأكيد", "بالطبع", "إليك", "الجملة هي", "هذا النص", "here is", "sure,",
    "certainly",
)

# An Arabic letter directly against a Latin one, no space or punctuation
# between them. Found by running the tightened prompt on the real corpus:
# assets_wearhouse_p05_prose_02's reask produced "تنفahi", an Arabic stem
# fused to three stray Latin letters, invisible to every check above since
# plain ASCII letters count toward neither arabic_ratio's numerator nor
# off_script_ratio (this corpus legitimately carries English system codes,
# BPM, MTZ, FIFO). Not banning Latin script; banning it touching Arabic with
# nothing between them. Checked against the real corpus before adding this:
# 02_chunks.json has 7 such adjacencies corpus-wide, all pre-existing OCR
# word-merges chunking.md already names as an unaddressed defect class
# ("BPMالبريد", "واMEPS J"), never a legitimate compound. A generated prefix
# reproducing one of those would be no loss either; the template prefix
# behind the fallback does not have the problem.
_MIXED_SCRIPT = re.compile(r"[؀-ۿ][A-Za-z]|[A-Za-z][؀-ۿ]")


def _tidy(raw: str) -> str:
    """Strip what a model routinely wraps a one-sentence answer in, before
    validate_prefix judges whether what remains is usable."""
    text = clean_text(raw).replace("\n", " ")
    text = _LABEL_PREFIX.sub("", text)
    text = text.strip(_QUOTE_CHARS + " ")
    return text.strip()


def validate_prefix(prefix: str, chunk_text: str) -> str | None:
    """Why this prefix is unusable, or None if it passes.

    Every rule here is something a 3B model measurably does on this corpus,
    not a hypothetical: llm.md's Arabic smoke test showed the model can write
    a correct, grounded sentence, but nothing stops it from also prefacing
    that sentence, quoting the chunk back verbatim, or answering in English
    on an off-day. Bounds are starting values; LEARNING/context.md records
    the distribution that either confirms or moves them.
    """
    if not prefix:
        return "empty"
    if not (MIN_PREFIX_CHARS <= len(prefix) <= MAX_PREFIX_CHARS):
        return f"length {len(prefix)} outside [{MIN_PREFIX_CHARS}, {MAX_PREFIX_CHARS}]"
    quality = assess_quality(prefix)
    if quality["arabic_ratio"] < MIN_ARABIC_RATIO:
        return f"arabic_ratio {quality['arabic_ratio']} under {MIN_ARABIC_RATIO}"
    # Reuses quality.py's own off-script threshold rather than inventing a
    # second one. Found by running the tightened prompt on the real corpus,
    # not anticipated: a single stray CJK character costs too little of a
    # ~160-character prefix to move arabic_ratio under 0.5, but it is exactly
    # what off_script_ratio exists to catch, and quality.py already derived
    # 0.01 as the line between an intact page and a damaged one.
    if quality["off_script_ratio"] > OFF_SCRIPT_LIMIT:
        return f"off_script_ratio {quality['off_script_ratio']} over {OFF_SCRIPT_LIMIT}"
    if _MIXED_SCRIPT.search(prefix):
        return "an Arabic letter sits directly against a Latin one, mid-word"
    if prefix in chunk_text:
        return "prefix is a verbatim substring of the chunk"
    # Containment alone cannot separate a lazy reformat from a genuine
    # synthesis in a fixed-register administrative corpus: "الموافقة
    # المطلوبة من طرف الجهة الطالبة" has no synonym, so a correct, useful
    # five-theme summary of a 2,500-character procedure block measurably
    # reused 100% of its tokens from the chunk, same as a prefix that just
    # swapped the chunk's newlines for commas and added nothing. What tells
    # them apart is whether real compression happened. Rejecting only when
    # the prefix is both high-containment and barely shorter than the chunk
    # it describes was checked against every procedure_block in the corpus
    # before being written: at this ceiling it rescues the genuine summaries
    # and leaves the nearly-verbatim ones rejected, with zero of the
    # already-passing chunks newly caught.
    compression = len(prefix) / len(chunk_text)
    if compression > COMPRESSION_CEILING:
        prefix_tokens = set(prefix.split())
        chunk_tokens = set(chunk_text.split())
        if prefix_tokens:
            containment = len(prefix_tokens & chunk_tokens) / len(prefix_tokens)
            if containment > CONTAINMENT_RATIO:
                return (f"{containment:.0%} of the prefix's tokens are copied from the "
                        f"chunk, and it is {compression:.0%} of the chunk's length")
    lowered = prefix.lower()
    if any(lowered.startswith(marker) for marker in _PREAMBLE_MARKERS):
        return "opens with a conversational preamble"
    return None


def _try_generate(ledger: Ledger, chunk: Chunk, header: str, section_text: str,
                   *, strict: bool, seed: int) -> tuple[str | None, str | None]:
    """One call, tidied and validated. (None, reason) if it was unusable for
    any reason, including a truncated completion.

    A truncated completion raises in client.py rather than returning
    (llm.md's rule: a cut-off answer must never read as a finished one), and
    that raise loses its row by client.py's own accepted-cost design, since
    the prompt was evaluated but ledger.call never reaches its record. Caught
    here rather than left to crash the other 356 calls in the run; the count
    of how often this happens is part of what gets reported, not hidden.
    """
    messages = _build_prompt(chunk, header, section_text, strict=strict)
    try:
        response = ledger.call("Contextualisation", messages, GENERATOR_MODEL,
                                max_tokens=200, seed=seed)
    except LLMError as error:
        return None, f"LLMError: {error}"
    prefix = _tidy(response.text)
    reason = validate_prefix(prefix, chunk.text)
    return (prefix, None) if reason is None else (None, reason)


def generate_prefix(ledger: Ledger, chunk: Chunk, header: str,
                     section_text: str) -> tuple[str, str, str | None]:
    """One llm prefix. Returns (prefix, llm_source, reason).

    llm_source is "model", "reask" or "template_fallback", recorded in the
    chunk's metadata so a fallback chunk can be counted rather than silently
    blended into the model's numbers. reason is the final rejection reason,
    kept only when the chunk fell back, so the notes can say why rather than
    just how many.
    """
    prefix, reason = _try_generate(ledger, chunk, header, section_text,
                                    strict=False, seed=0)
    if prefix is not None:
        return prefix, "model", None

    prefix, reason2 = _try_generate(ledger, chunk, header, section_text,
                                     strict=True, seed=1)
    if prefix is not None:
        return prefix, "reask", None

    return "", "template_fallback", reason2 or reason


def pick_samples(chunks: list[Chunk], n: int) -> list[Chunk]:
    """Up to n chunks spanning every chunk type before repeating one.

    Ordered by CHUNK_TYPES rather than alphabetically, so all eight types are
    represented within the first eight picks whenever n allows it, instead of
    exhausting n on one source's types before a second source is reached.
    """
    by_bucket: dict[tuple[str, str], Chunk] = {}
    for chunk in chunks:
        key = (chunk.metadata["chunk_type"], chunk.metadata["source"])
        by_bucket.setdefault(key, chunk)

    order = sorted(by_bucket, key=lambda k: (CHUNK_TYPES.index(k[0]), k[1]))
    chosen = [by_bucket[key] for key in order[:n]]
    if len(chosen) >= n:
        return chosen[:n]

    chosen_ids = {c.metadata["chunk_id"] for c in chosen}
    for chunk in chunks:
        if len(chosen) >= n:
            break
        if chunk.metadata["chunk_id"] not in chosen_ids:
            chosen.append(chunk)
            chosen_ids.add(chunk.metadata["chunk_id"])
    return chosen


def build_llm(base: list[Chunk], titles: dict[str, str | None], ledger: Ledger,
              subset: list[Chunk] | None = None,
              ) -> tuple[list[Chunk], dict[str, int], dict[str, str]]:
    """Every chunk's llm prefix, or just subset's when sampling.

    Grouped by section so consecutive calls share a prompt prefix, with one
    re-ask and a template fallback for whatever the model cannot produce.
    Prints its own progress: the full corpus is 357 calls and, per llm.md's
    measurement, roughly 18 minutes the first time, and minutes of silence
    would read as a hang rather than a normal run.

    Returns the built chunks, the model/reask/template_fallback/
    trimmed_sections counts, and a chunk_id-to-reason map for the fallbacks,
    for run() to report.
    """
    groups = _group_by_section(base)
    scaffold_chars = len(SYSTEM_PROMPT_STRICT) + 400

    results: dict[str, tuple[str, str]] = {}
    fallback_reasons: dict[str, str] = {}
    trimmed_sections = 0
    wanted_ids = None if subset is None else {c.metadata["chunk_id"] for c in subset}
    total = len(base) if wanted_ids is None else len(wanted_ids)
    done = 0

    for key in sorted(groups, key=lambda k: (k[0], k[1] or "")):
        source = key[0]
        members = groups[key]
        members_to_run = members if wanted_ids is None else [
            c for c in members if c.metadata["chunk_id"] in wanted_ids
        ]
        if not members_to_run:
            continue

        header = manual_header(source, titles.get(source), base)
        section_text, was_trimmed = _section_budget_text(header, members, scaffold_chars)
        if was_trimmed:
            trimmed_sections += 1

        for chunk in members_to_run:
            prefix, source_tag, reason = generate_prefix(ledger, chunk, header, section_text)
            if source_tag == "template_fallback":
                prefix = template_prefix(chunk, titles.get(source))
                if reason:
                    fallback_reasons[chunk.metadata["chunk_id"]] = reason
            results[chunk.metadata["chunk_id"]] = (prefix, source_tag)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total} contextualisation calls")

    counts = collections.Counter(tag for _, tag in results.values())
    counts["trimmed_sections"] = trimmed_sections

    built = []
    for chunk in base:
        chunk_id = chunk.metadata["chunk_id"]
        if chunk_id not in results:
            continue
        prefix, source_tag = results[chunk_id]
        new_chunk = copy.deepcopy(chunk)
        new_chunk.metadata["context_prefix"] = prefix
        new_chunk.metadata["llm_source"] = source_tag
        new_chunk.text = compose(prefix, chunk.text)
        built.append(new_chunk)

    return built, dict(counts), fallback_reasons


def write_samples(chunks: list[Chunk], path: Path) -> None:
    """The llm prefixes, next to the chunk text they describe, for reading.

    The only check no gate performs: whether a prefix is *true* rather than
    merely well-formed. Written to a file rather than the console for the
    reason probe.py gives: a Windows console is cp1252 and raises on the
    first Arabic letter, and would reverse the text even if it could encode
    it.
    """
    lines = []
    for chunk in chunks:
        meta = chunk.metadata
        prefix = meta.get("context_prefix", "")
        original = decompose(prefix, chunk.text)
        lines.append(f"=== {meta['chunk_id']} ({meta.get('llm_source', 'n/a')}) ===")
        lines.append(f"prefix:  {prefix}")
        lines.append(f"chunk:   {original[:300]}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --- verification -------------------------------------------------------------

def _template_gate_reason(chunk: Chunk, title: str | None) -> str | None:
    """Gate 4's per-chunk check: names a manual, a section and a page."""
    prefix = chunk.metadata.get("context_prefix") or ""
    if not prefix:
        return "empty"
    if "None" in prefix:
        return "contains the literal string 'None'"
    if title is None:
        return "manual title could not be read from the cover"
    if title not in prefix:
        return "does not name the manual"
    section_path = chunk.metadata.get("section_path")
    if section_path and section_path not in prefix:
        return "does not carry the section path"
    if "صفحة" not in prefix and "صفحات" not in prefix:
        return "does not name a page"
    return None


def verify(base: list[Chunk], variants: dict[str, list[Chunk]],
           titles: dict[str, str | None],
           llm_counts: dict[str, int] | None = None,
           ledger: Ledger | None = None) -> list[str]:
    """Check every gate the built variants allow checking. Empty when clean."""
    failures: list[str] = []
    base_ids = [c.metadata["chunk_id"] for c in base]
    base_by_id = {c.metadata["chunk_id"]: c for c in base}

    # Gate 1: identity.
    for name, chunks in variants.items():
        ids = [c.metadata["chunk_id"] for c in chunks]
        if ids != base_ids:
            failures.append(f"gate 1: {name} does not match 02_chunks.json "
                             f"in chunk_id count or order")

    # Gate 2: the baseline is untouched.
    if "none" in variants:
        mismatched = [
            c.metadata["chunk_id"] for c in variants["none"]
            if c.text != base_by_id[c.metadata["chunk_id"]].text
            or c.metadata.get("context_prefix") != ""
        ]
        if mismatched:
            failures.append(f"gate 2: {len(mismatched)} none-variant chunks "
                             f"are not byte-identical to 02_chunks.json "
                             f"({mismatched[:3]})")

    # Gate 3: every prefixed chunk recovers its original text.
    for name, chunks in variants.items():
        if name == "none":
            continue
        broken = []
        for chunk in chunks:
            original = base_by_id.get(chunk.metadata["chunk_id"])
            if original is None:
                continue
            try:
                recovered = decompose(chunk.metadata.get("context_prefix", ""), chunk.text)
            except ValueError:
                broken.append(chunk.metadata["chunk_id"])
                continue
            if recovered != original.text:
                broken.append(chunk.metadata["chunk_id"])
        if broken:
            failures.append(f"gate 3: {name} does not recover the original "
                             f"text for {len(broken)} chunks ({broken[:3]})")

    # Gate 4: template completeness.
    if "template" in variants:
        bad = [
            (c.metadata["chunk_id"], _template_gate_reason(c, titles.get(c.metadata["source"])))
            for c in variants["template"]
        ]
        bad = [entry for entry in bad if entry[1] is not None]
        if bad:
            failures.append(f"gate 4: {len(bad)} template prefixes incomplete "
                             f"or malformed ({bad[:3]})")

    # Gate 5: llm validity, and the fallback rate is not quietly high.
    if "llm" in variants and llm_counts is not None:
        total = sum(v for k, v in llm_counts.items() if k != "trimmed_sections")
        fallback = llm_counts.get("template_fallback", 0)
        if total and fallback / total > 0.10:
            failures.append(f"gate 5: {fallback} of {total} llm prefixes fell "
                             f"back to the template ({fallback / total:.0%}, "
                             f"over the 10% limit)")
        empty = [c.metadata["chunk_id"] for c in variants["llm"]
                  if not c.metadata.get("context_prefix")]
        if empty:
            failures.append(f"gate 5: {len(empty)} llm prefixes are empty "
                             f"({empty[:3]})")

    # Gate 6: no Contextualisation prompt was silently truncated. Uses the
    # server's own reported token count, not our character estimate, which
    # is the whole lesson of llm.md's context gate.
    if ledger is not None:
        contextual = [e for e in ledger.entries if e.step == "Contextualisation"]
        if contextual:
            peak = max(e.prompt_tokens for e in contextual)
            if peak >= LLM_CONTEXT * 0.9:
                failures.append(f"gate 6: a Contextualisation prompt reached "
                                 f"{peak} tokens against the {LLM_CONTEXT} "
                                 f"configured context")

    return failures


def oversized_prefixes(base_by_id: dict[str, Chunk],
                        chunks: list[Chunk]) -> list[str]:
    """Chunk ids whose prefix outweighs the chunk it describes.

    Measured rather than gated, the same choice quality.py makes for
    fragment_ratio, and for a reason that only became clear by running this
    over the real corpus rather than a handful of prose chunks: it is not
    mainly short chunks meeting a roughly fixed prefix cost. It is the full
    section_path, which by design carries the whole procedure title rather
    than just its last segment, meeting a genuinely short trailing block, a
    procedure's last single step, most often in Central Mail and Central
    Alarm, where a title can itself run past a hundred characters with a
    quoted sub-label inside it (``"البريد الأردني"``).
    Measured on the template variant: 43 of 357 chunks (12%), 35 of them
    procedure_block, 4 accounting_entry, 3 prose, 1 reference; every one of
    the 267 procedure_block hits comes from a block whose own text is a
    single short step. Both halves of that combination are deliberate:
    template_prefix carries the full path because the leaf alone loses what
    kind of content this is, and rows.py keeps a lone trailing step as its
    own block rather than merging it into its neighbour. Shortening the path
    or forcing a bigger block would be answering stage 6's question, whether
    the prefix helps retrieval, before it has been asked, so this stays a
    measurement rather than a rule.
    """
    return [
        c.metadata["chunk_id"] for c in chunks
        if len(c.metadata.get("context_prefix") or "")
           > len(base_by_id[c.metadata["chunk_id"]].text)
    ]


# --- entry points ---------------------------------------------------------

def run(variants: tuple[str, ...] = CONTEXT_VARIANTS,
        chunks_path: Path = CHUNKS_OUTPUT,
        layout_path: Path = LAYOUT_OUTPUT) -> int:
    """Build the requested context-prefix variants and report on the result."""
    if not chunks_path.exists():
        print(f"{chunks_path} not found. Run `python cli.py chunk` first.")
        return 0
    if not layout_path.exists():
        print(f"{layout_path} not found. Run `python cli.py layout` first.")
        return 0

    print(f"Building context prefixes: {', '.join(variants)}")
    started = time.time()

    base = load_chunks(chunks_path)
    documents = load_documents(layout_path)
    titles = manual_titles(documents)
    for source, title in titles.items():
        if title is None:
            print(f"  cover title unreadable for {source}")

    built: dict[str, list[Chunk]] = {}
    llm_counts: dict[str, int] | None = None
    fallback_reasons: dict[str, str] = {}
    ledger: Ledger | None = None

    if "none" in variants:
        built["none"] = build_none(base)
    if "template" in variants:
        built["template"] = build_template(base, titles)
    if "llm" in variants:
        ledger = Ledger(label="context-build")
        built["llm"], llm_counts, fallback_reasons = build_llm(base, titles, ledger)

    print()
    for name, chunks in built.items():
        save_chunks(chunks, CONTEXT_OUTPUTS[name])
        nonzero = sorted(len(c.metadata["context_prefix"]) for c in chunks
                          if c.metadata.get("context_prefix"))
        if nonzero:
            print(f"  {name:<10} {len(chunks)} chunks, prefix chars: "
                  f"median {nonzero[len(nonzero) // 2]}, max {nonzero[-1]}, "
                  f"written to {CONTEXT_OUTPUTS[name].name}")
        else:
            print(f"  {name:<10} {len(chunks)} chunks, no prefix, "
                  f"written to {CONTEXT_OUTPUTS[name].name}")

    if "llm" in built:
        print(f"\nllm route: {dict(llm_counts)}")
        if fallback_reasons:
            preview = list(fallback_reasons.items())[:5]
            print(f"  fallback reasons (first {len(preview)} of "
                  f"{len(fallback_reasons)}): {preview}")
        sample_ids = {c.metadata["chunk_id"] for c in pick_samples(base, 20)}
        write_samples(
            [c for c in built["llm"] if c.metadata["chunk_id"] in sample_ids],
            CONTEXT_SAMPLES_OUTPUT,
        )
        print(f"  sample transcript written to {CONTEXT_SAMPLES_OUTPUT}")
        print(f"\n{ledger.render()}")

    print(f"\nelapsed {time.time() - started:.1f}s")

    # Measured, not gated: see oversized_prefixes's docstring for why a
    # handful of the shortest chunks in the corpus legitimately fall under
    # their own prefix's length, and why a cutoff here would be invented
    # rather than derived.
    base_by_id = {c.metadata["chunk_id"]: c for c in base}
    print("\nPrefix against chunk length")
    for name, chunks in built.items():
        oversized = oversized_prefixes(base_by_id, chunks)
        if oversized:
            print(f"  {name:<10} {len(oversized)} of {len(chunks)} prefixes "
                  f"outweigh the chunk they describe ({oversized[:3]})")
        else:
            print(f"  {name:<10} no prefix outweighs its chunk")

    print("\nGates")
    failures = verify(base, built, titles, llm_counts, ledger)
    if ledger is not None:
        failures.extend(f"ledger: {f}" for f in ledger.verify())

    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1

    print("  ok  every built variant matches 02_chunks.json in count and order")
    if "none" in built:
        print("  ok  the none variant is byte-identical to 02_chunks.json")
    print("  ok  every prefixed chunk recovers its original text exactly")
    if "template" in built:
        print("  ok  every template prefix names a manual, a section and a page")
    if "llm" in built:
        print("  ok  llm fallbacks stay under the 10% limit")
        print("  ok  no Contextualisation prompt reached the context limit")
    return 0


def run_sample(n: int, chunks_path: Path = CHUNKS_OUTPUT,
               layout_path: Path = LAYOUT_OUTPUT,
               samples_path: Path = CONTEXT_SAMPLES_OUTPUT) -> int:
    """Generate the llm prefix for a stratified sample only, and write it for
    reading, without building or gating the full llm variant.

    The full run is roughly 18 minutes. This sends the same prompt to the
    same model over a handful of chunks chosen to span every chunk type, so
    the Arabic can be read and the prompt fixed before that is spent.
    """
    if not chunks_path.exists() or not layout_path.exists():
        print("Both chunk and layout output are required; run "
              "`python cli.py chunk` first.")
        return 0

    base = load_chunks(chunks_path)
    documents = load_documents(layout_path)
    titles = manual_titles(documents)
    sample_chunks = pick_samples(base, n)

    ledger = Ledger(label="context-sample")
    built, counts, fallback_reasons = build_llm(base, titles, ledger, subset=sample_chunks)

    write_samples(built, samples_path)
    print(f"\n{len(built)} sample prefixes written to {samples_path}")
    print(f"source split: {counts}")
    if fallback_reasons:
        print(f"fallback reasons: {fallback_reasons}")
    print(f"\n{ledger.render()}")
    return 0


if __name__ == "__main__":
    run()
