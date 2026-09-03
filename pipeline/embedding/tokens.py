"""Real token counts, before a single vector is computed.

chunking.md left this as its first open question for embedding: chunk size has
been in characters since stage 2, standing in at roughly 3 to 4 characters per
Arabic token because no tokenizer was installed. BGE-M3 arrives now with its
own tokenizer, and so does e5-large, with a much smaller one, 512 tokens
against 8,192. This module answers whether that gap matters on this corpus
before the bake-off spends anything on it.

It downloads nothing but tokenizer files and a config, a few tens of
megabytes rather than the multi-gigabyte weights embedder.py needs, so it can
run first and answer its question cheaply.

What this module does not do: it does not embed, and it does not decide
anything. It measures, and it names every chunk that would be silently
truncated, by id, so that fact is visible before it becomes a quiet gap in a
Recall@10 number three modules from now.
"""

from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from transformers import AutoConfig, AutoTokenizer

_T = TypeVar("_T")

from ..chunking.chunk import Chunk, load_chunks
from ..config import CONTEXT_OUTPUTS, EMBED_MODELS, TOKEN_CENSUS_OUTPUT

# Special tokens a sequence classification model adds around the content:
# CLS at the start, SEP at the end. Both candidates here add exactly two, but
# it is read from the tokenizer rather than assumed, since a wrong constant
# here would just move the truncation boundary by one token and never raise.
SPECIAL_TOKEN_OVERHEAD = 2

_tokenizer_cache: dict[str, Any] = {}
_max_length_cache: dict[str, int] = {}


def load_with_retry(
    load_fn: Callable[[], _T], label: str,
    attempts: int = 5, delay_s: float = 4.0,
) -> _T:
    """Retry a HuggingFace load through a transient MemoryError.

    Found during stage 8, not stage 6: loading BGE-M3's ~17 MB
    tokenizer.json inside a process that already has torch and
    qdrant_client imported intermittently raised MemoryError with
    several GB of free physical and virtual memory still reported at
    the OS level. Isolated directly, not guessed at, across several
    throwaway probes: the cached tokenizer.json read and sha256-verified
    correctly outside this process; the identical
    open(path, encoding="utf-8") plus json.load() call transformers
    itself makes succeeded instantly in a fresh interpreter; and the
    failure reproduces with or without a QdrantClient ever opened, so
    it is not specifically about this project's store. What is left is
    allocator fragmentation somewhere in this specific process, and it
    is genuinely intermittent: the exact same call has both failed and
    succeeded from one fresh process to the next with no controlled
    variable changed, tracked in LEARNING/router.md rather than solved
    here.

    Honest limit of this mitigation: gc.collect() only frees Python-level
    cyclic garbage, and does not defragment torch's or tokenizers' own
    native allocators, so an in-process retry sometimes rides out a
    short blip and sometimes does not, observed both ways while building
    this. A fresh process has reliably succeeded every time this was
    manually retried, which this function cannot offer without
    subprocess isolation per query, the cost QUERY_DEVICE = "cpu" was
    chosen to avoid. When every attempt here still fails, the honest
    next step is what embedding/run.py's own _SUBPROCESS_RETRIES message
    already models: report it as real, and rerun the command.
    """
    last_error: MemoryError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return load_fn()
        except MemoryError as error:
            last_error = error
            gc.collect()
            if attempt < attempts:
                time.sleep(delay_s)
    raise MemoryError(
        f"{label} failed after {attempts} attempts under retry, most "
        f"recently: {last_error}. See load_with_retry's own docstring "
        f"in tokens.py for what this was measured to be and not be."
    ) from last_error


def get_tokenizer(model_key: str):
    """Load and cache one model's tokenizer, by its short name in EMBED_MODELS."""
    if model_key not in _tokenizer_cache:
        model_id = EMBED_MODELS[model_key]
        _tokenizer_cache[model_key] = load_with_retry(
            lambda: AutoTokenizer.from_pretrained(model_id),
            f"tokenizer for {model_key}",
        )
    return _tokenizer_cache[model_key]


def max_content_length(model_key: str) -> int:
    """The safe *total* sequence length, content and special tokens
    combined, read from the model's own config rather than hardcoded.

    Not a content-only budget, despite the name. RoBERTa-family models,
    both candidates here, offset position ids by padding_idx before
    indexing the position embedding table, so a full max_position_embeddings
    sequence overflows it by SPECIAL_TOKEN_OVERHEAD: this subtracts that
    overhead once, here, so every caller reads the number that is actually
    safe to hand a tokenizer's max_length rather than the raw config value.
    embedder.py's truncation and this module's own census both read it from
    here so neither can silently disagree with the other about the limit.

    Confirmed the hard way: reproduced against e5-large as a real CUDA
    assertion, "vectorized gather kernel index out of bounds", by tokenizing
    to config.max_position_embeddings directly rather than subtracting this
    overhead first. BGE-M3's 8,192 cap never came close to being tested by
    this corpus, which is why the same bug sat unnoticed there.
    """
    if model_key not in _max_length_cache:
        config = AutoConfig.from_pretrained(EMBED_MODELS[model_key])
        _max_length_cache[model_key] = (
            config.max_position_embeddings - SPECIAL_TOKEN_OVERHEAD
        )
    return _max_length_cache[model_key]


def max_pure_content_tokens(model_key: str) -> int:
    """How many content tokens fit once CLS and SEP are set aside.

    max_content_length already is the total the tokenizer may safely
    produce; count_tokens below measures content alone, so the two are not
    directly comparable without subtracting SPECIAL_TOKEN_OVERHEAD once
    more here. This is the number a content-only count has to stay under.
    """
    return max_content_length(model_key) - SPECIAL_TOKEN_OVERHEAD


def count_tokens(text: str, model_key: str) -> int:
    """Real content-token count for one string under one model's tokenizer.

    add_special_tokens=False: this counts what the text itself costs, not
    the padded sequence length, so it is compared against
    max_pure_content_tokens rather than against max_content_length, which
    already reserves room for CLS and SEP.
    """
    tokenizer = get_tokenizer(model_key)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def census_one_variant(chunks: list[Chunk], model_key: str) -> list[dict[str, Any]]:
    """Token count, char count and overflow flag for every chunk, one model."""
    limit = max_pure_content_tokens(model_key)
    rows = []
    for chunk in chunks:
        tokens = count_tokens(chunk.text, model_key)
        rows.append({
            "chunk_id": chunk.metadata["chunk_id"],
            "chars": len(chunk.text),
            "tokens": tokens,
            "over_limit": tokens > limit,
        })
    return rows


def _stats_line(label: str, values: list[int]) -> str:
    if not values:
        return f"  {label}: no data"
    values_sorted = sorted(values)
    p90 = values_sorted[int(0.9 * (len(values_sorted) - 1))]
    return (
        f"  {label}: median {int(statistics.median(values))}, "
        f"p90 {p90}, max {max(values)}"
    )


def render_report(
    per_variant: dict[str, dict[str, list[dict[str, Any]]]],
) -> str:
    """The census as text for an editor, following probe.py's Windows-console
    reasoning: this corpus's chunk ids are ASCII, but the summary sits beside
    other stage-6 output that will not be, so the whole family goes to a file.
    """
    lines = ["Token census", "=" * 60, ""]

    for variant, by_model in per_variant.items():
        lines.append(f"variant: {variant}")
        for model_key, rows in by_model.items():
            limit = max_pure_content_tokens(model_key)
            tokens = [r["tokens"] for r in rows]
            chars = [r["chars"] for r in rows]
            ratios = [c / t for c, t in zip(chars, tokens) if t]
            over = [r for r in rows if r["over_limit"]]

            lines.append(f"  -- {model_key} (limit {limit} content tokens) --")
            lines.append(_stats_line("tokens", tokens))
            lines.append(_stats_line("chars", chars))
            lines.append(
                f"  chars per token: median {statistics.median(ratios):.2f}, "
                f"mean {statistics.mean(ratios):.2f} "
                f"(stage 2 assumed 3 to 4)"
            )
            lines.append(f"  over limit: {len(over)} of {len(rows)}")
            for row in over:
                lines.append(
                    f"    {row['chunk_id']}: {row['tokens']} tokens, "
                    f"{row['chars']} chars"
                )
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def run(
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
    output_path: Path = TOKEN_CENSUS_OUTPUT,
) -> bool:
    """Tokenize every chunk in every variant against both models, and report.

    True on completion; this stage measures rather than gates, so it fails
    only when an input file is missing, the same shape ingestion's own
    measurement-only stages take.
    """
    missing = [p for p in variant_paths.values() if not p.exists()]
    if missing:
        print(f"Missing {[str(p) for p in missing]}. Run `python cli.py context` first.")
        return False

    per_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant, path in variant_paths.items():
        chunks = load_chunks(path)
        per_variant[variant] = {
            model_key: census_one_variant(chunks, model_key)
            for model_key in EMBED_MODELS
        }
        print(f"{variant}: {len(chunks)} chunks tokenized against "
              f"{len(EMBED_MODELS)} models")

    report = render_report(per_variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nwritten to {output_path}")
    return True


if __name__ == "__main__":
    run()
