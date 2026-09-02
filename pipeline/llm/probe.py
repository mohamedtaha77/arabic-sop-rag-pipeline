"""Does any of this work on this machine.

The other four modules in this package are the contract. This one is the
measurement, and it exists because the plan's model sizing was decided against
a 4 GB card and 11.8 GB of system RAM rather than against a benchmark. Whether
a 3B model is fast enough to carry ten questions through eight techniques, and
whether its Arabic is worth reading, are questions no amount of design settles.

Four things get measured rather than assumed:

* **Cold against warm latency.** The first call includes loading the model off
  disk and onto a card that cannot hold all of it. Reporting only the first
  call would blame the model for the load; reporting only the second would
  hide a load that happens again every time the server evicts it.
* **Long-prompt throughput**, built from real chunks rather than filler. It is
  the prompt shape stage 9 will actually send, and short-prompt speed on a
  partially offloaded model says almost nothing about it.
* **Effective context.** Ollama truncates an over-long prompt in silence, so
  this sends one deliberately too big and reads back how many tokens the
  server says it evaluated. That number is the context, whatever config.py
  hoped for.
* **Arabic, by reading it.** No automated check tells you whether a 3B model
  writes coherent Arabic. This prints the output and leaves the judgement to a
  person, which is the same ground-truth method that produced ingestion's 44%
  against 97%.

The speed measurements deliberately bypass the cache. A benchmark that
silently measured a cache hit would report this machine as faster than any
machine has ever been.
"""

from __future__ import annotations

from pathlib import Path

from ..chunking.chunk import Chunk, load_chunks
from ..config import (
    CHUNKS_OUTPUT,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    LLM_BASE_URL,
    LLM_CONTEXT,
)
from .client import LLMError, list_models
from .ledger import Ledger

# Starting guess at how many characters of Arabic make one token, used only to
# size the calibration call below. The real ratio is then measured against the
# server and everything after that uses the measurement. chunking.md records
# the same 3-4 guess, and records that a tokenizer should replace it.
CHARS_PER_TOKEN_GUESS = 3

# Characters for the calibration call. Small enough to fit any plausible
# context, so it measures tokenisation rather than truncation.
CALIBRATION_CHARS = 7500

# Appended to measurement prompts. Their completions are never read, but the
# client refuses a truncated one, so they have to end quickly on their own.
BREVITY = "\n\nReply with the single word: ok."


def _corpus_text(chunks: list[Chunk]) -> str:
    """The whole chunked corpus as one string, for building sized prompts."""
    return "\n\n".join(chunk.text for chunk in chunks)


def run(source: Path = CHUNKS_OUTPUT) -> int:
    """Measure the local model layer and print what it found."""
    print("Local model probe")
    print(f"endpoint {LLM_BASE_URL}\n")

    try:
        available = list_models()
    except LLMError as error:
        print(f"  FAIL  {error}")
        return 1

    print("Models")
    missing = []
    for role, tag in (("generator", GENERATOR_MODEL), ("judge", JUDGE_MODEL)):
        if tag in available:
            print(f"  ok    {role:<10} {tag}")
        else:
            missing.append(tag)
            print(f"  FAIL  {role:<10} {tag} not pulled")
    if missing:
        print(f"\n  {len(available)} models present: {', '.join(available) or 'none'}")
        for tag in missing:
            print(f"  run `ollama pull {tag}`")
        return 1

    if GENERATOR_MODEL == JUDGE_MODEL:
        print("  FAIL  generator and judge are the same model; a judge that "
              "shares weights with the generator grades its own work")
        return 1
    print("  ok    generator and judge are different models\n")

    ledger = Ledger(label="probe")
    short = [{"role": "user", "content": "Reply with the single word: ready."}]

    # --- cold against warm --------------------------------------------------

    print("Speed")
    # Roomy rather than tight. The client raises on a completion cut off at
    # max_tokens, which is right for the pipeline and would otherwise make this
    # probe fail whenever a 3B model answered a one-word question with a
    # sentence. The deliberate truncation test below covers that guard instead.
    cold = ledger.call("Router", short, GENERATOR_MODEL,
                       max_tokens=64, use_cache=False)
    warm = ledger.call("Router", short, GENERATOR_MODEL,
                       max_tokens=64, seed=1, use_cache=False)
    print(f"  cold  {cold.latency_s:>6.1f}s  {cold.prompt_tokens:>5} in "
          f"{cold.completion_tokens:>4} out   (includes loading the model)")
    print(f"  warm  {warm.latency_s:>6.1f}s  {warm.prompt_tokens:>5} in "
          f"{warm.completion_tokens:>4} out")
    load_cost = cold.latency_s - warm.latency_s
    if load_cost < 0.5:
        print(f"  load  {load_cost:>6.1f}s  the model was already resident, so "
              f"the cold figure above is not a cold start")
    else:
        print(f"  load  {load_cost:>6.1f}s  paid again whenever the server evicts it")

    # --- long prompt, from real chunks --------------------------------------

    if not source.exists():
        print(f"\n  {source.name} not found; skipping the long-prompt and "
              f"Arabic checks. Run `python cli.py chunk` first.")
        return 1

    chunks = load_chunks(source)
    corpus = _corpus_text(chunks)

    # --- calibrate characters per token, then use the measurement -----------

    # Sizing a context test from a guessed ratio is how the first version of
    # this probe reported a wrong diagnosis. Measure the ratio first, on a
    # prompt small enough that nothing can truncate it.
    # The trailing instruction keeps the answer short enough that the
    # truncation guard is not tripped by a measurement call, whose completion
    # nobody reads. The ratio is computed from the exact string sent.
    calibration_text = corpus[:CALIBRATION_CHARS] + BREVITY
    calibration = ledger.call(
        "Compression",
        [{"role": "user", "content": calibration_text}],
        GENERATOR_MODEL, max_tokens=64, use_cache=False,
    )
    chars_per_token = len(calibration_text) / calibration.prompt_tokens
    print(f"  calib {calibration.latency_s:>6.1f}s  {calibration.prompt_tokens:>5} in "
          f"{calibration.completion_tokens:>4} out")

    # A prompt just under the configured context. If the server really serves
    # LLM_CONTEXT this is evaluated whole; if it serves less, it is truncated
    # and the gate below catches it.
    wanted = int(LLM_CONTEXT * 0.9)
    long_text = corpus[:int(wanted * chars_per_token)] + BREVITY
    long_call = ledger.call(
        "Final generation",
        [{"role": "user", "content": long_text}],
        GENERATOR_MODEL, max_tokens=64, use_cache=False,
    )
    print(f"  long  {long_call.latency_s:>6.1f}s  {long_call.prompt_tokens:>5} in "
          f"{long_call.completion_tokens:>4} out")
    # End to end, not generation speed: most of this is the prompt being read,
    # not tokens being written. It is still the number that decides whether ten
    # questions through eight techniques fit in an evening, which is the
    # decision this probe exists to inform.
    print(f"        {long_call.prompt_tokens / long_call.latency_s:>6.0f} prompt "
          f"tokens/s end to end, against {warm.prompt_tokens / warm.latency_s:.0f} "
          f"on the short prompt")

    # --- effective context --------------------------------------------------

    print("\nContext")
    print(f"  {chars_per_token:.1f} characters per token, measured on this "
          f"corpus with this tokenizer")
    print(f"  asked for {wanted} tokens ({len(long_text):,} characters), "
          f"server evaluated {long_call.prompt_tokens}")
    context_ok = long_call.prompt_tokens >= wanted * 0.9
    if context_ok:
        print(f"  ok    context serves at least {long_call.prompt_tokens} "
              f"tokens, against the {LLM_CONTEXT} config.py asks for")
    else:
        print(f"  FAIL  truncated at {long_call.prompt_tokens} tokens, under "
              f"the {LLM_CONTEXT} config.py asks for. Set "
              f"OLLAMA_CONTEXT_LENGTH={LLM_CONTEXT} and restart the server.")

    # --- what overflow actually does ----------------------------------------

    # Worth measuring rather than assuming, because it is the failure stage 9
    # would otherwise meet in production. Going over the context does not
    # raise: the server keeps roughly half and answers fluently from it.
    overflow = ledger.call(
        "CRAG evaluator",
        [{"role": "user",
          "content": corpus[:int(LLM_CONTEXT * 3 * chars_per_token)] + BREVITY}],
        GENERATOR_MODEL, max_tokens=64, use_cache=False,
    )
    print(f"  overflow: sent ~{LLM_CONTEXT * 3} tokens, server evaluated "
          f"{overflow.prompt_tokens}")
    print(f"  a prompt over context is silently cut to about half of it, not "
          f"refused.")
    print(f"  stage 9 has to budget its own prompt; the server will not say no.")

    # --- Arabic, printed to be read -----------------------------------------

    print("\nArabic smoke test")
    with_actor = next(
        (c for c in chunks if c.metadata.get("actor") and len(c.text) > 300),
        chunks[0],
    )
    question = [
        {"role": "system",
         "content": "أجب بالعربية فقط، وبإيجاز، اعتماداً على النص المرفق وحده."},
        {"role": "user",
         "content": f"النص:\n{with_actor.text}\n\nالسؤال: ما موضوع هذا النص؟"},
    ]
    # Cached, unlike the measurements above: this one is here to be read, and a
    # second run should demonstrate the cache rather than re-spend the tokens.
    answer = ledger.call("Presenter", question, GENERATOR_MODEL, max_tokens=400)

    # Written to a file rather than trusted to the console. A Windows console
    # is cp1252 by default and raises UnicodeEncodeError on the first Arabic
    # letter, which is a probe that fails for a reason having nothing to do
    # with the model. The file is also the only way to read the answer without
    # the right-to-left mangling ingestion.md already warns about.
    transcript = source.parent / "03_llm_probe_arabic.txt"
    transcript.write_text(
        f"chunk: {with_actor.metadata['chunk_id']}\n"
        f"actor: {with_actor.metadata.get('actor') or with_actor.metadata.get('unit')}\n"
        f"cached: {answer.cached}\n\n"
        f"--- chunk sent ---\n{with_actor.text}\n\n"
        f"--- model answer ---\n{answer.text.strip()}\n",
        encoding="utf-8",
    )
    print(f"  chunk  {with_actor.metadata['chunk_id']}")
    print(f"  served from cache: {answer.cached}")
    print(f"  {answer.completion_tokens} tokens of Arabic written to {transcript}")
    print("  read it there: this console cannot encode Arabic, "
          "and would reverse it if it could")

    # --- the truncation guard, tested by tripping it ------------------------

    # The one property here that cannot be confirmed by a call succeeding. A
    # completion cut off at max_tokens reads like a finished answer, so the
    # only honest check is to cause one deliberately and confirm it raises
    # rather than returning the half of it that looks fine.
    print("\nTruncation guard")
    truncation_ok = False
    try:
        ledger.call("Rewriter",
                    [{"role": "user", "content": "Count from one to twenty."}],
                    GENERATOR_MODEL, max_tokens=1, use_cache=False)
        print("  FAIL  a completion capped at 1 token came back as an answer")
    except LLMError as error:
        truncation_ok = True
        print(f"  ok    refused a truncated completion: {str(error)[:72]}...")

    # --- the CPU rows, which have no model yet -------------------------------

    # Reranking and Grounding guard are cross-encoder and NLI models that
    # arrive at stages 7 and 9. Their rows above therefore read No, correctly.
    # Recording an invented latency against them to make the table look
    # complete would put a fabricated number in a measurement, so the mechanism
    # is shown on a throwaway ledger instead and the real one stays honest.
    print("\nSchema check, not a measurement")
    demo = Ledger(label="schema check")
    demo.record_local("Reranking", 1.8, "stand-in for stage 7's cross-encoder")
    demo_row = next(r for r in demo.rows() if r["step"] == "Reranking")
    print(f"  record_local puts a CPU step in the table: {demo_row['calls']} call, "
          f"{demo_row['prompt_tokens']} tokens, {demo_row['latency_s']:.1f}s, "
          f"${demo_row['cost_usd']:.2f}")

    # --- the table ----------------------------------------------------------

    # Last, because it is the deliverable the other sections exist to fill in.
    # The Rewriter row reads No: its only call was the truncation test above,
    # which raised before it could be recorded, for the reason client.py gives.
    print("\nSection 7 table")
    print(ledger.render())

    print("\nGates")
    failures = ledger.verify()
    if not context_ok:
        failures.append(f"context truncates at {long_call.prompt_tokens} tokens")
    if not truncation_ok:
        failures.append("a truncated completion was returned instead of raising")
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1

    print("  ok  every call recorded tokens, latency and a notional cost")
    print("  ok  the table emits a row per declared step, run or not")
    print("  ok  the context the server serves matches what config.py asks for")
    print("  read the Arabic above before trusting this model for synthesis")
    return 0


if __name__ == "__main__":
    run()
