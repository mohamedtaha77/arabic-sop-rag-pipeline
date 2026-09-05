"""Project-wide paths, extraction settings and local model configuration."""

from __future__ import annotations

import os

# Set before transformers or huggingface_hub is ever imported anywhere in
# the process, which this achieves by being the first thing this module
# does and this module being the first import of nearly every entry
# point. Found during stage 8: transformers spawns a background thread on
# a HuggingFace model load to check the Hub API for a safetensors
# conversion PR, a live network call, racing against the main thread's
# own model construction from the very same checkpoint. faulthandler
# caught it directly: one thread blocked in
# safetensors_conversion.auto_conversion's HTTP request, the other inside
# torch's own nn.Linear construction, at the moment of a real crash. This
# is a genuine race condition, not the memory pressure it first looked
# like, and it is also pointless: every model this project uses is
# already downloaded and cached locally, the corpus this pipeline serves
# is internal-use bank material that runs entirely offline by design (see
# the README), and a query-time embed has no business reaching the
# network at all. Offline mode removes the thread and the race together;
# repeated loads after this was set never reproduced the crash again,
# against a machine that had reproduced it repeatedly before.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"

TEXTLAYER_OUTPUT = PROCESSED_DIR / "01_documents_textlayer.json"
OCR_OUTPUT = PROCESSED_DIR / "01_documents_ocr.json"
LAYOUT_OUTPUT = PROCESSED_DIR / "01_documents_layout.json"

CHUNKS_OUTPUT = PROCESSED_DIR / "02_chunks.json"

# Stage 4's three chunk sets, one per context-prefix strategy. The comparison
# needs all three: without the no-prefix baseline it can say which prefix wins
# but not whether prefixing helps at all.
#
# Three files rather than one keyed artifact, so each is a plain chunk list
# that chunk.load_chunks reads unchanged and stage 6 loops over the three paths
# uniformly. The baseline is a copy of CHUNKS_OUTPUT and is written anyway: a
# baseline stage 6 has to reach for differently is one that can drift from the
# other two without anything noticing.
CONTEXT_VARIANTS = ("none", "template", "llm")

CONTEXT_OUTPUTS = {
    name: PROCESSED_DIR / f"03_chunks_{name}.json" for name in CONTEXT_VARIANTS
}

# Prefixes written out for reading by eye. The only check on whether a
# generated prefix is true rather than merely well-formed, and a file rather
# than console output for the reason probe.py gives: a Windows console is
# cp1252 and would raise on the first Arabic letter, then reverse the rest.
CONTEXT_SAMPLES_OUTPUT = PROCESSED_DIR / "03_context_samples.txt"

# Page render resolution for OCR. 72 DPI (the PDF default) loses the dots that
# distinguish ب ت ث ن; 600 DPI quadruples runtime for no measured accuracy gain.
RENDER_DPI = 300

# EasyOCR language codes. Arabic first: it drives the recognition model and the
# corpus is ~85% Arabic. English is needed for embedded system names and codes.
OCR_LANGUAGES = ["ar", "en"]


# --- local model layer ------------------------------------------------------

LLM_CACHE_DIR = DATA_DIR / "llm_cache"

# Ollama's OpenAI-compatible endpoint, kept as a base URL rather than a full
# path. llama.cpp's server speaks the same dialect, so the documented fallback
# is a one-line change here rather than a rewrite of the client.
#
# 127.0.0.1, never "localhost", and this is worth about two seconds on every
# single LLM call. Measured on this machine, same model resident on the GPU,
# same prompt: via "localhost" a short call took 2.34 s wall while Ollama's own
# accounting reported 0.13 s of actual work; via 127.0.0.1 the same call takes
# 0.06 s. Windows resolves "localhost" to IPv6 ::1 first, Ollama listens on
# IPv4, and the stalled attempt has to fail before the fallback succeeds. At
# six calls for one advanced_rag question that alone was around 13 seconds of
# doing nothing.
LLM_BASE_URL = "http://127.0.0.1:11434/v1"

# Whether to tear every model down between steps.
#
# When True, techniques.rerank.apply() shuts the embedder worker down, evicts
# Ollama's resident model and kills its own worker after scoring, so only one
# heavy model is ever in memory at once. That is what stops this machine
# segfaulting, and it is also the single largest cost in the pipeline:
# reranking measured 29.0 s wall for a question whose LLM calls were all served
# from cache, nearly all of it reloading models that were just discarded (a
# cold Ollama reload alone measured 7.6 s).
#
# When False, everything stays resident and reranking drops to roughly the
# 1-2 s the scoring itself takes. That is only safe with real memory to spare:
# BGE-M3 and the reranker want about 2.2 GB each, and loading either with under
# ~2 GB free has segfaulted here repeatedly and reproducibly.
#
# So the default is measured rather than assumed, and re-measured on every run
# because what else is open on a laptop changes hour to hour. Override with
# GBG_LOW_MEMORY=0 (stay resident) or =1 (always tear down) when you want to
# decide for yourself.
LOW_MEMORY_HEADROOM_GB = 6.0

def _detect_low_memory() -> bool:
    forced = os.environ.get("GBG_LOW_MEMORY")
    if forced is not None:
        return forced.strip() not in {"0", "false", "False", ""}
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9 < LOW_MEMORY_HEADROOM_GB
    except Exception:
        # Never let a memory probe decide the pipeline cannot run: the
        # conservative answer is the one that has always worked here.
        return True

LOW_MEMORY_MODE = _detect_low_memory()

# Seconds. Generous on purpose: a 4 GB card holds only part of even a 3B model,
# so the remaining layers run on CPU and a long-context generation is measured
# in minutes. A timeout tight enough to feel responsive would fire on every
# real synthesiser call and look like a bug in the client.
LLM_TIMEOUT = 300

# Two models, from two different families. A judge sharing weights with the
# generator grades its own work generously; a judge sharing only a
# quantisation still shares every bias worth catching.
#
# 3B rather than the 7B the plan first assumed, because this machine measured
# 11.8 GB of system RAM beside the 4 GB card. Promotion to 7B for synthesis is
# a decision for the probe's numbers, not for this file.
GENERATOR_MODEL = "qwen2.5:3b-instruct-q4_K_M"
JUDGE_MODEL = "llama3.2:3b-instruct-q4_K_M"

# --- optional cloud provider (Groq), off unless explicitly asked for -------
#
# A same-day A/B experiment against the local models above, not a
# replacement: this pipeline's whole premise is running entirely local, and
# every other constant in this file stays exactly as it was. This override
# exists only because this particular corpus was confirmed non-sensitive and
# the point is to compare a frontier model against the 3B local ones on the
# same retrieval pipeline. Set LLM_PROVIDER=groq (cli.py serve --provider
# groq does this) to turn it on; anything else, including leaving it unset,
# leaves every line above untouched.
#
# Keys and model names load from .env (never hardcoded, never committed;
# .env already matches .gitignore) via python-dotenv, already an installed
# dependency though unused until now. Up to 4 keys because Groq's free tier
# rate-limits per key; client.py rotates to the next configured key on a 429
# rather than the whole pipeline stopping mid-question.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local")

# Populated only when LLM_PROVIDER is actually "groq", deliberately not
# just "whenever .env happens to have keys in it": .env is loaded
# unconditionally above (simplest way to make GROQ_GENERATOR_MODEL etc.
# available without a second load call), so leaving this keyed only on
# presence would mean a local run with a filled-in .env silently attaches
# a real Groq bearer token to every request sent to Ollama. Harmless in
# practice, since Ollama's endpoint does not check auth and ignores the
# header, but there is no reason for a key to leave this file's scope on
# a path that was never asked to use it.
GROQ_API_KEYS: list[str] = []

if LLM_PROVIDER == "groq":
    GROQ_API_KEYS = [
        key for key in (
            os.environ.get("GROQ_API_KEY_1", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", ""),
            os.environ.get("GROQ_API_KEY_4", ""),
        ) if key
    ]
    if not GROQ_API_KEYS:
        raise RuntimeError(
            "LLM_PROVIDER=groq but no GROQ_API_KEY_1..4 is set in .env. "
            "Copy .env.example to .env and fill in at least one key from "
            "console.groq.com."
        )
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
    # Checked live against this account's own /v1/models on 2026-09-05
    # (llama-3.3-70b-versatile and gemma2-9b-it, the first choice, had
    # both been retired by then). Two things ruled out qwen/qwen3.6-27b
    # as the judge despite otherwise fitting the "different family"
    # principle: measured directly, it prints its own chain of thought
    # as literal <think>...</think> text inside the answer content
    # itself, 201 completion tokens to say "pong", rather than keeping
    # it out of the field every JSON parser in this pipeline reads.
    # allam-2-7b (SDAIA/IBM, Arabic-specialised, a different family from
    # gpt-oss either way) came back clean, "pong" and nothing else, 4
    # tokens. Its own context is a comparatively tight 4096 tokens;
    # workable for entail.py's short premise+hypothesis pairs, not
    # verified for every possible chunk length in this corpus.
    GENERATOR_MODEL = os.environ.get("GROQ_GENERATOR_MODEL", "openai/gpt-oss-120b")
    JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "allam-2-7b")

# Tokens of context this pipeline needs, and what probe.py verifies the server
# actually provides rather than trusting.
#
# Ollama defaults to far less and truncates the overflow in silence. Eight
# retrieved chunks at the p90 size measured in chunking are roughly 6,000
# characters, near 2,000 Arabic tokens, before the prompt and the context
# prefixes stage 4 adds. The default would cut that mid-context and return a
# fluent answer built on half the evidence, which is quality.py's failure mode
# exactly: it returns, it reads well, and nothing appears in the logs.
LLM_CONTEXT = 8192


# --- golden set --------------------------------------------------------------

GOLDEN_DIR = DATA_DIR / "golden"
GOLDEN_SET = GOLDEN_DIR / "golden_set.json"
GOLDEN_PAGES_DIR = GOLDEN_DIR / "pages"

# The worksheet a person reads from, in the same register as
# CONTEXT_SAMPLES_OUTPUT: text meant for an editor, not JSON meant for code.
GOLDEN_WORKSHEET = GOLDEN_DIR / "worksheet.txt"

# Render resolution for a person reading a page on screen, not for OCR. RENDER_DPI
# is 300 because ingestion measured 600 as four times the runtime for no accuracy
# gain, and that measurement is about EasyOCR's recognition model, a fact about a
# recogniser rather than an eye. Reading by eye is a different consumer with a
# different failure mode, cropped or blurred detail rather than a misrecognised
# glyph, so it gets its own constant rather than borrowing one whose
# justification does not transfer.
GOLDEN_RENDER_DPI = 400


# --- embedding and the vector store -------------------------------------------

# The two candidates for the bake-off, short name to HuggingFace id.
#
# Only the identity lives here. How each one pools its hidden states, and which
# prefixes it needs on a query as against a passage, live in embedder.py beside
# the functions that implement them: that is behaviour, not configuration, and
# separating a pooling rule from its code is how the two drift apart.
#
# Both are XLM-RoBERTa-large. The library choice therefore decides nothing about
# Arabic and the weights decide everything, which is what makes this a bake-off
# rather than a preference.
EMBED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/multilingual-e5-large",
}

# The prior, named here so the bake-off's tie-break rule has something to point
# at rather than a string typed twice. One model producing both dense and
# learned sparse vectors removes a whole component from stage 7; that is an
# argument for a tie, never for a result.
PREFERRED_MODEL = "bge-m3"

# Vectors are cached to disk on the same principle as the OCR and LLM caches:
# a stage that gets re-run should pay for its GPU minutes once.
EMBEDDING_CACHE_DIR = DATA_DIR / "embeddings"

# BGE-M3's sparse (indices, values) pairs, cached separately from the dense
# .npy files above because they are a different shape entirely and mixing
# the two in one directory would need a naming convention to tell them
# apart. Unused until stage 7 makes sparse embedding a per-query cost paid
# repeatedly across an evaluation grid; the once-per-variant store build
# never needed this.
SPARSE_EMBEDDING_CACHE_DIR = DATA_DIR / "embeddings_sparse"

# Qdrant in local file mode. No Docker, decided in the plan and unchanged: one
# directory on disk, one collection per context variant.
QDRANT_PATH = DATA_DIR / "qdrant"
COLLECTION_PREFIX = "gbg_"

# 4096 MiB of VRAM holds one XLM-R-large at fp16 with room for activations, and
# does not hold two. Eight sequences of up to 8,192 tokens is the conservative
# starting point; embedder.py halves it on an OOM and says that it did, rather
# than dying or silently succeeding at a size nobody chose.
EMBED_BATCH_SIZE = 8

# The k the golden set is scored at, from the plan: Recall@10 and MRR. Named
# once so the metric functions, the store's search and gate 6's brute-force
# comparison cannot disagree about what "top 10" means.
RETRIEVAL_K = 10

TOKEN_CENSUS_OUTPUT = PROCESSED_DIR / "04_token_census.txt"
BAKEOFF_OUTPUT = PROCESSED_DIR / "04_bakeoff.md"

# Which model won, written by the bake-off and read by everything after it.
#
# A constant edited by hand after reading a table would let the store be built
# with a model the measurement never chose, and nothing would raise. Making the
# winner an artifact on disk means the store can refuse to build until the
# bake-off has actually run, the same way golden.py refuses to verify a set
# whose chunk file is missing.
BAKEOFF_DECISION = PROCESSED_DIR / "04_bakeoff_decision.json"


# --- hybrid retrieval ---------------------------------------------------------

# Standard textbook values, not tuned. Fitting k1 and b to 18 golden
# questions would fit the golden set rather than the corpus, the same
# objection stage 7's plan raises against tuning fusion weights.
BM25_K1 = 1.2
BM25_B = 0.75

# Reciprocal Rank Fusion's own constant, from the original paper and left
# there: RRF needs no score normalisation across dense, sparse and BM25,
# three systems whose raw scores are not on the same scale, which is the
# whole reason RRF rather than a weighted score sum is used at all.
RRF_K = 60

# How many candidates each leg contributes to fusion before ranks are
# combined. Wider than RETRIEVAL_K so a leg that ranks a gold chunk just
# outside the top 10 on its own can still lift it back in after fusion.
CANDIDATE_K = 50

# Diversity caps, enforcing "answer from the whole corpus" in code. Measured
# honestly rather than assumed to help: Q13's three gold chunks sit on two
# adjacent pages, where a per-page cap costs recall, while Q19's three sit
# one per manual, where a per-source cap is exactly what it needs.
# evaluate.py's grid reports both settings rather than picking one on
# instinct.
MAX_PER_SOURCE = 5
MAX_PER_PAGE = 3

# MMR's own trade-off between relevance and novelty. 1.0 would be no
# diversity term at all; this is a starting point evaluate.py's grid checks
# rather than asserts.
MMR_LAMBDA = 0.5

RETRIEVAL_OUTPUT = PROCESSED_DIR / "05_retrieval.md"
STOPWORD_REPORT = PROCESSED_DIR / "05_stopword_report.txt"

# What stage 7 decided, read by stages 8 through 10 the same way store.qdrant
# reads BAKEOFF_DECISION: an artifact on disk a downstream stage can refuse
# to build against until it exists, rather than a constant edited by hand
# after reading a table.
RETRIEVAL_DECISION = PROCESSED_DIR / "05_retrieval_decision.json"


# --- stage 8: router and techniques -------------------------------------------

# The cross-encoder for Reranking. XLM-R-large, the plan's own choice and the
# same family as BGE-M3. bge-reranker-base is the documented fallback if the
# v2-m3 checkpoint does not finish downloading against the deadline: about
# half the download, weaker on Arabic, and the report has to say so as a
# time decision rather than a measured one.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# How many of the fused candidates the cross-encoder actually scores.
# advanced-rag-plan.md's own component table budgets "around 20 pairs per
# query" for the CPU reranker; this is that budget, named rather than
# recomputed. Wider than RETRIEVAL_K so a chunk fusion ranked just outside
# the top 10 can still be promoted back in by the reranker.
RERANK_TOP_N = 20

# Paraphrases Multi-Query generates, beside the original question, all fused
# by fusion.reciprocal_rank_fusion. Not measured yet: a reasoned default
# rather than a number the golden set chose, and evaluate.py's per-technique
# grid is what gets to say whether more would help.
MULTIQUERY_N = 3

# Sub-questions Decomposition asks for at most. Q5's two-table structure,
# the case the golden set was built to exercise, is the worst case measured
# so far; four leaves room for a genuinely three-part question without
# inviting the router to over-decompose a two-part one.
DECOMPOSE_MAX = 4

# HyDE's hypothetical passage. Short on purpose: this is a retrieval query
# in prose form, not a synthesised answer, and a long one dilutes the
# embedding toward generic phrasing rather than the corpus's own vocabulary.
HYDE_MAX_TOKENS = 200

# CRAG's own adaptation, recorded in the plan: no web search in a closed
# domain, so "incorrect" falls back to one corpus re-query with a rewritten
# query, and refuses if that also grades low. One retry, not a loop: a
# second consecutive miss is evidence the corpus does not answer this,
# which is exactly Q10's case, not evidence the first rewrite was unlucky.
CRAG_MAX_REQUERIES = 1

# Characters per token, for turning LLM_CONTEXT into a character budget
# Compression can size against without calling a tokenizer per candidate
# chunk. llm.md measured 2.8 chars/token against Qwen2.5's own tokenizer on
# this corpus; this rounds down from that on purpose. Overestimating tokens
# only triggers compression that was not strictly needed; underestimating
# lets a prompt overflow, and llm.md also measured that an overflowing
# prompt does not raise or warn, it is silently cut to half the context and
# answers fluently from what remains. The asymmetry decides which way to
# round.
CHARS_PER_TOKEN = 2.5

# Tokens reserved out of LLM_CONTEXT for the system prompt, the question and
# the model's own answer, when Compression computes how much retrieved
# context is actually allowed to reach the prompt. Generous on purpose, for
# the same overflow failure mode CHARS_PER_TOKEN's comment describes.
GENERATION_RESERVE_TOKENS = 1500

ROUTER_OUTPUT = PROCESSED_DIR / "06_router.md"
TECHNIQUES_OUTPUT = PROCESSED_DIR / "06_techniques.md"

# What stage 8 decided about defaulting reranking on for advanced_rag, read
# back the same way RETRIEVAL_DECISION is: an artifact on disk stage 9 and
# stage 10 can refuse to build against until it exists, rather than a
# constant edited by hand after reading a table.
TECHNIQUE_DECISION = PROCESSED_DIR / "06_technique_decision.json"


# --- stage 9: two-stage generation --------------------------------------------

# The synthesiser's own completion cap, and what its prompt-budget preflight
# reserves out of LLM_CONTEXT for the answer itself. One number for both
# purposes, since they are the same fact stated from two directions: the
# server will not emit more than this many tokens, so the prompt has no
# business reserving less room than that.
#
# Raised from an initial 900 after synthesise.py's own gate measured it
# overflow for real on Q19, the six-roles-across-three-manuals question:
# once a prompt fix stopped the model echoing a fixed worked example
# (synthesise.py's own docstring has the full finding), it tried to write
# a genuine full reconciliation and hit exactly 900 completion tokens,
# raising client.py's finish_reason == "length" guard rather than
# returning a real, complete answer. 1,400 gives that answer shape real
# headroom above the corpus's own largest measured chunk (llm.md:
# ~1,040 tokens against Qwen2.5's tokenizer) while still leaving
# LLM_CONTEXT - SYNTHESIS_MAX_TOKENS = 6,792 tokens of room for retrieved
# context, comfortably above what even ten uncompressed chunks measure.
SYNTHESIS_MAX_TOKENS = 1400

# The presenter reformats an already-short synthesised answer; it has no
# reason to need more room than the synthesiser did; it only ever gets less
# to work with, since the whole point of splitting generation in two is that
# the presenter adds structure, not content.
PRESENTER_MAX_TOKENS = SYNTHESIS_MAX_TOKENS

# A refusal is one or two sentences explaining that the corpus does not
# cover a question, never a synthesised answer; generous relative to that
# real shape without inviting the model to pad a refusal into something
# longer than it has any reason to be.
REFUSAL_MAX_TOKENS = 200

# The "simple" route: a greeting reply or a brief general definition with
# no retrieval behind it (the task's own Q1, "What is RAG?", is this
# shape). Roomier than a refusal, since a definition runs longer than a
# one-line decline, still nowhere near a synthesised answer's own budget.
DIRECT_MAX_TOKENS = 300

# The entailment backend the guard's own bake-off measures against the LLM
# judge. mDeBERTa rather than a larger XNLI checkpoint, on the same reasoning
# rerank.py gives for its own cross-encoder pick: CPU latency at this size is
# acceptable and it leaves the card free. Its own 512-token position limit
# means a chunk longer than that gets truncated as the NLI premise, which
# entail.py's own module docstring records as a real, accepted limitation
# rather than something silently absorbed.
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Which entail.py backend the grounding guard uses, overriding the winner
# stored in 07_generation_decision.json. None means defer to that file
# (currently "llm").
#
# Tried switching this to "nli" to cut latency: the guard is the only step
# using JUDGE_MODEL while everything else uses GENERATOR_MODEL, and on a
# 4 GB card two 3B models forces an Ollama swap each way, measured at
# roughly 14 s of an 18 s answer. In isolation nli genuinely was faster
# (0.20-0.25 s warm against 8+ s). Reverted anyway: with the embedder
# worker and the reranker's own worker already contending for this
# machine's memory, adding the NLI worker as a third simultaneous torch
# process pushed Windows commit charge over its limit and caused new
# failures, including one where Ollama's own llama-server was killed as
# collateral. Three of six fresh questions failed outright with the nli
# backend active; that regression is worse than the 14 s it would have
# saved. Left as a documented option, not a live one, until this runs
# somewhere with real memory headroom to confirm it is actually safe.
GUARD_BACKEND: str | None = None

# What stage 9 decided: which entailment backend shipped, the CRAG threshold
# it measured, and the presenter's own block rate, read back the same way
# TECHNIQUE_DECISION is by stage 10, rather than a constant edited by hand
# after reading a table.
GENERATION_OUTPUT = PROCESSED_DIR / "07_generation.md"
GENERATION_DECISION = PROCESSED_DIR / "07_generation_decision.json"


# --- stage 10: evaluation and the report --------------------------------------

# One question answered by one of evaluation.record.ARMS, flattened and
# written once, read back by every file after harness.py rather than
# re-running local generation to re-render a table. RETRIEVAL_DECISION and
# TECHNIQUE_DECISION are read back the same way by the stages that consume
# them; this is that same discipline applied to a whole batch of answers
# instead of one number.
RUNS_OUTPUT = PROCESSED_DIR / "20_runs.json"

# ragas's own four scores per (arm, question), on a ledger separate from the
# one any run in RUNS_OUTPUT spent answering: a judge call is a real cost,
# but not the cost section 7 and section 14 both ask for, which is what
# answering the question spent, not what grading it afterwards spent.
JUDGE_OUTPUT = PROCESSED_DIR / "21_judge.json"

# The committed deliverable, at the repo root rather than under data/, since
# data/processed/ is gitignored on purpose (the corpus is Housing Bank
# material marked for internal use) and a report that only ever existed
# gitignored would never reach the repo this task is graded from.
REPORT_OUTPUT = PROJECT_ROOT / "REPORT.md"

# How many of the retrieved contexts a ragas judge call actually sees per
# question, capped independently of RETRIEVAL_K and RERANK_TOP_N. The
# adaptive and forced arms can hand generation up to RERANK_TOP_N (20)
# reranked chunks; a Context Relevance or Faithfulness prompt built from all
# 20 plus the question and the rubric risks exactly the silent overflow
# CHARS_PER_TOKEN's own comment already measured once, a prompt cut in half
# with no warning. Judging is scored against what was actually retrieved, so
# this caps the judge's own view rather than the generator's: raising it
# only costs judge tokens, never changes what an answer was grounded in.
JUDGE_TOP_K = 10

# The completion budget for one ragas metric call. A judge call's own
# output is a short structured extraction (a list of decomposed claims
# plus a verdict per claim, or a single score), never a full answer, so
# this sits well under SYNTHESIS_MAX_TOKENS on purpose: a reasoned
# starting point in the same spirit as MULTIQUERY_N and DECOMPOSE_MAX,
# not a number fit to this golden set.
JUDGE_LLM_MAX_TOKENS = 1024
