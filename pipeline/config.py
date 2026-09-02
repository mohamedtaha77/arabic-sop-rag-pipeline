"""Project-wide paths, extraction settings and local model configuration."""

from __future__ import annotations

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
LLM_BASE_URL = "http://localhost:11434/v1"

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
