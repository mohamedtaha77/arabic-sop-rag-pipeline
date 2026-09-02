# Arabic SOP retrieval pipeline

A retrieval-augmented generation pipeline built for Arabic standard operating
procedure manuals. The corpus is three internal procedure documents from Housing
Bank covering warehouse operations, central alarm, and central mail handling.
78 pages, roughly 115,000 characters, about 85% Arabic.

The pipeline runs locally. The source documents are classified for internal use,
which rules out cloud OCR, cloud embeddings, and hosted vision APIs regardless of
their accuracy.

## Status

Ingestion, chunking, the local model layer, context prefixes, the golden set,
embedding and the vector store are complete. Later stages are in progress.

| Stage | Status | Output |
|---|---|---|
| Ingestion | done | `Document` objects with provenance and quality metadata |
| Chunking | done | 357 chunks with section paths, actors and row provenance |
| Local model layer | done | tokens, latency and a per-step cost table for every call |
| Context prefixes | done | three chunk sets, none, template and LLM-generated, for the Contextual Retrieval comparison |
| Golden set | done | 20 Arabic questions with reference answers and gold chunk ids, read off rendered pages by hand |
| Embedding | done | BAAI/bge-m3, settled against intfloat/multilingual-e5-large on the golden set |
| Vector store | done | three Qdrant collections, dense and sparse, verified against a brute-force ranking |
| Retrieval | not started | hybrid BM25 and vector, with rank fusion |
| Generation | not started | grounded answers with version-aware citations |
| Evaluation | not started | accuracy against a labelled question set |

## Why ingestion needed two extraction routes

All three source PDFs have damaged ToUnicode tables. A PDF stores glyph numbers
plus a lookup table mapping each to a character, and in these files parts of that
table are wrong. Glyphs decode to the wrong codepoint, frequently producing a
valid Arabic word that is the wrong word.

| Correct | Meaning | Decoded as |
|---|---|---|
| على | on | عسى |
| مساعد | assistant | م اعد |
| الرقابة | control | الرقارة |
| نظام | system | نما |

In one manual the correct form of "on" appears 8 times and the corrupted form 87
times. The pages render correctly in any viewer, because the font still draws the
right shape. Only the machine-readable layer is wrong.

pypdf and PyMuPDF return byte-identical corruption on the affected pages,
including with PyMuPDF's `sort` and ligature flags, so the damage is in the files
rather than the reader. Switching libraries does not help. Rendering the pages
and running OCR does, because rasterisation resolves fonts the same way a viewer
does and never consults the broken table.

### Measured result

Scored against spellings verified by reading rendered page images:

| | text layer | OCR |
|---|---|---|
| Fidelity on known-corrupted words | 44% | 97% |
| Characters extracted | 115,028 | 147,751 |
| Fragmentation, share of 1 to 2 letter Arabic tokens | 13.6% to 17.6% | 5.0% to 5.9% |

The worst affected manual went from 707 to 1,964 characters per page, bringing it
in line with the other two rather than 60% below them.

### Both routes are retained

They fail in opposite directions, so neither is a strict replacement for the
other.

| | text layer | OCR |
|---|---|---|
| Prose accuracy on this corpus | 44% | 97% |
| Label to value adjacency in tables | preserved | destroyed |
| Speed per page | about 50ms | about 10s on a GPU |

Each page carries a version control header. pypdf reads in layout order, so each
value stays next to its label. EasyOCR groups the three labels onto one line and
places the values elsewhere, which destroys the binding. The version parser
matches on 28 of 29, 19 of 20 and 28 of 29 text layer pages, and on zero OCR
pages, so the OCR route borrows its version metadata from the text layer route.

One manual's header labels are themselves corrupted, truncated mid-word, which
left all three of its fields empty for a while. The digits beside them were
correct throughout, because digits are the one thing a broken glyph table does
not damage, so the version is recovered by anchoring on the surviving part of
the label. Its two dates truncate to the same prefix and cannot be told apart that
way, so those come from the header table's geometry instead.

The text layer route also runs 200 times faster and produces the quality
assessment that determines whether OCR is needed at all. On an undamaged PDF it
is the correct and only route required.

### A third route for table structure

Neither route above recovers columns. Several pages are grids binding an
executing unit to numbered action steps, and OCR flattens each row into a single
line, so column meaning is lost. That is a layout problem rather than a character
problem, and no OCR setting addresses it.

`layout.py` reads each page a third way, at `detail=1`, keeping the bounding box
per recognised fragment that the flat route discards. PyMuPDF's `find_tables`
supplies the cell geometry, and each fragment is assigned to the cell its centre
falls inside. Neither tool is sufficient alone: PyMuPDF knows where the cells
are, and its own cell text is both ToUnicode-corrupted and in visual rather than
reading order, so the characters have to come from OCR.

Three findings from measuring the corpus rather than assuming a table's shape:

- A detected table of two columns spanning most of the page height is not data.
  It is the numbered-list frame Word draws around a block of steps, usually with
  a real table nested inside it.
- Content tables are not uniform grids. PyMuPDF stores a merged cell once and
  marks every position it spans as empty, so rows are emitted as variable-length
  lists rather than padded to a fixed column count.
- Tables spanning a page break are common, 12 of them here, three running three
  pages deep. Each records where its continuation pages begin, since a heading
  announced on a later page belongs to the rows below that break.

### Known limitations

OCR introduces its own errors, including one Arabic conjunction misread as a
digit. These are fewer and less systematic than the font corruption they replace,
but they are not zero. Thirty individually confirmed misreadings are corrected by
an explicit list; a corpus sweep found 610 near-miss candidates, most of them
ordinary Arabic carrying a prefix, which is why the list stays hand-confirmed
rather than fuzzy-matched.

Words merged across a lost space, such as two nouns printed without the space
between them, are measured but not corrected. They are invisible to vector
retrieval and will cost lexical recall in the hybrid stage.

Arabic text extracted in visual rather than logical order is a common PDF failure
and worth ruling out on any new corpus. It does not occur here. Reversed spellings
of common words appear zero times in either route's output, so no bidirectional
reshaping pass is required.

## Chunking

After the tables are lifted out, the corpus is 110,793 characters of table cells
against 20,524 of prose. It is nine tenths table, so row shape decides how a
chunk is formed rather than any uniform size rule.

| Row shape | Count | Treatment |
|---|---|---|
| numbered step | 535 | grouped under the actor who performs it |
| actor label | 310 | becomes chunk metadata, not chunk content |
| continuation | 75 | joins the block it follows |
| data grid row | 52 | header re-attached, one chunk per row above six rows |
| account line | 37 | grouped per journal entry |
| heading | 23 | folds into the section path |
| ordinal sub-heading | 14 | folds into the section path |

A procedure block ends where the executor changes, the executing unit changes,
or a heading opens a new procedure. All three are boundaries the document itself
draws. A step number and a description mean little without knowing which unit
performs them, which is the failure nobody notices until an auditor asks.

Section paths need two heading sources. Two manuals put a procedure title in a
one-cell table row; the third prints it in prose on the page above and its tables
carry no title at all. Reading either source alone leaves most of one manual's
procedures unlabelled. A one-cell row is not reliably a heading either: of 107 in
the corpus, 37 are, and the rest are cover titles, approver names, account lines
and step text that lost its number cell.

Three gates run on every pass and print with the report:

- no chunk splits a table row
- every chunk has a section path
- every procedure block binds an executing actor

Verified against the source PDFs in both directions: of 22,936 token uses, none
appears in a chunk without appearing in the source, and three do not reach a
chunk, all from a single canonicalised heading. Everything else excluded is the
cover, the contents page, the version header band or the footer band, each
identified by its position on the page.

Chunk ids are scoped to a page and a type, `assets_wearhouse_p10_procedure_block_02`,
rather than sequential across the corpus. Under a corpus-wide sequence, inserting
one chunk renumbers everything after it and silently invalidates the gold ids the
evaluation set names by hand.

## Setup

Requires Python 3.13.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

OCR runs on CPU by default. On an NVIDIA GPU, install the CUDA build of PyTorch
to cut runtime from roughly 77 seconds per page to about 10:

```powershell
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

### Embedding models and the vector store

`BAAI/bge-m3` and `intfloat/multilingual-e5-large` download from
HuggingFace on first use, about 4.5 GB combined, public weights rather than
corpus data:

```powershell
$env:HF_HOME = "E:\hf"     # weights are a few GB; point somewhere with room
```

Qdrant runs embedded, local file mode, no server and no Docker: nothing
further to install. `data/qdrant/` and `data/embeddings/` are gitignored,
since a committed vector index would be the corpus in another form.

### The local model endpoint

Generation, routing and evaluation all run against a local OpenAI-compatible
endpoint. Nothing in `requirements.txt` covers it, because it is a server rather
than a Python package. Ollama is what this was built against; llama.cpp's server
speaks the same dialect and needs only a different `LLM_BASE_URL`.

Install Ollama, then start it with two settings that matter:

```powershell
$env:OLLAMA_MODELS = "E:\ollama\models"     # models are ~2 GB each
$env:OLLAMA_CONTEXT_LENGTH = "8192"          # see below
ollama serve
```

`OLLAMA_CONTEXT_LENGTH` is not optional. The default is smaller than the
prompts this pipeline sends, and an over-long prompt is truncated in
silence. The model answers fluently from half the retrieved evidence, and
nothing appears in the logs. `python cli.py llm` measures the context the server
actually serves and fails if it is short of what `pipeline/config.py` asks for.

Then pull the two models named in `pipeline/config.py`:

```powershell
ollama pull qwen2.5:3b-instruct-q4_K_M      # generator
ollama pull llama3.2:3b-instruct-q4_K_M     # judge
```

Two model families rather than one. The judge has to differ from the generator,
or it grades its own work generously, and sharing a quantisation is not
differing. `python cli.py llm` refuses to pass if the two tags are equal.

## Usage

Place PDFs in `data/raw/`, then:

```powershell
python cli.py textlayer    # fast extraction, assesses whether OCR is needed
python cli.py ocr          # rendering and OCR, cached per page
python cli.py compare      # score both routes against ground truth
python cli.py layout       # tables and prose from OCR plus page geometry
python cli.py chunk        # split the layout output into retrievable chunks
python cli.py llm          # measure the local model endpoint
python cli.py context      # build the three context-prefix chunk sets
python cli.py golden       # verify the golden set and report coverage
python cli.py embed --tokens    # real token counts, both models, no vectors
python cli.py embed --bakeoff   # score both models, write the decision
python cli.py embed             # embed the winner over all three variants
python cli.py store --probe     # check Qdrant local mode for sparse support
python cli.py store             # build and verify the three collections
python cli.py store --browse    # a local, offline page for looking at what got stored
```

Run `textlayer` first. It is cheap, and it populates the version metadata the
other two routes reuse. `layout` needs its output on disk; without it the version
fields fall back to what page geometry alone can recover.

Open the JSON outputs in an editor rather than a terminal. Windows consoles
render right-to-left text badly, and Arabic that looks reversed in a console is a
display problem rather than a data problem.

OCR results are cached under `data/ocr_cache/`, keyed by the source file's
content hash and the render resolution. Replacing a PDF with a corrected export
invalidates its cached pages automatically, even under the same filename. The
full 78 page pass takes about 14 minutes on an RTX 3050 Ti and is instant
thereafter.

## Layout

```
cli.py                          command line entry point
pipeline/
  config.py                     paths and extraction settings
  ingestion/
    document.py                 the Document contract
    cleaning.py                 Unicode normalisation and whitespace repair
    quality.py                  extraction quality scoring
    metadata.py                 version control field parsing
    storage.py                  JSON serialisation
    textlayer.py                pypdf extraction route
    ocr.py                      rendering and OCR route
    layout.py                   tables and prose from OCR plus page geometry
    compare.py                  scores the two routes against ground truth
  chunking/
    chunk.py                    the Chunk contract, stable ids, storage
    rows.py                     what one table row is
    sections.py                 section paths, from two heading sources
    tables.py                   a table's five kinds and their chunks
    prose.py                    running text, split on numbered rules
    chunker.py                  orchestration and the three gates
    context.py                  three context-prefix variants, none/template/llm
  llm/
    client.py                   one call to the local endpoint, and back
    cache.py                    how a call is not made twice
    pricing.py                  what a free call would have cost
    ledger.py                   which steps ran, and the cost table
    probe.py                    measures the endpoint on this machine
data/
  raw/                          source PDFs, not tracked
  processed/                    stage outputs, not tracked
  ocr_cache/                    per page OCR results, not tracked
  llm_cache/                    per call model results, not tracked
```

## Text normalisation

Applied identically to both routes so their outputs stay comparable. Every rule
below responds to a measurement on the corpus.

| Step | Effect measured |
|---|---|
| NFKC normalisation | Folded 8,990 Arabic presentation forms back to standard letters in one file |
| Remove bidi controls, private use glyphs, kashida | 963 kashida plus symbol font glyphs |
| Remove harakat | 52 total, inconsistent use splits a word's embeddings |
| Remove table of contents dot leaders | 4% to 8% of every file |
| Collapse space runs | 8% to 24% of every file, right-to-left layout encoded as literal spaces |

Presentation forms are the significant one. Arabic letters change shape by
position, and Unicode stores one codepoint per letter while the font selects the
shape. Some producers store the shapes instead, as codepoints in U+FB50 to
U+FEFF. Those are absent from embedding tokenizers and never match a normally
typed query, so without normalisation about a quarter of one manual is
unsearchable.

## The local model layer

Every later stage spends model calls, so the wrapper that makes them is built
before the stages that need it rather than retrofitted afterwards. It records
step, model, input and output tokens, latency and cost for every call, and
renders the per-technique table the evaluation needs.

The row schema is declared up front rather than collected from whatever ran. A
technique that never fired prints as a row reading `No` against zeros, because
which techniques were *skipped* is exactly what an ablation has to show. A step
name outside the schema raises rather than quietly dropping a row.

Local calls cost nothing, and a table of zeros distinguishes nothing. So token
counts are real, measured by the server, and priced against a published hosted
rate card to give the comparison a scale. Every cost figure is a counterfactual
and is labelled as one. Latency is not: running locally makes it the scarce
resource, so it sits beside cost in the table, and CPU-only steps such as the
reranker are recorded with real seconds against zero tokens.

Two model families rather than one, because a judge sharing weights with the
generator grades its own work generously. Measured on an RTX 3050 Ti: a 3B at
Q4_K_M with an 8,192 context fits entirely in the 4 GB card, 2,207 MB of it,
which is why a larger model is not used: a 7B Q4 exceeds the card before its KV
cache, and would run partly on a host with 11.8 GB of RAM.

Three failures are refused rather than returned, all of them cases where
plausible output is worse than an error:

- a completion cut off at `max_tokens`, which reads like a finished answer
- a response with no usage block, which would enter the cost table as free
- an unreachable server or missing model, each reporting the command that fixes it

A fourth cannot be refused and is measured instead. A prompt longer than the
context is not rejected: the server silently keeps about half and answers
fluently from it. Retrieval stages have to budget their own prompt size, since
nothing downstream will say no.

Responses are cached to disk by a hash of everything that changes the answer.
A cache hit replays the original token count and latency and is flagged as
cached, rather than reporting zero. Reporting zero would make every re-run show
a cost collapse that never happened.

## Context prefixes

Contextual Retrieval is a claim about retrieval, and the only way to test it
honestly is against a baseline that has no prefix at all. `pipeline/chunking/context.py`
builds three parallel chunk sets from `02_chunks.json`: none, a deterministic
template assembled from metadata already in the chunks, and one an LLM writes
after reading each chunk in the context of its own section.

| Variant | Total chars | Prefix chars, median |
|---|---|---|
| none | 141,276 | n/a |
| template | 194,804 | 150 |
| llm | 201,207 | 156 |

The template prefix names the manual, the full section path, the page, the
version and the issue date, and adds the actor or unit only when the chunk's
own text does not already carry them. The LLM prefix is generated from a
manual-level outline plus the chunk's whole section, not the whole document,
since a manual runs past the 8,192 context and an over-long prompt is cut in
silence rather than refused. Chunks are grouped by section when the calls are
made, so consecutive calls share an identical prompt prefix a KV cache can
reuse.

A prefix is rejected and re-asked once, then falls back to the template, when
it is empty, off length, mostly off-script, a verbatim or near-verbatim copy
of the chunk, or opens with a preamble. The one check that took two full runs
to get right: token overlap alone cannot tell a lazy copy from a genuine
synthesis in a fixed-register administrative corpus, where a correct summary
of a long procedure and a copy of a short one can both legitimately reuse
100% of their words. What separates them is compression, so a prefix is only
rejected on overlap when it is also not meaningfully shorter than the chunk
it describes. Final result: 281 of 357 prefixes generated cleanly, 49 needed
one re-ask, 27 fell back to the template, all recorded and none silently
blended into the others.

Both generated variants are recoverable back to `02_chunks.json`'s original
text by construction, and the none variant is byte-identical to it. Nothing
here decides whether prefixing helps; that is the embedding and evaluation
stages' question to answer against the golden set covered next.

## Golden set

Every stage before this one produces something that can only be checked
against itself: a table reconstruction against its own gates, three context
variants against one another. The golden set is the one anchor outside all
of that. `data/golden/golden_set.json` holds 20 Arabic questions with
reference answers and gold chunk ids, established by rendering the source
pages at high resolution and reading them directly, the same method that
produced ingestion's 44% against 97% fidelity figure. No model is called
anywhere in this stage.

The set covers both correctness and refusal. Most questions are answerable
from the corpus and carry the chunk ids a correct retriever has to return.
Two are not: one is entirely outside the corpus's domain and should be
refused before retrieval runs at all, and one is in domain but only returns
plausible, related, non-answering chunks, which a system has to catch
downstream rather than reject outright. A question one carried turn removed
from its context is included as well, since resolving it correctly depends
on state the harness has to carry forward rather than something visible in
the question text alone.

Eight checks run against the finished set before it is trusted: every gold
and distractor chunk id has to resolve identically across the base chunk
file and all three context variants, every quoted piece of supporting
evidence has to be a real substring of the chunk it names, an answerable
question has to carry at least one gold chunk and a refused one none, and
the whole set is bound by a content hash to the exact chunk build it was
read against, so a later re-chunk cannot silently invalidate it without
being noticed.

## Embedding

Two multilingual candidates, `BAAI/bge-m3` and
`intfloat/multilingual-e5-large`, scored against the golden set's 18
answerable questions before either one embeds the corpus for real: Recall@10
and MRR@10, both models over all three context variants, decided by a paired
bootstrap on the unconfounded none column rather than by a bare point
estimate from 18 questions. The two models tied closely enough that neither
95% CI excluded zero; BGE-M3 took the decision on the architectural
tie-break the rule named in advance, one model producing dense and learned
sparse vectors together, never on the numbers themselves.

The template context variant clearly outperformed both no prefix and the
LLM-generated prefix for both candidates, the first real answer to the
question Contextual Retrieval was built to ask.

Chunk size, measured in characters since chunking, was checked against both
models' real tokenizers: 3.55 to 3.57 characters per Arabic token, close
enough to the 3-4 assumed that nothing upstream needed revisiting.
e5-large's 512-token cap silently truncates about 1% of the corpus, named by
chunk id rather than smoothed into a percentage; BGE-M3's 8,192-token cap is
never approached.

## Vector store

Qdrant, local file mode, one collection per context variant, built with the
winning model and verified against a brute-force cosine ranking computed
independently of the store, since a wrong distance metric or a mis-mapped
payload would otherwise present itself as a retrieval result rather than a
store bug. BGE-M3's learned sparse vectors are written alongside the dense
ones, once the local store was confirmed to accept them by creating a real
collection rather than trusted to a changelog.

## Retrieval

Three legs, dense cosine, BGE-M3's own learned sparse vector, and a
hand-written Okapi BM25 over a corpus-measured Arabic tokenizer, fused by
Reciprocal Rank Fusion and optionally diversity-capped, decided against the
same golden set by the same paired-bootstrap rule that settled the model
bake-off: a candidate replaces the baseline only if its 95% CI excludes zero,
in its favour, on both Recall@10 and MRR@10.

No candidate cleared that bar. The baseline stage 6 already shipped, BGE-M3,
the template context variant, dense retrieval alone, ships forward
unchanged. Every leg stays implemented and independently switchable, so the
finding is measured rather than assumed: on this corpus, at this scale,
hybrid retrieval did not earn its place over a strong dense baseline once
the fusion weight was left undisturbed rather than searched until something
won. Diversity caps cost recall in every grid cell they were tried in and
are not part of the shipping configuration for the same reason.

Arabic BM25 needed more than splitting on spaces: a corpus-measured
stopword list (derived from the unprefixed chunk variant specifically,
since the context-prefixed variant's own boilerplate would otherwise look
like stopwords), conservative prefix-clitic stripping, and Arabic-Indic
digit folding. Clitic stripping alone lifted BM25's Recall@10 from 0.630 to
0.769 on the golden set, measured inside the real pipeline, not asserted.

Query-time embedding runs on CPU by default, a measured trade-off rather
than the GPU-everywhere default: BGE-M3's own dense and sparse query
vectors could load onto the same 4 GB card Ollama needs for generation,
about 6x faster per query, but at roughly 1.1 GB of VRAM held for as long
as the embedder stays resident. 178ms against a generation call timed out
at 300 seconds is negligible; 1.1 GB against a 4 GB budget is not.

## Planned design decisions

The manuals cross-reference governing policies that are not part of the corpus,
including a procurement policy and a delegation of authority document. Queries
landing on those references should be reported as outside the ingested set rather
than answered.

## Licence

MIT. See [LICENSE](LICENSE).
