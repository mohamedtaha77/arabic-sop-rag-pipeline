# Arabic SOP retrieval pipeline

A retrieval-augmented generation pipeline built for Arabic standard operating
procedure manuals. The corpus is three internal procedure documents from Housing
Bank covering warehouse operations, central alarm, and central mail handling.
78 pages, roughly 115,000 characters, about 85% Arabic.

The pipeline runs locally. The source documents are classified for internal use,
which rules out cloud OCR, cloud embeddings, and hosted vision APIs regardless of
their accuracy.

## Status

Ingestion is complete. Later stages are in progress.

| Stage | Status | Output |
|---|---|---|
| Ingestion | done | `Document` objects with provenance and quality metadata |
| Chunking | not started | retrievable units preserving table row structure |
| Embedding | not started | vectors from a multilingual model |
| Vector store | not started | searchable index |
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
matches on 28 of 29 and 19 of 20 text layer pages, and on zero OCR pages, so the
OCR route borrows its version metadata from the text layer route.

The text layer route also runs 200 times faster and produces the quality
assessment that determines whether OCR is needed at all. On an undamaged PDF it
is the correct and only route required.

### Known limitations

OCR does not recover table structure. Several pages are grids binding an
executing unit to numbered action steps, and OCR flattens each row into a single
line, so column meaning is lost. This is a layout problem rather than a character
problem, and no OCR setting addresses it. The bounding boxes discarded by
`detail=0` in `ocr_image` are where a fix would begin. Chunking has to solve it.

OCR also introduces its own errors, including one Arabic conjunction misread as a
digit. These are fewer and less systematic than the font corruption they replace,
but they are not zero.

Arabic text extracted in visual rather than logical order is a common PDF failure
and worth ruling out on any new corpus. It does not occur here. Reversed spellings
of common words appear zero times in either route's output, so no bidirectional
reshaping pass is required.

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

## Usage

Place PDFs in `data/raw/`, then:

```powershell
python cli.py textlayer    # fast extraction, assesses whether OCR is needed
python cli.py ocr          # rendering and OCR, cached per page
python cli.py compare      # score both routes against ground truth
```

Run `textlayer` first. It is cheap, and it populates the version metadata the OCR
route reuses.

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
    compare.py                  scores the two routes against ground truth
data/
  raw/                          source PDFs, not tracked
  processed/                    stage outputs, not tracked
  ocr_cache/                    per page OCR results, not tracked
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

## Planned design decisions

The embedding model has to be multilingual. The common tutorial default,
`all-MiniLM-L6-v2`, is English only and produces near-useless vectors for this
corpus. Candidates are `multilingual-e5`, `paraphrase-multilingual-mpnet`, and
`bge-m3`, to be benchmarked on the real documents.

Retrieval will be hybrid rather than vector only. Exact role names, rule numbers,
and form codes are lexical matching problems that embeddings blur and BM25
handles precisely. Vector-only retrieval performs acceptably in casual testing
and fails on exactly the questions this corpus exists to answer.

Chunks need a section path. Each manual restarts its own numbering, so a chunk
reading "step 9" is ambiguous across documents unless the section path travels
with it.

The manuals cross-reference governing policies that are not part of the corpus,
including a procurement policy and a delegation of authority document. Queries
landing on those references should be reported as outside the ingested set rather
than answered.

## Licence

MIT. See [LICENSE](LICENSE).
