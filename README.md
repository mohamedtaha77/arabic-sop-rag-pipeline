# Arabic SOP retrieval pipeline

A retrieval-augmented generation pipeline built for Arabic standard operating
procedure manuals. The corpus is three internal procedure documents from Housing
Bank covering warehouse operations, central alarm, and central mail handling.
78 pages, roughly 115,000 characters, about 85% Arabic.

The pipeline runs locally. The source documents are classified for internal use,
which rules out cloud OCR, cloud embeddings, and hosted vision APIs regardless of
their accuracy.

## Status

Ingestion and chunking are complete. Later stages are in progress.

| Stage | Status | Output |
|---|---|---|
| Ingestion | done | `Document` objects with provenance and quality metadata |
| Chunking | done | 357 chunks with section paths, actors and row provenance |
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
matches on 28 of 29, 19 of 20 and 28 of 29 text layer pages, and on zero OCR
pages, so the OCR route borrows its version metadata from the text layer route.

One manual's header labels are themselves corrupted, truncated mid-word, which
left all three of its fields empty for a while. The digits beside them were
correct throughout — digits are the one thing a broken glyph table does not
damage — so the version is recovered by anchoring on the surviving part of the
label. Its two dates truncate to the same prefix and cannot be told apart that
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

## Usage

Place PDFs in `data/raw/`, then:

```powershell
python cli.py textlayer    # fast extraction, assesses whether OCR is needed
python cli.py ocr          # rendering and OCR, cached per page
python cli.py compare      # score both routes against ground truth
python cli.py layout       # tables and prose from OCR plus page geometry
python cli.py chunk        # split the layout output into retrievable chunks
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

Chunk size is currently measured in characters, standing in for tokens at
roughly three to four characters per Arabic token. The embedding model's own
tokenizer should settle it, and the largest chunk at 2,903 characters needs
checking against a real token count before anything is indexed.

The manuals cross-reference governing policies that are not part of the corpus,
including a procurement policy and a delegation of authority document. Queries
landing on those references should be reported as outside the ingested set rather
than answered.

## Licence

MIT. See [LICENSE](LICENSE).
