"""Embedding stage: which model, then vectors from it.

Two candidates, BAAI/bge-m3 and intfloat/multilingual-e5-large, scored
against the golden set's 18 answerable questions before either one embeds a
corpus chunk for real. The comparison itself runs one model per OS process,
never two in one process: loading a second distinct checkpoint after the
first was loaded and released reproducibly crashed on the development
machine, confirmed unrelated to available memory. That constraint shapes
this whole package's module boundaries, including this file: it stays
empty of imports on purpose.

    tokens      real token counts against each model's own tokenizer,
                replacing chunking's 3-4 characters-per-token estimate
    embedder    the model contract: pooling, prefixes, the disk cache, one
                model resident at a time
    metrics     retrieval metrics, the bootstrap tie-break and the report,
                pure numpy, no torch, so the orchestrator below can import
                it without ever loading a model itself
    bakeoff     the worker: evaluates one model against the golden set,
                over all three context variants, in its own process
    run         the orchestrator: spawns bakeoff's worker once per model,
                each a subprocess, decides the winner, writes the report
    sparse      BGE-M3's sparse head, built only once the store confirms it
                can hold a sparse vector and BGE-M3 actually wins

Run in order: `python cli.py embed --tokens`, `python cli.py embed
--bakeoff`, then `python cli.py embed` to embed the winner. All after
`python cli.py context`.

An empty package __init__ rather than the re-exporting style chunking and
golden use: a package's __init__.py always executes before any of its
submodules, with no exception, so an eager `from .embedder import ...`
here would load torch the moment anything imports even `pipeline.embedding.run`,
silently defeating the exact property run.py exists to guarantee. Import
directly from the submodule that defines a name instead, e.g.
`from pipeline.embedding.embedder import embed_passages`.
"""
