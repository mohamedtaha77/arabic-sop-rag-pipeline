"""Vector store stage: Qdrant, local file mode, one collection per variant.

    probe       whether this local store accepts a sparse vector, measured
                by trying it against a throwaway collection rather than
                trusted to a changelog
    qdrant      build() and verify() for the three real collections, one
                per context variant, against whichever model
                pipeline.embedding.run's bake-off chose
    browse      a local, offline HTML page for looking at what is stored,
                nothing uploaded anywhere: the corpus is Housing Bank
                internal-use material

Run `python cli.py store --probe` any time; `python cli.py store` after
`python cli.py embed --bakeoff` has written a decision; `python cli.py
store --browse` after that to look at what got built.
"""

from .browse import run as run_browse
from .probe import probe_sparse_support
from .probe import run as run_probe
from .qdrant import PAYLOAD_FIELDS, build_collection, collection_name, verify
from .qdrant import run as run_store

__all__ = [
    "probe_sparse_support",
    "run_probe",
    "PAYLOAD_FIELDS",
    "build_collection",
    "collection_name",
    "verify",
    "run_store",
    "run_browse",
]
