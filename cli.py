"""Command line entry point.

    python cli.py textlayer     extract via the embedded text layer
    python cli.py ocr           extract via rendering and OCR
    python cli.py layout        extract tables and prose via OCR plus geometry
    python cli.py compare       score the two routes against each other
    python cli.py chunk         split the layout output into retrievable chunks
    python cli.py llm           measure the local model endpoint
    python cli.py context       build the three context-prefix chunk sets
    python cli.py golden        verify the golden set and report coverage
    python cli.py embed         token census, the bake-off, or the embed run
    python cli.py store         the sparse-vector probe, build, or browse it
"""

from __future__ import annotations

import argparse
import sys

from pipeline.config import CONTEXT_VARIANTS, RENDER_DPI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Ingestion pipeline for Arabic procedure documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("textlayer", help="extract via the embedded text layer")

    ocr_parser = sub.add_parser("ocr", help="extract via rendering and OCR")
    ocr_parser.add_argument(
        "--dpi", type=int, default=RENDER_DPI,
        help=f"render resolution, default {RENDER_DPI}",
    )

    layout_parser = sub.add_parser(
        "layout", help="extract tables and prose via OCR plus geometry"
    )
    layout_parser.add_argument(
        "--dpi", type=int, default=RENDER_DPI,
        help=f"render resolution, default {RENDER_DPI}",
    )

    sub.add_parser("compare", help="score both routes against ground truth")

    sub.add_parser("chunk", help="split the layout output into chunks")

    sub.add_parser("llm", help="measure the local model endpoint")

    context_parser = sub.add_parser(
        "context", help="build the three context-prefix chunk sets"
    )
    context_parser.add_argument(
        "--variant", action="append", choices=list(CONTEXT_VARIANTS),
        help="build only this variant; repeatable. Default: all three.",
    )
    context_parser.add_argument(
        "--sample", type=int, default=None,
        help="generate the llm prefix for a stratified sample of this many "
             "chunks and write it for reading, without building or gating "
             "the full llm variant",
    )

    golden_parser = sub.add_parser(
        "golden", help="verify the golden set and report coverage"
    )
    golden_parser.add_argument(
        "--pages", nargs="*", default=None, metavar="SLUG:PAGE",
        help="with no value, render every page the golden set points at and "
             "write the worksheet. With one or more SLUG:PAGE or "
             "SLUG:START-END values, e.g. alarm:5-8, render just those pages "
             "on demand without touching the worksheet.",
    )

    embed_parser = sub.add_parser(
        "embed", help="token census, the bake-off, or the embed run"
    )
    embed_group = embed_parser.add_mutually_exclusive_group()
    embed_group.add_argument(
        "--tokens", action="store_true",
        help="tokenize every chunk against both models' tokenizers and "
             "report real token counts; embeds nothing",
    )
    embed_group.add_argument(
        "--bakeoff", action="store_true",
        help="score both models over all three variants against the golden "
             "set and write the model decision",
    )

    store_parser = sub.add_parser(
        "store", help="the sparse-vector probe, build the store, or browse it"
    )
    store_group = store_parser.add_mutually_exclusive_group()
    store_group.add_argument(
        "--probe", action="store_true",
        help="check whether Qdrant local file mode accepts a sparse vector; "
             "builds nothing",
    )
    store_group.add_argument(
        "--browse", action="store_true",
        help="write a local, offline HTML page for browsing what is in the "
             "store; opens in a browser, nothing is uploaded anywhere",
    )

    args = parser.parse_args(argv)

    if args.command == "textlayer":
        from pipeline.ingestion import textlayer
        return 0 if textlayer.run() else 1

    if args.command == "ocr":
        from pipeline.ingestion import ocr
        return 0 if ocr.run(dpi=args.dpi) else 1

    if args.command == "layout":
        from pipeline.ingestion import layout
        return 0 if layout.run(dpi=args.dpi) else 1

    if args.command == "chunk":
        from pipeline.chunking import chunker
        return 0 if chunker.run() else 1

    if args.command == "llm":
        from pipeline.llm import probe
        return probe.run()

    if args.command == "context":
        from pipeline.chunking import context
        if args.sample is not None:
            return context.run_sample(args.sample)
        variants = tuple(args.variant) if args.variant else CONTEXT_VARIANTS
        return context.run(variants=variants)

    if args.command == "golden":
        from pipeline.golden import worksheet
        from pipeline.golden import golden as golden_module
        if args.pages is not None:
            if args.pages:
                return 0 if worksheet.run_pages(args.pages) else 1
            return 0 if worksheet.run() else 1
        return 0 if golden_module.run() else 1

    if args.command == "embed":
        if args.tokens:
            from pipeline.embedding import tokens
            return 0 if tokens.run() else 1
        if args.bakeoff:
            from pipeline.embedding import run as bakeoff_run
            return 0 if bakeoff_run.run() else 1
        from pipeline.embedding import bakeoff
        return 0 if bakeoff.run_embed() else 1

    if args.command == "store":
        if args.probe:
            from pipeline.store import probe as store_probe
            return 0 if store_probe.run() else 1
        if args.browse:
            from pipeline.store import browse
            return 0 if browse.run() else 1
        from pipeline.store import qdrant
        return 0 if qdrant.run() else 1

    from pipeline.ingestion import compare
    compare.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
