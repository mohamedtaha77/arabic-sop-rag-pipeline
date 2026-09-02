"""Command line entry point.

    python cli.py textlayer     extract via the embedded text layer
    python cli.py ocr           extract via rendering and OCR
    python cli.py layout        extract tables and prose via OCR plus geometry
    python cli.py compare       score the two routes against each other
    python cli.py chunk         split the layout output into retrievable chunks
    python cli.py llm           measure the local model endpoint
    python cli.py context       build the three context-prefix chunk sets
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

    from pipeline.ingestion import compare
    compare.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
