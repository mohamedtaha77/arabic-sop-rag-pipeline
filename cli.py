"""Command line entry point.

    python cli.py textlayer     extract via the embedded text layer
    python cli.py ocr           extract via rendering and OCR
    python cli.py layout        extract tables and prose via OCR plus geometry
    python cli.py compare       score the two routes against each other
"""

from __future__ import annotations

import argparse
import sys

from pipeline.config import RENDER_DPI


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

    from pipeline.ingestion import compare
    compare.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
