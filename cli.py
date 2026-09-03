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
    python cli.py retrieve      the retrieval grid, the device probe, or one query
    python cli.py route         one question through the pre-gate and router
    python cli.py techniques    one question through the full pipeline, or the reports
    python cli.py ask           two-stage generation: one question end to end, or the reports
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
    golden_parser.add_argument(
        "--refingerprint", action="store_true",
        help="rewrite corpus_fingerprint against the current chunk build, "
             "after re-checking gates 1-7 and, if --previous is given, "
             "proving every referenced chunk's text is unchanged",
    )
    golden_parser.add_argument(
        "--previous", type=str, default=None, metavar="PATH",
        help="a chunk file snapshot from before the re-chunk, checked "
             "against the current one so --refingerprint can prove text "
             "identity rather than merely trust gates 1-7",
    )
    golden_parser.add_argument(
        "--force", action="store_true",
        help="with --refingerprint and no --previous, proceed on gates 1-7 "
             "alone without independent proof of text identity",
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
    embed_group.add_argument(
        "--visualize", action="store_true",
        help="project the winning model's dense vectors to 2D with PCA and "
             "write a local, offline HTML scatter plot; embeds nothing new",
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

    retrieve_parser = sub.add_parser(
        "retrieve", help="hybrid retrieval: the grid, the device probe, or one query"
    )
    retrieve_group = retrieve_parser.add_mutually_exclusive_group()
    retrieve_group.add_argument(
        "--evaluate", action="store_true",
        help="score every mode over every context variant, with and "
             "without diversity caps, against the golden set, and write "
             "the shipping decision",
    )
    retrieve_group.add_argument(
        "--stopwords", action="store_true",
        help="write the document-frequency report the frozen stopword "
             "list in retrieval/text.py was chosen from; retrieves nothing",
    )
    retrieve_group.add_argument(
        "--device-probe", action="store_true",
        help="compare CPU and GPU query-time embedding latency, memory "
             "and cosine agreement, each in its own subprocess",
    )
    retrieve_group.add_argument(
        "query", nargs="?", default=None,
        help="run one question through the shipping configuration and "
             "print the ranked chunk ids",
    )

    route_parser = sub.add_parser(
        "route", help="one question through the pre-gate and router"
    )
    route_group = route_parser.add_mutually_exclusive_group()
    route_group.add_argument(
        "--evaluate", action="store_true",
        help="run every golden question through the pre-gate and router "
             "in order, report route agreement, and write the router "
             "report",
    )
    route_group.add_argument(
        "question", nargs="?", default=None,
        help="run one question through the pre-gate then the router, "
             "and print the route decision",
    )

    techniques_parser = sub.add_parser(
        "techniques",
        help="the eight techniques: one question through the full "
             "pipeline, or the router and technique reports",
    )
    techniques_group = techniques_parser.add_mutually_exclusive_group()
    techniques_group.add_argument(
        "--evaluate", action="store_true",
        help="write the router report, the technique report, and the "
             "reranking-default decision",
    )
    techniques_group.add_argument(
        "question", nargs="?", default=None,
        help="run one question through the pre-gate, the router, and "
             "whichever techniques it selects, and print the retrieved "
             "chunk ids and which techniques executed",
    )

    ask_parser = sub.add_parser(
        "ask",
        help="two-stage generation: one question through the full "
             "pipeline to a presented answer, or the generation report",
    )
    ask_group = ask_parser.add_mutually_exclusive_group()
    ask_group.add_argument(
        "--evaluate", action="store_true",
        help="run the crag-threshold measurement, the entailment "
             "bake-off, and every generation gate, and write the "
             "generation report and decision",
    )
    ask_group.add_argument(
        "question", nargs="?", default=None,
        help="answer one question end to end: route, retrieve, "
             "generate, guard, and print the presented answer",
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
        from pathlib import Path
        from pipeline.golden import worksheet
        from pipeline.golden import golden as golden_module
        if args.refingerprint:
            previous = Path(args.previous) if args.previous else None
            return 0 if golden_module.refingerprint(
                previous_chunks_path=previous, force=args.force
            ) else 1
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
        if args.visualize:
            from pipeline.embedding import visualize
            return 0 if visualize.run() else 1
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

    if args.command == "retrieve":
        if args.evaluate:
            from pipeline.retrieval import evaluate
            return 0 if evaluate.run() else 1
        if args.stopwords:
            import json as _json
            from pipeline.chunking.chunk import load_chunks
            from pipeline.config import CONTEXT_OUTPUTS, STOPWORD_REPORT
            from pipeline.retrieval.text import document_frequency_report
            chunks = load_chunks(CONTEXT_OUTPUTS["none"])
            report = document_frequency_report([c.text for c in chunks])
            STOPWORD_REPORT.parent.mkdir(parents=True, exist_ok=True)
            STOPWORD_REPORT.write_text(report, encoding="utf-8")
            print(f"written to {STOPWORD_REPORT}")
            return 0
        if args.device_probe:
            from pipeline.retrieval import device_probe
            return 0 if device_probe.run() else 1
        if args.query:
            from pipeline.retrieval.retriever import retrieve_shipping
            ranked = retrieve_shipping(args.query)
            for i, chunk_id in enumerate(ranked, 1):
                print(f"{i}. {chunk_id}")
            return 0
        retrieve_parser.print_help()
        return 1

    if args.command == "route":
        from pipeline.llm.ledger import Ledger
        from pipeline.router import gate
        from pipeline.router import router as router_module
        if args.evaluate:
            from pipeline.config import ROUTER_OUTPUT
            lines = router_module.verify_against_golden()
            ROUTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            ROUTER_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
            print(lines[0])
            print(f"written to {ROUTER_OUTPUT}")
            return 0
        if args.question:
            decision = gate.check(args.question)
            if decision is None:
                decision = router_module.route(args.question, Ledger(label="cli-route"))
            print(f"route: {decision.route}")
            print(f"reason: {decision.reason}")
            print(f"requested techniques: {list(decision.requested)}")
            if decision.refusal_kind:
                print(f"refusal_kind: {decision.refusal_kind}")
            return 0
        route_parser.print_help()
        return 1

    if args.command == "techniques":
        if args.evaluate:
            from pipeline.techniques import evaluate as techniques_evaluate
            return 0 if techniques_evaluate.run() else 1
        if args.question:
            from pipeline.llm.ledger import Ledger
            from pipeline.retrieval.retriever import open_shipping
            from pipeline.techniques import rerank
            from pipeline.techniques.run import answer
            # warm_up() before open_shipping(): a single ad-hoc question
            # could route to advanced_rag, which always reranks, and the
            # route is not known until the router runs inside answer();
            # see rerank.warm_up's own docstring for why this order is
            # load-bearing rather than a style preference.
            rerank.warm_up()
            with open_shipping() as handle:
                result = answer(args.question, Ledger(label="cli-techniques"), handle)
            print(f"route: {result.decision.route}")
            print(f"reason: {result.decision.reason}")
            print(f"executed techniques: {list(result.executed)}")
            print(f"retrieved chunk ids: {result.chunk_ids}")
            if result.crag_refused:
                print("CRAG refused: no confident answer found in the corpus")
            return 0
        techniques_parser.print_help()
        return 1

    if args.command == "ask":
        if args.evaluate:
            from pipeline.generation import evaluate as generation_evaluate
            return 0 if generation_evaluate.run() else 1
        if args.question:
            from pipeline.config import PROCESSED_DIR
            from pipeline.llm.ledger import Ledger
            from pipeline.retrieval.retriever import open_shipping
            from pipeline.techniques import rerank
            from pipeline.generation.run import answer

            # warm_up() before open_shipping(): the same load-bearing
            # order techniques' own cli-techniques branch already
            # follows, since a single ad-hoc question is not known to
            # need Reranking until the router runs inside answer().
            rerank.warm_up()
            with open_shipping() as handle:
                result = answer(args.question, Ledger(label="cli-ask"), handle)

            # Arabic to the console is the one thing every other command
            # in this file avoids, on purpose: llm.md and gate.py's own
            # __main__ both document a Windows console's cp1252 default
            # raising outright on the first Arabic letter. reconfigure()
            # is a genuine improvement over that crash, not a full fix
            # for every terminal's own rendering, so the full answer is
            # also written to a file, the same fallback every Arabic
            # output in this project already relies on.
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

            print(f"kind: {result.kind}")
            if result.refusal_kind:
                print(f"refusal_kind: {result.refusal_kind}")
            print(f"citations: {list(result.citations)}")
            if result.presenter_rejected:
                print("presenter rejected by the guard; showing the "
                      "synthesised text instead")
            print(f"\n{result.text}")

            out_path = PROCESSED_DIR / "_last_ask.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                f"question: {result.question}\n"
                f"kind: {result.kind}\n"
                f"refusal_kind: {result.refusal_kind}\n"
                f"citations: {list(result.citations)}\n"
                f"presenter_rejected: {result.presenter_rejected}\n\n"
                f"text:\n{result.text}\n\n"
                f"synthesised:\n{result.synthesised}\n\n"
                f"presented:\n{result.presented}\n",
                encoding="utf-8",
            )
            print(f"\n(also written to {out_path})")
            return 0
        ask_parser.print_help()
        return 1

    from pipeline.ingestion import compare
    compare.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
