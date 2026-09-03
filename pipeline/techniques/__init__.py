"""Techniques stage: the eight advanced-RAG techniques, and the
orchestration that decides which ones a question actually needs.

Position: pipeline/router/ has already decided whether a question needs
retrieval at all, and if so, which techniques it shows a real signal for.
This package is where those techniques live and where run.py's answer()
turns a RouteDecision into an actual retrieved, and for advanced_rag,
transformed, reranked, compressed, CRAG-checked context.

    run          the orchestration: schema.TechniqueSet dispatch,
                 QuestionRun, the record section 2 of the task asks for

Filled in one technique at a time as each is built; not every name below
exists yet, and run.py is written against that: a technique the router
requests but this package has not built yet raises a clear
NotImplementedError naming the file, not an ImportError three frames deep.

    rewrite, multiquery, decompose, hyde, selfquery   query transformation
    rerank, compress, crag                            retrieval improvement

Run `python cli.py techniques "..."` for one question through the full
pipeline, once every technique above exists.
"""
