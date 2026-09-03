"""Router stage: a question in, a routed and technique-tagged decision out.

Position: retriever.retrieve_shipping already answers a question the same
way every time, dense retrieval, top 10, done. This stage is what decides
whether that is even the right thing to do for a given question, and if
not, which of the eight techniques techniques/run.py should reach for.

    schema   the RouteDecision and TechniqueSet contracts, and the
             vocabulary both are built from
    gate     the deterministic, zero-cost pre-gate: catches a greeting or a
             fragment before any model is called
    router   the one structured call that decides in_scope, route, and
             which techniques a question shows a real signal for

Run `python cli.py route "..."` for one question, or
`python -m pipeline.router.router` to check route agreement against the
whole golden set and write `data/processed/06_router_probe.txt`.
"""

from .gate import check as pre_gate_check
from .router import route
from .schema import ROUTES, TECHNIQUES, RouteDecision, TechniqueSet

__all__ = [
    "ROUTES",
    "TECHNIQUES",
    "RouteDecision",
    "TechniqueSet",
    "pre_gate_check",
    "route",
]
