"""Read a FastAPI/APIRouter route table across the fastapi 0.137 break.

fastapi>=0.137 stopped flattening ``include_router()`` sub-routes into the
parent's ``.routes`` list: each inclusion is now a single private
``_IncludedRouter`` wrapper
(https://github.com/fastapi/fastapi/releases/tag/0.137.0). Reading
``[r.path for r in app.routes]`` therefore raises
``AttributeError: '_IncludedRouter' object has no attribute 'path'`` under
fastapi 0.141.1 / starlette 1.6.0.

Neither obvious repair works, and both fail QUIETLY rather than loudly:

* ``if hasattr(r, "path")`` drops every included route, so a route-table
  assertion either fails confusingly or passes vacuously over the four
  default docs routes;
* a hand-rolled recursion into ``r.routes`` finds nothing either — measured
  on this tree, the wrapper exposes NEITHER ``.path`` NOR ``.routes``.

``fastapi.routing.iter_route_contexts`` (public, added in 0.137.2) is the
accessor that resolves either shape, yielding contexts whose ``.path`` is the
real, fully-prefixed path. On fastapi<0.137.2 the list is already flat, so the
fallback below is exactly the pre-0.137 read.
"""

from __future__ import annotations

from collections.abc import Iterable

try:
    from fastapi.routing import iter_route_contexts
except ImportError:  # fastapi < 0.137.2: .routes is already flat
    iter_route_contexts = None  # type: ignore[assignment]


def route_paths(routes: Iterable[object]) -> set[str]:
    """Every effective path reachable from ``routes`` (an app's or a router's)."""
    if iter_route_contexts is None:
        return {r.path for r in routes if hasattr(r, "path")}
    return {ctx.path for ctx in iter_route_contexts(routes) if ctx.path is not None}
