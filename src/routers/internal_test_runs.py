"""Cluster-internal routes of Rust test runs (never in the public OpenAPI or the gateway).

``GET /internal/test-runs/{run_id}/archive``: the runner Job of a Rust test run
downloads the run's working-tree archive, once, with the per-run token it was
started with (``X-Run-Token``). Anything else (unknown run, wrong token, a second
download) is the same 404. The one-time token is the real control: the user
AuthMiddleware lets this prefix through (``middleware.auth.INTERNAL_PREFIX``), and the
gitops mesh rule (policies/cross-namespace/coding-test-runs-access.yaml) is L4 only:
it admits the coding-test-runs identity to port 8689, any path.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Response

from services.rust_test_runs import take_archive

router = APIRouter(include_in_schema=False)

TOKEN_HEADER = "X-Run-Token"


@router.get("/test-runs/{run_id}/archive")
async def get_run_archive(run_id: str, token: str = Header("", alias=TOKEN_HEADER)) -> Response:
    data = await take_archive(run_id, token) if token else None
    if data is None:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(content=data, media_type="application/gzip")
