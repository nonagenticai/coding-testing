"""POST /api/tests/results/{container_name}/computed — the R10-A2 producer route.

Driven through this service's REAL ``AuthMiddleware`` (licence introspection is
the only thing mocked, exactly as in test_fork_vouching.py) and the REAL
Postgres store via ``pg_db``. Only the coding-containers lookup is patched.

Why the real middleware: the service-account gate keys on
``tenant_license_verified`` — state that only the licence arm records. Stubbing
identity with ``override_tenant_auth`` would bypass the very thing under test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.auth import _default_tenant as dt_mod
from coding_client.auth import _trusted_networks as tn_mod
from coding_client.auth.tenant import clear_license_caches
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.route_paths import route_paths

SVC_SUBJECT = "019ff970-e327-74cf-aa8c-e0ce39997d44"
HUMAN_SUBJECT = "4b7f2a3e-0000-4000-8000-000000000001"
SVC_TENANT = "service-own-tenant-0001"
CUSTOMER_TENANT = "customer-tenant-0042"
OTHER_TENANT = "customer-tenant-0099"
CONTAINER = "card-repo-main"
PATH = f"/api/tests/results/{CONTAINER}/computed"

BODY = {
    "test_id": "aaqa:card-77:corr-1",
    "test_name": "r3.qa.coding",
    "language": "python",
    "test_type": "unit",
    "test_framework": "pytest",
    "exit_code": 0,
    "success": True,
    "execution_time_ms": 1234.0,
    "stdout": "12 passed",
    "stderr": "",
    "resource_usage": {"tests": 12, "failures": 0, "errors": 0, "skipped": 0, "passed": 12},
    "card_id": "card-77",
    "commit_sha": "abc123",
    "correlation_id": "corr-1",
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, trusted_networks):
    trusted_networks("")
    import middleware.auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "ENABLE_AUTH", True)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://test-licenses/license/introspect")
    monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", SVC_SUBJECT)
    monkeypatch.setattr(tn_mod, "TRUSTED_NETWORKS", [])
    monkeypatch.setattr(dt_mod, "DEFAULT_TENANT_ID", None)
    clear_license_caches()
    yield
    clear_license_caches()


def _introspection(subject: str, tenant: str = SVC_TENANT):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"active": True, "tenant_id": tenant, "subject_id": subject}
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=client)


def _container(owner_tenant: str | None = CUSTOMER_TENANT, missing: bool = False):
    async def _fake(container_name: str, tenant_id: str | None = None):
        if missing:
            raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")
        from shared.container_validation import ContainerInfo

        return ContainerInfo(container_name=container_name, tenant_id=owner_tenant or "")

    return patch("routers.test_results_data.validate_and_get_container_for_tenant", _fake)


def _client() -> TestClient:
    # Imported inside, never at module scope — see test_fork_vouching._client.
    from middleware.auth import AuthMiddleware
    from routers.test_results_data import router

    app = FastAPI()
    app.include_router(router, prefix="/api/tests")
    app.add_middleware(AuthMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _post(
    subject: str = SVC_SUBJECT,
    body: dict | None = None,
    tenant: str = CUSTOMER_TENANT,
    project: str | None = "proj-5",
):
    headers = {"Authorization": "Bearer svc-token", "X-Tenant-Id": tenant}
    if project is not None:
        headers["X-Project-Id"] = project
    with _introspection(subject):
        return _client().post(PATH, json=body or BODY, headers=headers)


async def _rows(pg_db) -> list:
    async with pg_db() as session:
        return list(
            (
                await session.execute(
                    text(
                        "SELECT test_id, tenant_id, card_id, project_id, commit_sha, "
                        "correlation_id, source, container_name FROM test_results"
                    )
                )
            ).all()
        )


def test_route_is_mounted_on_the_real_app():
    from app_factory import create_app

    assert "/api/tests/results/{container_name}/computed" in route_paths(create_app().routes)


async def test_service_account_creates_row_with_card_columns(pg_db):
    with _container():
        r = _post()
    assert r.status_code == 201, r.text
    assert r.json()["created"] is True
    assert r.json()["result"]["card_id"] == "card-77"
    rows = await _rows(pg_db)
    assert len(rows) == 1
    assert tuple(rows[0]) == (
        "aaqa:card-77:corr-1",
        CUSTOMER_TENANT,  # the vouched (card's) tenant, never the service's own
        "card-77",
        "proj-5",
        "abc123",
        "corr-1",
        "aa_qa",
        CONTAINER,
    )


async def test_repeat_post_is_idempotent(pg_db):
    with _container():
        first = _post()
        second = _post(body={**BODY, "stdout": "changed", "success": False})
    assert first.status_code == 201
    assert second.status_code == 200, second.text
    assert second.json()["created"] is False
    # The stored row is returned unchanged — a repeat never overwrites.
    assert second.json()["result"]["stdout"] == "12 passed"
    assert len(await _rows(pg_db)) == 1


async def test_project_comes_from_the_header_never_the_body(pg_db):
    with _container():
        missing = _post(project=None)
        blank = _post(project=" ")
        in_body = _post(body={**BODY, "project_id": "proj-5"})
    for r in (missing, blank):
        assert r.status_code == 422 and "X-Project-Id" in r.text, r.text
    assert in_body.status_code == 422  # extra="forbid": project_id is not a body field
    assert await _rows(pg_db) == []


async def test_tenant_id_in_body_is_rejected(pg_db):
    with _container():
        r = _post(body={**BODY, "tenant_id": OTHER_TENANT})
    assert r.status_code == 422
    assert await _rows(pg_db) == []


async def test_container_of_another_tenant_is_403(pg_db):
    with _container(owner_tenant=OTHER_TENANT):
        r = _post()
    assert r.status_code == 403, r.text
    assert await _rows(pg_db) == []


async def test_unknown_container_is_404(pg_db):
    with _container(missing=True):
        r = _post()
    assert r.status_code == 404, r.text
    assert await _rows(pg_db) == []


async def test_non_service_account_licence_is_403(pg_db):
    """A verified licence whose subject is NOT allowlisted, using its own tenant."""
    with _container(owner_tenant=SVC_TENANT):
        r = _post(subject=HUMAN_SUBJECT, tenant=SVC_TENANT)
    assert r.status_code == 403, r.text
    assert "SERVICE_ACCOUNT_REQUIRED" in r.text
    assert await _rows(pg_db) == []


async def test_forged_header_identity_is_not_a_service_account(pg_db, monkeypatch):
    """With header trust ON, an x-user-id naming the SA subject is a CLAIM, not a licence."""
    monkeypatch.setenv("TRUST_TENANT_HEADER", "true")
    with _container():
        r = _client().post(
            PATH,
            json=BODY,
            headers={
                "X-User-Id": SVC_SUBJECT,
                "X-Tenant-Id": CUSTOMER_TENANT,
                "X-User-Roles": "developer,admin",
            },
        )
    assert r.status_code == 403, r.text
    assert await _rows(pg_db) == []


async def test_test_id_reused_in_another_container_is_409_without_leaking(pg_db):
    with _container():
        assert _post().status_code == 201
        with _introspection(SVC_SUBJECT):
            r = _client().post(
                "/api/tests/results/other-container/computed",
                json=BODY,
                headers={
                    "Authorization": "Bearer svc-token",
                    "X-Tenant-Id": CUSTOMER_TENANT,
                    "X-Project-Id": "proj-5",
                },
            )
    assert r.status_code == 409, r.text
    assert "12 passed" not in r.text
    assert len(await _rows(pg_db)) == 1


async def _stored_success(pg_db) -> list:
    async with pg_db() as session:
        return list((await session.execute(text("SELECT success FROM test_results"))).scalars())


async def test_omitted_success_is_stored_as_false(pg_db):
    """The generated node omits a false boolean; absence is not a pass claim."""
    body = {k: v for k, v in BODY.items() if k != "success"} | {"exit_code": 1}
    with _container():
        r = _post(body=body)
    assert r.status_code == 201, r.text
    assert r.json()["result"]["success"] is False
    assert await _stored_success(pg_db) == [False]


async def test_explicit_success_true_is_stored_as_true(pg_db):
    with _container():
        r = _post()
    assert r.status_code == 201, r.text
    assert r.json()["result"]["success"] is True
    assert await _stored_success(pg_db) == [True]


async def test_omitted_exit_code_is_still_422(pg_db):
    body = {k: v for k, v in BODY.items() if k != "exit_code"}
    with _container():
        r = _post(body=body)
    assert r.status_code == 422, r.text
    assert await _rows(pg_db) == []


# ── junit_summary: the service derives the outcome fields (B.3, R10-A2 step 3 re-plan) ──────────────
SUMMARY = {
    "tests": 12,
    "failures": 0,
    "errors": 0,
    "skipped": 1,
    "time": 1.2345,
    "failing": [],
    "errored": [],
}
SUMMARY_BODY = {
    "exit_code": 0,
    "test_name": "r3.qa.coding",
    "junit_summary": SUMMARY,
    "card_id": "card-77",
    "commit_sha": "abc123",
}


def _model(**kw):
    from routers.test_results_data import ComputedTestResultRequest

    return ComputedTestResultRequest(**kw)


def test_summary_pass_derives_every_field():
    m = _model(**SUMMARY_BODY)
    assert m.success is True
    assert m.execution_time_ms == 1234.5
    assert m.stdout == "" and m.stderr == ""
    assert m.resource_usage == {"tests": 12, "failures": 0, "errors": 0, "skipped": 1, "passed": 11}
    # The id is derived by the route (tenant + container are not body fields).
    assert m.test_id is None
    tid = _derive(m)
    assert tid.startswith("aa_qa:card-77:") and len(tid.split(":")[2]) == 32


def _derive(m, tenant: str = CUSTOMER_TENANT, container: str = CONTAINER) -> str:
    from routers.test_results_data import derive_computed_test_id

    return derive_computed_test_id(m, tenant_id=tenant, container_name=container)


def test_all_skipped_summary_is_never_a_pass():
    m = _model(**{**SUMMARY_BODY, "junit_summary": {**SUMMARY, "tests": 3, "skipped": 3}})
    assert m.resource_usage["passed"] == 0
    assert m.success is False
    # passed sent explicitly as 0 is the same answer
    m0 = _model(
        **{**SUMMARY_BODY, "junit_summary": {**SUMMARY, "tests": 3, "skipped": 3, "passed": 0}}
    )
    assert m0.success is False


@pytest.mark.parametrize("exit_code", [1, 3, 5, -1])
def test_nonzero_exit_with_a_clean_summary_is_never_a_pass(exit_code):
    assert _model(**{**SUMMARY_BODY, "exit_code": exit_code}).success is False


def test_exit_zero_clean_summary_with_passes_is_a_pass():
    m = _model(**{**SUMMARY_BODY, "exit_code": 0, "junit_summary": {**SUMMARY, "passed": 11}})
    assert m.success is True


@pytest.mark.parametrize(
    "patch_",
    [
        {"tests": 3, "passed": 4},
        {"tests": 3, "failures": 2, "errors": 1, "skipped": 1},
        {"tests": 0, "skipped": 1},
    ],
    ids=["passed>tests", "f+e+s>tests", "skipped-with-zero-tests"],
)
def test_inconsistent_summary_counts_are_rejected(patch_):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="exceeds tests"):
        _model(**{**SUMMARY_BODY, "junit_summary": {**SUMMARY, **patch_}})


@pytest.mark.parametrize(
    "patch_",
    [{"failures": 1}, {"errors": 1}, {"tests": 0, "skipped": 0}],
    ids=["failures", "errors", "zero-tests"],
)
def test_summary_failure_shapes_are_never_a_pass(patch_):
    assert _model(**{**SUMMARY_BODY, "junit_summary": {**SUMMARY, **patch_}}).success is False


def test_summary_text_fields_and_caps():
    long = [f"t::case_{i:04d}" for i in range(2000)]
    m = _model(
        **{
            **SUMMARY_BODY,
            "junit_summary": {
                **SUMMARY,
                "failures": 2,
                "reason": "2 failed",
                "failing": long,
                "errored": long,
            },
        }
    )
    assert m.stdout.startswith("2 failed\nfailing:\nt::case_0000")
    assert m.stdout.endswith("chars]") and "…[truncated " in m.stdout and len(m.stdout) < 4100
    assert m.stderr.startswith("t::case_0000\nt::case_0001") and "…[truncated " in m.stderr


def test_summary_passed_is_taken_when_sent_and_unknown_keys_ignored():
    m = _model(**{**SUMMARY_BODY, "junit_summary": {**SUMMARY, "passed": 7, "future_key": 1}})
    assert m.resource_usage["passed"] == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", True),
        ("execution_time_ms", 1.0),
        ("stdout", "x"),
        ("stderr", ""),
        ("resource_usage", {}),
    ],
)
def test_summary_mixed_with_an_explicit_derived_field_is_rejected(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cannot be combined"):
        _model(**{**SUMMARY_BODY, field: value})


def test_derived_test_id_needs_card_id_and_explicit_form_needs_test_id():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="card_id"):
        _model(**{k: v for k, v in SUMMARY_BODY.items() if k != "card_id"})
    with pytest.raises(ValidationError, match="test_id is required"):
        _model(**{k: v for k, v in BODY.items() if k != "test_id"})


def test_derived_test_id_is_stable_and_outcome_keyed():
    a = _derive(_model(**SUMMARY_BODY))
    reordered = {
        "junit_summary": dict(reversed(list(SUMMARY.items()))),
        **{k: v for k, v in SUMMARY_BODY.items() if k != "junit_summary"},
    }
    assert _derive(_model(**reordered)) == a
    assert _derive(_model(**{**SUMMARY_BODY, "exit_code": 1})) != a
    assert _derive(_model(**{**SUMMARY_BODY, "junit_summary": {**SUMMARY, "failures": 1}})) != a
    assert _derive(_model(**{**SUMMARY_BODY, "commit_sha": "def456"})) != a
    assert _derive(_model(**{**SUMMARY_BODY, "card_id": "card-78"})) != a
    assert _derive(_model(**SUMMARY_BODY), tenant=OTHER_TENANT) != a
    assert _derive(_model(**SUMMARY_BODY), container="other-repo") != a
    assert _model(**{**SUMMARY_BODY, "test_id": "explicit-1"}).test_id == "explicit-1"


async def _derived_row(pg_db) -> list:
    async with pg_db() as session:
        return list(
            (
                await session.execute(
                    text(
                        "SELECT test_id, success, execution_time_ms, stdout, stderr, resource_usage, tenant_id, card_id FROM test_results"
                    )
                )
            ).all()
        )


async def test_summary_only_body_stores_a_derived_row_and_repeats_idempotently(pg_db):
    body = {
        **SUMMARY_BODY,
        "junit_summary": {**SUMMARY, "failures": 1, "reason": "1 failed", "failing": ["t::a"]},
    }
    with _container():
        r = _post(body=body)
    assert r.status_code == 201, r.text
    rows = await _derived_row(pg_db)
    assert len(rows) == 1
    test_id, success, ms, out, err, usage, tenant, card = rows[0]
    assert success is False and ms == 1234.5 and out == "1 failed\nfailing:\nt::a" and err == ""
    assert usage == {"tests": 12, "failures": 1, "errors": 0, "skipped": 1, "passed": 10}
    assert tenant == CUSTOMER_TENANT and card == "card-77" and test_id.startswith("aa_qa:card-77:")
    with _container():
        r2 = _post(body=body)
    assert r2.status_code == 200, r2.text
    assert r2.json()["created"] is False and r2.json()["result"]["test_id"] == test_id
    assert len(await _derived_row(pg_db)) == 1
    with _container():
        r3 = _post(body={**body, "junit_summary": {**SUMMARY}})
    assert r3.status_code == 201, r3.text
    assert len(await _derived_row(pg_db)) == 2


async def test_summary_mixed_body_is_422_and_stores_nothing(pg_db):
    with _container():
        r = _post(body={**SUMMARY_BODY, "success": True})
    assert r.status_code == 422, r.text
    assert await _derived_row(pg_db) == []


async def test_derived_id_is_scoped_per_tenant_and_per_container(pg_db):
    """Same card + outcome in another tenant / another container is its OWN row, never a 409."""
    with _container():
        r1 = _post(body=SUMMARY_BODY)
    assert r1.status_code == 201, r1.text
    with _container(owner_tenant=OTHER_TENANT):
        r2 = _post(body=SUMMARY_BODY, tenant=OTHER_TENANT)
    assert r2.status_code == 201, r2.text
    other_path = PATH.replace(CONTAINER, "card-repo-retry")
    with _container(), _introspection(SVC_SUBJECT):
        r3 = _client().post(
            other_path,
            json=SUMMARY_BODY,
            headers={
                "Authorization": "Bearer svc-token",
                "X-Tenant-Id": CUSTOMER_TENANT,
                "X-Project-Id": "proj-5",
            },
        )
    assert r3.status_code == 201, r3.text
    ids = {r.json()["result"]["test_id"] for r in (r1, r2, r3)}
    assert len(ids) == 3
    rows = await _rows(pg_db)
    assert sorted((t, c) for _, t, *_rest, c in rows) == sorted(
        [
            (CUSTOMER_TENANT, CONTAINER),
            (OTHER_TENANT, CONTAINER),
            (CUSTOMER_TENANT, "card-repo-retry"),
        ]
    )


async def test_derived_id_identical_repeat_in_same_scope_collapses(pg_db):
    with _container():
        r1 = _post(body=SUMMARY_BODY)
        r2 = _post(body=SUMMARY_BODY)
    assert r1.status_code == 201, r1.text
    assert r2.status_code == 200, r2.text
    assert r2.json()["created"] is False
    assert r2.json()["result"]["test_id"] == r1.json()["result"]["test_id"]
    assert len(await _rows(pg_db)) == 1
