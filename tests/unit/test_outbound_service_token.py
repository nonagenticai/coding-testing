"""Every outbound coding-* sibling call must carry this service's own licence.

Today those calls send only a bare ``X-Tenant-Id`` — and one of them sent nothing
at all. A bare tenant header is not a credential: it is accepted only because the
callees still trust it. As each callee flips to ``TRUST_TENANT_HEADER=false``,
these calls 401, and every single one of them is wrapped in a broad ``except``
that turns the refusal into a plausible-looking answer:

* container validation -> ``404 Container not found``
* quality analysis     -> ``QualityResult(success=False)``, i.e. "you scored 0"
* security validation  -> ``is_safe=True`` (fail-open!)

So each test here asserts BOTH halves that must hold on the wire:

1. ``Authorization: Bearer <CODING_SERVICE_TOKEN>`` — who *we* are.
2. ``X-Tenant-Id: <customer>`` — who we are acting *for*, unchanged.

Asserting (2) is not redundant. The tempting shortcut is to put the credential
into ``X-Tenant-Id``, which collapses every customer into coding-testing's own
tenant because the callee then resolves the tenant from the token. A test that
only checked "some credential went out" passes on exactly that bug.

The negative case matters as much: with no token configured there must be NO
``Authorization`` header at all. A bare ``Bearer `` is a malformed credential,
not "no credential", and it burns an introspection round-trip on the fleet's
hottest auth path.

The ``*-test-executor`` services are deliberately NOT covered: they are not
coding-* siblings and are not licence-gated, so they keep the bare tenant header
``_executor_headers`` has always sent them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

_TOKEN = "svc-token-abc"  # noqa: S105
_TENANT = "0195f48b-d000-7000-8000-000000000003"


@pytest.fixture(autouse=True)
def _no_ambient_service_token(monkeypatch):
    """Delete the token rather than assume it is absent.

    A developer (or a runner) with the real ``CODING_SERVICE_TOKEN`` exported
    would otherwise make every "no token => no header" test pass for the wrong
    reason. Each positive test sets it itself.
    """
    monkeypatch.delenv("CODING_SERVICE_TOKEN", raising=False)


def _recording_client(seen: list[httpx.Request], response: httpx.Response):
    """A factory that yields a real httpx.AsyncClient over a MockTransport.

    Real client, real header assembly, real URL assembly — only the transport is
    faked, so neither a header nor a URL the production code fails to build can
    be supplied by the double.

    Every kwarg the caller passes is forwarded and only ``transport`` is
    overridden. That matters since container_validation moved to
    ``CodingWorkspacesClient``: the SDK sets ``base_url`` on the client and then
    requests a RELATIVE path, so a factory that swallowed ``base_url`` left
    httpx with a bare "/api/workspaces/repa" and raised
    ``ValueError: unknown url type``. Forwarding also means the recorded
    request carries the fully-joined absolute URL, so a test can assert on it.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    # Bind the REAL class now: the factory replaces ``httpx.AsyncClient``
    # module-wide for the duration of the patch, so looking it up by name inside
    # the factory would call the factory again.
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return factory


# ---------------------------------------------------------------------------
# src/auth/outbound.py — the SDK re-export itself
# ---------------------------------------------------------------------------


class TestOutboundModuleIsTheSdkImplementation:
    """The local module must BE the SDK's, not a second copy of it.

    A local re-implementation would compile, pass its own tests, and drift — the
    exact failure the SDK promotion exists to end. Identity (``is``) is therefore
    the assertion, not behavioural equivalence.
    """

    def test_the_helpers_are_the_sdk_objects_not_local_copies(self):
        from coding_client.auth import outbound as sdk

        from auth import outbound

        assert outbound.tenant_headers is sdk.tenant_headers
        assert outbound.optional_tenant_headers is sdk.optional_tenant_headers
        assert outbound.TenantMissingError is sdk.TenantMissingError
        assert outbound.TENANT_ID_HEADER is sdk.TENANT_ID_HEADER

    def test_the_credential_never_lands_in_the_tenant_header(self):
        from auth.outbound import optional_tenant_headers, tenant_headers

        scoped = tenant_headers("t1", operation="probe", service_token=_TOKEN)
        assert scoped == {"X-Tenant-Id": "t1", "Authorization": f"Bearer {_TOKEN}"}

        # No tenant to name, but still able to prove who is calling.
        assert optional_tenant_headers(None, service_token=_TOKEN) == {
            "Authorization": f"Bearer {_TOKEN}"
        }

    def test_a_missing_tenant_is_an_error_never_a_default(self):
        from auth.outbound import TenantMissingError, tenant_headers

        with pytest.raises(TenantMissingError):
            tenant_headers(None, operation="probe", service_token=_TOKEN)
        with pytest.raises(TenantMissingError):
            tenant_headers("", operation="probe", service_token=_TOKEN)

    def test_service_token_is_read_from_the_environment_per_call(self, monkeypatch):
        """Per call, so a rotated secret needs no pod restart."""
        from auth.outbound import service_token

        assert service_token() is None
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        assert service_token() == _TOKEN
        monkeypatch.setenv("CODING_SERVICE_TOKEN", "")
        assert service_token() is None


# ---------------------------------------------------------------------------
# src/shared/container_validation.py -> coding-containers
# ---------------------------------------------------------------------------


class TestContainerValidationSendsTheLicence:
    @staticmethod
    def _ok() -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "container": {
                    "container_name": "repa",
                    "status": "running",
                    "codebase_path": "/workspace",
                }
            },
        )

    async def _call(self, tenant_id: str | None) -> httpx.Request:
        import shared.container_validation as mod

        seen: list[httpx.Request] = []
        with patch.object(mod.httpx, "AsyncClient", _recording_client(seen, self._ok())):
            await mod.validate_and_get_container_for_tenant("repa", tenant_id=tenant_id)
        assert len(seen) == 1
        return seen[0]

    async def test_licence_and_customer_tenant_both_go_out(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        request = await self._call(_TENANT)

        # The SDK owns URL assembly now (base_url + relative path). Pin the
        # joined result: a client pointed at the wrong host, or a path that
        # silently lost its prefix, is invisible in a headers-only assertion.
        assert str(request.url).endswith("/api/workspaces/repa")
        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        # coding-containers resolves the container against the asserted tenant.
        # A token in this slot would look up every customer's container under
        # coding-testing's own tenant — and this function turns "not found"
        # into a 404 the caller reads as "your container does not exist".
        assert request.headers["x-tenant-id"] == _TENANT

    async def test_no_token_means_no_authorization_header_at_all(self):
        request = await self._call(_TENANT)

        assert "authorization" not in request.headers
        assert request.headers["x-tenant-id"] == _TENANT

    async def test_the_legacy_untenanted_path_still_proves_who_is_calling(self, monkeypatch):
        """``validate_and_get_container()`` deliberately passes ``tenant_id=None``.

        Absent stays absent — no fabricated tenant — but the credential still
        goes out: proving who we are is not a claim about whose data we want.
        """
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        request = await self._call(None)

        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        assert "x-tenant-id" not in request.headers


# ---------------------------------------------------------------------------
# src/services/code_execution_service.py -> coding-analysis / coding-security
# ---------------------------------------------------------------------------


class TestQualityAnalysisSendsTheLicence:
    """``analyze_quality`` / ``analyze_container_quality`` -> coding-analysis."""

    @staticmethod
    def _ok() -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "analysis_id": "qa-1",
                "success": True,
                "overall_score": 85.0,
                "total_issues": 0,
                "issues": [],
                "tools_results": {},
                "execution_time": 0.5,
                "language": "python",
            },
        )

    @staticmethod
    def _request(**overrides: Any) -> SimpleNamespace:
        base = {
            # whole-workspace shape: inline code without code_path is a 422 now
            "code_content": None,
            "code": None,
            "language": "python",
            "tools": None,
            "container_name": "repa",
            "fix_issues": False,
            "custom_rules": None,
            "tenant_id": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    async def _call(self, method: str, tenant_id: str | None) -> httpx.Request:
        import services.code_execution_service as mod

        seen: list[httpx.Request] = []
        svc = mod.CodeExecutionService()
        with patch.object(mod.httpx, "AsyncClient", _recording_client(seen, self._ok())):
            await getattr(svc, method)(self._request(), tenant_id=tenant_id)
        assert len(seen) == 1
        return seen[0]

    @pytest.mark.parametrize("method", ["analyze_quality", "analyze_container_quality"])
    async def test_licence_and_customer_tenant_both_go_out(self, method, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        request = await self._call(method, _TENANT)

        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        assert request.headers["x-tenant-id"] == _TENANT

    @pytest.mark.parametrize("method", ["analyze_quality", "analyze_container_quality"])
    async def test_no_token_means_no_authorization_header_at_all(self, method):
        request = await self._call(method, _TENANT)

        assert "authorization" not in request.headers
        assert request.headers["x-tenant-id"] == _TENANT

    async def test_the_tenant_falls_back_exactly_as_the_executor_path_does(self, monkeypatch):
        """Adding the credential must not change the tenant policy.

        ``_analysis_headers`` delegates resolution to ``_executor_headers`` so
        there is one policy and one failure mode; conftest pins the default to
        ``test-default-tenant``.
        """
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        request = await self._call("analyze_quality", None)

        assert request.headers["x-tenant-id"] == "test-default-tenant"
        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"

    async def test_no_tenant_anywhere_still_fails_loud(self, monkeypatch):
        """A credential is not a substitute for a tenant — the loud failure stays."""
        import services.code_execution_service as mod

        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        monkeypatch.setattr(mod, "DEFAULT_TENANT_ID", None)
        with pytest.raises(RuntimeError, match="DEFAULT_TENANT_ID is unset"):
            mod._analysis_headers(None)


class TestSecurityValidationSendsTheLicence:
    """``_validate_code_security`` -> coding-security.

    This one sent no identity headers at all. Its failure mode is the worst of
    the three: a 401 lands in ``except httpx.HTTPStatusError`` and the code is
    reported unsafe, while an unreachable service fails OPEN.
    """

    @staticmethod
    def _ok() -> httpx.Response:
        return httpx.Response(200, json={"is_safe": True, "violations": [], "risk_score": 0})

    async def _call(self) -> httpx.Request:
        import services.code_execution_service as mod

        seen: list[httpx.Request] = []
        client = _recording_client(seen, self._ok())()
        with patch.object(mod, "_get_security_http_client", return_value=client):
            await mod.CodeExecutionService()._validate_code_security("print(1)", "python")
        await client.aclose()
        assert len(seen) == 1
        return seen[0]

    async def test_the_licence_goes_out(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        request = await self._call()

        assert request.headers.get("authorization") == f"Bearer {_TOKEN}"
        # No tenant is asserted: the security verdict is a property of the code,
        # not of a customer's data. Absent, never fabricated.
        assert "x-tenant-id" not in request.headers

    async def test_no_token_means_no_authorization_header_at_all(self):
        request = await self._call()

        assert "authorization" not in request.headers


# ---------------------------------------------------------------------------
# src/processing/test_execution/http_test_runner.py -> coding-analysis
# ---------------------------------------------------------------------------


class TestHttpRunnerQualityAnalysisSendsIdentity:
    """The call that sent NOTHING while its sibling two methods up sent a tenant.

    ``HTTPTestRunner.execute_test`` has resolved and sent ``X-Tenant-Id`` for
    every request for a long time. ``HTTPTestRunner.analyze_quality`` — same
    class, same session, a licence-gated coding-* sibling on the other end — sent
    only ``Content-Type``. It reached coding-analysis solely via the trusted
    pod-CIDR fallback, which is a network position and not a credential.
    """

    @staticmethod
    def _session_capturing(captured: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={
                "analysis_id": "qa-1",
                "success": True,
                "language": "python",
                "tools_results": {},
                "overall_score": 90.0,
                "total_issues": 0,
                "execution_time": 0.1,
            }
        )
        session = MagicMock()

        def _post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session.post = _post
        return session

    async def _call(self, tenant_id: str | None = _TENANT) -> dict[str, Any]:
        from models.testing_models import QualityRequest
        from processing.test_execution.http_test_runner import HTTPTestRunner

        runner = HTTPTestRunner()
        captured: dict[str, Any] = {}
        runner._get_session = AsyncMock(return_value=self._session_capturing(captured))
        await runner.analyze_quality(
            QualityRequest(container_name="repa", code_content="x = 1", language="python"),
            tenant_id=tenant_id,
        )
        return captured

    async def test_licence_and_customer_tenant_both_go_out(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        captured = await self._call()

        assert captured["headers"].get("Authorization") == f"Bearer {_TOKEN}"
        assert captured["headers"]["X-Tenant-Id"] == _TENANT
        # Content-Type is not collateral damage of adding the identity headers.
        assert captured["headers"]["Content-Type"] == "application/json"

    async def test_no_token_still_sends_the_tenant(self):
        captured = await self._call()

        assert "Authorization" not in captured["headers"]
        assert captured["headers"]["X-Tenant-Id"] == _TENANT

    async def test_the_tenant_falls_back_like_execute_test_does(self, monkeypatch):
        monkeypatch.setenv("CODING_SERVICE_TOKEN", _TOKEN)
        captured = await self._call(tenant_id=None)

        assert captured["headers"]["X-Tenant-Id"] == "test-default-tenant"

    async def test_no_tenant_anywhere_fails_loud_rather_than_sending_an_empty_header(
        self, monkeypatch
    ):
        import processing.test_execution.http_test_runner as mod
        from models.testing_models import QualityRequest

        monkeypatch.setattr(mod, "DEFAULT_TENANT_ID", None)
        runner = mod.HTTPTestRunner()
        runner._get_session = AsyncMock(return_value=self._session_capturing({}))
        with pytest.raises(RuntimeError, match="DEFAULT_TENANT_ID is unset"):
            await runner.analyze_quality(
                QualityRequest(container_name="repa", code_content="x = 1", language="python")
            )
