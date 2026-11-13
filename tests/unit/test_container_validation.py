"""Unit tests for container validation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from coding_client.clients import ServiceError
from fastapi import HTTPException


def _patch_workspaces_client(*, result=None, error=None):
    """Stand in for CodingWorkspacesClient as used by container_validation.

    The production code constructs the client, calls ``get_workspace``, and
    closes it in a finally. Faking at that seam keeps these tests about the
    translation from SDK outcomes to HTTP semantics rather than about httpx.

    Returns the (patcher, fake_client) pair; the fake exposes ``get_workspace``
    so call sites can assert on the headers that were sent, and ``close`` so
    they can assert the connection is not leaked.
    """
    fake = AsyncMock()
    if error is not None:
        fake.get_workspace = AsyncMock(side_effect=error)
    else:
        fake.get_workspace = AsyncMock(return_value=result)
    ctor = patch("shared.container_validation.CodingWorkspacesClient", return_value=fake)
    return ctor, fake


class TestContainerValidation:
    """Test validate_and_get_container_for_tenant.

    The status-code matrix below is the contract this call site had before it
    moved onto the typed SDK client, and it is pinned here deliberately: this is
    a validator, so "service returned 500" and "service is down" must both keep
    answering *invalid* (HTTPException 404) rather than escaping to the caller,
    and a real 404 must never be admitted as a valid container.
    """

    async def test_successful_validation(self):
        """200 with a JSON body -> parsed ContainerInfo."""
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, fake = _patch_workspaces_client(
            result={
                "container_name": "my-container",
                "status": "running",
                "codebase_path": "/code",
                "tenant_id": "n8n-dev",
            }
        )
        with ctor:
            result = await validate_and_get_container_for_tenant("my-container", "n8n-dev")
        assert result.container_name == "my-container"
        assert result.status.value == "running"
        assert result.codebase_path == "/code"
        assert result.tenant_id == "n8n-dev"
        fake.close.assert_awaited_once()

    async def test_successful_validation_unwraps_container_key(self):
        """200 whose payload nests the record under "container" is unwrapped."""
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, _ = _patch_workspaces_client(
            result={
                "container": {
                    "container_name": "nested",
                    "status": "stopped",
                    "codebase_path": "/src",
                    "tenant_id": "t9",
                }
            }
        )
        with ctor:
            result = await validate_and_get_container_for_tenant("nested", "t9")
        assert result.container_name == "nested"
        assert result.status.value == "stopped"
        assert result.codebase_path == "/src"

    async def test_container_not_found(self):
        """404 from coding-containers -> HTTPException 404 (invalid)."""
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, _ = _patch_workspaces_client(
            error=ServiceError(404, "GET /api/workspaces/missing returned 404", body=None)
        )
        with ctor, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("missing", "n8n-dev")
        assert exc_info.value.status_code == 404
        assert "missing" in exc_info.value.detail

    @pytest.mark.parametrize("status_code", [400, 401, 403, 409, 422, 500, 502, 503])
    async def test_other_error_statuses_raise_404(self, status_code):
        """Any other non-2xx -> HTTPException 404, never the upstream status.

        The pre-SDK code funnelled these through ``raise_for_status()`` into the
        same ``except httpx.HTTPError`` arm that produced a 404, so leaking a 500
        or a 401 to the caller here would be a regression.
        """
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, _ = _patch_workspaces_client(
            error=ServiceError(status_code, f"returned {status_code}", body={"detail": "nope"})
        )
        with ctor, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("container", "n8n-dev")
        assert exc_info.value.status_code == 404

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("Connection refused"),
            httpx.ReadTimeout("timed out"),
            httpx.PoolTimeout("pool exhausted"),
            httpx.RemoteProtocolError("bad framing"),
        ],
    )
    async def test_network_failure_raises_404(self, exc):
        """Service down / unreachable -> HTTPException 404, not a raw httpx error.

        The SDK retries the first three of these three times and then re-raises
        the original exception, so what reaches this function is still an
        ``httpx.HTTPError`` and must be translated, not propagated. A blip that
        escaped as an unhandled exception would 500 every test run.
        """
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, fake = _patch_workspaces_client(error=exc)
        with ctor, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("container", "n8n-dev")
        assert exc_info.value.status_code == 404
        # An outage must not also leak the client's connection pool.
        fake.close.assert_awaited_once()

    @pytest.mark.parametrize("payload", [None, "", "not json at all", []])
    async def test_unusable_payload_raises_404(self, payload):
        """204/empty/non-JSON bodies are rejected, not dereferenced.

        The SDK degrades an empty body to ``None`` and an unparseable one to raw
        text instead of raising, so without an explicit guard these would reach
        ``.get()`` and surface to the caller as an AttributeError 500.
        """
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, _ = _patch_workspaces_client(result=payload)
        with ctor, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("container", "n8n-dev")
        assert exc_info.value.status_code == 404

    async def test_missing_service_url_raises_503(self):
        """An unconfigured CONTAINERS_SERVICE_URL stays a 503, not a 404.

        "We are misconfigured" is a different answer from "that container does
        not exist", and the check must run before the client is constructed —
        the SDK would otherwise raise RuntimeError from its own env fallback.
        """
        from shared import container_validation

        unconfigured = SimpleNamespace(CONTAINERS_SERVICE_URL="")
        # Also patch the constructor: if the guard ever moved below it, the SDK
        # would raise RuntimeError here and this test must fail loudly rather
        # than pass on the wrong exception.
        with (
            patch.object(container_validation, "get_settings", return_value=unconfigured),
            patch.object(
                container_validation,
                "CodingWorkspacesClient",
                side_effect=AssertionError("client built despite unconfigured URL"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await container_validation.validate_and_get_container_for_tenant("c", "t")
        assert exc_info.value.status_code == 503

    async def test_sends_tenant_header(self):
        """The CUSTOMER tenant travels in X-Tenant-Id via the client's headers=."""
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, fake = _patch_workspaces_client(
            result={"container_name": "c1", "status": "running", "tenant_id": "t1"}
        )
        with ctor:
            await validate_and_get_container_for_tenant("c1", "t1")

        fake.get_workspace.assert_called_once()
        call = fake.get_workspace.call_args
        assert call.args[0] == "c1"
        assert call.kwargs["headers"]["X-Tenant-Id"] == "t1"

    async def test_omits_tenant_header_when_unscoped(self):
        """tenant_id=None sends no X-Tenant-Id — absent stays absent.

        ``optional_tenant_headers`` must not fabricate a default; the legacy
        un-scoped path depends on the header simply not being there.
        """
        from shared.container_validation import validate_and_get_container_for_tenant

        ctor, fake = _patch_workspaces_client(result={"container_name": "c1", "status": "running"})
        with ctor:
            await validate_and_get_container_for_tenant("c1", None)

        headers = fake.get_workspace.call_args.kwargs["headers"]
        assert "X-Tenant-Id" not in headers

    async def test_client_built_from_settings_url(self):
        """base_url comes from settings, not the SDK's CODING_CONTAINERS_URL env."""
        from shared import container_validation

        fake = AsyncMock()
        fake.get_workspace = AsyncMock(return_value={"container_name": "c1", "status": "running"})
        configured = SimpleNamespace(CONTAINERS_SERVICE_URL="http://coding-containers:8691")
        with (
            patch.object(container_validation, "get_settings", return_value=configured),
            patch.object(container_validation, "CodingWorkspacesClient", return_value=fake) as ctor,
        ):
            await container_validation.validate_and_get_container_for_tenant("c1", "t1")

        assert ctor.call_args.kwargs["base_url"] == "http://coding-containers:8691"
        # The old hand-rolled call used a 10s budget; the SDK default is 30s.
        assert ctor.call_args.kwargs["timeout"] == 10.0


class TestContainerInfoDataclass:
    """Test ContainerInfo dataclass."""

    def test_default_values(self):
        """Test ContainerInfo default values."""
        from shared.container_validation import ContainerInfo, ContainerStatus

        info = ContainerInfo(container_name="test")
        assert info.container_name == "test"
        assert info.status == ContainerStatus.RUNNING
        assert info.codebase_path == ""
        assert info.tenant_id == ""
        assert info.pr_number is None


class TestContainerStatus:
    """Test ContainerStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        from shared.container_validation import ContainerStatus

        assert ContainerStatus.CREATING == "creating"
        assert ContainerStatus.RUNNING == "running"
        assert ContainerStatus.STOPPED == "stopped"
        assert ContainerStatus.ERROR == "error"
        assert ContainerStatus.REMOVING == "removing"


class TestLegacyValidation:
    """Test legacy validate_and_get_container function."""

    async def test_legacy_calls_tenant_version(self):
        """Test legacy function delegates to tenant-aware version."""
        from shared.container_validation import validate_and_get_container

        with patch(
            "shared.container_validation.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = AsyncMock()
            await validate_and_get_container("my-container")
            mock.assert_called_once_with("my-container", tenant_id=None)
