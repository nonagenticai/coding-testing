"""coding-containers exec transport shared by the workspace-run services.

One implementation of the three exec operations (sync exec, async exec submit,
async job poll) used by the workspace test runner (A2), the OpenAPI fetcher (A5b)
and the behaviour-parity runner (A5c). Each of those keeps its own error mapping
and file-staging helpers; only the wire transport is shared.
"""

from __future__ import annotations

from typing import Any, Protocol

from coding_client.clients import CodingWorkspacesClient
from fastapi import HTTPException

from auth.outbound import optional_tenant_headers, service_token
from config.settings import get_settings


class WorkspaceTransport(Protocol):
    """The three coding-containers exec operations a workspace run needs."""

    async def exec(
        self, container: str, tenant_id: str | None, argv: list[str], workdir: str, timeout: int
    ) -> Any: ...

    async def exec_async(
        self, container: str, tenant_id: str | None, argv: list[str], workdir: str, timeout: int
    ) -> Any: ...

    async def get_exec_job(self, container: str, tenant_id: str | None, job_id: str) -> Any: ...


class ContainersTransport:
    """coding-containers over coding-client.

    ``exec_async`` / ``get_exec_job`` go through the generic ``post``/``get``
    because coding-client 1.19.x has no typed methods for them yet
    (coding-client#65). The async submit uses ``max_retries=1``: a retried POST
    after a read timeout would start a SECOND job (idempotency there is keyed
    on ``callback_url``, which this caller does not send).
    """

    def _client(self, timeout: float, max_retries: int = 3) -> CodingWorkspacesClient:
        url = get_settings().CONTAINERS_SERVICE_URL
        if not url:
            raise HTTPException(status_code=503, detail="CONTAINERS_SERVICE_URL is not configured")
        return CodingWorkspacesClient(base_url=url, timeout=timeout, max_retries=max_retries)

    @staticmethod
    def _headers(tenant_id: str | None) -> dict[str, str]:
        return optional_tenant_headers(tenant_id, service_token=service_token())

    async def exec(
        self, container: str, tenant_id: str | None, argv: list[str], workdir: str, timeout: int
    ) -> Any:
        client = self._client(float(timeout) + 10.0)
        try:
            return await client.exec(
                container,
                headers=self._headers(tenant_id),
                command=argv,
                workdir=workdir,
                timeout=timeout,
            )
        finally:
            await client.close()

    async def exec_async(
        self, container: str, tenant_id: str | None, argv: list[str], workdir: str, timeout: int
    ) -> Any:
        client = self._client(30.0, max_retries=1)
        try:
            return await client.post(
                f"/api/workspaces/{container}/exec/async",
                headers=self._headers(tenant_id),
                json={"command": argv, "workdir": workdir, "timeout": timeout},
            )
        finally:
            await client.close()

    async def get_exec_job(self, container: str, tenant_id: str | None, job_id: str) -> Any:
        client = self._client(30.0)
        try:
            return await client.get(
                f"/api/workspaces/{container}/exec/jobs/{job_id}",
                headers=self._headers(tenant_id),
            )
        finally:
            await client.close()
