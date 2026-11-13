"""Container validation - calls coding-containers service."""

import logging
from dataclasses import dataclass
from enum import StrEnum

import httpx
from coding_client.clients import CodingWorkspacesClient, ServiceError
from fastapi import HTTPException

from auth.outbound import optional_tenant_headers, service_token
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ContainerStatus(StrEnum):
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    REMOVING = "removing"


@dataclass
class ContainerInfo:
    container_name: str
    status: ContainerStatus = ContainerStatus.RUNNING
    codebase_path: str = ""
    repo_url: str = ""
    branch: str = ""
    api_endpoint: str = ""
    internal_endpoint: str = ""
    public_endpoint: str = ""
    description: str = ""
    pr_number: int | None = None
    provision_type: str = ""
    tenant_id: str = ""

    def __post_init__(self) -> None:
        if not self.public_endpoint:
            self.public_endpoint = self.api_endpoint
        if not self.api_endpoint:
            self.api_endpoint = self.public_endpoint or self.internal_endpoint


async def validate_and_get_container_for_tenant(
    container_name: str,
    tenant_id: str | None = None,
) -> ContainerInfo:
    """Validate container exists and belongs to tenant via coding-containers service."""
    containers_url = get_settings().CONTAINERS_SERVICE_URL
    if not containers_url:
        raise HTTPException(
            status_code=503,
            detail="CONTAINERS_SERVICE_URL is not configured",
        )
    # X-Tenant-Id is the CUSTOMER's; Authorization is this service's own
    # licence. coding-containers resolves the container against the asserted
    # tenant and, once it stops trusting a bare header, will only honour that
    # assertion when coding-testing's subject is in its CODING_ONBEHALF_SUBJECTS.
    # The two headers must stay separate: a token in X-Tenant-Id would resolve
    # every lookup against coding-testing's own tenant, i.e. answer "not found"
    # for every real customer container — which this function turns into a 404.
    #
    # ``optional_`` rather than ``tenant_headers`` because
    # validate_and_get_container() deliberately passes tenant_id=None (the legacy
    # un-scoped path); refusing that call would be a regression. Absent stays
    # absent — never a fabricated default.
    headers = optional_tenant_headers(tenant_id, service_token=service_token())

    # The typed SDK client replaces a hand-rolled GET whose URL was assembled
    # here. It does NOT behave like raw httpx and the difference is the whole
    # risk of this call site, so the failure translation below is exhaustive:
    #
    #   * >=400 -> ServiceError(status_code=...), never a returned response.
    #   * ConnectError / ReadTimeout / PoolTimeout -> retried 3x, then re-raised
    #     as the original httpx exception.
    #   * success -> parsed JSON (or None on 204 / empty body).
    #
    # This is a VALIDATOR: every downstream failure — missing container, 500,
    # or the service being flat-out down — must answer "invalid" (HTTP 404) the
    # way the httpx version did. An outage must not become an exception that
    # escapes to callers, and a 404 must never become a false "valid".
    #
    # try/finally rather than ``async with``: ServiceClient.__aenter__ is
    # annotated ``-> ServiceClient``, so the bound name would narrow to the base
    # class and ``ty`` would reject get_workspace as unknown. The finally clause
    # gives the same one-client-per-call lifetime the old ``httpx.AsyncClient``
    # context manager had — the pooled connection is closed rather than leaking
    # for the life of the process. timeout=10.0 pins the previous budget instead
    # of inheriting the SDK's 30.0 default.
    client = CodingWorkspacesClient(base_url=containers_url, timeout=10.0)
    try:
        data = await client.get_workspace(container_name, headers=headers)
    except ServiceError as e:
        # Both arms answer 404, exactly as the pre-SDK code did — a 404 and a 500
        # are equally "not a container you may run tests in". They are split only
        # so the log distinguishes "no such container" from "coding-containers is
        # broken", which are very different pages to wake someone for. Do NOT
        # start forwarding e.status_code to the caller: a transient upstream 500
        # would then surface as a 500 from every endpoint that validates first.
        if e.status_code == 404:
            logger.warning(f"Container '{container_name}' not found in coding-containers")
        else:
            logger.warning(
                f"Container validation failed for '{container_name}': "
                f"coding-containers returned {e.status_code} (body={e.body!r})"
            )
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")
    except httpx.HTTPError as e:
        # Transport-level: connect refused, timeouts (already retried by the
        # SDK), protocol errors. The old code caught the same base class here.
        logger.warning(f"Container validation failed for '{container_name}': {e}")
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")
    finally:
        await client.close()

    if not isinstance(data, dict):
        # 204 / empty / non-JSON body. The SDK degrades these to None or to raw
        # text instead of raising, so an unusable payload has to be rejected
        # explicitly — otherwise it would reach ``.get()`` and blow up with an
        # AttributeError that callers see as a 500.
        logger.warning(
            f"Container validation failed for '{container_name}': "
            f"unusable response payload of type {type(data).__name__}"
        )
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")

    # Response wraps container data under "container" key
    container = data.get("container", data)
    return ContainerInfo(
        container_name=container.get("container_name", container_name),
        status=ContainerStatus(container.get("status", "running")),
        codebase_path=container.get("codebase_path", ""),
        api_endpoint=container.get("api_endpoint", ""),
        internal_endpoint=container.get("internal_endpoint", ""),
        public_endpoint=container.get("public_endpoint", ""),
        tenant_id=container.get("tenant_id", tenant_id or ""),
    )


async def validate_and_get_container(container_name: str) -> ContainerInfo:
    """Legacy validation without tenant filtering."""
    return await validate_and_get_container_for_tenant(container_name, tenant_id=None)
