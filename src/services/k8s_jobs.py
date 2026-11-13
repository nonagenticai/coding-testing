"""One throwaway Kubernetes Job per Rust test run (plain httpx against the API server).

The Job runs the Rust runner image's ``rust-repo-run`` with::

    ARCHIVE_URL    {TESTING_INTERNAL_URL}/internal/test-runs/{run_id}/archive
    ARCHIVE_TOKEN  the run's one-time token (sent back as the X-Run-Token header)
    PRESETS        comma-separated: test,clippy,fmt,audit,deny,coverage
    WORKDIR        checkout-relative crate directory ("" = the root)
    TIMEOUT_S      the run's budget in seconds
    DATABASE_URL   with_postgres only: the postgres sidecar on 127.0.0.1:5432
    NEXTEST_EXTRA_ARGS  the request's extra_args, a JSON array of strings, appended
                   to the cargo nextest argv (absent when there are none)

and prints one ``RESULT_JSON:<document>`` line (``services.rust_test_runs`` maps it).
This module creates the Job, polls its status, reads the pod's log and deletes the
Job.

In-cluster access uses the pod's ServiceAccount token (re-read per request: it is a
rotating projected token) and the cluster CA. RBAC needed in ``TEST_RUNS_NAMESPACE``
(gitops services/coding-test-runs/rbac.yaml): ``jobs`` create/get/delete, ``pods``
list, ``pods/log`` get.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
RESULT_PREFIX = "RESULT_JSON:"
LOG_TAIL_LINES = 200
#: Seconds between Job status reads.
POLL_INTERVAL_S = 5.0
#: activeDeadlineSeconds = timeout + this (setup: a cold image pull on a CI node,
#: the archive download).
JOB_DEADLINE_SLACK_S = 300
#: How long past the Job's own deadline this service keeps waiting before giving up.
WAIT_SLACK_S = 120
#: A finished Job (and its result log) is kept this long PAST the run's own poll
#: deadline (created + timeout + JOB_DEADLINE_SLACK_S + WAIT_SLACK_S + 60, see
#: rust_test_runs.GRACE_S), so a poll from a replica that did not run it can still
#: read the result until then.
JOB_TTL_S = 600
RUNNER_COMMAND = ["rust-repo-run"]

#: Placement: the CI nodes (cargo build load stays off the serving / stateful nodes).
#: Same label + taint as the ARC runner pools (gitops services/arc-runners).
NODE_SELECTOR: dict[str, str] = {"node-role/ci-runner": "true"}
TOLERATIONS: list[dict[str, str]] = [
    {"key": "ci-runner", "operator": "Equal", "value": "true", "effect": "NoSchedule"}
]
#: with_postgres: a postgres:18 native sidecar, trust auth on 127.0.0.1 only (the pod's
#: own loopback); the runner starts once its startupProbe passes.
PG_PORT = 5432
PG_URL = f"postgres://postgres@127.0.0.1:{PG_PORT}/postgres"
PG_UID = 999


def postgres_sidecar(image: str) -> dict[str, Any]:
    """Native sidecar (initContainers + restartPolicy Always), restricted-PSA compliant."""
    return {
        "name": "postgres",
        "image": image,
        "restartPolicy": "Always",
        "args": ["-c", "listen_addresses=127.0.0.1"],
        "env": [
            {"name": "POSTGRES_HOST_AUTH_METHOD", "value": "trust"},
            {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"},
        ],
        "volumeMounts": [{"name": "pgdata", "mountPath": "/var/lib/postgresql/data"}],
        "startupProbe": {
            "exec": {
                "command": ["pg_isready", "-h", "127.0.0.1", "-p", str(PG_PORT), "-U", "postgres"]
            },
            "periodSeconds": 2,
            "failureThreshold": 60,
        },
        "resources": {
            "requests": {"cpu": "100m", "memory": "256Mi", "ephemeral-storage": "256Mi"},
            "limits": {"cpu": "1", "memory": "1Gi", "ephemeral-storage": "2Gi"},
        },
        "securityContext": {
            "runAsUser": PG_UID,
            "runAsGroup": PG_UID,
            "runAsNonRoot": True,
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
    }


class KubeError(RuntimeError):
    """The API server refused or could not be reached, or the Job gave no result.

    ``status``: the API server's HTTP status when it answered one (404: not found).
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def job_name(run_id: str) -> str:
    return f"rust-run-{run_id}"


def build_job_spec(
    *,
    run_id: str,
    namespace: str,
    image: str,
    archive_url: str,
    archive_token: str,
    presets: list[str],
    workdir: str,
    timeout_s: int,
    resources: dict[str, dict[str, str]],
    postgres_image: str | None = None,
    nextest_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """The batch/v1 Job of one Rust run (pure: no I/O). ``postgres_image``: with_postgres."""
    labels = {
        "app.kubernetes.io/name": "rust-repo-run",
        "app.kubernetes.io/managed-by": "coding-testing",
        "coding-testing/run-id": run_id,
    }
    env = {
        "ARCHIVE_URL": archive_url,
        "ARCHIVE_TOKEN": archive_token,
        "PRESETS": ",".join(presets),
        "WORKDIR": workdir,
        "TIMEOUT_S": str(timeout_s),
    }
    if postgres_image:
        env["DATABASE_URL"] = PG_URL
    if nextest_extra_args:
        env["NEXTEST_EXTRA_ARGS"] = json.dumps(nextest_extra_args)
    pod: dict[str, Any] = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "nodeSelector": NODE_SELECTOR,
        "tolerations": TOLERATIONS,
        "securityContext": {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "runner",
                "image": image,
                "command": RUNNER_COMMAND,
                "env": [{"name": k, "value": v} for k, v in env.items()],
                "resources": resources,
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
            }
        ],
    }
    if postgres_image:
        pod["initContainers"] = [postgres_sidecar(postgres_image)]
        pod["volumes"] = [{"name": "pgdata", "emptyDir": {"sizeLimit": "2Gi"}}]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name(run_id), "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": timeout_s + JOB_DEADLINE_SLACK_S,
            # Counted from the Job's finish, which is >= its creation: outlasts the
            # run's poll deadline whenever the Job finishes.
            "ttlSecondsAfterFinished": timeout_s + JOB_DEADLINE_SLACK_S + WAIT_SLACK_S + JOB_TTL_S,
            "template": {"metadata": {"labels": labels}, "spec": pod},
        },
    }


def job_outcome(job: dict[str, Any]) -> str | None:
    """'succeeded' / 'failed' once the Job is finished, None while it runs."""
    status = job.get("status") or {}
    for cond in status.get("conditions") or []:
        if cond.get("status") == "True" and cond.get("type") in ("Complete", "Failed"):
            return "succeeded" if cond["type"] == "Complete" else "failed"
    if status.get("succeeded"):
        return "succeeded"
    if status.get("failed"):
        return "failed"
    return None


def job_failure_reason(job: dict[str, Any]) -> str:
    for cond in (job.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "Failed" and cond.get("status") == "True":
            return f"{cond.get('reason') or 'Failed'}: {cond.get('message') or ''}".strip()
    return "the Job failed"


def pod_failure_reason(pod: dict[str, Any] | None) -> str:
    """Why a runner pod did not produce a result, from its own status (no events RBAC)."""
    if pod is None:
        return "no runner pod was ever created (namespace quota refused it?)"
    status = pod.get("status") or {}
    parts: list[str] = []
    for cond in status.get("conditions") or []:
        if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
            parts.append(f"unschedulable: {cond.get('message') or cond.get('reason')}")
    statuses = (status.get("initContainerStatuses") or []) + (status.get("containerStatuses") or [])
    for cs in statuses:
        state = cs.get("state") or {}
        term = state.get("terminated") or (cs.get("lastState") or {}).get("terminated")
        if term:
            parts.append(
                f"{cs.get('name')}: {term.get('reason') or 'terminated'} "
                f"(exit {term.get('exitCode')})"
            )
        elif (state.get("waiting") or {}).get("reason"):
            parts.append(f"{cs.get('name')}: {state['waiting']['reason']}")
    return "; ".join(parts) or f"pod phase {status.get('phase') or 'unknown'}"


def parse_result_log(log: str) -> Any:
    """The document of the LAST ``RESULT_JSON:`` line of a runner log.

    Raises :class:`KubeError` when there is no such line or it is not JSON.
    """
    lines = [ln for ln in log.splitlines() if ln.startswith(RESULT_PREFIX)]
    if not lines:
        raise KubeError("the runner printed no RESULT_JSON line")
    try:
        return json.loads(lines[-1][len(RESULT_PREFIX) :])
    except ValueError as e:
        raise KubeError(f"the runner's RESULT_JSON line is not JSON: {e}") from e


class KubeApi:
    """The API-server calls a run needs."""

    def _client(self) -> httpx.AsyncClient:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        if not host:
            raise KubeError("not running in a cluster (KUBERNETES_SERVICE_HOST is unset)")
        try:
            with open(f"{SA_DIR}/token", encoding="utf-8") as fh:
                token = fh.read().strip()
        except OSError as e:
            raise KubeError(f"no ServiceAccount token: {e}") from e
        return httpx.AsyncClient(
            base_url=f"https://{host}:{port}",
            headers={"Authorization": f"Bearer {token}"},
            verify=f"{SA_DIR}/ca.crt",
            timeout=30.0,
        )

    async def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        async with self._client() as client:
            try:
                resp = await client.request(method, path, **kw)
            except httpx.HTTPError as e:
                raise KubeError(f"API server unreachable: {e!r}") from e
        if resp.status_code >= 400:
            raise KubeError(
                f"{method} {path} answered {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
            )
        return resp

    async def create_job(self, namespace: str, spec: dict[str, Any]) -> None:
        await self._call("POST", f"/apis/batch/v1/namespaces/{namespace}/jobs", json=spec)

    async def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        return (
            await self._call("GET", f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}")
        ).json()

    async def job_pods(self, namespace: str, name: str) -> list[dict[str, Any]]:
        resp = await self._call(
            "GET",
            f"/api/v1/namespaces/{namespace}/pods",
            params={"labelSelector": f"job-name={name}"},
        )
        return resp.json().get("items") or []

    async def pod_log(self, namespace: str, pod: str) -> str:
        resp = await self._call(
            "GET",
            f"/api/v1/namespaces/{namespace}/pods/{pod}/log",
            params={"container": "runner", "tailLines": str(LOG_TAIL_LINES)},
        )
        return resp.text

    async def delete_job(self, namespace: str, name: str) -> None:
        await self._call(
            "DELETE",
            f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}",
            params={"propagationPolicy": "Background"},
        )


async def run_job(api: KubeApi, spec: dict[str, Any], timeout_s: int) -> str:
    """Create the Job, wait for it, return its log (:func:`finished_log`).

    The Job is left in place: the caller stores the result, THEN deletes it
    (:func:`delete_job_quietly`), so a Job that is gone always means "nobody will
    report it" (``rust_test_runs.reconcile``). Raises :class:`KubeError` when the Job
    could not be created or read, or did not finish in time.
    """
    namespace, name = spec["metadata"]["namespace"], spec["metadata"]["name"]
    await api.create_job(namespace, spec)
    deadline = time.monotonic() + timeout_s + JOB_DEADLINE_SLACK_S + WAIT_SLACK_S
    last_pod: dict[str, Any] | None = None
    while True:
        job = await api.get_job(namespace, name)
        if job_outcome(job) is not None:
            return await finished_log(api, namespace, name, job, last_pod)
        # Remembered: on DeadlineExceeded the controller deletes the pod, and with
        # it the reason it never ran (unschedulable, image pull, OOMKilled).
        seen = await api.job_pods(namespace, name)
        last_pod = seen[-1] if seen else last_pod
        if time.monotonic() > deadline:
            raise KubeError(f"Job {name} did not finish within its deadline")
        await asyncio.sleep(POLL_INTERVAL_S)


async def finished_log(
    api: KubeApi,
    namespace: str,
    name: str,
    job: dict[str, Any],
    last_pod: dict[str, Any] | None = None,
) -> str:
    """The runner log of FINISHED Job ``job``.

    Raises :class:`KubeError` when the Job failed; the message carries the Job's
    failure reason (DeadlineExceeded, BackoffLimitExceeded) and the pod's
    (unschedulable, OOMKilled, image pull, no pod at all).
    """
    pods = await api.job_pods(namespace, name)
    pod = pods[-1] if pods else last_pod
    log = ""
    if pod is not None:
        try:
            log = await api.pod_log(namespace, pod["metadata"]["name"])
        except KubeError as e:  # never started (quota, unschedulable, image pull...)
            log = f"(no log: {e})"
    # rust-repo-run exits 0 whenever the run happened (failing tests included) and 2
    # only for runner errors, so a Failed Job never carries a gradable result: its
    # log tail (with the runner's errors) goes into the failure message.
    if job_outcome(job) == "succeeded":
        return log
    raise KubeError(f"{job_failure_reason(job)}; {pod_failure_reason(pod)}; log tail: {log[-500:]}")


async def delete_job_quietly(api: KubeApi, namespace: str, name: str) -> None:
    """Delete the Job; a failure is only logged (ttlSecondsAfterFinished reaps it)."""
    try:
        await api.delete_job(namespace, name)
    except KubeError as e:
        if e.status != 404:
            logger.warning(
                "could not delete Job %s (ttlSecondsAfterFinished reaps it): %s", name, e
            )
