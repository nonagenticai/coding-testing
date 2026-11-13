"""Code execution service.

This provides the CodeExecutionService interface. In the microservice architecture,
test execution is forwarded to the language-specific test executor services via HTTP.
"""

import logging
import posixpath
import re
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

import httpx
from coding_client.clients import ServiceError
from fastapi import HTTPException

from auth.outbound import optional_tenant_headers, service_token, tenant_headers
from auth.tenant import DEFAULT_TENANT_ID, TENANT_ID_HEADER
from config.settings import get_settings
from services.intelligent_testing_service import (
    WORKSPACE_ROOT,
    WorkspaceExec,
    WorkspaceReadError,
    _containers_exec,
)

logger = logging.getLogger(__name__)

# coding-analysis analyze-sync budget. A whole-workspace run can be slow; a
# scoped run (``paths``) analyses only the named files inside the workspace.
QUALITY_ANALYSIS_TIMEOUT_SECONDS = 120
# quality-check with code_path stats the file first (cheap existence check).
QUALITY_FILE_CHECK_TIMEOUT_SECONDS = 30
INLINE_CODE_NEEDS_WORKSPACE = (
    "inline code quality needs a workspace (container_name + code_path): "
    "the linters exist only in workspace containers — not computed"
)

_EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def _str_or_none(value: Any) -> str | None:
    """Only a real, non-blank string counts (callers also pass Mocks/None)."""
    return value if isinstance(value, str) and value.strip() else None


def _workspace_relative(path: str | None, field: str) -> str:
    """Normalise a caller path to a workspace-relative POSIX path ("." = root).

    Accepts ``/workspace/...`` (the workspace root as coding-containers mounts
    it) and strips it; rejects any other absolute path, ``..`` segments, a
    leading ``-`` (it would be read as an option by the exec'd tool) and NUL.
    """
    if path is None:
        return "."
    raw = path.strip()
    if "\x00" in raw:
        raise HTTPException(status_code=422, detail=f"{field} contains a NUL byte")
    if raw == WORKSPACE_ROOT or raw.startswith(WORKSPACE_ROOT + "/"):
        raw = raw[len(WORKSPACE_ROOT) :].lstrip("/") or "."
    if raw.startswith("/"):
        raise HTTPException(
            status_code=422,
            detail=f"{field} must be workspace-relative (e.g. 'src/app.py'), got '{path}'",
        )
    if ".." in PurePosixPath(raw).parts:
        raise HTTPException(status_code=422, detail=f"{field} must not contain '..'")
    norm = posixpath.normpath(raw)
    if norm.startswith("-"):
        raise HTTPException(status_code=422, detail=f"{field} must not start with '-'")
    return norm


def _language_from_path(rel_path: str) -> str | None:
    return _EXT_LANGUAGE.get(PurePosixPath(rel_path).suffix.lower())


def _path_in_scope(file_path: Any, scope: str) -> bool:
    """Whether an issue's file path lies under the workspace-relative ``scope``.

    coding-analysis may report paths relative to the workspace or absolute under
    its own checkout root, so an absolute path matches on a ``/<scope>/`` segment.
    """
    if not isinstance(file_path, str) or not file_path:
        return False
    p = file_path[2:] if file_path.startswith("./") else file_path
    if p == scope or p.startswith(scope + "/"):
        return True
    return p.startswith("/") and (f"/{scope}/" in p or p.endswith(f"/{scope}"))


def _echoes_scope(data: dict[str, Any], scope: list[str]) -> bool:
    """True only when coding-analysis says it analysed exactly ``scope``.

    An older coding-analysis drops the unknown ``paths`` field and analyses the
    whole workspace; the only proof the scope was honoured is the echo.
    """
    # coding-analysis (fix/quality-sync-paths) answers ``scope: [...]`` (null when
    # unscoped); ``paths`` is accepted too so either spelling proves the scope.
    echoed = data.get("scope")
    if not isinstance(echoed, list):
        echoed = data.get("paths")
    if not isinstance(echoed, list):
        return False
    try:
        return sorted(_workspace_relative(str(p), "paths") for p in echoed) == sorted(scope)
    except HTTPException:
        return False


# Language to executor URL mapping using settings
_LANGUAGE_EXECUTOR_MAP: dict[str, str] = {}


def _executor_headers(tenant_id: str | None = None) -> dict[str, str]:
    """Outbound headers for executor calls.

    The executors are orchestrator-only and reject requests without an
    X-Tenant-Id header. Forward the caller's tenant when known, else fall back
    to DEFAULT_TENANT_ID. Since coding-client 1.9.0 that default is ``None`` when
    unset (no more "n8n-dev" literal), so fail loud rather than send an empty
    tenant header — a misconfiguration must surface, not silently mis-scope.
    """
    tenant = tenant_id or DEFAULT_TENANT_ID
    if tenant is None:
        raise RuntimeError(
            "No tenant to send to the executor: caller passed none and "
            "DEFAULT_TENANT_ID is unset. Set DEFAULT_TENANT_ID on the Deployment."
        )
    return {TENANT_ID_HEADER: tenant}


def _analysis_headers(tenant_id: str | None = None) -> dict[str, str]:
    """Outbound headers for calls to coding-analysis (a licence-gated sibling).

    Deliberately NOT the same builder as :func:`_executor_headers`. The
    ``*-test-executor`` services are not coding-* siblings and are not on the
    licence-authz path, so sending them a credential would be noise. coding-
    analysis IS a sibling: once it sets ``TRUST_TENANT_HEADER=false`` a bare
    ``X-Tenant-Id`` proves nothing and every quality analysis 401s.

    ``Authorization`` carries THIS service's licence, never a forwarded copy of
    the caller's, and it does not touch ``X-Tenant-Id`` — which still names the
    customer. Folding the token into the tenant header would make coding-analysis
    resolve every request against coding-testing's own tenant, so every
    container-scoped analysis would look at the wrong tenant's container.

    The tenant RESOLUTION is shared with ``_executor_headers`` on purpose: one
    policy, one failure mode. That policy (caller's tenant, else
    ``DEFAULT_TENANT_ID``, else fail loud) is unchanged here — this function adds
    the credential and nothing else.
    """
    return tenant_headers(
        _executor_headers(tenant_id)[TENANT_ID_HEADER],
        operation="coding-analysis quality analysis",
        service_token=service_token(),
    )


def _security_headers() -> dict[str, str]:
    """Outbound headers for calls to coding-security (a licence-gated sibling).

    This call site has no tenant to name — ``_validate_code_security`` receives
    only the code and its language, and the security verdict is a property of the
    code, not of a customer's data. So the tenant header is omitted rather than
    fabricated (``optional_``, and ``None`` passed explicitly), while the licence
    still goes out: proving who is calling is not a claim about whose data we
    want, and it costs nothing.
    """
    return optional_tenant_headers(None, service_token=service_token())


def _get_executor_url(language: str) -> str:
    """Resolve language to its executor service URL.

    Raises HTTPException(400) for unsupported languages,
    HTTPException(503) if the executor URL is not configured.
    """
    lang = language.lower()
    # Normalise aliases
    if lang in ("typescript", "nodejs"):
        lang = "javascript"

    s = get_settings()
    url_map: dict[str, str] = {
        "python": s.PYTHON_TEST_SERVICE_URL,
        "javascript": s.JAVASCRIPT_TEST_SERVICE_URL,
        "java": s.JAVA_TEST_SERVICE_URL,
        "go": s.GO_TEST_SERVICE_URL,
        "rust": s.RUST_TEST_SERVICE_URL,
        "csharp": s.CSHARP_TEST_SERVICE_URL,
        "cpp": s.CPP_TEST_SERVICE_URL,
    }

    if lang not in url_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported: {sorted(url_map.keys())}",
        )

    url = url_map[lang]
    if not url:
        raise HTTPException(
            status_code=503,
            detail=f"Executor for language '{language}' is not configured (env var missing)",
        )
    return url


# Module-level shared httpx client for executor calls
_executor_http_client: httpx.AsyncClient | None = None


def _get_executor_http_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient for executor communication."""
    global _executor_http_client
    if _executor_http_client is None:
        _executor_http_client = httpx.AsyncClient(timeout=60)
    return _executor_http_client


# Shared httpx client for security service calls
_security_http_client: httpx.AsyncClient | None = None


def _get_security_http_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient for security service communication."""
    global _security_http_client
    if _security_http_client is None:
        _security_http_client = httpx.AsyncClient(timeout=30)
    return _security_http_client


class CodeExecutionService:
    """Code execution service.

    Forwards test execution requests to language-specific executor microservices
    and security validation to the coding-security service.
    """

    def __init__(
        self,
        docker_service: Any = None,
        cache_manager: Any = None,
        security_service: Any = None,
        workspace_exec: WorkspaceExec | None = None,
    ):
        self.docker_service = docker_service
        # coding-containers exec transport (tests inject a fake).
        self._workspace_exec: WorkspaceExec = workspace_exec or _containers_exec
        self.cache_manager = cache_manager
        self.security_service = security_service
        self.test_orchestrator: Any = None
        self.settings: Any = None
        logger.info("CodeExecutionService initialized")

    async def execute_code(
        self,
        code: str,
        language: str,
        timeout: int = 30,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute code via the language-specific executor service."""
        try:
            executor_url = _get_executor_url(language)
        except HTTPException as e:
            return {
                "stdout": "",
                "stderr": f"Executor not configured: {e.detail}",
                "exit_code": 1,
                "execution_time": 0.0,
            }
        client = _get_executor_http_client()

        payload = {"code": code, "language": language, "timeout": timeout, **kwargs}
        try:
            resp = await client.post(
                f"{executor_url}/execute", json=payload, headers=_executor_headers(tenant_id)
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Executor returned error for {language}: {e.response.status_code}")
            return {
                "stdout": "",
                "stderr": f"Executor error: {e.response.text[:500]}",
                "exit_code": 1,
                "execution_time": 0.0,
            }
        except httpx.ConnectError as e:
            logger.error(f"Cannot reach executor for {language} at {executor_url}: {e}")
            return {
                "stdout": "",
                "stderr": f"Executor unreachable: {executor_url}",
                "exit_code": 1,
                "execution_time": 0.0,
            }

    async def execute_tests(
        self,
        code: str,
        test_code: str,
        language: str,
        timeout: int = 60,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute tests via the language-specific executor service."""
        try:
            executor_url = _get_executor_url(language)
        except HTTPException as e:
            return {
                "stdout": "",
                "stderr": f"Executor not configured: {e.detail}",
                "exit_code": 1,
                "execution_time": 0.0,
                "tests_passed": 0,
                "tests_failed": 0,
            }
        client = _get_executor_http_client()

        payload = {
            "code": code,
            "test_code": test_code,
            "language": language,
            "timeout": timeout,
            **kwargs,
        }
        try:
            resp = await client.post(
                f"{executor_url}/execute", json=payload, headers=_executor_headers(tenant_id)
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Executor returned error for {language}: {e.response.status_code}")
            return {
                "stdout": "",
                "stderr": f"Executor error: {e.response.text[:500]}",
                "exit_code": 1,
                "execution_time": 0.0,
                "tests_passed": 0,
                "tests_failed": 0,
            }
        except httpx.ConnectError as e:
            logger.error(f"Cannot reach executor for {language} at {executor_url}: {e}")
            return {
                "stdout": "",
                "stderr": f"Executor unreachable: {executor_url}",
                "exit_code": 1,
                "execution_time": 0.0,
                "tests_passed": 0,
                "tests_failed": 0,
            }

    async def _execute_in_workspace(
        self, request: Any, language: str, tenant_id: str | None
    ) -> Any:
        """Run the workspace's OWN tests, in the workspace (container_name, no inline code).

        The executors only take inline ``code``/``test_code``; this mode used to
        forward ``test_path`` to them and come back 502 "body.code Field required"
        for every workspace call. Python runs pytest inside the workspace via
        coding-containers exec, so the repo's imports and conftest apply. Other
        languages are refused (422) rather than answered with a guess.
        """
        from models.testing_models import TestResult, TestSecurityLevel
        from services.intelligent_testing_service import _containers_exec

        if str(language).lower() != "python":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Workspace test execution for '{language}' is not computed here; run the "
                    "repo's test command with POST /api/workspaces/{container_name}/exec, or "
                    "send inline code + test_code."
                ),
            )
        target = (
            getattr(request, "test_file_path", None) or getattr(request, "test_path", None) or "."
        )
        if target.startswith(("/", "-")) or ".." in target.split("/"):
            # A leading "-" would be read by pytest as an option, not a path.
            raise HTTPException(status_code=400, detail="test_path must be workspace-relative")
        timeout = int(getattr(request, "timeout", None) or 300)
        argv = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--", target]
        started = datetime.now(UTC)
        try:
            payload = await _containers_exec(request.container_name, tenant_id, argv, timeout)
        except Exception as e:  # noqa: BLE001 - any transport failure is "not run"
            raise HTTPException(
                status_code=502, detail=f"Workspace test run failed to execute: {e}"
            ) from e
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Workspace test run returned no result")
        if not isinstance(payload.get("exit_code"), int):
            raise HTTPException(
                status_code=502, detail="Workspace test run returned no exit code: not computed"
            )
        exit_code = payload["exit_code"]
        stdout = payload.get("stdout") or ""
        stderr = payload.get("stderr") or ""
        counts = {
            k: int(m.group(1)) if (m := re.search(rf"(\d+) {k}\b", stdout)) else 0
            for k in ("passed", "failed", "error", "errors")
        }
        failed = counts["failed"] + counts["error"] + counts["errors"]
        error = None
        if exit_code == 5:
            error = "no tests collected: nothing was verified (pytest exit 5)"
        elif exit_code == 124:
            error = f"timed out after {timeout}s"
        elif payload.get("truncated"):
            error = "output truncated; counts may be incomplete"
        security_level = getattr(request, "security_level", TestSecurityLevel.HIGH)
        if isinstance(security_level, str):
            security_level = TestSecurityLevel(security_level)
        return TestResult(
            test_id=str(uuid.uuid4()),
            # Exit 0 with at least one passed test, and nothing else, is a pass.
            success=exit_code == 0 and counts["passed"] > 0 and not payload.get("truncated"),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time=float(payload.get("duration") or 0.0),
            created_at=started,
            language="python",
            security_level=security_level,
            test_type=getattr(getattr(request, "test_type", None), "value", None)
            or getattr(request, "test_type", None),
            container_name=request.container_name,
            error=error,
            tests_run=counts["passed"] + failed,
            tests_passed=counts["passed"],
            tests_failed=failed,
            no_tests_run=exit_code == 5,
            test_metadata={"mode": "workspace", "command": argv},
        )

    async def execute_test(self, request: Any, tenant_id: str | None = None) -> Any:
        """Execute a single test by forwarding to the language-specific executor."""
        from models.testing_models import TestResult, TestSecurityLevel, read_not_run_flag

        language = getattr(request, "language", None)
        if not language:
            raise HTTPException(status_code=400, detail="Language is required in the test request")

        request_tenant = getattr(request, "tenant_id", None)
        tenant_id = tenant_id or (request_tenant if isinstance(request_tenant, str) else None)
        if (
            getattr(request, "container_name", None)
            and not getattr(request, "code", None)
            and not getattr(request, "test_code", None)
        ):
            return await self._execute_in_workspace(request, language, tenant_id)

        executor_url = _get_executor_url(language)
        client = _get_executor_http_client()

        # Build payload from request attributes
        payload: dict[str, Any] = {}
        for attr in (
            "code",
            "test_code",
            "language",
            "timeout",
            "test_type",
            "container_name",
            "code_path",
            "test_path",
            "dependencies",
            "security_level",
            "with_hypothesis",
            "resource_limits",
        ):
            val = getattr(request, attr, None)
            if val is not None:
                # Convert enums/models to serialisable form
                if hasattr(val, "value"):
                    val = val.value
                elif hasattr(val, "model_dump"):
                    val = val.model_dump()
                payload[attr] = val

        try:
            resp = await client.post(
                f"{executor_url}/execute", json=payload, headers=_executor_headers(tenant_id)
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError as e:
            logger.error(f"Cannot reach executor for {language} at {executor_url}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Test executor for '{language}' is unreachable at {executor_url}",
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Executor returned error for {language}: {e.response.status_code}")
            raise HTTPException(
                status_code=502,
                detail=f"Test executor error: {e.response.text[:500]}",
            )

        # Parse response into TestResult
        security_level = getattr(request, "security_level", TestSecurityLevel.HIGH)
        if isinstance(security_level, str):
            security_level = TestSecurityLevel(security_level)

        return TestResult(
            test_id=data.get("test_id", str(__import__("uuid").uuid4())),
            success=data.get("success", False),
            # The executors' /execute contract returns the combined stream as
            # `output`; asking only for `stdout`/`stderr` yielded "" every single
            # time, which is why none of these failures were ever visible.
            stdout=data.get("stdout") or data.get("output") or "",
            stderr=data.get("stderr") or "",
            exit_code=data.get("exit_code", 1),
            execution_time=data.get("execution_time", 0.0),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(UTC),
            language=data.get("language", language),
            security_level=security_level,
            test_type=data.get("test_type"),
            container_name=data.get("container_name"),
            coverage_data=data.get("coverage_data"),
            test_metadata=data.get("test_metadata"),
            error=data.get("error"),
            tests_run=data.get("tests_run", 0),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            # Same three spellings as the other executor boundary; see
            # models.testing_models.NOT_RUN_FIELD_ALIASES.
            no_tests_run=read_not_run_flag(data),
        )

    async def _post_analyze_sync(
        self, payload: dict[str, Any], tenant_id: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """POST coding-analysis ``/api/quality/analyze-sync``.

        Returns ``(data, None)`` on a 2xx JSON object, else ``(None, reason)``.
        Every no-verdict path (unreachable, timeout, 4xx/5xx, non-JSON) comes back
        as a reason so the caller can answer ``success=False`` with an explicit
        error — never a 0-issue pass and never an opaque 500.
        """
        url = get_settings().QUALITY_ANALYSIS_SERVICE_URL
        if not url:
            raise HTTPException(
                status_code=503,
                detail="QUALITY_ANALYSIS_SERVICE_URL is not configured",
            )
        try:
            async with httpx.AsyncClient(timeout=QUALITY_ANALYSIS_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{url}/api/quality/analyze-sync",
                    json=payload,
                    headers=_analysis_headers(tenant_id),
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            return None, f"coding-analysis timed out: {e!r} — quality not computed"
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300] if e.response is not None else ""
            return None, (
                f"coding-analysis answered {e.response.status_code}: {body} — quality not computed"
            )
        except httpx.HTTPError as e:
            return None, f"coding-analysis unreachable: {e!r} — quality not computed"
        except ValueError as e:
            return None, f"coding-analysis returned non-JSON: {e} — quality not computed"
        if not isinstance(data, dict):
            return None, "coding-analysis returned no result object — quality not computed"
        return data, None

    @staticmethod
    def _quality_result(
        data: dict[str, Any],
        *,
        language: str | None,
        scope: list[str] | None = None,
        scope_applied: bool | None = None,
        issues: list[Any] | None = None,
        out_of_scope_issues: int = 0,
    ) -> Any:
        from models.testing_models import QualityResult

        success = bool(data.get("success", False))
        error = data.get("error")
        score = data.get("overall_score")
        if not isinstance(score, int | float):
            # coding-analysis sends null for an inconclusive run; that is no
            # score, and must never surface as success.
            score = 0.0
            if success:
                success = False
                error = error or "coding-analysis returned no overall_score — not computed"
        return QualityResult(
            analysis_id=data.get("analysis_id") or "",
            success=success,
            overall_score=score,
            total_issues=data.get("total_issues", 0),
            issues=data.get("issues", []) if issues is None else issues,
            tools_results=data.get("tools_results") or {},
            execution_time=data.get("execution_time") or 0.0,
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(UTC),
            language=data.get("language") or language or "unknown",
            error=error,
            scope=scope,
            scope_applied=scope_applied,
            out_of_scope_issues=out_of_scope_issues,
        )

    @staticmethod
    def _not_computed(error: str, language: str | None, scope: list[str] | None = None) -> Any:
        from models.testing_models import QualityResult

        result = QualityResult.create_failed_result(
            error_message=error, language=language or "unknown"
        )
        result.scope = scope
        return result

    async def _check_workspace_file(
        self, container_name: str, tenant_id: str | None, rel_path: str
    ) -> str | None:
        """Prove ``rel_path`` is a regular, non-empty file in the workspace.

        Returns ``None`` when it is, else a no-verdict reason. Raises 404 when the
        file does not exist and 422 when it is a directory — both are the
        caller's input, not an outage.
        """
        # `stat` is not on coding-containers' exec allowlist (400, measured live);
        # `find` is. rel_path never starts with "-" (validated), so find reads it
        # as a start point, not an expression. Output: "<type letter> <size>".
        argv = ["find", rel_path, "-maxdepth", "0", "-printf", "%y %s"]
        try:
            res = await self._workspace_exec(
                container_name, tenant_id, argv, QUALITY_FILE_CHECK_TIMEOUT_SECONDS
            )
        except (ServiceError, httpx.HTTPError, WorkspaceReadError) as e:
            return f"checking {rel_path} in the workspace failed: {e} — not computed"
        if not isinstance(res, dict):
            return f"checking {rel_path}: coding-containers returned no result — not computed"
        exit_code = res.get("exit_code")
        stderr = str(res.get("stderr") or "")
        if exit_code != 0:
            if "No such file" in stderr:
                raise HTTPException(
                    status_code=404,
                    detail=f"code_path '{rel_path}' not found in workspace — quality not computed",
                )
            if exit_code == 124:
                return f"checking {rel_path} timed out — not computed"
            return f"checking {rel_path} exited {exit_code}: {stderr[:200]} — not computed"
        type_letter, _, size = str(res.get("stdout") or "").strip().partition(" ")
        kind = {"f": "regular file", "d": "directory"}.get(type_letter, type_letter or "unknown")
        if kind == "regular file" and size.strip() == "0":
            kind = "regular empty file"
        if kind == "directory":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"code_path '{rel_path}' is a directory; use /api/tests/quality-scan "
                    "with scan_path for a directory"
                ),
            )
        if kind == "regular empty file":
            return f"code_path '{rel_path}' is empty — nothing to analyse, not computed"
        if kind != "regular file":
            return f"code_path '{rel_path}' is not a regular file ({kind!r}) — not computed"
        return None

    async def analyze_quality(self, request: Any, tenant_id: str | None = None) -> Any:
        """Analyze code quality via coding-analysis service.

        The linters exist only in workspace containers, so every run happens IN
        the workspace (``container_name``):

        * ``code_path`` naming one workspace file -> ``paths: [code_path]``; the
          scope is only reported as applied when coding-analysis echoes it
          (an older coding-analysis ignores ``paths`` and analyses the whole
          workspace — then out-of-scope issues are dropped and counted);
        * neither code nor path (``code_path`` "." or unset) -> the whole workspace;
        * inline ``code``/``code_content`` without ``code_path`` -> 422: coding-
          analysis's ad-hoc ``code_content`` mode has no linters and can never
          produce a verdict.
        """
        request_tenant = getattr(request, "tenant_id", None)
        tenant_id = tenant_id or (request_tenant if isinstance(request_tenant, str) else None)
        container_name = _str_or_none(getattr(request, "container_name", None))
        language = _str_or_none(getattr(request, "language", None))
        code = _str_or_none(getattr(request, "code_content", None)) or _str_or_none(
            getattr(request, "code", None)
        )
        rel_path = _workspace_relative(
            _str_or_none(getattr(request, "code_path", None)), "code_path"
        )

        scope: list[str] | None = None
        if rel_path != ".":
            if not container_name:
                raise HTTPException(status_code=422, detail="code_path requires container_name")
            language = language or _language_from_path(rel_path)
            if not language:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"cannot infer language from '{rel_path}'; pass language "
                        f"(known extensions: {sorted(_EXT_LANGUAGE)})"
                    ),
                )
            scope = [rel_path]
            reason = await self._check_workspace_file(container_name, tenant_id, rel_path)
            if reason is not None:
                return self._not_computed(reason, language, scope)
        elif code:
            raise HTTPException(status_code=422, detail=INLINE_CODE_NEEDS_WORKSPACE)
        elif not container_name:
            raise HTTPException(status_code=422, detail="quality-check requires container_name")

        payload: dict[str, Any] = {
            "container_name": container_name,
            "paths": scope,
            "language": language,
            "tools": getattr(request, "tools", None),
            "fix_issues": getattr(request, "fix_issues", False),
            "custom_rules": getattr(request, "custom_rules", None),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        data, reason = await self._post_analyze_sync(payload, tenant_id)
        if data is None:
            return self._not_computed(reason or "not computed", language, scope)
        return self._scoped_result(data, language, scope)

    def _scoped_result(
        self, data: dict[str, Any], language: str | None, scope: list[str] | None
    ) -> Any:
        """Trust a scope only when coding-analysis echoes it; else filter + count."""
        if scope is None:
            return self._quality_result(data, language=language)
        if _echoes_scope(data, scope):
            return self._quality_result(data, language=language, scope=scope, scope_applied=True)
        raw = data.get("issues") or []
        kept = [
            i
            for i in raw
            if isinstance(i, dict) and any(_path_in_scope(i.get("file_path"), s) for s in scope)
        ]
        return self._quality_result(
            data,
            language=language,
            scope=scope,
            scope_applied=False,
            issues=kept,
            out_of_scope_issues=len(raw) - len(kept),
        )

    async def analyze_container_quality(self, request: Any, tenant_id: str | None = None) -> Any:
        """Analyze container quality via coding-analysis service.

        ``scan_path`` is sent as ``paths: [scan_path]``. coding-analysis versions
        that do not know ``paths`` silently ignore it and analyse the whole
        workspace, so the scope is only reported as applied when the response
        echoes it back; otherwise ``scope_applied`` is False and out-of-scope
        issues are dropped (counted in ``out_of_scope_issues``).
        """
        request_tenant = getattr(request, "tenant_id", None)
        tenant_id = tenant_id or (request_tenant if isinstance(request_tenant, str) else None)
        language = _str_or_none(getattr(request, "language", None))
        rel_path = _workspace_relative(
            _str_or_none(getattr(request, "scan_path", None)), "scan_path"
        )
        scope = None if rel_path == "." else [rel_path]
        payload = {
            "container_name": getattr(request, "container_name", None),
            "language": language,
            "tools": getattr(request, "tools", None),
            "fix_issues": getattr(request, "fix_issues", False),
            "paths": scope,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        data, reason = await self._post_analyze_sync(payload, tenant_id)
        if data is None:
            return self._not_computed(reason or "not computed", language, scope)
        return self._scoped_result(data, language, scope)

    async def _validate_code_security(self, code: str, language: str) -> dict[str, Any]:
        """Validate code security via the coding-security service.

        Returns coding-security's verdict (``is_safe``, ``violations``,
        ``risk_score``) with ``computed: True``, or ``is_safe: None`` and
        ``computed: False`` with a ``reason`` when no verdict exists.

        Two defects lived here: coding-security answers
        ``{"status", "validation": {"is_safe", ...}}`` and this read ``is_safe``
        from the TOP level, so the route's ``.get("is_safe", True)`` made every
        answer "safe"; and an unconfigured or unreachable service "failed open"
        as safe. Neither was a verdict.
        """

        def not_computed(reason: str) -> dict[str, Any]:
            return {
                "is_safe": None,
                "computed": False,
                "violations": [],
                "risk_score": None,
                "reason": reason,
            }

        security_url = get_settings().SECURITY_SERVICE_URL
        if not security_url:
            logger.warning("SECURITY_SERVICE_URL not configured; security NOT computed")
            return not_computed("security service not configured")
        client = _get_security_http_client()

        try:
            resp = await client.post(
                f"{security_url}/api/security/validate",
                json={"code": code, "language": language},
                headers=_security_headers(),
            )
            resp.raise_for_status()
            body = resp.json()
        except httpx.ConnectError as e:
            logger.warning(f"Security service unreachable at {security_url}: {e}")
            return not_computed("security service unreachable")
        except httpx.HTTPStatusError as e:
            logger.error(f"Security service error: {e.response.status_code}")
            return not_computed(
                f"security service answered {e.response.status_code}: {e.response.text[:300]}"
            )
        verdict = body.get("validation", body) if isinstance(body, dict) else None
        if not isinstance(verdict, dict) or not isinstance(verdict.get("is_safe"), bool):
            return not_computed("security service returned no is_safe verdict")
        return {**verdict, "computed": True}

    # Known quality tools per language — matches what's installed in executor Docker images
    _QUALITY_TOOLS: dict[str, list[str]] = {
        "python": ["ruff", "ruff-format", "ty", "bandit", "radon"],
        "javascript": ["eslint", "prettier", "jshint"],
        "typescript": ["eslint", "prettier", "tsc"],
        "java": ["checkstyle", "pmd", "spotbugs"],
        "go": ["golint", "staticcheck", "gofmt"],
        "rust": ["clippy", "rustfmt"],
        "csharp": ["dotnet-format", "roslynator"],
        "cpp": ["clang-tidy", "cppcheck", "clang-format"],
    }

    def _get_supported_tools(self, language: str) -> list[str]:
        """Get supported quality tools for a language."""
        return self._QUALITY_TOOLS.get(language.lower(), [])

    async def get_service_health(self) -> dict[str, Any]:
        """Get service health status (stub)."""
        return {"status": "stub", "message": "CodeExecutionService stub"}


_code_execution_service: CodeExecutionService | None = None


def get_code_execution_service() -> CodeExecutionService:
    """Get CodeExecutionService singleton."""
    global _code_execution_service
    if _code_execution_service is None:
        _code_execution_service = CodeExecutionService()
    return _code_execution_service
