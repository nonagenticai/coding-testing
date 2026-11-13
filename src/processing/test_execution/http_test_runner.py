"""
HTTP Test Runner - Communicates with containerized test services
Replaces direct container management with HTTP API calls to test service containers.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from aiohttp import ClientError, ClientSession, ClientTimeout

from auth.outbound import service_token, tenant_headers
from auth.tenant import DEFAULT_TENANT_ID, TENANT_ID_HEADER
from config.settings import get_settings
from models.testing_models import (
    QualityRequest,
    QualityResult,
    SecurityLevel,
    TestRequest,
    TestResult,
    read_not_run_flag,
)
from services.stubs import RedisCacheManager

logger = logging.getLogger(__name__)


class HTTPTestRunner:
    """
    Test runner that communicates with HTTP-based test service containers.
    Provides a clean interface between the main application and containerized test services.
    """

    def __init__(self, cache_manager: RedisCacheManager | None = None):
        self.cache_manager = cache_manager

        # Service URLs from settings — all must be set in production via
        # gitops-production env; the production validator enforces this.
        s = get_settings()
        self.python_service_url = s.PYTHON_TEST_SERVICE_URL
        self.javascript_service_url = s.JAVASCRIPT_TEST_SERVICE_URL
        self.quality_service_url = s.QUALITY_ANALYSIS_SERVICE_URL
        self.java_service_url = s.JAVA_TEST_SERVICE_URL
        self.rust_service_url = s.RUST_TEST_SERVICE_URL
        self.csharp_service_url = s.CSHARP_TEST_SERVICE_URL
        self.go_service_url = s.GO_TEST_SERVICE_URL

        # HTTP client configuration
        self.timeout = ClientTimeout(total=120, connect=10)  # 2 minutes total, 10s connect
        self.session: ClientSession | None = None

    async def _get_session(self) -> ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def __aenter__(self) -> "HTTPTestRunner":
        """Enter an async context that owns the lazily-created ClientSession.

        Use ``async with HTTPTestRunner() as runner: ...`` so the session (and its
        connection pool) is closed deterministically on exit. The factory returns
        ownerless runners, so without an explicit owner a forgotten ``.close()``
        leaks the session.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _get_service_url(self, language: str) -> str:
        """Get appropriate service URL based on language."""
        language_lower = language.lower()

        if language_lower in ["python", "py"]:
            return self.python_service_url
        if language_lower in ["javascript", "js", "typescript", "ts"]:
            return self.javascript_service_url
        if language_lower in ["java"]:
            return self.java_service_url
        if language_lower in ["rust", "rs"]:
            return self.rust_service_url
        if language_lower in ["csharp", "c#", "cs"]:
            return self.csharp_service_url
        if language_lower in ["go", "golang"]:
            return self.go_service_url
        # Default to Python for unknown languages
        logger.warning("Unknown language '%s', defaulting to Python service", language)
        return self.python_service_url

    async def execute_test(self, request: TestRequest, tenant_id: str | None = None) -> TestResult:
        """Execute test via HTTP service.

        The test-executor services are orchestrator-only and reject requests
        without an X-Tenant-Id header. We forward the caller's tenant when
        available, falling back to DEFAULT_TENANT_ID so this internal path is
        never rejected.
        """
        test_id = str(uuid7())
        start_time = datetime.now(UTC)

        # Since coding-client 1.9.0 DEFAULT_TENANT_ID is None when unset (no more
        # "n8n-dev" literal), so resolve the outbound tenant up front and fail
        # loud rather than send an empty X-Tenant-Id the executor would reject.
        outbound_tenant = tenant_id or DEFAULT_TENANT_ID
        if outbound_tenant is None:
            raise RuntimeError(
                "No tenant to send to the test executor: caller passed none and "
                "DEFAULT_TENANT_ID is unset. Set DEFAULT_TENANT_ID on the Deployment."
            )

        try:
            logger.info("HTTP test execution %s: %s", test_id, request.language)

            # Get appropriate service URL
            service_url = self._get_service_url(request.language)

            # Prepare request payload
            payload = {
                "code": request.code,
                "test_code": request.test_code,
                "code_file_path": request.code_file_path,
                "test_file_path": request.test_file_path,
                "language": request.language,
                "timeout": request.timeout,
                "with_hypothesis": request.with_hypothesis,
                "security_level": (
                    request.security_level.value
                    if isinstance(request.security_level, SecurityLevel)
                    else request.security_level
                ),
            }

            # Remove None values to avoid issues
            payload = {k: v for k, v in payload.items() if v is not None}

            # Make HTTP request to test service
            session = await self._get_session()
            async with session.post(
                f"{service_url}/execute",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    TENANT_ID_HEADER: outbound_tenant,
                },
            ) as response:
                if response.status == 200:
                    result_data = await response.json()

                    # Convert response to TestResult
                    return TestResult(
                        test_id=result_data.get("test_id", test_id),
                        success=result_data.get("success", False),
                        # Executors return the combined stream as `output`; a
                        # bare `stdout`/`stderr` lookup was always "" and hid
                        # every failure message from every caller.
                        stdout=result_data.get("stdout") or result_data.get("output") or "",
                        stderr=result_data.get("stderr") or "",
                        exit_code=result_data.get("exit_code", 1),
                        execution_time=result_data.get("execution_time", 0.0),
                        created_at=datetime.fromisoformat(
                            result_data.get("created_at", start_time.isoformat())
                        ),
                        language=result_data.get("language", request.language),
                        security_level=result_data.get("security_level", request.security_level),
                        coverage_data=None,
                        test_metadata=None,
                        test_type=None,
                        container_name=None,
                        quality_report=None,
                        error=result_data.get("error"),
                        tests_run=result_data.get("tests_run", 0),
                        tests_passed=result_data.get("tests_passed", 0),
                        tests_failed=result_data.get("tests_failed", 0),
                        # Read every spelling: csharp and cpp emit
                        # `noTestsRun`, which a bare `no_tests_run` lookup
                        # silently read as "tests did run".
                        no_tests_run=read_not_run_flag(result_data),
                    )
                # Handle HTTP error
                error_text = await response.text()
                logger.error("Test service HTTP error %s: %s", response.status, error_text)

                return TestResult(
                    test_id=test_id,
                    success=False,
                    stdout="",
                    stderr=f"HTTP {response.status}: {error_text}",
                    exit_code=response.status,
                    execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                    created_at=start_time,
                    language=request.language,
                    security_level=request.security_level,
                    coverage_data=None,
                    test_metadata=None,
                    test_type=None,
                    container_name=None,
                    quality_report=None,
                    error=f"HTTP {response.status}: {error_text}",
                )

        except TimeoutError:
            logger.error("Test execution %s timed out", test_id)
            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr="Test execution timed out",
                exit_code=124,  # Standard timeout exit code
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                created_at=start_time,
                language=request.language,
                security_level=request.security_level,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=None,
                quality_report=None,
                error="Test execution timed out",
            )

        except (ClientError, OSError) as e:
            logger.error("Test execution %s failed: %s", test_id, e)
            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                created_at=start_time,
                language=request.language,
                security_level=request.security_level,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=None,
                quality_report=None,
                error=str(e),
            )

    async def analyze_quality(
        self, request: QualityRequest, tenant_id: str | None = None
    ) -> QualityResult:
        """Run quality analysis via coding-analysis.

        ⚠️ This call used to send ``Content-Type`` and NOTHING else — no tenant,
        no credential — while its sibling ``execute_test`` two methods up sent
        ``X-Tenant-Id`` on every request. coding-analysis is a licence-gated
        coding-* sibling, so this call reached it only through the trusted-pod-CIDR
        fallback (a network position, not a credential) and would 401 the moment
        that fallback or ``TRUST_TENANT_HEADER`` is turned off. Worse, the failure
        would be invisible: the ``except`` below turns any error into a
        ``QualityResult(success=False)``, i.e. "your code scored 0".

        It now uses the SAME resolution as ``execute_test``: the caller's tenant
        when known, ``DEFAULT_TENANT_ID`` otherwise, and a loud failure when
        neither exists rather than an empty header. ``Authorization`` carries this
        service's own licence, separately, so the asserted tenant can be vouched.
        """
        analysis_id = str(uuid7())
        start_time = datetime.now(UTC)

        outbound_tenant = tenant_id or DEFAULT_TENANT_ID
        if outbound_tenant is None:
            raise RuntimeError(
                "No tenant to send to coding-analysis: caller passed none and "
                "DEFAULT_TENANT_ID is unset. Set DEFAULT_TENANT_ID on the Deployment."
            )

        try:
            logger.info("HTTP quality analysis %s: %s", analysis_id, request.language)

            # Prepare request payload
            payload = {
                "code_path": request.code_path,
                "code_content": request.code_content,
                "language": request.language,
                "tools": request.tools,
                "fix_issues": request.fix_issues,
                "custom_rules": request.custom_rules,
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            # Make HTTP request to quality service
            session = await self._get_session()
            async with session.post(
                f"{self.quality_service_url}/api/quality/analyze",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **tenant_headers(
                        outbound_tenant,
                        operation="coding-analysis quality analysis",
                        service_token=service_token(),
                    ),
                },
            ) as response:
                if response.status == 200:
                    result_data = await response.json()

                    # Convert response to QualityResult
                    return QualityResult(
                        analysis_id=result_data.get("analysis_id", analysis_id),
                        success=result_data.get("success", False),
                        language=result_data.get("language", request.language or "unknown"),
                        tools_results=result_data.get("tools_results", {}),
                        overall_score=result_data.get("overall_score", 0.0),
                        total_issues=result_data.get("total_issues", 0),
                        execution_time=result_data.get("execution_time", 0.0),
                        created_at=datetime.fromisoformat(
                            result_data.get("created_at", start_time.isoformat())
                        ),
                        error=result_data.get("error"),
                    )
                # Handle HTTP error
                error_text = await response.text()
                logger.error("Quality service HTTP error %s: %s", response.status, error_text)

                return QualityResult(
                    analysis_id=analysis_id,
                    success=False,
                    language=request.language or "unknown",
                    tools_results={},
                    overall_score=0.0,
                    total_issues=0,
                    execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                    created_at=start_time,
                    error=f"HTTP {response.status}: {error_text}",
                )

        except TimeoutError:
            logger.error("Quality analysis %s timed out", analysis_id)
            return QualityResult(
                analysis_id=analysis_id,
                success=False,
                language=request.language or "unknown",
                tools_results={},
                overall_score=0.0,
                total_issues=0,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                created_at=start_time,
                error="Quality analysis timed out",
            )

        except (ClientError, OSError) as e:
            logger.error("Quality analysis %s failed: %s", analysis_id, e)
            return QualityResult(
                analysis_id=analysis_id,
                success=False,
                language=request.language or "unknown",
                tools_results={},
                overall_score=0.0,
                total_issues=0,
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                created_at=start_time,
                error=str(e),
            )

    async def health_check(self, service: str = "all") -> dict[str, Any]:
        """Check health of test services."""
        results = {}
        services = {
            "python": self.python_service_url,
            "javascript": self.javascript_service_url,
            "quality": self.quality_service_url,
        }

        if service != "all":
            service_url = services.get(service)
            if service_url:
                services = {service: service_url}
            else:
                services = {}

        session = await self._get_session()

        for service_name, service_url in services.items():
            if service_url:
                try:
                    async with session.get(
                        f"{service_url}/health", timeout=ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            results[service_name] = {
                                "status": "healthy",
                                "url": service_url,
                                "response_time": response.headers.get("response-time", "unknown"),
                                "details": health_data,
                            }
                        else:
                            results[service_name] = {
                                "status": "unhealthy",
                                "url": service_url,
                                "error": f"HTTP {response.status}",
                            }
                except (ClientError, OSError) as e:
                    results[service_name] = {
                        "status": "error",
                        "url": service_url,
                        "error": str(e),
                    }

        return results

    async def get_service_info(self, service: str) -> dict[str, Any]:
        """Get information about a specific service."""
        service_urls = {
            "python": self.python_service_url,
            "javascript": self.javascript_service_url,
            "quality": self.quality_service_url,
        }

        service_url = service_urls.get(service)
        if not service_url:
            raise ValueError(f"Unknown service: {service}")

        session = await self._get_session()

        try:
            async with session.get(f"{service_url}/info") as response:
                if response.status == 200:
                    result = await response.json()
                    return dict(result) if result else {}
                error_text = await response.text()
                raise ConnectionError(f"HTTP {response.status}: {error_text}")
        except (ClientError, OSError) as e:
            logger.error("Failed to get info for %s: %s", service, e)
            raise


# Factory function for compatibility with existing code
def create_container_test_runner(docker_service, config) -> HTTPTestRunner:
    """Factory function to create HTTP test runner (replaces container-based runner)."""
    logger.info("Creating HTTP-based test runner instead of container-based runner")
    return HTTPTestRunner()


# For backward compatibility
ContainerTestRunner = HTTPTestRunner
