"""
Test Orchestrator Service - Phase 1 Implementation
Centralized test execution orchestration for multi-language test containers.
"""

import json
import logging
import os
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid7

from pydantic import BaseModel, Field, model_validator

from models.testing_models import (
    TestRequest,
    TestSecurityLevel,
    enforce_not_run_is_never_a_pass,
    read_not_run_flag,
)

try:
    from config.test_framework_config import TestConfig
except ImportError:
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class TestConfig:
        max_execution_time: int = 300
        max_memory_mb: int = 1024
        max_file_size_mb: int = 10
        max_code_length: int = 50000
        allowed_imports: list[str] | None = None


from config.test_framework_security_manager import (
    TestValidationSecurityConfig,
)
from services.stubs import DockerService
from services.test_execution.security_manager import ExecutionSecurityManager
from services.test_execution.test_execution_container_registry import (
    TestExecutionContainerRegistry,
)

logger = logging.getLogger(__name__)


class TestLanguage(StrEnum):
    """Supported programming languages for test execution.

    This enum defines all programming languages that can be executed
    in containerized test environments. Each language has a corresponding
    container with appropriate runtime and testing tools installed.
    """

    __test__ = False  # Not a pytest test class

    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    NODEJS = "nodejs"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


class OrchestratorType(StrEnum):
    """Types of tests that can be executed.

    This enum defines various test types supported by the orchestrator,
    including standard test types (unit, integration) and language-specific
    tools (clippy for Rust, vet for Go).
    """

    UNIT = "unit"
    INTEGRATION = "integration"
    BENCHMARK = "benchmark"
    COVERAGE = "coverage"
    SECURITY = "security"
    DOC = "doc"
    CLIPPY = "clippy"
    FMT = "fmt"
    VET = "vet"
    RACE = "race"


# ---------------------------------------------------------------------------
# "No test command defined" is a NOT-RUN outcome, never a pass.
#
# Until 2026-09-23 an unmapped (language, test_type) pair fell back to
# ["echo", "No test command defined"]. `echo` exits 0 and the verdict is
# `success = exit_code == 0`, so every such pair reported a GREEN suite having
# executed no tests at all: python+{integration,security,benchmark}, js/ts
# non-unit, and every test_type for java/csharp/cpp.
#
# The fix is a distinct, explicitly non-passing outcome rather than a generic
# failure: `success=False` plus `not_run=True` and a machine-readable
# `not_run_reason`, so a caller that legitimately has no tests for a pair can
# tell "nothing to run" from "the tests failed" — while nothing can read it
# as "the tests passed".
# ---------------------------------------------------------------------------

#: POSIX "command not found". There was no command, so nothing was verified.
NO_TEST_COMMAND_EXIT_CODE = 127

#: Stable machine-readable marker carried on TestExecutionResult.not_run_reason.
NO_TEST_COMMAND_REASON = "no_test_command_defined"


def _read_not_run(raw: dict[str, Any]) -> bool:
    """Read the not-run fact out of raw execution output, under either spelling.

    The orchestrator's own raw dicts use ``not_run``; the language executors'
    ``/execute`` contract uses ``no_tests_run``; csharp and cpp have drifted to
    ``noTestsRun``. All three mean "nothing was executed", and a reader that
    knows only one of them drops the others on the floor at exactly the
    boundary where it stops being recoverable.

    Delegates to :func:`read_not_run_flag` so this path and the two HTTP
    boundaries cannot drift into different answers to the same question.
    """
    return read_not_run_flag(raw)


def _no_test_command_message(language: Any, test_type: Any) -> str:
    """Human-readable explanation for a not-run outcome."""
    lang = getattr(language, "value", language)
    ttype = getattr(test_type, "value", test_type)
    return (
        f"No test command is defined for language={lang} test_type={ttype}. "
        "Nothing was executed, so this run verified nothing. Reported as "
        f"not-run (exit {NO_TEST_COMMAND_EXIT_CODE}); it is never a pass."
    )


class TestExecutionRequest(BaseModel):
    """Request model for test execution via orchestrator."""

    __test__ = False  # Not a pytest test class

    language: TestLanguage
    test_type: OrchestratorType
    source_code: str
    container_name: str | None = Field(None, description="Container name for test execution")
    tenant_id: str | None = Field(None, description="Tenant ID for container validation")
    test_files: dict[str, str] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=300)
    resource_limits: dict[str, Any] | None = None
    security_level: TestSecurityLevel = TestSecurityLevel.MEDIUM
    project_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestExecutionResult(BaseModel):
    """Result model for test execution."""

    __test__ = False  # Not a pytest test class

    test_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: float
    coverage_report: dict[str, Any] | None = None
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    performance_metrics: dict[str, float] | None = None
    security_issues: list[str] = Field(default_factory=list)
    container_id: str | None = None
    container_name: str | None = Field(None, description="Container name used for execution")
    language: TestLanguage
    test_type: OrchestratorType
    created_at: datetime
    # Distinguishes "the tests failed" from "no tests were executed at all".
    # `not_run=True` always implies `success=False`; the converse does not hold.
    not_run: bool = Field(
        default=False,
        description="True when no tests were executed (e.g. no test command is defined)",
    )
    not_run_reason: str | None = Field(
        default=None,
        description="Machine-readable reason nothing was executed, when not_run is True",
    )

    @model_validator(mode="after")
    def _not_run_is_never_a_pass(self) -> "TestExecutionResult":
        """Enforce the invariant the comment above only asserted.

        ``_build_not_run_result`` sets ``success=False`` itself, but this model
        is also rehydrated from JSONB (``suite_from_data``) and built from raw
        executor output, where nothing stopped a row carrying
        ``success=True`` alongside ``not_run=True``. Shares one implementation
        with ``TestResult.no_tests_run``.
        """
        enforce_not_run_is_never_a_pass(self, "not_run")
        return self


class TestOrchestrator:
    """
    Centralized test orchestration service that manages test execution
    across multiple language-specific containers.
    """

    def __init__(self, docker_service: DockerService, config: TestConfig):
        self.docker_service = docker_service
        self.config = config
        self.container_registry = TestExecutionContainerRegistry()
        self.active_containers: dict[str, str] = {}  # test_id -> container_id
        # test_id -> tenant_id (parallel to active_containers, for tenant-scoped
        # enumeration/cleanup). In-memory + per-replica: authoritative only for
        # containers this replica started; correctness (no cross-tenant leak),
        # not distributed completeness, is the guarantee.
        self.active_container_tenants: dict[str, str | None] = {}
        # language -> cached container info
        self.container_cache: dict[str, Any] = {}

        # Enable Kubernetes mode when running in-cluster or explicitly requested
        self.use_k8s = os.environ.get("KUBERNETES_IN_CLUSTER", "false").lower() == "true"
        self._k8s_executor = None  # Lazy-loaded

        # Initialize security manager
        security_config = TestValidationSecurityConfig(
            max_execution_time=getattr(config, "max_execution_time", 300),
            max_memory_mb=getattr(config, "max_memory_mb", 1024),
            max_file_size_mb=getattr(config, "max_file_size_mb", 10),
            max_code_length=getattr(config, "max_code_length", 50000),
            allowed_modules=getattr(config, "allowed_imports", ["pytest", "unittest"]),
        )
        self.security_manager = ExecutionSecurityManager(security_config)

        logger.info("TestOrchestrator initialized with container registry")

    @property
    def k8s_executor(self):
        """Lazy-load the Kubernetes test executor."""
        if self._k8s_executor is None and self.use_k8s:
            try:
                from services.test_execution.kubernetes_test_executor import (
                    KubernetesTestExecutor,
                )

                self._k8s_executor = KubernetesTestExecutor()
            except ImportError:
                logger.warning("KubernetesTestExecutor not available")
        return self._k8s_executor

    async def execute_tests(
        self, request: TestExecutionRequest, tenant_id: str | None = None
    ) -> TestExecutionResult:
        """
        Execute tests in appropriate language container with full orchestration.

        Args:
            request: The test execution request.
            tenant_id: Authenticated tenant of the caller. When provided it is the
                authoritative tenant used for container validation and for tagging
                the active-container entry (so ``/execute/active`` and
                ``/cleanup`` stay tenant-scoped). Falls back to
                ``request.tenant_id`` when not supplied.
        """
        test_id = str(uuid7())
        start_time = datetime.now(UTC)
        container_id = None
        effective_tenant_id = tenant_id if tenant_id is not None else request.tenant_id

        try:
            logger.info(
                f"Starting orchestrated test execution {test_id} for {request.language.value}"
            )
            logger.info(f"Container name: {request.container_name}")

            # Validate container exists before executing
            if request.container_name:
                try:
                    from shared.container_validation import (
                        validate_and_get_container_for_tenant,
                    )

                    await validate_and_get_container_for_tenant(
                        request.container_name, tenant_id=effective_tenant_id
                    )
                    logger.debug(
                        f"Container '{request.container_name}' validated for test orchestration"
                    )
                except ImportError:
                    logger.debug("Container validation module not available, skipping validation")
                except Exception as e:
                    logger.error(f"Container validation failed: {e}")
                    raise ValueError(f"Container '{request.container_name}' validation failed: {e}")

            # Security validation
            await self._validate_execution_request(request)

            # Get container specification
            container_spec = self.container_registry.get_container_spec(request.language.value)
            if not container_spec:
                raise ValueError(
                    f"No container specification found for language: {request.language.value}"
                )

            # Refuse to start anything when there is no test command to run.
            # This is checked before container/Job creation so BOTH the Docker
            # and the Kubernetes path are covered by one guard.
            if self._get_test_command(request.language, request.test_type) is None:
                return self._build_not_run_result(request, test_id, start_time)

            # Kubernetes path: run tests via K8s Job instead of Docker
            if self.use_k8s and self.k8s_executor:
                test_command = self._get_test_command(request.language, request.test_type)
                k8s_result = await self.k8s_executor.run(request, container_spec, test_command)

                execution_result = await self._build_result_from_raw(
                    raw=k8s_result,
                    request=request,
                    test_id=test_id,
                    start_time=start_time,
                )

                logger.info("Test execution %s completed via Kubernetes Job", test_id)
                return execution_result

            # Create and configure container
            container_id = await self._create_test_container(request, test_id, container_spec)
            self.active_containers[test_id] = container_id
            self.active_container_tenants[test_id] = effective_tenant_id

            # Prepare test environment
            await self._prepare_test_environment(container_id, request)

            # Execute tests
            result = await self._execute_in_container(container_id, request, test_id)

            # Collect and process results
            execution_result = await self._collect_results(
                container_id, result, request, test_id, start_time
            )

            logger.info(f"Test execution {test_id} completed successfully")
            return execution_result

        except Exception as e:
            logger.error(f"Test execution {test_id} failed: {e}")
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return TestExecutionResult(
                test_id=test_id,
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time_ms=execution_time,
                container_id=container_id,
                container_name=request.container_name,
                language=request.language,
                test_type=request.test_type,
                created_at=start_time,
            )
        finally:
            # Always cleanup
            if container_id:
                await self._cleanup_container(test_id, container_id)

    async def get_supported_languages(self) -> list[str]:
        """Get list of supported languages for test execution."""
        return self.container_registry.get_supported_languages()

    async def get_supported_test_types(self, language: str) -> list[str]:
        """Get supported test types for a specific language."""
        container_spec = self.container_registry.get_container_spec(language)
        if not container_spec:
            return []

        # Language-specific test types
        test_types_map = {
            "go": ["unit", "benchmark", "coverage", "race", "security", "vet"],
            "rust": [
                "unit",
                "doc",
                "benchmark",
                "coverage",
                "security",
                "clippy",
                "fmt",
            ],
            "java": ["unit", "integration", "coverage", "security"],
            "csharp": ["unit", "integration", "coverage", "security"],
            "cpp": ["unit", "coverage", "security"],
            "python": ["unit", "integration", "coverage", "security"],
            "javascript": ["unit", "integration", "coverage", "security"],
            "typescript": ["unit", "integration", "coverage", "security"],
        }

        return test_types_map.get(language, ["unit", "coverage"])

    async def _validate_execution_request(self, request: TestExecutionRequest) -> None:
        """Validate test execution request using security manager."""
        # Convert to TestRequest for security validation
        test_request = TestRequest(
            code=request.source_code,
            test_code=("\n".join(request.test_files.values()) if request.test_files else None),
            language=request.language.value,
            timeout=request.timeout_seconds,
            security_level=request.security_level,
            code_path=None,
            code_file_path=None,
            test_path=None,
            test_file_path=None,
            container_name=(request.container_name if hasattr(request, "container_name") else None),
            test_type=None,
            dependencies={},
            resource_limits=None,
            run_quality_check=False,
            quality_tools=None,
        )

        # Validate and sanitize
        await self.security_manager.validate_and_sanitize(test_request)

        # Additional orchestrator-specific validation
        if request.timeout_seconds > 600:  # 10 minutes max
            raise ValueError("Timeout exceeds maximum allowed (600 seconds)")

        if len(request.source_code) > 100000:  # 100KB max
            raise ValueError("Source code exceeds maximum size (100KB)")

    async def _create_test_container(
        self, request: TestExecutionRequest, test_id: str, container_spec: Any
    ) -> str:
        """Create and configure test container."""
        # Get security configuration
        security_config = self.security_manager.get_container_security_config(
            request.security_level
        )

        # Configure container
        container_config = {
            "image": container_spec.image,
            "working_dir": "/workspace",
            "mem_limit": security_config.get("mem_limit", container_spec.memory_limit),
            "cpu_quota": security_config.get("cpu_quota", container_spec.cpu_quota),
            "network_disabled": security_config.get("network_disabled", True),
            "read_only": security_config.get("read_only", True),
            "user": security_config.get("user", "testuser"),
            "security_opt": security_config.get("security_opt", ["no-new-privileges:true"]),
            "tmpfs": security_config.get("tmpfs", {"/tmp": "noexec,nosuid,size=100m"}),
            "cap_drop": security_config.get("cap_drop", ["ALL"]),
            "labels": {
                "test_id": test_id,
                "service": "test_execution",
                "language": request.language.value,
                "test_type": request.test_type.value,
            },
            "language": request.language.value,
            "environment": container_spec.environment,
        }

        # Validate security configuration
        if not self.security_manager.validate_container_runtime_security(container_config):
            raise ValueError("Container configuration failed security validation")

        # Create container
        container_id = await self.docker_service.create_test_container(container_config)
        await self.docker_service.start_container(container_id)

        logger.info(f"Created secure test container {container_id} for {request.language.value}")
        return container_id

    async def _prepare_test_environment(
        self, container_id: str, request: TestExecutionRequest
    ) -> None:
        """Prepare test environment in container."""
        # Create source files
        if request.source_code:
            extension = self._get_file_extension(request.language.value)
            filename = f"main{extension}"
            await self.docker_service.create_file_in_container(
                container_id, f"/workspace/{filename}", request.source_code
            )

        # Create test files
        for filename, content in request.test_files.items():
            await self.docker_service.create_file_in_container(
                container_id, f"/workspace/{filename}", content
            )

        # Create language-specific configuration files
        await self._create_language_config_files(container_id, request)

        logger.debug(f"Prepared test environment in container {container_id}")

    async def _create_language_config_files(
        self, container_id: str, request: TestExecutionRequest
    ) -> None:
        """Create language-specific configuration files."""
        if request.language == TestLanguage.GO:
            # Create go.mod if not exists
            go_mod_content = """module test-module

go 1.21

{self._format_go_dependencies(request.dependencies)}
"""
            await self.docker_service.create_file_in_container(
                container_id, "/workspace/go.mod", go_mod_content
            )

        elif request.language == TestLanguage.RUST:
            # Create Cargo.toml if not exists
            cargo_toml_content = """[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"

[dependencies]
{self._format_rust_dependencies(request.dependencies)}
"""
            await self.docker_service.create_file_in_container(
                container_id, "/workspace/Cargo.toml", cargo_toml_content
            )

        elif request.language in [
            TestLanguage.JAVASCRIPT,
            TestLanguage.TYPESCRIPT,
        ]:
            # Create package.json
            package_json: dict[str, Any] = {
                "name": "test-project",
                "version": "1.0.0",
                "scripts": {
                    "test": (
                        "jest"
                        if request.language == TestLanguage.JAVASCRIPT
                        else "jest --preset ts-jest"
                    )
                },
                "devDependencies": {"jest": "^29.0.0", **request.dependencies},
            }

            if request.language == TestLanguage.TYPESCRIPT:
                ts_deps: dict[str, str] = {
                    "ts-jest": "^29.0.0",
                    "@types/jest": "^29.0.0",
                    "typescript": "^5.0.0",
                }
                dev_deps = package_json["devDependencies"]
                if isinstance(dev_deps, dict):
                    dev_deps.update(ts_deps)

            await self.docker_service.create_file_in_container(
                container_id,
                "/workspace/package.json",
                json.dumps(package_json, indent=2),
            )

    def _build_not_run_result(
        self,
        request: TestExecutionRequest,
        test_id: str,
        start_time: datetime,
    ) -> TestExecutionResult:
        """Build the distinct NOT-RUN result for an unmapped (language, test_type)."""
        message = _no_test_command_message(request.language, request.test_type)
        logger.warning("Test execution %s not run: %s", test_id, message)
        return TestExecutionResult(
            test_id=test_id,
            success=False,
            exit_code=NO_TEST_COMMAND_EXIT_CODE,
            stdout="",
            stderr=message,
            execution_time_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
            container_id=None,
            container_name=request.container_name,
            language=request.language,
            test_type=request.test_type,
            created_at=start_time,
            not_run=True,
            not_run_reason=NO_TEST_COMMAND_REASON,
        )

    async def _execute_in_container(
        self, container_id: str, request: TestExecutionRequest, test_id: str
    ) -> dict[str, Any]:
        """Execute test in container using language-specific test runner."""
        # Get test command
        test_command = self._get_test_command(request.language, request.test_type)

        # Defence in depth: this method is independently callable, so it must
        # not silently succeed when there is nothing to execute.
        if test_command is None:
            return {
                "exit_code": NO_TEST_COMMAND_EXIT_CODE,
                "stdout": "",
                "stderr": _no_test_command_message(request.language, request.test_type),
                "execution_time_ms": 0.0,
                "not_run": True,
                "not_run_reason": NO_TEST_COMMAND_REASON,
            }

        # Execute with timeout
        start_time = datetime.now(UTC)

        try:
            result = await self.docker_service.exec_in_container(
                container_id, test_command, timeout=request.timeout_seconds
            )

            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return {
                "exit_code": result.get("exit_code", 1),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time_ms": execution_time,
            }

        except TimeoutError:
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.warning(f"Test execution timed out for {test_id}")

            return {
                "exit_code": 124,  # Standard timeout exit code
                "stdout": "",
                "stderr": f"Test execution timed out after {request.timeout_seconds} seconds",
                "execution_time_ms": execution_time,
            }

    async def _collect_results(
        self,
        container_id: str,
        result: dict[str, Any],
        request: TestExecutionRequest,
        test_id: str,
        start_time: datetime,
    ) -> TestExecutionResult:
        """Collect and process test results."""
        # Collect additional artifacts based on test type
        coverage_report = None
        performance_metrics = None
        security_issues = []

        if request.test_type == OrchestratorType.COVERAGE:
            coverage_report = await self._collect_coverage_report(container_id, request.language)

        if request.test_type == OrchestratorType.BENCHMARK:
            performance_metrics = await self._collect_performance_metrics(
                container_id, request.language
            )

        if request.test_type == OrchestratorType.SECURITY:
            security_issues = await self._collect_security_issues(container_id, request.language)

        # Parse test results
        test_results = await self._parse_test_output(result.get("stdout", ""), request.language)

        return TestExecutionResult(
            test_id=test_id,
            success=result.get("exit_code", 1) == 0,
            exit_code=result.get("exit_code", 1),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            execution_time_ms=result.get("execution_time_ms", 0.0),
            coverage_report=coverage_report,
            test_results=test_results,
            performance_metrics=performance_metrics,
            security_issues=security_issues,
            container_id=container_id,
            container_name=request.container_name,
            language=request.language,
            test_type=request.test_type,
            created_at=start_time,
            # `no_tests_run` is the name this fact carries on the executor
            # `/execute` contract; `not_run` is the name it carries here. Read
            # both, or a runner that reports the executor's spelling has its
            # answer silently dropped at this boundary.
            not_run=_read_not_run(result),
            not_run_reason=result.get("not_run_reason"),
        )

    async def _build_result_from_raw(
        self,
        raw: dict[str, Any],
        request: TestExecutionRequest,
        test_id: str,
        start_time: datetime,
    ) -> TestExecutionResult:
        """Convert raw K8s execution output into TestExecutionResult."""

        test_results = await self._parse_test_output(raw.get("stdout", ""), request.language)

        execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

        return TestExecutionResult(
            test_id=test_id,
            success=raw.get("exit_code", 1) == 0,
            exit_code=raw.get("exit_code", 1),
            stdout=raw.get("stdout", ""),
            stderr=raw.get("stderr", ""),
            execution_time_ms=raw.get("execution_time_ms", execution_time_ms),
            coverage_report=None,
            test_results=test_results,
            performance_metrics=None,
            security_issues=[],
            container_id=None,
            container_name=request.container_name,
            language=request.language,
            test_type=request.test_type,
            created_at=start_time,
            # Same two spellings as the Docker path above (see `_read_not_run`).
            not_run=_read_not_run(raw),
            not_run_reason=raw.get("not_run_reason"),
        )

    async def _cleanup_container(self, test_id: str, container_id: str) -> None:
        """Cleanup test container and resources."""
        try:
            await self.docker_service.stop_container(container_id)
            await self.docker_service.remove_container_by_id(container_id)

            # Remove from active containers
            if test_id in self.active_containers:
                del self.active_containers[test_id]
            self.active_container_tenants.pop(test_id, None)

            logger.debug(f"Cleaned up container {container_id} for test {test_id}")

        except Exception as e:
            logger.warning(f"Failed to cleanup container {container_id}: {e}")

    def _get_test_command(
        self, language: TestLanguage, test_type: OrchestratorType
    ) -> list[str] | None:
        """Get language-specific test command, or None when no command is defined.

        Returning None (rather than a command that exits 0) is what makes an
        unmapped (language, test_type) pair unable to be graded as a pass.
        """
        commands = {
            TestLanguage.GO: {
                OrchestratorType.UNIT: ["/usr/local/bin/go-test-runner.sh", "unit"],
                OrchestratorType.BENCHMARK: [
                    "/usr/local/bin/go-test-runner.sh",
                    "benchmark",
                ],
                OrchestratorType.COVERAGE: [
                    "/usr/local/bin/go-test-runner.sh",
                    "coverage",
                ],
                OrchestratorType.RACE: ["/usr/local/bin/go-test-runner.sh", "race"],
                OrchestratorType.SECURITY: [
                    "/usr/local/bin/go-test-runner.sh",
                    "security",
                ],
                OrchestratorType.VET: ["/usr/local/bin/go-test-runner.sh", "vet"],
            },
            TestLanguage.RUST: {
                OrchestratorType.UNIT: [
                    "/usr/local/bin/rust-test-runner.sh",
                    "unit",
                ],
                OrchestratorType.DOC: ["/usr/local/bin/rust-test-runner.sh", "doc"],
                OrchestratorType.BENCHMARK: [
                    "/usr/local/bin/rust-test-runner.sh",
                    "benchmark",
                ],
                OrchestratorType.COVERAGE: [
                    "/usr/local/bin/rust-test-runner.sh",
                    "coverage",
                ],
                OrchestratorType.SECURITY: [
                    "/usr/local/bin/rust-test-runner.sh",
                    "security",
                ],
                OrchestratorType.CLIPPY: [
                    "/usr/local/bin/rust-test-runner.sh",
                    "clippy",
                ],
                OrchestratorType.FMT: ["/usr/local/bin/rust-test-runner.sh", "fmt"],
            },
            TestLanguage.PYTHON: {
                OrchestratorType.UNIT: [
                    "python",
                    "-m",
                    "pytest",
                    "-v",
                    "--tb=short",
                ],
                OrchestratorType.COVERAGE: [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=.",
                    "--cov-report=json",
                ],
            },
            TestLanguage.JAVASCRIPT: {
                OrchestratorType.UNIT: ["npm", "test"],
                OrchestratorType.COVERAGE: ["npm", "run", "test:coverage"],
            },
            TestLanguage.TYPESCRIPT: {
                OrchestratorType.UNIT: ["npm", "test"],
                OrchestratorType.COVERAGE: ["npm", "run", "test:coverage"],
            },
        }

        # No fallback command. An unmapped pair yields None, which every caller
        # turns into a not-run (non-passing) result.
        return commands.get(language, {}).get(test_type)

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
            "csharp": ".cs",
            "cpp": ".cpp",
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
        }
        return extensions.get(language, ".txt")

    def _format_go_dependencies(self, dependencies: dict[str, str]) -> str:
        """Format Go dependencies for go.mod."""
        if not dependencies:
            return ""

        lines = []
        for dep, version in dependencies.items():
            lines.append(f"require {dep} {version}")
        return "\n".join(lines)

    def _format_rust_dependencies(self, dependencies: dict[str, str]) -> str:
        """Format Rust dependencies for Cargo.toml."""
        if not dependencies:
            return ""

        lines = []
        for dep, version in dependencies.items():
            lines.append(f'{dep} = "{version}"')
        return "\n".join(lines)

    async def _collect_coverage_report(
        self, container_id: str, language: TestLanguage
    ) -> dict[str, Any] | None:
        """Collect coverage report from container."""
        try:
            if language == TestLanguage.GO:
                # Try to read Go coverage output
                result = await self.docker_service.exec_in_container(
                    container_id, ["cat", "/tmp/coverage.out"], timeout=10
                )
                if result.get("exit_code") == 0:
                    return {
                        "format": "go_coverage",
                        "data": result.get("stdout", ""),
                    }

            elif language == TestLanguage.RUST:
                # Try to read Rust tarpaulin output
                result = await self.docker_service.exec_in_container(
                    container_id,
                    ["cat", "/tmp/tarpaulin-report.json"],
                    timeout=10,
                )
                if result.get("exit_code") == 0:
                    return {
                        "format": "tarpaulin_json",
                        "data": result.get("stdout", ""),
                    }

        except Exception as e:
            logger.warning(f"Failed to collect coverage report: {e}")

        return None

    async def _collect_performance_metrics(
        self, container_id: str, language: TestLanguage
    ) -> dict[str, float] | None:
        """Collect performance metrics from container."""
        try:
            if language == TestLanguage.GO:
                # Parse Go benchmark output
                result = await self.docker_service.exec_in_container(
                    container_id, ["cat", "/tmp/test-output.json"], timeout=10
                )
                if result.get("exit_code") == 0:
                    # Basic parsing - would need more sophisticated parsing in production
                    return {"benchmark_data": len(result.get("stdout", ""))}

        except Exception as e:
            logger.warning(f"Failed to collect performance metrics: {e}")

        return None

    async def _collect_security_issues(
        self, container_id: str, language: TestLanguage
    ) -> list[str]:
        """Collect security issues from container."""
        issues = []

        try:
            if language == TestLanguage.GO:
                # Read golangci-lint output
                result = await self.docker_service.exec_in_container(
                    container_id,
                    ["cat", "/tmp/security-report.json"],
                    timeout=10,
                )
                if result.get("exit_code") == 0 and result.get("stdout"):
                    issues.append(f"golangci-lint: {len(result.get('stdout', ''))} bytes of output")

            elif language == TestLanguage.RUST:
                # Read cargo audit output
                result = await self.docker_service.exec_in_container(
                    container_id, ["cat", "/tmp/audit-report.json"], timeout=10
                )
                if result.get("exit_code") == 0 and result.get("stdout"):
                    issues.append(f"cargo-audit: {len(result.get('stdout', ''))} bytes of output")

        except Exception as e:
            logger.warning(f"Failed to collect security issues: {e}")

        return issues

    async def _parse_test_output(self, stdout: str, language: TestLanguage) -> list[dict[str, Any]]:
        """Parse test output into structured results."""
        results: list[dict[str, Any]] = []

        if not stdout:
            return results

        try:
            # Basic parsing - would need more sophisticated parsing in production
            if language == TestLanguage.GO:
                # Go test output parsing
                lines = stdout.split("\n")
                for line in lines:
                    if "PASS" in line or "FAIL" in line:
                        results.append({"line": line, "parsed": True})

            elif language == TestLanguage.RUST:
                # Rust test output parsing
                lines = stdout.split("\n")
                for line in lines:
                    if "test result:" in line:
                        results.append({"line": line, "parsed": True})

        except Exception as e:
            logger.warning(f"Failed to parse test output: {e}")

        return results


def create_test_orchestrator(docker_service: DockerService, config: TestConfig) -> TestOrchestrator:
    """Factory function to create TestOrchestrator."""
    return TestOrchestrator(docker_service, config)
