# /src/services/testing/testpy_integration_adapter.py
"""
Integration Adapter for Python Test Suite
Bridges your existing pythonTestSuite with the code_graph_helper architecture.
"""

import asyncio
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid7

# Import your existing Python test suite components
# Adjust these paths based on your actual pythonTestSuite structure
try:
    import sys

    sys.path.append("../../python_test_suite/src")

    from config.test_framework_config import MetricsManager
    from config.test_framework_config import (
        SecurityManager as OriginalSecurityManager,
    )
    from config.test_framework_config import (
        TestCacheManager as OriginalCacheManager,
    )
    from config.test_framework_config import (
        TestPyConfig as OriginalTestPyConfig,
    )
    from config.test_framework_config import (
        TestRunner as OriginalTestRunner,
    )

except ImportError as e:
    # Create temporary logger for error before module logger is initialized
    import logging

    _temp_logger = logging.getLogger(__name__)
    _temp_logger.warning(f"Could not import Python test suite components: {e}")

    # Provide fallback implementations
    class FallbackTestRunner:
        """Fallback implementation for running tests when TestRunner is not available."""

        def run_tests(
            self, code: str, test_code: str, timeout: int | None = None
        ) -> dict[str, Any]:
            """Run tests on the provided code and test code with a timeout."""
            return {"success": False, "error": "TestRunner not available"}

    class FallbackSecurityManager:
        """Fallback implementation for security validation when SecurityManager is not available."""

        def validate_code(self, code: str) -> Any:
            """Validate the provided code for security issues."""
            return {"valid": True}

    class FallbackTestPyConfig:
        """Fallback configuration for TestPy when original config is not available."""

    class FallbackCacheManager:
        """Fallback cache manager implementation when original cache manager is not available."""

    class FallbackMetricsManager:
        """Fallback metrics manager implementation when original metrics manager is not available."""

    # Use fallback classes
    OriginalTestRunner: Any = FallbackTestRunner
    OriginalSecurityManager: Any = FallbackSecurityManager
    OriginalTestPyConfig: Any = FallbackTestPyConfig
    OriginalCacheManager: Any = FallbackCacheManager
    MetricsManager: Any = FallbackMetricsManager


from models.common_responses_models import ResourceMetrics
from models.testing_models import (
    SecurityLevel,
    TestRequest,
    TestResult,
)
from services.stubs import DockerService

logger = logging.getLogger(__name__)


class TestPyIntegrationAdapter:
    """
    Adapter that integrates your existing Python test suite functionality
    with the code_graph_helper Docker-based execution environment.
    """

    def __init__(
        self,
        docker_service: DockerService,
        test_runner: Any | None = None,
        security_manager: Any | None = None,
        metrics_manager: Any | None = None,
        config: Any | None = None,
    ):
        self.docker_service = docker_service
        self.config = config or self._get_default_config()
        self.test_runner = test_runner
        self.security_manager = security_manager
        self.metrics_manager = metrics_manager

        # Initialize your existing components
        self._init_testpy_components()

        logger.info("TestPy Integration Adapter initialized")

    def _get_default_config(self):
        """Get default configuration for TestPy integration."""
        return {
            "security": {
                "max_execution_time": 30,
                "max_memory_mb": 512,
                "network_access": False,
            },
            "testing": {
                "default_timeout": 30,
                "hypothesis_enabled": True,
            },
        }

    def _init_testpy_components(self):
        """Initialize your existing Python test suite components."""
        try:
            # Initialize your existing components if not provided
            if not self.test_runner:
                self.test_runner = OriginalTestRunner()
            if not self.security_manager:
                self.security_manager = OriginalSecurityManager()
            # Initialize metrics if available and not provided
            if not self.metrics_manager:
                try:
                    self.metrics_manager = MetricsManager()
                except Exception as e:
                    logger.warning(
                        f"MetricsManager initialization failed, metrics will be disabled: {e}"
                    )
                    self.metrics_manager = None

        except Exception as e:
            logger.warning(f"Could not initialize TestPy components: {e}")
            # Use fallback implementations
            self.test_runner = OriginalTestRunner()
            self.security_manager = OriginalSecurityManager()
            self.metrics_manager = None

    def _convert_config_format(self) -> Any:
        """Convert our TestPyConfig to your original config format."""
        try:
            # Create a config object that matches your original structure
            class LegacyConfig:
                """Legacy configuration adapter for compatibility with original config structure."""

                def __init__(self, config: Any):
                    """Initialize with the original TestPy configuration."""
                    self.security = config.security
                    self.test = config.test
                    self.logging = config.logging

            return LegacyConfig(self.config)

        except Exception as e:
            logger.warning(f"Config conversion failed: {e}")
            return self.config  # Fallback to new config

    def _create_legacy_logger(self):
        """Create a logger adapter for your existing logging system."""

        class LoggerAdapter:
            """Adapter for integrating with the existing logging system."""

            def __init__(self, logger):
                """Initialize the adapter with a logger instance."""
                self.logger = logger

            def info(self, msg, **kwargs):
                """Log an info message."""
                self.logger.info(msg)

            def error(self, msg, **kwargs):
                """Log an error message."""
                self.logger.error(msg)

            def warning(self, msg, **kwargs):
                """Log a warning message."""
                self.logger.warning(msg)

            def debug(self, msg, **kwargs):
                """Log a debug message."""
                self.logger.debug(msg)

            def component_context(self, component: str):
                """Provide context manager for component logging."""
                return self

            def __enter__(self):
                """Enter the context manager."""
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                """Exit the context manager, handling exceptions."""
                pass

        return LoggerAdapter(logger)

    async def execute_test_with_testpy(self, request: TestRequest) -> TestResult:
        """
        Execute a test using your existing TestPy logic but in Docker containers.
        This bridges your proven test execution with container isolation.
        """
        test_id = str(uuid7())

        try:
            logger.info(f"Executing test {test_id} using TestPy integration")

            # Step 1: Security validation using your existing security manager
            await self._validate_with_testpy_security(request)

            # Step 2: Prepare test environment using your patterns
            test_env = await self._prepare_testpy_environment(request)

            # Step 3: Execute in container using your test runner logic
            result = await self._execute_in_container_with_testpy(test_id, request, test_env)

            # Step 4: Record metrics using your existing metrics system
            if self.metrics_manager:
                await self._record_testpy_metrics(test_id, result)

            return result

        except Exception as e:
            logger.error(f"TestPy execution failed for {test_id}: {e}")
            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time=0.0,
                created_at=datetime.now(UTC),
                language=request.language,
                security_level=request.security_level or SecurityLevel.MEDIUM,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=None,
                quality_report=None,
                error=str(e),
            )

    async def _validate_with_testpy_security(self, request: TestRequest) -> None:
        """Use your existing security validation logic."""
        try:
            if request.code and self.security_manager:
                # Use your existing validation
                validation_result = self.security_manager.validate_code(request.code)

                if hasattr(validation_result, "passed") and not validation_result.passed:
                    raise ValueError(f"Security validation failed: {validation_result.error}")
                elif hasattr(validation_result, "valid") and not validation_result.valid:
                    raise ValueError("Security validation failed")

            if request.test_code and self.security_manager:
                validation_result = self.security_manager.validate_code(request.test_code)

                if hasattr(validation_result, "passed") and not validation_result.passed:
                    raise ValueError(
                        f"Test code security validation failed: {validation_result.error}"
                    )
                elif hasattr(validation_result, "valid") and not validation_result.valid:
                    raise ValueError("Test code security validation failed")

        except Exception as e:
            logger.error(f"Security validation error: {e}")
            raise ValueError(f"Security validation failed: {str(e)}")

    async def _prepare_testpy_environment(self, request: TestRequest) -> dict[str, Any]:
        """Prepare test environment using your existing patterns."""
        # Create temporary directory for test files
        temp_dir = tempfile.mkdtemp(prefix=f"testpy_env_{uuid7().hex[:8]}_")
        temp_path = Path(temp_dir)

        try:
            # Get code content
            code_content = request.code
            if request.code_file_path:
                code_content = Path(request.code_file_path).read_text()

            # Get test content
            test_content = request.test_code
            if request.test_file_path:
                test_content = Path(request.test_file_path).read_text()

            # Ensure test_content is not None
            if not test_content:
                test_content = "# Empty test"

            # Use your existing test processing logic
            if self.test_runner and hasattr(self.test_runner, "_process_test_code"):
                module_name = f"test_module_{uuid7().hex[:8]}"
                processed_test_code = self.test_runner._process_test_code(test_content, module_name)  # type: ignore[call-non-callable]  # ty: ignore[not-callable]
            else:
                # Fallback processing
                module_name = f"test_module_{uuid7().hex[:8]}"
                if "import" not in test_content:
                    processed_test_code = f"from {module_name} import *\n\n{test_content}"
                else:
                    processed_test_code = test_content

            # Ensure processed_test_code is not None
            if not processed_test_code:
                processed_test_code = "# Empty processed test"

            # Create files
            code_file = temp_path / f"{module_name}.py"
            test_file = temp_path / f"test_{module_name}.py"

            code_file.write_text(code_content or "")
            test_file.write_text(processed_test_code or "")

            return {
                "temp_dir": temp_dir,
                "code_file": str(code_file),
                "test_file": str(test_file),
                "module_name": module_name,
                "test_content": processed_test_code,
                "code_content": code_content,
            }

        except Exception as e:
            # Cleanup on error
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    async def _execute_in_container_with_testpy(
        self, test_id: str, request: TestRequest, test_env: dict[str, Any]
    ) -> TestResult:
        """Execute test in container using your TestPy patterns."""
        docker_id = None
        start_time = asyncio.get_event_loop().time()

        try:
            # Get container image based on language
            image_name = self.docker_service.get_language_container_image(request.language)

            # Create container configuration for TestPy execution
            container_config = {
                "image": image_name,
                "language": request.language,
                "mem_limit": "512m",
                "cpu_quota": 50000,
                "network_disabled": True,
                "read_only": False,  # Need write access for test files
                "labels": {
                    "test_id": test_id,
                    "type": "testpy_execution",
                    "integration": "testpy",
                },
                "working_dir": "/app/test",
                "tmpfs": {"/tmp": "noexec,nosuid,size=100m"},
            }

            # Create test container using DockerService
            docker_id = await self.docker_service.create_test_container(container_config)
            logger.info(f"Created TestPy container {docker_id} for test {test_id}")

            # Start the container
            await self.docker_service.start_container(docker_id)

            # Copy test files to container
            test_file_content = test_env.get("test_content", "")
            test_file_name = Path(test_env["test_file"]).name

            await self.docker_service.create_file_in_container(
                docker_id, f"/app/test/{test_file_name}", test_file_content
            )

            # Copy source code if provided
            if request.code:
                source_file = "main.py" if request.language == "python" else "main.js"
                await self.docker_service.create_file_in_container(
                    docker_id, f"/app/test/{source_file}", request.code
                )

            # Use your existing test execution logic adapted for containers
            result = await self._run_testpy_in_container(docker_id, test_env, request)

            execution_time = asyncio.get_event_loop().time() - start_time

            return TestResult(
                test_id=test_id,
                success=result["success"],
                stdout=result.get("output") or "",
                stderr=result.get("error") or "",
                exit_code=0 if result["success"] else 1,
                execution_time=execution_time,
                created_at=datetime.now(UTC),
                language=request.language,
                security_level=request.security_level or SecurityLevel.MEDIUM,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=docker_id,
                quality_report=None,
                error=(result.get("error", "") if not result["success"] else None),
            )

        except TimeoutError:
            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr=f"Test execution timed out after {request.timeout}s",
                exit_code=124,  # timeout exit code
                execution_time=float(request.timeout or 30),
                created_at=datetime.now(UTC),
                language=request.language,
                security_level=request.security_level or SecurityLevel.MEDIUM,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=None,
                quality_report=None,
                error=f"Test execution timed out after {request.timeout}s",
            )
        except Exception as e:
            logger.error(f"TestPy container execution failed: {e}")
            return TestResult(
                test_id=test_id,
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time=asyncio.get_event_loop().time() - start_time,
                created_at=datetime.now(UTC),
                language=request.language,
                security_level=request.security_level or SecurityLevel.MEDIUM,
                coverage_data=None,
                test_metadata=None,
                test_type=None,
                container_name=None,
                quality_report=None,
                error=str(e),
            )

        finally:
            # Cleanup
            if docker_id:
                await self._cleanup_testpy_container(docker_id)
            await self._cleanup_testpy_environment(test_env)

    async def _run_testpy_in_container(
        self, docker_id: str, test_env: dict[str, Any], request: TestRequest
    ) -> dict[str, Any]:
        """Run tests using your existing TestPy command structure."""

        test_file_name = Path(test_env["test_file"]).name

        # Build command similar to your existing test runner
        if request.language == "python":
            cmd = [
                "python",
                "-m",
                "pytest",
                f"/app/test/{test_file_name}",
                "-v",
            ]

            # Add hypothesis support if requested
            if request.with_hypothesis:
                cmd.extend(["--hypothesis-show-statistics"])

        else:
            # Support for other languages
            cmd = [
                "python",
                "-m",
                "pytest",
                f"/app/test/{test_file_name}",
                "-v",
            ]

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.docker_service.exec_in_container(docker_id, cmd),
                timeout=request.timeout,
            )

            return {
                "success": result.get("exit_code", 1) == 0,
                "output": result.get("stdout", ""),
                "error": (result.get("stderr") if result.get("exit_code", 1) != 0 else None),
            }

        except TimeoutError:
            raise
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _build_testpy_container_config(
        self, request: TestRequest, test_env: dict[str, Any]
    ) -> dict[str, Any]:
        """Build container config using your security settings."""

        config = {
            "image": "code-graph-test-python:latest",  # Use your container
            "working_dir": "/app/test",
            "volumes": {test_env["temp_dir"]: {"bind": "/app/test", "mode": "rw"}},
            "environment": {
                "PYTHONPATH": "/app/test:/app/testpy",
                "TEST_MODE": "true",
                "TESTPY_MODE": "true",
            },
            "mem_limit": f"{self.config.security.max_memory_mb}m",
            "cpu_quota": 50000,  # 0.5 CPU
            "network_disabled": not self.config.security.network_access,
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],
            "read_only": True,
            "tmpfs": {"/tmp": "size=100m,noexec,nosuid,nodev"},
            "user": "testuser",
        }

        return config

    async def _get_container_resource_metrics(self, docker_id: str) -> ResourceMetrics:
        """Get resource metrics from container."""
        try:
            # Use existing Docker service method if available
            if hasattr(self.docker_service, "get_container_stats"):
                stats = await self.docker_service.get_container_stats(docker_id)
                return ResourceMetrics(
                    cpu_usage=stats.get("cpu_percent", 0.0),
                    memory_usage=stats.get("memory_usage", 0) / (1024 * 1024),
                    disk_usage=0.0,
                    network_connections=0,
                )
        except Exception as e:
            logger.warning(f"Failed to get resource metrics: {e}")

        return ResourceMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_connections=0,
        )

    async def _record_testpy_metrics(self, test_id: str, result: TestResult) -> None:
        """Record metrics using your existing metrics system."""
        if not self.metrics_manager:
            return

        try:
            # Use your existing metrics recording
            if hasattr(self.metrics_manager, "record_metric"):
                self.metrics_manager.record_metric(  # type: ignore[call-non-callable]
                    "test_execution_time",
                    result.execution_time,
                    "seconds",
                    f"Test execution time for {test_id}",
                )

                self.metrics_manager.record_metric(  # type: ignore[call-non-callable]
                    "test_success_rate",
                    1.0 if result.success else 0.0,
                    "ratio",
                    f"Test success for {test_id}",
                )

                if result.test_metadata and "memory_usage_mb" in result.test_metadata:
                    self.metrics_manager.record_metric(  # type: ignore[call-non-callable]
                        "test_memory_usage",
                        result.test_metadata["memory_usage_mb"],
                        "MB",
                        f"Memory usage for {test_id}",
                    )

        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")

    async def _cleanup_testpy_container(self, docker_id: str) -> None:
        """Cleanup container using existing patterns."""
        try:
            await self.docker_service.stop_container(docker_id)
            await self.docker_service.remove_container_by_id(docker_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup container {docker_id}: {e}")

    async def _cleanup_testpy_environment(self, test_env: dict[str, Any]) -> None:
        """Cleanup test environment using your existing cleanup patterns."""
        try:
            import shutil

            if "temp_dir" in test_env:
                shutil.rmtree(test_env["temp_dir"], ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup test environment: {e}")

    def get_testpy_config(self) -> dict[str, Any]:
        """Get your original TestPy configuration."""
        return {
            "security": {
                "max_memory_mb": self.config.security.max_memory_mb,
                "max_execution_time": self.config.security.max_execution_time,
                "allowed_modules": self.config.security.allowed_modules,
                "sandbox_enabled": self.config.security.sandbox_enabled,
            },
            "test": {
                "timeout": self.config.test.timeout,
                "parallel": self.config.test.parallel,
                "max_workers": self.config.test.max_workers,
            },
            "logging": {
                "level": self.config.logging.level.value,
                "console_output": self.config.logging.console_output,
            },
        }

    async def health_check(self) -> dict[str, Any]:
        """Health check for TestPy integration."""
        try:
            return {
                "testpy_integration": "healthy",
                "test_runner_available": self.test_runner is not None,
                "security_manager_available": self.security_manager is not None,
                "metrics_manager_available": self.metrics_manager is not None,
                "docker_service_available": self.docker_service.is_docker_available(),
                "config_valid": True,
            }
        except Exception as e:
            return {"testpy_integration": "unhealthy", "error": str(e)}


# Factory function for integration
def create_testpy_adapter(
    docker_service: DockerService,
    test_runner: Any | None = None,
    security_manager: Any | None = None,
    metrics_manager: Any | None = None,
    config: Any | None = None,
) -> TestPyIntegrationAdapter:
    """Create TestPy integration adapter."""
    return TestPyIntegrationAdapter(
        docker_service=docker_service,
        test_runner=test_runner,
        security_manager=security_manager,
        metrics_manager=metrics_manager,
        config=config,
    )
