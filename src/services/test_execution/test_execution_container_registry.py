"""
Container Registry Service - Phase 1 Implementation
Manages container specifications and configurations for multi-language test execution.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from config.settings import get_settings

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp: Any = None

try:
    import docker

    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False
    docker: Any = None

logger = logging.getLogger(__name__)


class ContainerTestExecutionStatus(Enum):
    """Status of a test execution container."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TestExecutionSpec:
    """Specification for a test execution container."""

    image: str
    version: str
    memory_limit: str
    cpu_quota: int
    timeout_seconds: int
    environment: dict[str, str]
    volumes: dict[str, str]
    working_dir: str = "/workspace"
    user: str = "1001:1001"
    entrypoint: list[str] | None = None
    supported_test_types: list[str] | None = None
    mounts: list[str] | None = None
    security_opts: list[str] | None = None

    def __post_init__(self):
        """Initialize defaults for optional fields."""
        if self.mounts is None:
            self.mounts = []
        if self.security_opts is None:
            self.security_opts = ["no-new-privileges:true"]


class ContainerExecutionStatus(StrEnum):
    """Container status enumeration."""

    AVAILABLE = "available"
    BUILDING = "building"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class TestExecutionContainerRegistry:
    """
    Registry service for managing test execution container specifications.
    Provides centralized configuration and availability checking for language containers.
    """

    def __init__(self, enable_periodic_refresh: bool = True):
        """Initialize the TestExecutionContainerRegistry service.

        This initializes the TestExecutionContainerRegistry with required dependencies
        and configuration for service operations.

        Args:
            enable_periodic_refresh: Whether to enable periodic health check refresh

        Raises:
            Exception: If operation encounters an error
        """
        self.docker_client = None
        self.specs = self._load_container_specs()
        self.container_status: dict[str, ContainerExecutionStatus] = {}
        self.last_health_check = 0
        self.health_check_interval = 30
        self._refresh_thread = None
        self._stop_refresh = False
        self._initial_check_delay = 60

        # Check if Kubernetes mode is enabled; skip Docker/HTTP executors in that case
        k8s_mode = os.environ.get("KUBERNETES_IN_CLUSTER", "false").lower() == "true"

        # Check if we're in CI/test mode where test executors are not available
        # CI environments typically don't have the test executor containers running
        ci_mode = os.environ.get("CI", "false").lower() == "true"
        skip_executors = os.environ.get("SKIP_TEST_EXECUTORS", "false").lower() == "true"

        if k8s_mode or ci_mode or skip_executors:
            mode_name = "K8s" if k8s_mode else ("CI" if ci_mode else "SKIP_TEST_EXECUTORS")
            logger.info(f"{mode_name} mode - skipping Docker/HTTP test executors")
            self._docker_available = False
            self.docker_client = None
            # Mark all services as unavailable (tests will use K8s Jobs or be skipped)
            for lang in self.specs:
                self.container_status[lang] = ContainerExecutionStatus.UNAVAILABLE
            return

        # Initialize Docker client
        self._docker_available = False
        if HAS_DOCKER:
            try:
                self.docker_client = docker.from_env()  # type: ignore[possibly-missing-attribute]
                # Verify Docker daemon is actually reachable
                self.docker_client.ping()
                self._docker_available = True
                logger.info("TestExecutionContainerRegistry initialized with Docker client")
            except Exception as e:
                logger.warning(f"Docker client initialization failed: {e}")
                self.docker_client = None
        else:
            logger.warning("docker module not available - Docker functionality disabled")
            self.docker_client = None

        # Check initial container availability
        self._check_container_availability()

        # Give containers a moment to start up (skip in test/CI environment without Docker)
        if self._docker_available:
            logger.info(
                f"Waiting {self._initial_check_delay} seconds for containers to initialize..."
            )
            time.sleep(self._initial_check_delay)
        else:
            logger.info("Docker not available, skipping container initialization delay")

        # Do an immediate health check after initial delay
        logger.info("Performing initial health check of all services...")
        self.check_all_services_health_sync()

        # Start periodic health check refresh if enabled
        if enable_periodic_refresh:
            self._start_periodic_refresh()

    def _load_container_specs(self) -> dict[str, TestExecutionSpec]:
        """Load container specifications for all supported languages."""
        settings = get_settings()
        cpu_quota_standard = getattr(settings, "container_cpu_quota", 50000)
        cpu_quota_high = getattr(settings, "container_cpu_quota_high", 75000)

        return {
            "go": TestExecutionSpec(
                image="codegraph/test-executor-go:latest",
                version="1.21",
                memory_limit="512m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=300,
                environment={
                    "GOCACHE": "/tmp/go-cache",
                    "GOMODCACHE": "/tmp/go-mod-cache",
                    "GOPATH": "/go",
                    "PATH": "/usr/local/go/bin:/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/go-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "benchmark",
                    "coverage",
                    "race",
                    "security",
                    "vet",
                ],
            ),
            "rust": TestExecutionSpec(
                image="codegraph/test-executor-rust:latest",
                version="1.75",
                memory_limit="1g",
                cpu_quota=cpu_quota_high,
                timeout_seconds=600,
                environment={
                    "CARGO_HOME": "/tmp/cargo",
                    "CARGO_TARGET_DIR": "/tmp/target",
                    "RUST_BACKTRACE": "1",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/rust-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "doc",
                    "benchmark",
                    "coverage",
                    "security",
                    "clippy",
                    "fmt",
                ],
            ),
            "java": TestExecutionSpec(
                image="codegraph/test-executor-java:latest",
                version="17",
                memory_limit="1g",
                cpu_quota=cpu_quota_high,
                timeout_seconds=600,
                environment={
                    "GRADLE_USER_HOME": "/tmp/gradle",
                    "MAVEN_OPTS": "-Xmx512m",
                    "JAVA_OPTS": "-Xmx512m",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/java-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "integration",
                    "coverage",
                    "security",
                ],
            ),
            "csharp": TestExecutionSpec(
                image="codegraph/test-executor-csharp:latest",
                version="8.0",
                memory_limit="768m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=450,
                environment={
                    "DOTNET_CLI_HOME": "/tmp/dotnet",
                    "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
                    "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/csharp-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "integration",
                    "coverage",
                    "security",
                ],
            ),
            "cpp": TestExecutionSpec(
                image="codegraph/test-executor-cpp:latest",
                version="20",
                memory_limit="512m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=300,
                environment={
                    "CXX": "g++",
                    "CC": "gcc",
                    "CMAKE_BUILD_TYPE": "Debug",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/cpp-test-runner.sh"],
                supported_test_types=["unit", "coverage", "security"],
            ),
            "python": TestExecutionSpec(
                image="codegraph/test-executor-python:latest",
                version="3.11",
                memory_limit="512m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=300,
                environment={
                    "PYTHONPATH": "/workspace",
                    "PYTHONUNBUFFERED": "1",
                    "PYTEST_CURRENT_TEST": "1",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/python-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "integration",
                    "coverage",
                    "security",
                ],
            ),
            "nodejs": TestExecutionSpec(
                image="codegraph/test-executor-nodejs:latest",
                version="20",
                memory_limit="512m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=300,
                environment={
                    "NODE_ENV": "test",
                    "NPM_CONFIG_CACHE": "/tmp/npm",
                    "NODE_OPTIONS": "--max-old-space-size=256",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/nodejs-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "integration",
                    "coverage",
                    "security",
                ],
            ),
            "javascript": TestExecutionSpec(
                image="codegraph/test-executor-nodejs:latest",
                version="20",
                memory_limit="512m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=300,
                environment={
                    "NODE_ENV": "test",
                    "NPM_CONFIG_CACHE": "/tmp/npm",
                    "NODE_OPTIONS": "--max-old-space-size=256",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/nodejs-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "integration",
                    "coverage",
                    "security",
                ],
            ),
            "typescript": TestExecutionSpec(
                image="codegraph/test-executor-nodejs:latest",
                version="20",
                memory_limit="512m",
                cpu_quota=cpu_quota_standard,
                timeout_seconds=300,
                environment={
                    "NODE_ENV": "test",
                    "NPM_CONFIG_CACHE": "/tmp/npm",
                    "NODE_OPTIONS": "--max-old-space-size=256",
                    "TS_NODE_TRANSPILE_ONLY": "true",
                },
                volumes={"/tmp": "/tmp"},
                entrypoint=["/usr/local/bin/nodejs-test-runner.sh"],
                supported_test_types=[
                    "unit",
                    "integration",
                    "coverage",
                    "security",
                ],
            ),
        }

    def get_container_spec(self, language: str) -> TestExecutionSpec | None:
        """Get container specification for a specific language."""
        return self.specs.get(language.lower())

    def get_supported_languages(self) -> list[str]:
        """Get list of all supported languages."""
        return list(self.specs.keys())

    def get_available_languages(self) -> list[str]:
        """Get list of languages with available containers."""
        available = []
        for language, status in self.container_status.items():
            if status == ContainerExecutionStatus.AVAILABLE:
                available.append(language)
        return available

    def get_container_status(self, language: str) -> ContainerExecutionStatus:
        """Get current status of a container."""
        return self.container_status.get(language.lower(), ContainerExecutionStatus.UNAVAILABLE)

    def is_container_available(self, language: str) -> bool:
        """Check if a container is available for use."""
        return self.get_container_status(language) == ContainerExecutionStatus.AVAILABLE

    def refresh_container_availability(self) -> None:
        """Refresh the availability status of all containers."""
        self._check_container_availability()

    def get_supported_test_types(self, language: str) -> list[str]:
        """Get supported test types for a language."""
        spec = self.get_container_spec(language)
        if spec and spec.supported_test_types:
            return spec.supported_test_types
        return ["unit"]  # Default fallback

    def _check_container_availability(self) -> None:
        """Check availability of all registered containers via docker-compose services."""
        if not self.docker_client:
            # Mark all as unavailable if no Docker client
            for language in self.specs:
                self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE
            return

        # Map languages to their actual docker-compose service container names
        service_mappings = {
            "python": "python-test-executor",
            "javascript": "javascript-test-executor",
            "typescript": "javascript-test-executor",
            "nodejs": "javascript-test-executor",
            "java": "java-test-executor",
            "rust": "rust-test-executor",
            "csharp": "csharp-test-executor",
            "go": "go-test-executor",
            "cpp": "cpp-test-executor",  # Enable C++ support
        }

        for language, spec in self.specs.items():
            service_name = service_mappings.get(language)
            if not service_name:
                self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE
                logger.warning(f"No service mapping for language: {language}")
                continue

            try:
                # Check if the docker-compose service container is running
                containers = self.docker_client.containers.list(
                    filters={"name": f"code_graph_helper-{service_name}-1"}
                )

                if containers and containers[0].status == "running":
                    # Container is running - assume building state since this is sync context
                    # Health checks are performed asynchronously by the periodic refresh thread
                    self.container_status[language] = ContainerExecutionStatus.BUILDING
                    logger.debug(
                        f"Service {service_name} is running but not ready - {language} building"
                    )
                else:
                    self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE
                    logger.warning(f"Service {service_name} not running - {language} unavailable")

            except Exception as e:
                self.container_status[language] = ContainerExecutionStatus.FAILED
                logger.error(f"Error checking service {service_name} for {language}: {e}")

    def get_service_health_check(self, language: str) -> str | None:
        """Get the health check endpoint URL for a language service."""
        service_mappings = {
            "python": "http://python-test-executor:8087/health",
            "javascript": "http://javascript-test-executor:8085/health",
            "typescript": "http://javascript-test-executor:8085/health",
            "nodejs": "http://javascript-test-executor:8085/health",
            "java": "http://java-test-executor:8088/health",
            "rust": "http://rust-test-executor:8089/health",
            "csharp": "http://csharp-test-executor:8090/health",
            "go": "http://go-test-executor:8091/health",
            "cpp": "http://cpp-test-executor:8094/health",
        }
        return service_mappings.get(language.lower())

    async def check_service_health(self, language: str, timeout: float = 5.0) -> bool:
        """Check if a test executor service is healthy via HTTP health check."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not available - cannot perform health checks")
            return False

        health_url = self.get_service_health_check(language)
        if not health_url:
            return False

        # Go services take longer to start, increase timeout
        if language == "go":
            timeout = 10.0

        try:
            async with aiohttp.ClientSession(  # type: ignore[possibly-missing-attribute]
                timeout=aiohttp.ClientTimeout(total=timeout, connect=timeout / 2)  # type: ignore[possibly-missing-attribute]
            ) as session:
                async with session.get(health_url) as response:
                    is_healthy = response.status == 200

                    # Update container status based on health check result
                    if is_healthy:
                        if (
                            self.container_status.get(language)
                            != ContainerExecutionStatus.AVAILABLE
                        ):
                            logger.debug(f"Service {language} is now available")
                            self.container_status[language] = ContainerExecutionStatus.AVAILABLE
                        logger.debug(f"Service health check passed for {language}")
                    else:
                        if self.container_status.get(language) != ContainerExecutionStatus.BUILDING:
                            self.container_status[language] = ContainerExecutionStatus.BUILDING
                        logger.warning(
                            f"Service health check failed for {language}: {response.status}"
                        )
                    return is_healthy
        except Exception as e:
            # On error, mark as building if container exists, otherwise unavailable
            if self.container_status.get(language) == ContainerExecutionStatus.AVAILABLE:
                self.container_status[language] = ContainerExecutionStatus.BUILDING

            # Use INFO level for connection errors (expected during startup), WARNING for other errors
            error_str = str(e)
            if any(
                conn_error in error_str.lower()
                for conn_error in ["connect", "connection", "timeout", "unreachable"]
            ):
                logger.info(
                    f"Service {language} not ready yet (connection error - will retry): {e}"
                )
            else:
                logger.warning(f"Service health check error for {language}: {e}")
            return False

    def check_all_services_health(self) -> dict[str, bool]:
        """Check health of all test executor services synchronously (deprecated - use check_all_services_health_sync)."""
        logger.warning(
            "check_all_services_health is deprecated, use check_all_services_health_sync instead"
        )
        return self.check_all_services_health_sync()

    def refresh_container_status(self) -> None:
        """Refresh the status of all containers."""
        self._check_container_availability()
        logger.info("Container status refreshed")

    def build_container(self, language: str, dockerfile_path: str | None = None) -> bool:
        """Build a container for a specific language."""
        if not self.docker_client:
            logger.error("Docker client not available for building")
            return False

        spec = self.get_container_spec(language)
        if not spec:
            logger.error(f"No specification found for language: {language}")
            return False

        # Mark as building
        self.container_status[language] = ContainerExecutionStatus.BUILDING

        try:
            # Determine dockerfile path
            if not dockerfile_path:
                dockerfile_path = f"containers/test-executors/{language}/Dockerfile"

            # Check if dockerfile exists
            if not Path(dockerfile_path).exists():
                logger.error(f"Dockerfile not found at {dockerfile_path}")
                self.container_status[language] = ContainerExecutionStatus.FAILED
                return False

            # Build the image
            build_context = Path(dockerfile_path).parent

            logger.info(f"Building container {spec.image} from {dockerfile_path}")

            image, build_logs = self.docker_client.images.build(
                path=str(build_context),
                dockerfile=Path(dockerfile_path).name,
                tag=spec.image,
                rm=True,
                forcerm=True,
            )

            # Log build output
            for log in build_logs:
                if isinstance(log, dict) and "stream" in log and isinstance(log["stream"], str):
                    logger.debug(f"Build: {log['stream'].strip()}")

            self.container_status[language] = ContainerExecutionStatus.AVAILABLE
            logger.info(f"Successfully built container {spec.image}")
            return True

        except Exception as e:
            logger.error(f"Failed to build container for {language}: {e}")
            self.container_status[language] = ContainerExecutionStatus.FAILED
            return False

    def pull_container(self, language: str) -> bool:
        """Pull a container image for a specific language."""
        if not self.docker_client:
            logger.error("Docker client not available for pulling")
            return False

        spec = self.get_container_spec(language)
        if not spec:
            logger.error(f"No specification found for language: {language}")
            return False

        try:
            logger.info(f"Pulling container {spec.image}")
            self.docker_client.images.pull(spec.image)

            self.container_status[language] = ContainerExecutionStatus.AVAILABLE
            logger.info(f"Successfully pulled container {spec.image}")
            return True

        except Exception as e:
            logger.error(f"Failed to pull container {spec.image}: {e}")
            self.container_status[language] = ContainerExecutionStatus.FAILED
            return False

    def get_container_info(self, language: str) -> dict[str, Any]:
        """Get detailed information about a container."""
        spec = self.get_container_spec(language)
        if not spec:
            return {}

        status = self.get_container_status(language)

        return {
            "language": language,
            "image": spec.image,
            "status": status.value,
            "memory_limit": spec.memory_limit,
            "cpu_quota": spec.cpu_quota,
            "timeout_seconds": spec.timeout_seconds,
            "supported_test_types": spec.supported_test_types or [],
            "environment": spec.environment,
            "working_dir": spec.working_dir,
            "user": spec.user,
            "available": status == ContainerExecutionStatus.AVAILABLE,
        }

    def get_registry_status(self) -> dict[str, Any]:
        """Get overall registry status."""
        total_containers = len(self.specs)
        available_containers = len(self.get_available_languages())

        return {
            "total_containers": total_containers,
            "available_containers": available_containers,
            "availability_rate": (
                available_containers / total_containers if total_containers > 0 else 0
            ),
            "docker_client_available": self.docker_client is not None,
            "supported_languages": self.get_supported_languages(),
            "available_languages": self.get_available_languages(),
            "container_status": {
                lang: status.value for lang, status in self.container_status.items()
            },
        }

    def validate_container_spec(self, language: str) -> dict[str, Any]:
        """Validate a container specification."""
        spec = self.get_container_spec(language)
        if not spec:
            return {
                "valid": False,
                "error": f"No specification found for {language}",
            }

        validation_results: dict[str, Any] = {
            "valid": True,
            "language": language,
            "checks": [],
        }
        checks: list[dict[str, Any]] = validation_results["checks"]

        # Check image name format
        if not spec.image or ":" not in spec.image:
            checks.append(
                {
                    "check": "image_format",
                    "passed": False,
                    "message": "Image name should include tag",
                }
            )
            validation_results["valid"] = False
        else:
            checks.append(
                {
                    "check": "image_format",
                    "passed": True,
                    "message": "Image name format is valid",
                }
            )

        # Check resource limits
        if spec.memory_limit and not spec.memory_limit.endswith(("m", "g", "k")):
            checks.append(
                {
                    "check": "memory_limit",
                    "passed": False,
                    "message": "Memory limit should have unit (m, g, k)",
                }
            )
            validation_results["valid"] = False
        else:
            checks.append(
                {
                    "check": "memory_limit",
                    "passed": True,
                    "message": "Memory limit format is valid",
                }
            )

        # Check CPU quota
        if spec.cpu_quota <= 0 or spec.cpu_quota > 100000:
            checks.append(
                {
                    "check": "cpu_quota",
                    "passed": False,
                    "message": "CPU quota should be between 1 and 100000",
                }
            )
            validation_results["valid"] = False
        else:
            checks.append(
                {
                    "check": "cpu_quota",
                    "passed": True,
                    "message": "CPU quota is valid",
                }
            )

        # Check timeout
        if spec.timeout_seconds <= 0 or spec.timeout_seconds > 3600:
            checks.append(
                {
                    "check": "timeout",
                    "passed": False,
                    "message": "Timeout should be between 1 and 3600 seconds",
                }
            )
            validation_results["valid"] = False
        else:
            checks.append(
                {
                    "check": "timeout",
                    "passed": True,
                    "message": "Timeout is valid",
                }
            )

        # Check supported test types
        if not spec.supported_test_types:
            checks.append(
                {
                    "check": "test_types",
                    "passed": False,
                    "message": "No supported test types defined",
                }
            )
            validation_results["valid"] = False
        else:
            checks.append(
                {
                    "check": "test_types",
                    "passed": True,
                    "message": f"Supports {len(spec.supported_test_types)} test types",
                }
            )

        return validation_results

    def _start_periodic_refresh(self) -> None:
        """Start periodic health check refresh in background thread."""
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_refresh = False
            self._refresh_thread = threading.Thread(
                target=self._periodic_refresh_worker, daemon=True
            )
            self._refresh_thread.start()
            logger.info("Started periodic health check refresh thread")

            # Schedule an aggressive startup check for slow-starting services
            threading.Thread(target=self._aggressive_startup_check, daemon=True).start()

    def _aggressive_startup_check(self) -> None:
        """Aggressively check for slow-starting services during initial startup."""
        logger.info("Starting aggressive health checks for slow-starting services (like Go)")
        max_attempts = 12  # Try for up to 2 minutes
        attempt = 0

        while attempt < max_attempts and not self._stop_refresh:
            try:
                # Check specifically for services that are building/starting
                building_services = [
                    lang
                    for lang, status in self.container_status.items()
                    if status
                    in [
                        ContainerExecutionStatus.BUILDING,
                        ContainerExecutionStatus.UNAVAILABLE,
                    ]
                ]

                if not building_services:
                    logger.info("All services are available, stopping aggressive checks")
                    break

                logger.debug(f"Aggressive check #{attempt + 1} for services: {building_services}")

                # Do a health check
                results = self.check_all_services_health_sync()

                # Log newly available services
                for lang, is_healthy in results.items():
                    if (
                        is_healthy
                        and self.container_status.get(lang) != ContainerExecutionStatus.AVAILABLE
                    ):
                        logger.info(f"Service {lang} is now available after aggressive check")

                attempt += 1
                time.sleep(45)

            except Exception as e:
                logger.error(f"Error during aggressive startup check: {e}")
                break

        logger.info("Aggressive startup checks completed")

    def stop_periodic_refresh(self) -> None:
        """Stop periodic health check refresh."""
        self._stop_refresh = True
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
            logger.info("Stopped periodic health check refresh thread")

    def _periodic_refresh_worker(self) -> None:
        """Background worker for periodic health check refresh."""
        while not self._stop_refresh:
            try:
                current_time = time.time()
                if current_time - self.last_health_check > self.health_check_interval:
                    logger.debug("Running periodic container availability and health check refresh")
                    # First check container availability
                    self._check_container_availability()
                    # Then perform actual health checks to update status from building to available
                    health_results = self.check_all_services_health_sync()
                    self.last_health_check = current_time

                    # Log status changes
                    available_services = [
                        lang for lang, healthy in health_results.items() if healthy
                    ]
                    if available_services:
                        logger.debug(f"Services now available: {available_services}")
                    logger.debug(
                        f"Container status: {[f'{lang}: {status.value}' for lang, status in self.container_status.items()]}"
                    )

                # Sleep for 30 seconds before next check
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in periodic refresh worker: {e}")
                time.sleep(60)  # Wait longer on error

    def force_health_check_refresh(self) -> dict[str, bool]:
        """Force immediate health check refresh for all services."""
        logger.info("Forcing immediate health check refresh")
        self._check_container_availability()
        return self.check_all_services_health_sync()

    def check_all_services_health_sync(self) -> dict[str, bool]:
        """Check health of all test executor services synchronously with better error handling."""

        async def _check_all():
            health_results = {}

            # Create health check tasks
            tasks = []
            for language in self.specs:
                task = asyncio.create_task(
                    self._safe_health_check(language), name=f"health_check_{language}"
                )
                tasks.append((language, task))

            # Wait for all tasks with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                    timeout=15.0,
                )
            except TimeoutError:
                logger.warning("Health check timeout - some services may be slow")

            # Collect results
            for language, task in tasks:
                try:
                    if task.done():
                        result = task.result()
                        health_results[language] = (
                            result if not isinstance(result, Exception) else False
                        )
                    else:
                        task.cancel()
                        health_results[language] = False
                except Exception as e:
                    logger.warning(f"Health check exception for {language}: {e}")
                    health_results[language] = False

            return health_results

        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to handle differently
                logger.warning("Running health check in existing event loop context")
                return {lang: False for lang in self.specs}  # Return conservative result
            except RuntimeError:
                # No running loop, create one
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(_check_all())
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous health check: {e}")
            return {lang: False for lang in self.specs}

    async def _safe_health_check(self, language: str) -> bool:
        """Safely perform health check with timeout and error handling."""
        try:
            return await asyncio.wait_for(
                self.check_service_health(language, timeout=3.0), timeout=5.0
            )
        except TimeoutError:
            logger.info(f"Service {language} not ready yet (timeout - will retry)")
            return False
        except Exception as e:
            # Use INFO level for connection errors (expected during startup), WARNING for other errors
            error_str = str(e)
            if any(
                conn_error in error_str.lower()
                for conn_error in ["connect", "connection", "timeout", "unreachable"]
            ):
                logger.info(
                    f"Service {language} not ready yet (connection error - will retry): {e}"
                )
            else:
                logger.warning(f"Health check error for {language}: {e}")
            return False

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for all containers and services."""
        current_time = time.time()

        # Get container status
        container_status = {lang: status.value for lang, status in self.container_status.items()}

        # Get health check results
        health_results = self.check_all_services_health_sync()

        return {
            "timestamp": current_time,
            "last_health_check": self.last_health_check,
            "health_check_interval": self.health_check_interval,
            "seconds_since_last_check": current_time - self.last_health_check,
            "docker_client_available": self.docker_client is not None,
            "periodic_refresh_enabled": self._refresh_thread is not None
            and self._refresh_thread.is_alive(),
            "container_status": container_status,
            "health_results": health_results,
            "overall_health": {
                "total_services": len(self.specs),
                "healthy_services": sum(1 for h in health_results.values() if h),
                "available_containers": sum(
                    1 for s in container_status.values() if s == "available"
                ),
                "health_rate": (
                    sum(1 for h in health_results.values() if h) / len(health_results)
                    if health_results
                    else 0
                ),
                "availability_rate": (
                    sum(1 for s in container_status.values() if s == "available")
                    / len(container_status)
                    if container_status
                    else 0
                ),
            },
        }

    def register_custom_container(self, language: str, spec: TestExecutionSpec) -> bool:
        """Register a custom container specification."""
        try:
            # Validate the spec
            validation = self.validate_container_spec_object(spec)
            if not validation["valid"]:
                logger.error(f"Invalid container spec for {language}: {validation['error']}")
                return False

            # Register the spec
            self.specs[language] = spec
            self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE

            # Check availability
            self._check_single_container_availability(language)

            logger.info(f"Registered custom container for {language}")
            return True

        except Exception as e:
            logger.error(f"Failed to register custom container for {language}: {e}")
            return False

    def validate_container_spec_object(self, spec: TestExecutionSpec) -> dict[str, Any]:
        """Validate a ContainerSpec object."""
        if not spec.image:
            return {"valid": False, "error": "Image name is required"}

        if not spec.memory_limit:
            return {"valid": False, "error": "Memory limit is required"}

        if spec.cpu_quota <= 0:
            return {"valid": False, "error": "CPU quota must be positive"}

        if spec.timeout_seconds <= 0:
            return {"valid": False, "error": "Timeout must be positive"}

        return {"valid": True}

    def _check_single_container_availability(self, language: str) -> None:
        """Check availability of a single container."""
        if not self.docker_client:
            self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE
            return

        spec = self.get_container_spec(language)
        if not spec:
            self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE
            return

        try:
            self.docker_client.images.get(spec.image)
            self.container_status[language] = ContainerExecutionStatus.AVAILABLE
        except Exception as e:
            if (
                HAS_DOCKER
                and hasattr(docker, "errors")
                and isinstance(e, docker.errors.ImageNotFound)
            ):
                self.container_status[language] = ContainerExecutionStatus.UNAVAILABLE
            else:
                self.container_status[language] = ContainerExecutionStatus.FAILED


def create_container_registry(
    enable_periodic_refresh: bool = True,
) -> TestExecutionContainerRegistry:
    """Factory function to create TestExecutionContainerRegistry.

    Args:
        enable_periodic_refresh: Whether to enable periodic health check refresh
    """
    return TestExecutionContainerRegistry(enable_periodic_refresh=enable_periodic_refresh)
