"""Service stubs for cross-service dependencies.

These replace monolith services that belong to other microservices.
They provide the same interface but use HTTP calls or no-ops.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContainerRuntimeUnavailableError(RuntimeError):
    """Raised when real containerized test execution is required but unavailable.

    coding-testing does not run containers itself. In production, real test
    execution must go through EITHER the in-cluster Kubernetes executor
    (``KUBERNETES_IN_CLUSTER=true`` → ``KubernetesTestExecutor``) OR the
    language-specific test-executor microservices reached over HTTP via
    ``CodeExecutionService`` (the ``/api/tests`` path).

    This ``DockerService`` is a no-op stub that only exists to satisfy the
    interface. It MUST NEVER fabricate an execution result: the container
    lifecycle methods raise this error so an unexecuted test can never be
    reported as a pass.
    """


_NO_RUNTIME_MSG = (
    "No container runtime is available in coding-testing: refusing to fabricate a "
    "test result for a test that was never executed. Real execution requires the "
    "in-cluster Kubernetes executor (set KUBERNETES_IN_CLUSTER=true) or the "
    "language test-executor services. This DockerService stub is a no-op and will "
    "not run tests."
)


class DockerService:
    """Stub for docker service (belongs to coding-containers).

    Query/cleanup methods are benign no-ops (Docker is genuinely unavailable
    here). The container-lifecycle methods that would otherwise *manufacture* a
    successful execution — create/start/write-file/exec — fail loud with
    :class:`ContainerRuntimeUnavailableError` so the orchestration path returns
    an honest failure instead of a green result from nothing.
    """

    async def find_container_by_name(self, name: str) -> Any:
        logger.warning(f"DockerService stub called for '{name}' - use coding-containers service")
        return None

    def get_language_container_image(self, language: str) -> str:
        """Get container image name for a language."""
        images = {
            "python": "python:3.13-slim",
            "javascript": "node:22-slim",
            "typescript": "node:22-slim",
            "go": "golang:1.23-alpine",
            "rust": "rust:1.83-slim",
            "java": "eclipse-temurin:21-jdk",
            "csharp": "mcr.microsoft.com/dotnet/sdk:9.0",
            "cpp": "gcc:14",
        }
        return images.get(language, "python:3.13-slim")

    async def create_test_container(self, config: dict[str, Any]) -> str:
        """Fail loud: this stub cannot create a real test container.

        Returning a fake container id here is what let the orchestration path
        pretend a container existed and later report a pass. Raise instead.
        """
        logger.error("DockerService stub: create_test_container called - no runtime available")
        raise ContainerRuntimeUnavailableError(_NO_RUNTIME_MSG)

    async def start_container(self, container_id: str) -> None:
        """Fail loud: this stub cannot start a real container."""
        logger.error("DockerService stub: start_container called - no runtime available")
        raise ContainerRuntimeUnavailableError(_NO_RUNTIME_MSG)

    async def create_file_in_container(
        self, container_id: str, file_path: str, content: str
    ) -> bool:
        """Fail loud: this stub cannot write files into a real container."""
        logger.error("DockerService stub: create_file_in_container called - no runtime available")
        raise ContainerRuntimeUnavailableError(_NO_RUNTIME_MSG)

    async def exec_in_container(
        self,
        docker_id: str,
        command: list[str],
        timeout: int = 30,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Fail loud: this stub cannot execute anything.

        Previously returned ``{"exit_code": 0, ...}`` unconditionally, which the
        orchestrator read as ``success = (exit_code == 0) == True`` — reporting
        every unexecuted test as passed in milliseconds. Raise instead so the
        orchestration path fails honestly.
        """
        logger.error("DockerService stub: exec_in_container called - no runtime available")
        raise ContainerRuntimeUnavailableError(_NO_RUNTIME_MSG)

    async def install_tools_in_container(
        self, container_id: str, tools: list[str], language: str
    ) -> None:
        """Install tools in a container (stub)."""
        logger.warning(f"DockerService stub: install_tools_in_container called for {container_id}")

    async def stop_container(self, container_id: str) -> None:
        """Stop a container (stub)."""
        logger.warning(f"DockerService stub: stop_container called for {container_id}")

    async def remove_container_by_id(self, container_id: str) -> None:
        """Remove a container by ID (stub)."""
        logger.warning(f"DockerService stub: remove_container_by_id called for {container_id}")

    async def remove_container(self, container_id: str) -> None:
        """Remove a container (stub)."""
        logger.warning(f"DockerService stub: remove_container called for {container_id}")

    async def remove_container_by_name(self, name: str) -> None:
        """Remove a container by name (stub)."""
        logger.warning(f"DockerService stub: remove_container_by_name called for {name}")

    async def list_containers(self, **kwargs: Any) -> list[Any]:
        """List containers (stub)."""
        return []

    async def is_docker_available(self) -> bool:
        """Check if Docker is available (stub)."""
        return False

    async def get_container_stats(self, container_id: str) -> dict[str, Any]:
        """Get container statistics (stub)."""
        return {"memory_usage_mb": 0, "cpu_percent": 0.0}

    async def create_container(self, config: Any) -> Any:
        """Fail loud: this stub cannot create a real container."""
        logger.error("DockerService stub: create_container called - no runtime available")
        raise ContainerRuntimeUnavailableError(_NO_RUNTIME_MSG)


_docker_service: DockerService | None = None


def get_docker_service() -> DockerService:
    global _docker_service
    if _docker_service is None:
        _docker_service = DockerService()
    return _docker_service


class RedisCacheManager:
    """Stub for cache manager."""

    async def get(self, key: str) -> Any:
        return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        pass

    async def delete(self, key: str) -> None:
        pass

    async def get_redis(self) -> Any:
        """Get redis client (stub)."""
        return None


_cache_manager: RedisCacheManager | None = None


def get_cache_manager() -> RedisCacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
    return _cache_manager


class MetricsService:
    """Stub for metrics service."""

    async def record_test_execution(self, metrics_data: dict) -> None:
        logger.debug(f"Metrics stub: {metrics_data.get('event_type', 'unknown')}")

    async def record_quality_analysis(self, analysis_data: dict[str, Any] | None = None) -> None:
        """Record quality analysis metrics (stub)."""
        pass

    async def get_test_execution_analytics(self, **kwargs: Any) -> dict[str, Any]:
        """Get test execution analytics (stub)."""
        return {"total_executions": 0, "success_rate": 0.0}


_metrics_service: MetricsService | None = None


def get_metrics_service() -> MetricsService:
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
