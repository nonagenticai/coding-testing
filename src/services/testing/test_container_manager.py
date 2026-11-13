"""Docker container integration for test execution."""

import logging
from datetime import UTC, datetime
from typing import Any

from models.testing_models import TestRequest, TestResult
from services.stubs import DockerService

logger = logging.getLogger(__name__)


class TestContainerManager:
    """
    Manages Docker container lifecycle for test execution including
    container creation, file preparation, execution, and cleanup.
    """

    def __init__(self, docker_service: DockerService):
        self.docker_service = docker_service

    async def execute_in_container(
        self, request: TestRequest, test_id: str, config_manager, result_processor
    ) -> TestResult:
        """Execute test in Docker container using TestPy patterns."""
        container_id = None
        start_time = datetime.now(UTC)

        try:
            # Get container image for language
            image_name = config_manager.get_container_image(request.language)

            # Create container configuration
            container_config = config_manager.create_container_config(request, test_id, image_name)

            # Create test container
            container_id = await self.docker_service.create_test_container(container_config)
            logger.info(f"Created test container {container_id} for test {test_id}")

            # Start the container
            await self.docker_service.start_container(container_id)

            # Prepare and copy test files to container
            test_files = config_manager.prepare_test_files(request)
            for file_path, content in test_files.items():
                container_path = f"/app/test/{file_path}"
                await self.docker_service.create_file_in_container(
                    container_id, container_path, content
                )

            # Determine test command based on language
            test_command = config_manager.get_test_command(request.language, test_files)

            # Execute tests in container
            exec_result = await self.docker_service.exec_in_container(
                docker_id=container_id,
                command=test_command,
                timeout=request.timeout or 30,
                user="testuser",
            )

            # Get container stats for metrics
            stats = await self.docker_service.get_container_stats(container_id)

            # Calculate execution time
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # Create container result
            container_result = {
                "exit_code": exec_result.get("exit_code", 1),
                "stdout": exec_result.get("stdout", ""),
                "stderr": exec_result.get("stderr", ""),
                "execution_time": execution_time,
                "memory_usage": stats.get("memory_usage_mb", 0),
                "cpu_usage": stats.get("cpu_percent", 0),
            }

            # Convert container result to TestResult
            return result_processor.convert_container_result(container_result, test_id, request)

        except Exception as e:
            logger.error(f"Container execution failed for test {test_id}: {e}")
            return self._create_error_result(test_id, start_time, request, str(e))

        finally:
            # Cleanup container
            if container_id:
                await self._cleanup_container(container_id)

    async def _cleanup_container(self, container_id: str) -> None:
        """Clean up a test container."""
        try:
            await self.docker_service.stop_container(container_id)
            await self.docker_service.remove_container_by_id(container_id)
            logger.info(f"Cleaned up test container {container_id}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup container {container_id}: {cleanup_error}")

    def _create_error_result(
        self, test_id: str, start_time: datetime, request: TestRequest, error: str
    ) -> TestResult:
        """Create an error TestResult for container failures."""
        return TestResult(
            test_id=test_id,
            success=False,
            stdout="",
            stderr=error,
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
            error=error,
        )

    async def cleanup_test_containers(self, test_id: str) -> None:
        """Clean up any containers created for this test."""
        # Note: Since DockerService doesn't have list_containers method,
        # container cleanup is handled in the finally block of execute_in_container
        # This method is kept for API compatibility
        logger.debug(f"Container cleanup for test {test_id} handled during execution")

    async def health_check(self) -> dict[str, Any]:
        """Health check for the container manager."""
        try:
            # DockerService doesn't have health_check method, check basic functionality
            docker_healthy = hasattr(self.docker_service, "client")
            return {
                "status": "healthy" if docker_healthy else "unhealthy",
                "docker_service": docker_healthy,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def get_container_stats(self, container_id: str) -> dict[str, Any]:
        """Get container statistics."""
        try:
            return await self.docker_service.get_container_stats(container_id)
        except Exception as e:
            logger.error(f"Failed to get container stats for {container_id}: {e}")
            return {}

    async def create_test_file_in_container(
        self, container_id: str, file_path: str, content: str
    ) -> None:
        """Create a file in the test container."""
        try:
            await self.docker_service.create_file_in_container(container_id, file_path, content)
        except Exception as e:
            logger.error(f"Failed to create file {file_path} in container {container_id}: {e}")
            raise

    async def execute_command_in_container(
        self, container_id: str, command: list[str], timeout: int = 30
    ) -> dict[str, Any]:
        """Execute a command in the test container."""
        try:
            return await self.docker_service.exec_in_container(
                docker_id=container_id,
                command=command,
                timeout=timeout,
                user="testuser",
            )
        except Exception as e:
            logger.error(f"Failed to execute command in container {container_id}: {e}")
            raise

    async def stop_container(self, container_id: str) -> None:
        """Stop a test container."""
        try:
            await self.docker_service.stop_container(container_id)
        except Exception as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            raise

    async def remove_container(self, container_id: str) -> None:
        """Remove a test container."""
        try:
            await self.docker_service.remove_container_by_id(container_id)
        except Exception as e:
            logger.error(f"Failed to remove container {container_id}: {e}")
            raise

    async def create_container(self, config: dict[str, Any]) -> str:
        """Create a new test container."""
        try:
            return await self.docker_service.create_test_container(config)
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            raise

    async def start_container(self, container_id: str) -> None:
        """Start a test container."""
        try:
            await self.docker_service.start_container(container_id)
        except Exception as e:
            logger.error(f"Failed to start container {container_id}: {e}")
            raise
