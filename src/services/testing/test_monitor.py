"""Test monitoring and logging for the testing service."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from models.testing_models import TestResult
from services.stubs import RedisCacheManager

logger = logging.getLogger(__name__)


class TestMonitor:
    """
    Provides monitoring, health checks, and active test tracking
    for the testing service.
    """

    def __init__(self, cache_manager: RedisCacheManager | None = None):
        self.cache_manager = cache_manager
        self.active_tests: dict[str, asyncio.Task[TestResult]] = {}

    async def health_check(
        self,
        container_manager,
        config_manager,
        result_processor,
        intelligent_analyzer=None,
    ) -> dict[str, Any]:
        """Health check for the testing service."""
        health_status: dict[str, Any] = {
            "service": "test_execution",
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Check container manager
        try:
            container_health = await container_manager.health_check()
            health_status["checks"]["containers"] = container_health
        except Exception as e:
            health_status["checks"]["containers"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check cache service
        if self.cache_manager:
            try:
                await self.cache_manager.get("health_check")
                health_status["checks"]["cache"] = {"status": "healthy"}
            except Exception as e:
                health_status["checks"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
        else:
            health_status["checks"]["cache"] = {"status": "disabled"}

        # Check configuration manager
        try:
            config_validation = config_manager.validate_configuration()
            health_status["checks"]["config"] = {
                "status": "healthy" if config_validation["valid"] else "degraded",
                "details": config_validation["details"],
            }
        except Exception as e:
            health_status["checks"]["config"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check intelligent analyzer
        if intelligent_analyzer:
            try:
                analyzer_health = await intelligent_analyzer.health_check()
                health_status["checks"]["intelligent_analyzer"] = analyzer_health
            except Exception as e:
                health_status["checks"]["intelligent_analyzer"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
        else:
            health_status["checks"]["intelligent_analyzer"] = {"status": "disabled"}

        # Overall status
        unhealthy_checks = [
            check
            for check in health_status["checks"].values()
            if check.get("status") not in ["healthy", "disabled"]
        ]

        if unhealthy_checks:
            health_status["status"] = (
                "degraded" if len(unhealthy_checks) < len(health_status["checks"]) else "unhealthy"
            )

        return health_status

    async def get_test_results(self, test_id: str) -> TestResult:
        """Get cached test results."""
        if self.cache_manager:
            cache_key = f"test_result:{test_id}"
            cached_data = await self.cache_manager.get(cache_key)

            if cached_data:
                return TestResult(**cached_data)

        # Check active tests
        if test_id in self.active_tests:
            task = self.active_tests[test_id]
            if task.done():
                try:
                    return await task
                except Exception as e:
                    logger.error(f"Failed to get result for active test {test_id}: {str(e)}")
                    raise
            else:
                raise ValueError(f"Test {test_id} is still running")

        raise ValueError(f"Test results not found for ID: {test_id}")

    async def cleanup_test_environment(self, test_id: str, container_manager) -> bool:
        """Cleanup test environment and cached results."""
        try:
            cleanup_success = True

            # Cancel and remove active test
            if test_id in self.active_tests:
                task = self.active_tests[test_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.warning(f"Error cancelling test {test_id}: {str(e)}")
                        cleanup_success = False

                del self.active_tests[test_id]

            # Clean up cached results
            if self.cache_manager:
                try:
                    await self.cache_manager.delete(f"test_result:{test_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning cache for test {test_id}: {str(e)}")
                    cleanup_success = False

            # Clean up any temporary containers
            try:
                await container_manager.cleanup_test_containers(test_id)
            except Exception as e:
                logger.warning(f"Error cleaning containers for test {test_id}: {str(e)}")
                cleanup_success = False

            logger.info(f"Cleanup completed for test {test_id}, success: {cleanup_success}")
            return cleanup_success

        except Exception as e:
            logger.error(f"Failed to cleanup test environment {test_id}: {str(e)}")
            return False

    def track_active_test(self, test_id: str, task: asyncio.Task[TestResult]) -> None:
        """Track an active test task."""
        self.active_tests[test_id] = task
        logger.info(f"Started tracking test {test_id}")

    def untrack_test(self, test_id: str) -> asyncio.Task[TestResult] | None:
        """Stop tracking a test and return the task if it exists."""
        task = self.active_tests.pop(test_id, None)
        if task:
            logger.info(f"Stopped tracking test {test_id}")
        return task

    def get_active_test_ids(self) -> list[str]:
        """Get list of active test IDs."""
        return list(self.active_tests.keys())

    def get_active_test_count(self) -> int:
        """Get count of active tests."""
        return len(self.active_tests)

    async def cancel_test(self, test_id: str) -> bool:
        """Cancel an active test."""
        if test_id not in self.active_tests:
            return False

        task = self.active_tests[test_id]
        if task.done():
            return False

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Cancelled test {test_id}")
            return True
        except Exception as e:
            logger.warning(f"Error cancelling test {test_id}: {e}")
            return False
        return False

    async def cleanup_completed_tests(self) -> int:
        """Clean up completed test tasks and return count of cleaned up tests."""
        completed_tests = [test_id for test_id, task in self.active_tests.items() if task.done()]

        for test_id in completed_tests:
            del self.active_tests[test_id]

        if completed_tests:
            logger.info(f"Cleaned up {len(completed_tests)} completed tests")

        return len(completed_tests)

    async def get_test_status(self, test_id: str) -> dict[str, Any]:
        """Get status information for a specific test."""
        if test_id not in self.active_tests:
            return {"status": "not_found", "test_id": test_id}

        task = self.active_tests[test_id]

        if task.done():
            try:
                result = await task
                return {
                    "status": "completed",
                    "test_id": test_id,
                    "success": result.success,
                    "execution_time": result.execution_time,
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "test_id": test_id,
                    "error": str(e),
                }
        else:
            return {"status": "running", "test_id": test_id}

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system metrics for monitoring."""
        return {
            "active_tests": self.get_active_test_count(),
            "cache_status": "enabled" if self.cache_manager else "disabled",
            "timestamp": datetime.now(UTC).isoformat(),
        }
