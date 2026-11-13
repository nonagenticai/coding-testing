"""
Health check monitoring system for test execution containers.

This module provides comprehensive health monitoring for container instances,
including liveness checks, readiness probes, and automated recovery mechanisms.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import aiohttp
import docker

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class HealthCheckType(StrEnum):
    """Types of health checks."""

    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    DEPENDENCY = "dependency"


class RecoveryAction(StrEnum):
    """Recovery actions for unhealthy containers."""

    RESTART = "restart"
    RECREATE = "recreate"
    SCALE_DOWN = "scale_down"
    ALERT_ONLY = "alert_only"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    check_type: HealthCheckType
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1
    initial_delay_seconds: int = 0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_type": self.check_type.value,
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "initial_delay_seconds": self.initial_delay_seconds,
            "enabled": self.enabled,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    timestamp: datetime
    check_type: HealthCheckType
    status: HealthStatus
    response_time_ms: float
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "check_type": self.check_type.value,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class ContainerHealth:
    """Overall health state of a container."""

    container_id: str
    language: str
    overall_status: HealthStatus
    last_updated: datetime
    check_results: dict[HealthCheckType, list[HealthCheckResult]] = field(default_factory=dict)
    consecutive_failures: dict[HealthCheckType, int] = field(default_factory=dict)
    consecutive_successes: dict[HealthCheckType, int] = field(default_factory=dict)
    recovery_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "container_id": self.container_id,
            "language": self.language,
            "overall_status": self.overall_status.value,
            "last_updated": self.last_updated.isoformat(),
            "check_results": {
                check_type.value: [result.to_dict() for result in results]
                for check_type, results in self.check_results.items()
            },
            "consecutive_failures": {
                check_type.value: count for check_type, count in self.consecutive_failures.items()
            },
            "consecutive_successes": {
                check_type.value: count for check_type, count in self.consecutive_successes.items()
            },
            "recovery_attempts": self.recovery_attempts,
        }


class ContainerHealthChecker:
    """Base class for health check implementations."""

    def __init__(self, config: HealthCheckConfig):
        """Initialize the ContainerHealthChecker service.

        This initializes the ContainerHealthChecker with required dependencies
        and configuration for service operations.

        Args:
            config (HealthCheckConfig): Configuration settings for config
        """
        self.config = config

    async def check_health(self, container_id: str) -> HealthCheckResult:
        """Perform health check on container."""
        raise NotImplementedError


class DockerHealthChecker(ContainerHealthChecker):
    """Health checker using Docker container stats."""

    def __init__(self, config: HealthCheckConfig):
        """Initialize the DockerHealthChecker service.

        This initializes the DockerHealthChecker with required dependencies
        and configuration for service operations.

        Args:
            config (HealthCheckConfig): Configuration settings for config
        """
        super().__init__(config)
        self.docker_client = docker.from_env()

    async def check_health(self, container_id: str) -> HealthCheckResult:
        """Check container health using Docker API."""
        start_time = time.time()

        try:
            container = self.docker_client.containers.get(container_id)

            # Check container status
            status = container.status
            if status == "running":
                # Get container stats
                stats = container.stats(stream=False)

                # Check CPU and memory usage
                cpu_usage = self._calculate_cpu_usage(stats)
                memory_usage = self._calculate_memory_usage(stats)

                # Determine health based on resource usage
                if cpu_usage < 95 and memory_usage < 95:
                    health_status = HealthStatus.HEALTHY
                    message = f"Container running normally (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)"
                else:
                    health_status = HealthStatus.DEGRADED
                    message = f"Container under stress (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)"

                metadata = {
                    "container_status": status,
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory_usage,
                }

            elif status in ["created", "restarting"]:
                health_status = HealthStatus.STARTING
                message = f"Container starting (status: {status})"
                metadata = {"container_status": status}

            elif status == "exited":
                health_status = HealthStatus.UNHEALTHY
                exit_code = container.attrs["State"]["ExitCode"]
                message = f"Container exited with code {exit_code}"
                metadata = {"container_status": status, "exit_code": exit_code}

            else:
                health_status = HealthStatus.UNKNOWN
                message = f"Unknown container status: {status}"
                metadata = {"container_status": status}

        except docker.errors.NotFound:
            health_status = HealthStatus.UNHEALTHY
            message = "Container not found"
            metadata = {}

        except (
            docker.errors.APIError,
            docker.errors.ContainerError,
            AttributeError,
        ) as e:
            health_status = HealthStatus.UNKNOWN
            message = f"Health check failed: {str(e)}"
            metadata = {"error": str(e)}

        response_time_ms = (time.time() - start_time) * 1000

        return HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=self.config.check_type,
            status=health_status,
            response_time_ms=response_time_ms,
            message=message,
            metadata=metadata,
        )

    def _calculate_cpu_usage(self, stats: dict[str, Any]) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            )

            if system_delta > 0:
                result = (
                    (cpu_delta / system_delta)
                    * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
                    * 100
                )
                return float(result)
        except (KeyError, ZeroDivisionError):  # fmt: skip
            pass

        return 0.0

    def _calculate_memory_usage(self, stats: dict[str, Any]) -> float:
        """Calculate memory usage percentage from Docker stats."""
        try:
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            result = (memory_usage / memory_limit) * 100
            return float(result)
        except (KeyError, ZeroDivisionError):  # fmt: skip
            pass

        return 0.0


class HTTPHealthChecker(ContainerHealthChecker):
    """Health checker using HTTP endpoint."""

    def __init__(self, config: HealthCheckConfig, endpoint: str = "/health"):
        """Initialize the HTTPHealthChecker service.

        This initializes the HTTPHealthChecker with required dependencies
        and configuration for service operations.

        Args:
            config (HealthCheckConfig): Configuration settings for config
            endpoint (str): Endpoint
        """
        super().__init__(config)
        self.endpoint = endpoint

    async def check_health(self, container_id: str) -> HealthCheckResult:
        """Check container health using HTTP endpoint."""
        start_time = time.time()

        try:
            # Get container IP
            docker_client = docker.from_env()
            container = docker_client.containers.get(container_id)

            # Get container IP address
            network_settings = container.attrs["NetworkSettings"]
            ip_address = network_settings.get("IPAddress")

            if not ip_address:
                # Try to get from networks
                networks = network_settings.get("Networks", {})
                for _, network_info in networks.items():
                    if network_info.get("IPAddress"):
                        ip_address = network_info["IPAddress"]
                        break

            if not ip_address:
                raise ValueError("Could not determine container IP address")

            # Make HTTP request
            url = f"http://{ip_address}:8080{self.endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        health_status = HealthStatus.HEALTHY
                        message = "HTTP health check passed"
                    else:
                        health_status = HealthStatus.UNHEALTHY
                        message = f"HTTP health check failed with status {response.status}"

                    metadata = {
                        "http_status": response.status,
                        "response_text": response_text[:200],  # Truncate
                        "url": url,
                    }

        except TimeoutError:
            health_status = HealthStatus.UNHEALTHY
            message = "HTTP health check timed out"
            metadata = {"timeout_seconds": self.config.timeout_seconds}

        except (aiohttp.ClientError, docker.errors.APIError, ValueError) as e:
            health_status = HealthStatus.UNHEALTHY
            message = f"HTTP health check failed: {str(e)}"
            metadata = {"error": str(e)}

        response_time_ms = (time.time() - start_time) * 1000

        return HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=self.config.check_type,
            status=health_status,
            response_time_ms=response_time_ms,
            message=message,
            metadata=metadata,
        )


class ContainerHealthMonitor:
    """Monitor health of individual containers."""

    def __init__(self, container_id: str, language: str):
        """Initialize the ContainerHealthMonitor service.

        This initializes the ContainerHealthMonitor with required dependencies
        and configuration for service operations.

        Args:
            container_id (str): Unique identifier for the container id
            language (str): Language
        """
        self.container_id = container_id
        self.language = language
        self.health = ContainerHealth(
            container_id=container_id,
            language=language,
            overall_status=HealthStatus.STARTING,
            last_updated=datetime.now(UTC),
        )

        self.health_checkers: dict[HealthCheckType, ContainerHealthChecker] = {}
        self.monitoring_tasks: dict[HealthCheckType, asyncio.Task] = {}
        self.recovery_handlers: dict[RecoveryAction, Callable] = {}
        self.monitoring_active = False

        # Setup default health checkers
        self._setup_default_checkers()

    def _setup_default_checkers(self) -> None:
        """Setup default health checkers."""
        # Liveness check using Docker API
        liveness_config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            interval_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
        )
        self.health_checkers[HealthCheckType.LIVENESS] = DockerHealthChecker(liveness_config)

        # Readiness check using HTTP (if available)
        readiness_config = HealthCheckConfig(
            check_type=HealthCheckType.READINESS,
            interval_seconds=15,
            timeout_seconds=5,
            failure_threshold=2,
        )
        self.health_checkers[HealthCheckType.READINESS] = HTTPHealthChecker(readiness_config)

    def add_health_checker(self, checker: ContainerHealthChecker) -> None:
        """Add custom health checker."""
        self.health_checkers[checker.config.check_type] = checker

    def add_recovery_handler(self, action: RecoveryAction, handler: Callable) -> None:
        """Add recovery action handler."""
        self.recovery_handlers[action] = handler

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        self.monitoring_active = True

        for check_type, checker in self.health_checkers.items():
            if checker.config.enabled:
                task = asyncio.create_task(self._monitor_health_check(check_type, checker))
                self.monitoring_tasks[check_type] = task

        logger.info("Started health monitoring for container %s", self.container_id)

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False

        for task in self.monitoring_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)

        self.monitoring_tasks.clear()
        logger.info("Stopped health monitoring for container %s", self.container_id)

    async def _monitor_health_check(
        self, check_type: HealthCheckType, checker: ContainerHealthChecker
    ) -> None:
        """Monitor a specific health check."""
        # Initial delay
        if checker.config.initial_delay_seconds > 0:
            await asyncio.sleep(checker.config.initial_delay_seconds)

        while self.monitoring_active:
            try:
                # Perform health check
                result = await checker.check_health(self.container_id)

                # Store result
                if check_type not in self.health.check_results:
                    self.health.check_results[check_type] = []

                self.health.check_results[check_type].append(result)

                # Keep only recent results (last 100)
                if len(self.health.check_results[check_type]) > 100:
                    self.health.check_results[check_type] = self.health.check_results[check_type][
                        -100:
                    ]

                # Update consecutive counters
                if result.status == HealthStatus.HEALTHY:
                    self.health.consecutive_successes[check_type] = (
                        self.health.consecutive_successes.get(check_type, 0) + 1
                    )
                    self.health.consecutive_failures[check_type] = 0
                else:
                    self.health.consecutive_failures[check_type] = (
                        self.health.consecutive_failures.get(check_type, 0) + 1
                    )
                    self.health.consecutive_successes[check_type] = 0

                # Check if recovery is needed
                if self.health.consecutive_failures[check_type] >= checker.config.failure_threshold:
                    await self._trigger_recovery(check_type, result)

                # Update overall status
                self._update_overall_status()

                logger.debug(
                    "Health check %s for %s: %s",
                    check_type.value,
                    self.container_id,
                    result.status.value,
                )

            except (asyncio.CancelledError, RuntimeError) as e:
                logger.error("Health check error for %s: %s", self.container_id, e)

            await asyncio.sleep(checker.config.interval_seconds)

    async def _trigger_recovery(
        self, check_type: HealthCheckType, result: HealthCheckResult
    ) -> None:
        """Trigger recovery actions for failed health checks."""
        self.health.recovery_attempts += 1

        # Determine recovery action based on failure count and type
        if self.health.recovery_attempts == 1:
            recovery_action = RecoveryAction.RESTART
        elif self.health.recovery_attempts <= 3:
            recovery_action = RecoveryAction.RECREATE
        else:
            recovery_action = RecoveryAction.ALERT_ONLY

        # Execute recovery handler
        handler = self.recovery_handlers.get(recovery_action)
        if handler:
            try:
                await handler(self.container_id, check_type, result)
                logger.warning(
                    "Recovery action %s triggered for %s",
                    recovery_action.value,
                    self.container_id,
                )
            except (RuntimeError, docker.errors.APIError) as e:
                logger.error(
                    "Recovery action %s failed for %s: %s",
                    recovery_action.value,
                    self.container_id,
                    e,
                )
        else:
            logger.warning("No recovery handler for action %s", recovery_action.value)

    def _update_overall_status(self) -> None:
        """Update overall container health status."""
        self.health.last_updated = datetime.now(UTC)

        # Get latest status from each check type
        statuses = []
        for _, results in self.health.check_results.items():
            if results:
                statuses.append(results[-1].status)

        if not statuses:
            self.health.overall_status = HealthStatus.UNKNOWN
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            self.health.overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            self.health.overall_status = HealthStatus.DEGRADED
        elif any(status == HealthStatus.STARTING for status in statuses):
            self.health.overall_status = HealthStatus.STARTING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            self.health.overall_status = HealthStatus.HEALTHY
        else:
            self.health.overall_status = HealthStatus.UNKNOWN

    def get_health_status(self) -> ContainerHealth:
        """Get current health status."""
        return self.health

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary."""
        return {
            "container_id": self.container_id,
            "language": self.language,
            "overall_status": self.health.overall_status.value,
            "last_updated": self.health.last_updated.isoformat(),
            "recovery_attempts": self.health.recovery_attempts,
            "check_types": list(self.health_checkers.keys()),
            "recent_failures": {
                check_type.value: count
                for check_type, count in self.health.consecutive_failures.items()
                if count > 0
            },
        }


class HealthMonitoringService:
    """Central health monitoring service for all containers."""

    def __init__(self):
        """Initialize the HealthMonitoringService service.

        This initializes the HealthMonitoringService with required dependencies
        and configuration for service operations.
        """
        self.container_monitors: dict[str, ContainerHealthMonitor] = {}
        self.system_health_checkers: list[ContainerHealthChecker] = []
        self.monitoring_active = False
        self.system_monitoring_task: asyncio.Task | None = None

        # Setup default recovery handlers
        self._setup_recovery_handlers()

    def _setup_recovery_handlers(self) -> None:
        """Setup default recovery handlers."""
        self.default_recovery_handlers = {
            RecoveryAction.RESTART: self._restart_container,
            RecoveryAction.RECREATE: self._recreate_container,
            RecoveryAction.ALERT_ONLY: self._alert_only,
        }

    async def _restart_container(
        self,
        container_id: str,
        check_type: HealthCheckType,
        result: HealthCheckResult,  # noqa: ARG002
    ) -> None:
        """Restart unhealthy container."""
        try:
            docker_client = docker.from_env()
            container = docker_client.containers.get(container_id)
            container.restart()
            logger.info(
                "Restarted container %s due to failed %s check",
                container_id,
                check_type.value,
            )
        except (docker.errors.NotFound, docker.errors.APIError) as e:
            logger.error("Failed to restart container %s: %s", container_id, e)

    async def _recreate_container(
        self,
        container_id: str,
        check_type: HealthCheckType,
        result: HealthCheckResult,  # noqa: ARG002
    ) -> None:
        """Recreate unhealthy container."""
        try:
            docker_client = docker.from_env()
            container = docker_client.containers.get(container_id)

            # Get container configuration
            config = container.attrs["Config"]
            host_config = container.attrs["HostConfig"]

            # Stop and remove container
            container.stop()
            container.remove()

            # Create new container with same configuration
            new_container = docker_client.containers.run(
                image=config["Image"], detach=True, **host_config
            )

            logger.info(
                "Recreated container %s as %s due to failed %s check",
                container_id,
                new_container.id,
                check_type.value,
            )

        except (
            docker.errors.NotFound,
            docker.errors.APIError,
            docker.errors.ContainerError,
        ) as e:
            logger.error("Failed to recreate container %s: %s", container_id, e)

    async def _alert_only(
        self,
        container_id: str,
        check_type: HealthCheckType,
        result: HealthCheckResult,
    ) -> None:
        """Alert only without taking action."""
        logger.critical(
            "Container %s failed %s check: %s",
            container_id,
            check_type.value,
            result.message,
        )

    async def start_monitoring(self) -> None:
        """Start the health monitoring service."""
        self.monitoring_active = True
        self.system_monitoring_task = asyncio.create_task(self._monitor_system_health())
        logger.info("Health monitoring service started")

    async def stop_monitoring(self) -> None:
        """Stop the health monitoring service."""
        self.monitoring_active = False

        # Stop all container monitors
        for monitor in self.container_monitors.values():
            await monitor.stop_monitoring()

        # Stop system monitoring
        if self.system_monitoring_task:
            self.system_monitoring_task.cancel()
            try:
                await self.system_monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitoring service stopped")

    async def add_container_monitor(self, container_id: str, language: str) -> None:
        """Add monitoring for a new container."""
        if container_id in self.container_monitors:
            logger.warning("Container %s already being monitored", container_id)
            return

        monitor = ContainerHealthMonitor(container_id, language)

        # Add default recovery handlers
        for action, handler in self.default_recovery_handlers.items():
            monitor.add_recovery_handler(action, handler)

        self.container_monitors[container_id] = monitor
        await monitor.start_monitoring()

        logger.info("Added health monitoring for container %s", container_id)

    async def remove_container_monitor(self, container_id: str) -> None:
        """Remove monitoring for a container."""
        monitor = self.container_monitors.pop(container_id, None)
        if monitor:
            await monitor.stop_monitoring()
            logger.info("Removed health monitoring for container %s", container_id)

    def get_container_health(self, container_id: str) -> ContainerHealth | None:
        """Get health status for a specific container."""
        monitor = self.container_monitors.get(container_id)
        return monitor.get_health_status() if monitor else None

    def get_all_container_health(self) -> dict[str, ContainerHealth]:
        """Get health status for all monitored containers."""
        return {
            container_id: monitor.get_health_status()
            for container_id, monitor in self.container_monitors.items()
        }

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary."""
        total_containers = len(self.container_monitors)
        healthy_containers = 0
        unhealthy_containers = 0
        degraded_containers = 0

        for monitor in self.container_monitors.values():
            status = monitor.get_health_status().overall_status
            if status == HealthStatus.HEALTHY:
                healthy_containers += 1
            elif status == HealthStatus.UNHEALTHY:
                unhealthy_containers += 1
            elif status == HealthStatus.DEGRADED:
                degraded_containers += 1

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_containers": total_containers,
            "healthy_containers": healthy_containers,
            "unhealthy_containers": unhealthy_containers,
            "degraded_containers": degraded_containers,
            "health_percentage": (
                (healthy_containers / total_containers * 100) if total_containers > 0 else 0
            ),
            "monitored_containers": [
                monitor.get_health_summary() for monitor in self.container_monitors.values()
            ],
        }

    async def _monitor_system_health(self) -> None:
        """Monitor overall system health."""
        while self.monitoring_active:
            try:
                # Perform system-level health checks
                summary = self.get_health_summary()

                # Log system health status
                if summary["unhealthy_containers"] > 0:
                    logger.warning(
                        "System health: %s unhealthy containers",
                        summary["unhealthy_containers"],
                    )

                # Check for system-wide issues
                if summary["health_percentage"] < 50:
                    logger.critical("System health critical: less than 50% of containers healthy")

            except (RuntimeError, ValueError) as e:
                logger.error("System health monitoring error: %s", e)

            await asyncio.sleep(60)  # Check every minute

    def export_health_data(self) -> str:
        """Export health data as JSON."""
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": self.get_health_summary(),
            "containers": {
                container_id: monitor.get_health_status().to_dict()
                for container_id, monitor in self.container_monitors.items()
            },
        }
        return json.dumps(data, indent=2)


# Global health monitoring service
_health_service = None


def get_health_service() -> HealthMonitoringService:
    """Get the global health monitoring service instance."""
    global _health_service
    if _health_service is None:
        _health_service = HealthMonitoringService()
    return _health_service
