"""
Resource usage monitoring and alerting system for test execution infrastructure.

This module provides comprehensive resource monitoring with alert thresholds,
resource quotas, and automated resource management for container-based test execution.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

# Try to import psutil, but make it optional
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # psutil integration is optional - application will work without resource monitoring
    psutil: Any = None

# Try to import docker, but make it optional
try:
    import docker

    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False
    # Docker integration is optional
    docker: Any = None

logger = logging.getLogger(__name__)


class ResourceType(StrEnum):
    """Types of system resources."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    CONTAINER_COUNT = "container_count"
    OPEN_FILES = "open_files"


class AlertLevel(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertAction(StrEnum):
    """Actions to take when alerts are triggered."""

    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    TERMINATE_CONTAINERS = "terminate_containers"
    SCALE_DOWN = "scale_down"


@dataclass
class ResourceThreshold:
    """Resource threshold configuration."""

    resource_type: ResourceType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    unit: str
    duration_seconds: int = 60  # Sustained threshold duration

    def get_alert_level(self, value: float) -> AlertLevel | None:
        """Get alert level for a resource value."""
        if value >= self.emergency_threshold:
            return AlertLevel.EMERGENCY
        elif value >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif value >= self.warning_threshold:
            return AlertLevel.WARNING
        return None


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""

    timestamp: datetime
    resource_type: ResourceType
    current_value: float
    max_value: float
    usage_percent: float
    unit: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "current_value": self.current_value,
            "max_value": self.max_value,
            "usage_percent": self.usage_percent,
            "unit": self.unit,
            "metadata": self.metadata,
        }


@dataclass
class ResourceAlert:
    """Resource usage alert."""

    timestamp: datetime
    resource_type: ResourceType
    alert_level: AlertLevel
    current_value: float
    threshold_value: float
    usage_percent: float
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "alert_level": self.alert_level.value,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "usage_percent": self.usage_percent,
            "message": self.message,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


class ResourceCollector:
    """Collects system resource usage metrics."""

    def __init__(self):
        """Initialize the ResourceCollector service.

        This initializes the ResourceCollector with required dependencies
        and configuration for service operations.
        """
        self.docker_client = None
        if HAS_DOCKER:
            try:
                self.docker_client = docker.from_env()  # type: ignore[possibly-missing-attribute]
            except Exception as e:
                logger.warning(f"Docker client initialization failed: {e}")

    def collect_system_resources(self) -> list[ResourceUsage]:
        """Collect system-wide resource usage."""
        resources = []
        now = datetime.now(UTC)

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)  # type: ignore[possibly-missing-attribute]
        resources.append(
            ResourceUsage(
                timestamp=now,
                resource_type=ResourceType.CPU,
                current_value=cpu_percent,
                max_value=100.0,
                usage_percent=cpu_percent,
                unit="percent",
            )
        )

        # Memory usage
        memory = psutil.virtual_memory()  # type: ignore[possibly-missing-attribute]
        resources.append(
            ResourceUsage(
                timestamp=now,
                resource_type=ResourceType.MEMORY,
                current_value=memory.used / (1024**3),  # GB
                max_value=memory.total / (1024**3),  # GB
                usage_percent=memory.percent,
                unit="GB",
                metadata={
                    "available_gb": memory.available / (1024**3),
                    "buffers_gb": getattr(memory, "buffers", 0) / (1024**3),
                    "cached_gb": getattr(memory, "cached", 0) / (1024**3),
                },
            )
        )

        # Disk usage
        disk = psutil.disk_usage("/")  # type: ignore[possibly-missing-attribute]
        resources.append(
            ResourceUsage(
                timestamp=now,
                resource_type=ResourceType.DISK,
                current_value=disk.used / (1024**3),  # GB
                max_value=disk.total / (1024**3),  # GB
                usage_percent=(disk.used / disk.total) * 100,
                unit="GB",
                metadata={
                    "free_gb": disk.free / (1024**3),
                    "mount_point": "/",
                },
            )
        )

        # Network usage (if available)
        try:
            network = psutil.net_io_counters()  # type: ignore[possibly-missing-attribute]
            resources.append(
                ResourceUsage(
                    timestamp=now,
                    resource_type=ResourceType.NETWORK,
                    current_value=network.bytes_sent + network.bytes_recv,
                    max_value=0,  # No max for network
                    usage_percent=0,  # No percentage for network
                    unit="bytes",
                    metadata={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv,
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Failed to collect network stats: {e}")

        # Container count
        try:
            if self.docker_client is None:
                raise RuntimeError("Docker client not available")
            containers = self.docker_client.containers.list()
            test_containers = [c for c in containers if "test-executor" in c.name]

            resources.append(
                ResourceUsage(
                    timestamp=now,
                    resource_type=ResourceType.CONTAINER_COUNT,
                    current_value=len(test_containers),
                    max_value=50,  # Configurable max
                    usage_percent=(len(test_containers) / 50) * 100,
                    unit="count",
                    metadata={
                        "total_containers": len(containers),
                        "test_containers": len(test_containers),
                        "running_containers": len([c for c in containers if c.status == "running"]),
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Failed to collect container stats: {e}")

        return resources

    def collect_container_resources(self, container_id: str) -> list[ResourceUsage]:
        """Collect resource usage for a specific container."""
        resources = []
        now = datetime.now(UTC)

        try:
            if self.docker_client is None:
                raise RuntimeError("Docker client not available")
            container = self.docker_client.containers.get(container_id)
            stats: dict[str, Any] = container.stats(stream=False)

            # CPU usage
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            )

            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (
                    (cpu_delta / system_delta)
                    * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
                    * 100
                )

            resources.append(
                ResourceUsage(
                    timestamp=now,
                    resource_type=ResourceType.CPU,
                    current_value=cpu_percent,
                    max_value=100.0,
                    usage_percent=cpu_percent,
                    unit="percent",
                    metadata={"container_id": container_id},
                )
            )

            # Memory usage
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            memory_percent = (memory_usage / memory_limit) * 100

            resources.append(
                ResourceUsage(
                    timestamp=now,
                    resource_type=ResourceType.MEMORY,
                    current_value=memory_usage / (1024**2),  # MB
                    max_value=memory_limit / (1024**2),  # MB
                    usage_percent=memory_percent,
                    unit="MB",
                    metadata={
                        "container_id": container_id,
                        "cache_mb": stats["memory_stats"].get("cache", 0) / (1024**2),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to collect container {container_id} stats: {e}")

        return resources


class AlertManager:
    """Manages resource alerts and notifications."""

    def __init__(self):
        """Initialize the AlertManager service.

        This initializes the AlertManager with required dependencies
        and configuration for service operations.
        """
        self.alert_handlers: dict[AlertAction, Callable] = {}
        self.active_alerts: dict[str, ResourceAlert] = {}
        self.alert_history: list[ResourceAlert] = []
        self.suppressed_alerts: dict[str, datetime] = {}
        self.lock = threading.Lock()

    def register_alert_handler(self, action: AlertAction, handler: Callable) -> None:
        """Register a handler for alert actions."""
        self.alert_handlers[action] = handler

    def create_alert(
        self,
        resource_usage: ResourceUsage,
        threshold: ResourceThreshold,
        alert_level: AlertLevel,
    ) -> ResourceAlert:
        """Create a new resource alert."""
        threshold_value = {
            AlertLevel.WARNING: threshold.warning_threshold,
            AlertLevel.CRITICAL: threshold.critical_threshold,
            AlertLevel.EMERGENCY: threshold.emergency_threshold,
        }[alert_level]

        message = (
            f"{resource_usage.resource_type.value.upper()} usage "
            f"{alert_level.value.upper()}: "
            f"{resource_usage.current_value:.1f} {resource_usage.unit} "
            f"({resource_usage.usage_percent:.1f}%) "
            f"exceeds {threshold_value} {resource_usage.unit} threshold"
        )

        return ResourceAlert(
            timestamp=datetime.now(UTC),
            resource_type=resource_usage.resource_type,
            alert_level=alert_level,
            current_value=resource_usage.current_value,
            threshold_value=threshold_value,
            usage_percent=resource_usage.usage_percent,
            message=message,
            metadata=resource_usage.metadata,
        )

    def process_alert(self, alert: ResourceAlert) -> None:
        """Process and potentially trigger actions for an alert."""
        alert_key = f"{alert.resource_type.value}_{alert.alert_level.value}"

        with self.lock:
            # Check if alert is suppressed
            if alert_key in self.suppressed_alerts:
                suppressed_until = self.suppressed_alerts[alert_key]
                if datetime.now(UTC) < suppressed_until:
                    return

            # Store alert
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)

            # Keep only last 1000 alerts in history
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]

        # Trigger alert actions
        self._trigger_alert_actions(alert)

        logger.warning(f"Resource alert triggered: {alert.message}")

    def _trigger_alert_actions(self, alert: ResourceAlert) -> None:
        """Trigger appropriate actions based on alert level."""
        actions = self._get_actions_for_alert(alert)

        for action in actions:
            handler = self.alert_handlers.get(action)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler {action} failed: {e}")

    def _get_actions_for_alert(self, alert: ResourceAlert) -> list[AlertAction]:
        """Get actions to take for an alert."""
        actions = [AlertAction.LOG]  # Always log

        if alert.alert_level == AlertLevel.WARNING:
            actions.extend([AlertAction.EMAIL])
        elif alert.alert_level == AlertLevel.CRITICAL:
            actions.extend([AlertAction.EMAIL, AlertAction.SLACK])
        elif alert.alert_level == AlertLevel.EMERGENCY:
            actions.extend(
                [
                    AlertAction.EMAIL,
                    AlertAction.SLACK,
                    AlertAction.WEBHOOK,
                    AlertAction.SCALE_DOWN,
                ]
            )

            # Emergency CPU/Memory alerts may require container termination
            if alert.resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                actions.append(AlertAction.TERMINATE_CONTAINERS)

        return actions

    def acknowledge_alert(self, alert_key: str) -> bool:
        """Acknowledge an active alert."""
        with self.lock:
            if alert_key in self.active_alerts:
                self.active_alerts[alert_key].acknowledged = True
                return True
        return False

    def suppress_alert(self, alert_key: str, duration_minutes: int = 60) -> None:
        """Suppress an alert type for a specified duration."""
        suppressed_until = datetime.now(UTC) + timedelta(minutes=duration_minutes)
        with self.lock:
            self.suppressed_alerts[alert_key] = suppressed_until

    def get_active_alerts(self) -> list[ResourceAlert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> list[ResourceAlert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        with self.lock:
            return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class ResourceMonitor:
    """Main resource monitoring service."""

    def __init__(self, monitoring_interval: int = 30):
        """Initialize the ResourceMonitor service.

        This initializes the ResourceMonitor with required dependencies
        and configuration for service operations.

        Args:
            monitoring_interval (int): Monitoring interval
        """
        self.monitoring_interval = monitoring_interval
        self.collector = ResourceCollector()
        self.alert_manager = AlertManager()
        self.thresholds = self._load_default_thresholds()
        self.resource_history: list[ResourceUsage] = []
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None
        self.lock = threading.Lock()

        # Register default alert handlers
        self._register_default_handlers()

    def _load_default_thresholds(
        self,
    ) -> dict[ResourceType, ResourceThreshold]:
        """Load default resource thresholds."""
        return {
            ResourceType.CPU: ResourceThreshold(
                resource_type=ResourceType.CPU,
                warning_threshold=70.0,
                critical_threshold=85.0,
                emergency_threshold=95.0,
                unit="percent",
            ),
            ResourceType.MEMORY: ResourceThreshold(
                resource_type=ResourceType.MEMORY,
                warning_threshold=80.0,
                critical_threshold=90.0,
                emergency_threshold=95.0,
                unit="percent",
            ),
            ResourceType.DISK: ResourceThreshold(
                resource_type=ResourceType.DISK,
                warning_threshold=80.0,
                critical_threshold=90.0,
                emergency_threshold=95.0,
                unit="percent",
            ),
            ResourceType.CONTAINER_COUNT: ResourceThreshold(
                resource_type=ResourceType.CONTAINER_COUNT,
                warning_threshold=30,
                critical_threshold=40,
                emergency_threshold=45,
                unit="count",
            ),
        }

    def _register_default_handlers(self) -> None:
        """Register default alert handlers."""
        self.alert_manager.register_alert_handler(AlertAction.LOG, self._log_alert_handler)
        self.alert_manager.register_alert_handler(
            AlertAction.TERMINATE_CONTAINERS,
            self._terminate_containers_handler,
        )
        self.alert_manager.register_alert_handler(AlertAction.SCALE_DOWN, self._scale_down_handler)

    def _log_alert_handler(self, alert: ResourceAlert) -> None:
        """Default log handler for alerts."""
        logger.warning(f"RESOURCE ALERT: {alert.message}")

    def _terminate_containers_handler(self, alert: ResourceAlert) -> None:
        """Handler to terminate containers during emergency."""
        try:
            if self.collector.docker_client is None:
                raise RuntimeError("Docker client not available")
            containers = self.collector.docker_client.containers.list()
            test_containers = [c for c in containers if "test-executor" in c.name]

            # Terminate oldest containers first
            test_containers.sort(key=lambda x: x.attrs["Created"])

            terminated_count = 0
            # Terminate up to 5 containers
            for container in test_containers[:5]:
                try:
                    container.stop(timeout=10)
                    terminated_count += 1
                    logger.warning(
                        f"Terminated container {container.name} due to resource emergency"
                    )
                except Exception as e:
                    logger.error(f"Failed to terminate container {container.name}: {e}")

            logger.info(f"Terminated {terminated_count} containers due to resource emergency")

        except Exception as e:
            logger.error(f"Emergency container termination failed: {e}")

    def _scale_down_handler(self, alert: ResourceAlert) -> None:
        """Handler to scale down operations during emergency."""
        logger.warning("Scaling down operations due to resource emergency")
        # This would integrate with orchestration system to reduce container limits

    async def start_monitoring(self) -> None:
        """Start the resource monitoring service."""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the resource monitoring service."""
        self.monitoring_active = False
        if self.monitoring_task:
            await self.monitoring_task
        logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system resources
                resources = self.collector.collect_system_resources()

                # Store resource history
                with self.lock:
                    self.resource_history.extend(resources)

                    # Keep only last 24 hours of data
                    cutoff_time = datetime.now(UTC) - timedelta(hours=24)
                    self.resource_history = [
                        r for r in self.resource_history if r.timestamp >= cutoff_time
                    ]

                # Check thresholds and create alerts
                for resource in resources:
                    threshold = self.thresholds.get(resource.resource_type)
                    if threshold:
                        alert_level = threshold.get_alert_level(resource.usage_percent)
                        if alert_level:
                            alert = self.alert_manager.create_alert(
                                resource, threshold, alert_level
                            )
                            self.alert_manager.process_alert(alert)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

            await asyncio.sleep(self.monitoring_interval)

    def get_current_usage(self) -> list[ResourceUsage]:
        """Get current resource usage snapshot."""
        return self.collector.collect_system_resources()

    def get_resource_history(self, hours: int = 24) -> list[ResourceUsage]:
        """Get resource usage history."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        with self.lock:
            return [r for r in self.resource_history if r.timestamp >= cutoff_time]

    def get_resource_summary(self) -> dict[str, Any]:
        """Get resource usage summary."""
        current_usage = self.get_current_usage()
        active_alerts = self.alert_manager.get_active_alerts()

        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "current_usage": {
                resource.resource_type.value: {
                    "value": resource.current_value,
                    "usage_percent": resource.usage_percent,
                    "unit": resource.unit,
                }
                for resource in current_usage
            },
            "active_alerts": len(active_alerts),
            "alert_breakdown": {
                level.value: len([a for a in active_alerts if a.alert_level == level])
                for level in AlertLevel
            },
            "thresholds": {
                resource_type.value: {
                    "warning": threshold.warning_threshold,
                    "critical": threshold.critical_threshold,
                    "emergency": threshold.emergency_threshold,
                    "unit": threshold.unit,
                }
                for resource_type, threshold in self.thresholds.items()
            },
        }

        return summary

    def set_threshold(self, resource_type: ResourceType, threshold: ResourceThreshold) -> None:
        """Set custom threshold for a resource type."""
        self.thresholds[resource_type] = threshold

    def export_data(self, hours: int = 24) -> str:
        """Export monitoring data as JSON."""
        data = {
            "resource_history": [r.to_dict() for r in self.get_resource_history(hours)],
            "alert_history": [a.to_dict() for a in self.alert_manager.get_alert_history(hours)],
            "active_alerts": [a.to_dict() for a in self.alert_manager.get_active_alerts()],
            "thresholds": {k.value: v.__dict__ for k, v in self.thresholds.items()},
        }
        return json.dumps(data, indent=2, default=str)


# Global resource monitor instance
_resource_monitor = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor
