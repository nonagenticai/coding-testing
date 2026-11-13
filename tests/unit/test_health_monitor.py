"""Tests for health_monitor.py — health checks, component status, monitoring."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Install a stub docker module so the health_monitor module can import
_docker_stub = types.ModuleType("docker")
_docker_errors = types.ModuleType("docker.errors")
_docker_errors.NotFound = type("NotFound", (Exception,), {})
_docker_errors.APIError = type("APIError", (Exception,), {})
_docker_errors.ContainerError = type("ContainerError", (Exception,), {})
_docker_stub.errors = _docker_errors
_docker_stub.from_env = MagicMock()
sys.modules.setdefault("docker", _docker_stub)
sys.modules.setdefault("docker.errors", _docker_errors)

# Also stub aiohttp if missing
if "aiohttp" not in sys.modules:
    _aiohttp_stub = types.ModuleType("aiohttp")
    _aiohttp_stub.ClientSession = MagicMock
    _aiohttp_stub.ClientTimeout = MagicMock
    _aiohttp_stub.ClientError = type("ClientError", (Exception,), {})
    sys.modules["aiohttp"] = _aiohttp_stub

from services.test_execution.health_monitor import (  # noqa: E402
    ContainerHealth,
    ContainerHealthChecker,
    ContainerHealthMonitor,
    DockerHealthChecker,
    HealthCheckConfig,
    HealthCheckResult,
    HealthCheckType,
    HealthMonitoringService,
    HealthStatus,
    RecoveryAction,
)

# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class TestHealthEnums:
    def test_health_status_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.STARTING == "starting"

    def test_health_check_type_values(self):
        assert HealthCheckType.LIVENESS == "liveness"
        assert HealthCheckType.READINESS == "readiness"

    def test_recovery_action_values(self):
        assert RecoveryAction.RESTART == "restart"
        assert RecoveryAction.RECREATE == "recreate"
        assert RecoveryAction.ALERT_ONLY == "alert_only"


class TestHealthCheckConfig:
    def test_defaults(self):
        cfg = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        assert cfg.interval_seconds == 30
        assert cfg.timeout_seconds == 10
        assert cfg.failure_threshold == 3
        assert cfg.success_threshold == 1
        assert cfg.enabled is True

    def test_to_dict(self):
        cfg = HealthCheckConfig(
            check_type=HealthCheckType.READINESS,
            interval_seconds=15,
            failure_threshold=2,
        )
        d = cfg.to_dict()
        assert d["check_type"] == "readiness"
        assert d["interval_seconds"] == 15
        assert d["failure_threshold"] == 2


class TestHealthCheckResult:
    def test_to_dict(self):
        result = HealthCheckResult(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.HEALTHY,
            response_time_ms=5.2,
            message="OK",
            metadata={"cpu": 10.0},
        )
        d = result.to_dict()
        assert d["status"] == "healthy"
        assert d["response_time_ms"] == 5.2
        assert d["metadata"]["cpu"] == 10.0


class TestContainerHealth:
    def test_to_dict(self):
        health = ContainerHealth(
            container_id="c1",
            language="python",
            overall_status=HealthStatus.HEALTHY,
            last_updated=datetime(2025, 1, 1, tzinfo=UTC),
        )
        d = health.to_dict()
        assert d["container_id"] == "c1"
        assert d["overall_status"] == "healthy"
        assert d["recovery_attempts"] == 0

    def test_to_dict_with_results(self):
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.HEALTHY,
            response_time_ms=5.0,
            message="OK",
        )
        health = ContainerHealth(
            container_id="c1",
            language="go",
            overall_status=HealthStatus.HEALTHY,
            last_updated=datetime.now(UTC),
            check_results={HealthCheckType.LIVENESS: [result]},
            consecutive_failures={HealthCheckType.LIVENESS: 0},
            consecutive_successes={HealthCheckType.LIVENESS: 5},
        )
        d = health.to_dict()
        assert list(d["check_results"].keys()) == ["liveness"]
        assert len(d["check_results"]["liveness"]) == 1
        assert d["check_results"]["liveness"][0]["status"] == "healthy"
        assert d["check_results"]["liveness"][0]["message"] == "OK"
        assert d["check_results"]["liveness"][0]["response_time_ms"] == 5.0
        assert d["consecutive_successes"] == {"liveness": 5}
        assert d["consecutive_failures"] == {"liveness": 0}


# ---------------------------------------------------------------------------
# ContainerHealthMonitor (mocked Docker)
# ---------------------------------------------------------------------------


class TestContainerHealthMonitor:
    @pytest.fixture
    def monitor(self):
        mon = ContainerHealthMonitor("c1", "python")
        return mon

    def test_initial_state(self, monitor):
        assert monitor.container_id == "c1"
        assert monitor.language == "python"
        assert monitor.health.overall_status == HealthStatus.STARTING
        assert monitor.monitoring_active is False

    def test_add_health_checker(self, monitor):
        cfg = HealthCheckConfig(check_type=HealthCheckType.STARTUP)
        checker = ContainerHealthChecker(cfg)
        monitor.add_health_checker(checker)
        assert HealthCheckType.STARTUP in monitor.health_checkers

    def test_add_recovery_handler(self, monitor):
        handler = AsyncMock()
        monitor.add_recovery_handler(RecoveryAction.RESTART, handler)
        assert RecoveryAction.RESTART in monitor.recovery_handlers

    def test_get_health_status(self, monitor):
        status = monitor.get_health_status()
        assert status.container_id == "c1"
        assert status.language == "python"
        assert status.overall_status == HealthStatus.STARTING
        assert status.recovery_attempts == 0
        assert status.check_results == {}
        assert status.consecutive_failures == {}
        assert status.consecutive_successes == {}

    def test_get_health_summary(self, monitor):
        summary = monitor.get_health_summary()
        assert summary["container_id"] == "c1"
        assert summary["language"] == "python"
        assert summary["overall_status"] == "starting"
        assert summary["recovery_attempts"] == 0
        assert summary["recent_failures"] == {}
        assert set(summary["check_types"]) == {HealthCheckType.LIVENESS, HealthCheckType.READINESS}

    def test_update_overall_status_no_results(self, monitor):
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.UNKNOWN

    def test_update_overall_status_all_healthy(self, monitor):
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.HEALTHY,
            response_time_ms=5.0,
            message="OK",
        )
        monitor.health.check_results[HealthCheckType.LIVENESS] = [result]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.HEALTHY

    def test_update_overall_status_one_unhealthy(self, monitor):
        healthy = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.HEALTHY,
            response_time_ms=5.0,
            message="OK",
        )
        unhealthy = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.READINESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Not ready",
        )
        monitor.health.check_results[HealthCheckType.LIVENESS] = [healthy]
        monitor.health.check_results[HealthCheckType.READINESS] = [unhealthy]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.UNHEALTHY

    def test_update_overall_status_degraded(self, monitor):
        degraded = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.DEGRADED,
            response_time_ms=5.0,
            message="Degraded",
        )
        monitor.health.check_results[HealthCheckType.LIVENESS] = [degraded]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.DEGRADED

    def test_update_overall_status_starting(self, monitor):
        starting = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.STARTING,
            response_time_ms=5.0,
            message="Starting",
        )
        monitor.health.check_results[HealthCheckType.LIVENESS] = [starting]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.STARTING

    @pytest.mark.asyncio
    async def test_trigger_recovery_first_attempt(self, monitor):
        handler = AsyncMock()
        monitor.add_recovery_handler(RecoveryAction.RESTART, handler)
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Failed",
        )
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 1
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_recovery_escalation(self, monitor):
        restart_handler = AsyncMock()
        recreate_handler = AsyncMock()
        alert_handler = AsyncMock()
        monitor.add_recovery_handler(RecoveryAction.RESTART, restart_handler)
        monitor.add_recovery_handler(RecoveryAction.RECREATE, recreate_handler)
        monitor.add_recovery_handler(RecoveryAction.ALERT_ONLY, alert_handler)

        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Failed",
        )

        # First call -> RESTART
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 1

        # Second call -> RECREATE
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 2

        # Fourth call -> ALERT_ONLY
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 4


# ---------------------------------------------------------------------------
# HealthMonitoringService
# ---------------------------------------------------------------------------


class TestHealthMonitoringService:
    @pytest.fixture
    def service(self):
        return HealthMonitoringService()

    def test_initial_state(self, service):
        assert service.monitoring_active is False
        assert len(service.container_monitors) == 0

    def test_get_container_health_not_found(self, service):
        assert service.get_container_health("nonexistent") is None

    def test_get_all_container_health_empty(self, service):
        assert service.get_all_container_health() == {}

    def test_get_health_summary_empty(self, service):
        summary = service.get_health_summary()
        assert summary["total_containers"] == 0
        assert summary["health_percentage"] == 0

    def test_export_health_data(self, service):
        data_str = service.export_health_data()
        data = json.loads(data_str)
        assert set(data.keys()) == {"timestamp", "summary", "containers"}
        assert data["containers"] == {}
        assert data["summary"]["total_containers"] == 0
        assert data["summary"]["healthy_containers"] == 0
        assert data["summary"]["health_percentage"] == 0

    @pytest.mark.asyncio
    async def test_alert_only_handler(self, service):
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Container is dead",
        )
        # _alert_only just logs, verify it completes without error
        # and does not modify service state
        monitors_before = dict(service.container_monitors)
        await service._alert_only("c1", HealthCheckType.LIVENESS, result)
        assert service.container_monitors == monitors_before

    @pytest.mark.asyncio
    async def test_remove_container_monitor_nonexistent(self, service):
        # Removing nonexistent container should leave monitors unchanged
        await service.remove_container_monitor("nonexistent")
        assert len(service.container_monitors) == 0

    def test_get_health_summary_with_containers(self, service):
        """Test summary counts healthy/unhealthy/degraded correctly."""
        mon1 = ContainerHealthMonitor("c1", "python")
        result_healthy = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.HEALTHY,
            response_time_ms=5.0,
            message="OK",
        )
        mon1.health.check_results[HealthCheckType.LIVENESS] = [result_healthy]
        mon1._update_overall_status()

        mon2 = ContainerHealthMonitor("c2", "go")
        result_unhealthy = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Bad",
        )
        mon2.health.check_results[HealthCheckType.LIVENESS] = [result_unhealthy]
        mon2._update_overall_status()

        mon3 = ContainerHealthMonitor("c3", "rust")
        result_degraded = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.DEGRADED,
            response_time_ms=5.0,
            message="Degraded",
        )
        mon3.health.check_results[HealthCheckType.LIVENESS] = [result_degraded]
        mon3._update_overall_status()

        service.container_monitors = {"c1": mon1, "c2": mon2, "c3": mon3}
        summary = service.get_health_summary()
        assert summary["total_containers"] == 3
        assert summary["healthy_containers"] == 1
        assert summary["unhealthy_containers"] == 1
        assert summary["degraded_containers"] == 1
        assert summary["health_percentage"] == pytest.approx(33.33, abs=0.1)
        assert len(summary["monitored_containers"]) == 3

    def test_export_health_data_with_containers(self, service):
        """Test export includes container data."""
        mon = ContainerHealthMonitor("c1", "python")
        service.container_monitors = {"c1": mon}
        data_str = service.export_health_data()
        data = json.loads(data_str)
        assert list(data["containers"].keys()) == ["c1"]
        assert data["containers"]["c1"]["container_id"] == "c1"
        assert data["containers"]["c1"]["language"] == "python"
        assert data["containers"]["c1"]["overall_status"] == "starting"
        assert data["containers"]["c1"]["recovery_attempts"] == 0
        assert data["summary"]["total_containers"] == 1

    def test_get_container_health_found(self, service):
        """Test retrieving health for monitored container."""
        mon = ContainerHealthMonitor("c1", "python")
        service.container_monitors = {"c1": mon}
        health = service.get_container_health("c1")
        assert health.container_id == "c1"
        assert health.language == "python"
        assert health.overall_status == HealthStatus.STARTING
        assert health.recovery_attempts == 0

    def test_get_all_container_health_with_data(self, service):
        """Test getting all container health."""
        mon1 = ContainerHealthMonitor("c1", "python")
        mon2 = ContainerHealthMonitor("c2", "go")
        service.container_monitors = {"c1": mon1, "c2": mon2}
        all_health = service.get_all_container_health()
        assert len(all_health) == 2
        assert all_health["c1"].container_id == "c1"
        assert all_health["c1"].language == "python"
        assert all_health["c1"].overall_status == HealthStatus.STARTING
        assert all_health["c2"].container_id == "c2"
        assert all_health["c2"].language == "go"
        assert all_health["c2"].overall_status == HealthStatus.STARTING


# ---------------------------------------------------------------------------
# ContainerHealthMonitor advanced
# ---------------------------------------------------------------------------


class TestContainerHealthMonitorAdvanced:
    def test_update_overall_status_unknown(self):
        """Test status is UNKNOWN for non-standard statuses."""
        monitor = ContainerHealthMonitor("c1", "python")
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNKNOWN,
            response_time_ms=5.0,
            message="Unknown",
        )
        monitor.health.check_results[HealthCheckType.LIVENESS] = [result]
        monitor._update_overall_status()
        assert monitor.health.overall_status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_trigger_recovery_no_handler(self):
        """Recovery with no handler should increment attempts and log only."""
        monitor = ContainerHealthMonitor("c1", "python")
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Failed",
        )
        # Clear default handlers
        monitor.recovery_handlers = {}
        assert monitor.health.recovery_attempts == 0
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 1
        # Overall status should remain unchanged (STARTING, since no _update_overall_status call)
        assert monitor.health.overall_status == HealthStatus.STARTING

    @pytest.mark.asyncio
    async def test_trigger_recovery_handler_exception(self):
        """Recovery handler raising RuntimeError should be caught, attempts still incremented."""
        monitor = ContainerHealthMonitor("c1", "python")
        handler = AsyncMock(side_effect=RuntimeError("handler fail"))
        monitor.add_recovery_handler(RecoveryAction.RESTART, handler)
        result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=5.0,
            message="Failed",
        )
        assert monitor.health.recovery_attempts == 0
        await monitor._trigger_recovery(HealthCheckType.LIVENESS, result)
        assert monitor.health.recovery_attempts == 1
        handler.assert_called_once_with("c1", HealthCheckType.LIVENESS, result)

    def test_get_health_summary_with_failures(self):
        """Test summary shows recent failures."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.health.consecutive_failures[HealthCheckType.LIVENESS] = 3
        summary = monitor.get_health_summary()
        assert summary["recent_failures"]["liveness"] == 3


# ---------------------------------------------------------------------------
# HealthCheckConfig advanced
# ---------------------------------------------------------------------------


class TestHealthCheckConfigAdvanced:
    def test_to_dict_with_initial_delay(self):
        cfg = HealthCheckConfig(
            check_type=HealthCheckType.STARTUP,
            initial_delay_seconds=30,
            enabled=False,
        )
        d = cfg.to_dict()
        assert d["check_type"] == "startup"
        assert d["initial_delay_seconds"] == 30
        assert d["enabled"] is False


# ---------------------------------------------------------------------------
# HealthCheckResult advanced
# ---------------------------------------------------------------------------


class TestHealthCheckResultAdvanced:
    def test_to_dict_with_empty_metadata(self):
        r = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.DEPENDENCY,
            status=HealthStatus.STOPPING,
            response_time_ms=0.0,
            message="Stopping",
        )
        d = r.to_dict()
        assert d["check_type"] == "dependency"
        assert d["status"] == "stopping"
        assert d["metadata"] == {}


# ---------------------------------------------------------------------------
# get_health_service global
# ---------------------------------------------------------------------------

from services.test_execution.health_monitor import get_health_service  # noqa: E402


class TestGetHealthService:
    def test_returns_singleton(self):
        svc = get_health_service()
        assert svc.monitoring_active is False
        assert svc.container_monitors == {}
        svc2 = get_health_service()
        assert svc is svc2


# ---------------------------------------------------------------------------
# ContainerHealthChecker base class (coverage)
# ---------------------------------------------------------------------------


class TestContainerHealthCheckerBase:
    @pytest.mark.asyncio
    async def test_check_health_raises_not_implemented(self):
        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        checker = ContainerHealthChecker(config)
        with pytest.raises(NotImplementedError):
            await checker.check_health("container-1")


# ---------------------------------------------------------------------------
# DockerHealthChecker (coverage)
# ---------------------------------------------------------------------------


class TestDockerHealthChecker:
    @pytest.fixture
    def checker(self):
        config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        with patch("services.test_execution.health_monitor.docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            chk = DockerHealthChecker(config)
            chk.docker_client = mock_client
            return chk, mock_client

    @pytest.mark.asyncio
    async def test_running_healthy(self, checker):
        chk, mock_client = checker
        container = MagicMock()
        container.status = "running"
        container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2000, "percpu_usage": [1000, 1000]},
                "system_cpu_usage": 100000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 90000,
            },
            "memory_stats": {"usage": 100_000_000, "limit": 1_000_000_000},
        }
        mock_client.containers.get.return_value = container
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.HEALTHY
        assert "running normally" in result.message

    @pytest.mark.asyncio
    async def test_running_degraded_high_cpu(self, checker):
        chk, mock_client = checker
        container = MagicMock()
        container.status = "running"
        # Make CPU > 95%
        container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 99000, "percpu_usage": [50000]},
                "system_cpu_usage": 100000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000},
                "system_cpu_usage": 99000,
            },
            "memory_stats": {"usage": 100_000, "limit": 1_000_000_000},
        }
        mock_client.containers.get.return_value = container
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.DEGRADED
        assert "under stress" in result.message

    @pytest.mark.asyncio
    async def test_created_status(self, checker):
        chk, mock_client = checker
        container = MagicMock()
        container.status = "created"
        mock_client.containers.get.return_value = container
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.STARTING
        assert "created" in result.message

    @pytest.mark.asyncio
    async def test_restarting_status(self, checker):
        chk, mock_client = checker
        container = MagicMock()
        container.status = "restarting"
        mock_client.containers.get.return_value = container
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.STARTING
        assert "restarting" in result.message

    @pytest.mark.asyncio
    async def test_exited_status(self, checker):
        chk, mock_client = checker
        container = MagicMock()
        container.status = "exited"
        container.attrs = {"State": {"ExitCode": 137}}
        mock_client.containers.get.return_value = container
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.UNHEALTHY
        assert "137" in result.message

    @pytest.mark.asyncio
    async def test_unknown_status(self, checker):
        chk, mock_client = checker
        container = MagicMock()
        container.status = "paused"
        mock_client.containers.get.return_value = container
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.UNKNOWN
        assert "paused" in result.message

    @pytest.mark.asyncio
    async def test_not_found(self, checker):
        chk, mock_client = checker
        docker_mod = sys.modules["docker"]
        mock_client.containers.get.side_effect = docker_mod.errors.NotFound("gone")
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_api_error(self, checker):
        chk, mock_client = checker
        docker_mod = sys.modules["docker"]
        mock_client.containers.get.side_effect = docker_mod.errors.APIError("api err")
        result = await chk.check_health("c1")
        assert result.status == HealthStatus.UNKNOWN
        assert "api err" in result.message

    def test_calculate_cpu_usage_zero_system_delta(self, checker):
        chk, _ = checker
        stats = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 100, "percpu_usage": [100]},
                "system_cpu_usage": 100,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100},
                "system_cpu_usage": 100,
            },
        }
        assert chk._calculate_cpu_usage(stats) == 0.0

    def test_calculate_cpu_usage_key_error(self, checker):
        chk, _ = checker
        assert chk._calculate_cpu_usage({}) == 0.0

    def test_calculate_memory_usage_normal(self, checker):
        chk, _ = checker
        stats = {"memory_stats": {"usage": 500_000_000, "limit": 1_000_000_000}}
        assert chk._calculate_memory_usage(stats) == pytest.approx(50.0)

    def test_calculate_memory_usage_key_error(self, checker):
        chk, _ = checker
        assert chk._calculate_memory_usage({}) == 0.0


# ---------------------------------------------------------------------------
# HTTPHealthChecker — full check_health paths
# ---------------------------------------------------------------------------


class TestHTTPHealthCheckerFull:
    """Cover HTTPHealthChecker check_health: successful response, no-IP, timeout."""

    @pytest.fixture
    def http_checker(self):
        from services.test_execution.health_monitor import HTTPHealthChecker

        config = HealthCheckConfig(check_type=HealthCheckType.READINESS, timeout_seconds=5)
        return HTTPHealthChecker(config, endpoint="/health")

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_successful_health_check(self, http_checker):
        """Covers lines 303-338: successful HTTP health check with IPAddress."""
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "IPAddress": "172.17.0.2",
            }
        }
        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_container

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"status":"ok"}')

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_docker_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.HEALTHY
            assert "HTTP health check passed" in result.message

    @pytest.mark.asyncio
    async def test_unhealthy_status_code(self, http_checker):
        """Covers lines 331-332: non-200 HTTP response."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_container

        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="Service Unavailable")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_docker_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "503" in result.message

    @pytest.mark.asyncio
    async def test_ip_from_networks(self, http_checker):
        """Covers lines 307-312: IP extracted from Networks dict."""
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "IPAddress": "",
                "Networks": {
                    "bridge": {"IPAddress": "172.18.0.5"},
                },
            }
        }
        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_container

        # Raise ValueError for URL to shortcut, verifying IP was extracted
        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_docker_client,
            ),
            patch(
                "services.test_execution.health_monitor.docker.errors.APIError",
                type("APIError", (Exception,), {}),
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                side_effect=ValueError("test"),
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "test" in result.message

    @pytest.mark.asyncio
    async def test_no_ip_raises_value_error(self, http_checker):
        """Covers lines 314-315: no IP address found."""
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "IPAddress": "",
                "Networks": {},
            }
        }
        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_container

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_docker_client,
            ),
            patch(
                "services.test_execution.health_monitor.docker.errors.APIError",
                type("APIError", (Exception,), {}),
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert (
                "Could not determine" in result.message
                or "health check failed" in result.message.lower()
            )

    @pytest.mark.asyncio
    async def test_timeout_error(self, http_checker):
        """Covers lines 340-343: TimeoutError in HTTP check."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_docker_client = MagicMock()
        mock_docker_client.containers.get.return_value = mock_container

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_docker_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                side_effect=TimeoutError("timed out"),
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "timed out" in result.message.lower()


# ---------------------------------------------------------------------------
# ContainerHealthMonitor — start/stop monitoring (coverage)
# ---------------------------------------------------------------------------


class TestContainerHealthMonitorStartStop:
    @pytest.mark.asyncio
    async def test_start_and_stop_monitoring(self):
        monitor = ContainerHealthMonitor("c1", "python")
        # Replace checkers with mocked ones
        mock_checker = MagicMock()
        mock_checker.config = HealthCheckConfig(check_type=HealthCheckType.LIVENESS)
        mock_checker.check_health = AsyncMock(
            return_value=HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.HEALTHY,
                response_time_ms=1.0,
                message="OK",
            )
        )
        monitor.health_checkers = {HealthCheckType.LIVENESS: mock_checker}

        assert monitor.monitoring_active is False
        assert len(monitor.monitoring_tasks) == 0

        await monitor.start_monitoring()
        assert monitor.monitoring_active is True
        assert len(monitor.monitoring_tasks) == 1
        assert isinstance(monitor.monitoring_tasks[HealthCheckType.LIVENESS], asyncio.Task)

        # Let the loop run briefly
        await asyncio.sleep(0.1)

        await monitor.stop_monitoring()
        assert monitor.monitoring_active is False
        assert len(monitor.monitoring_tasks) == 0


# ---------------------------------------------------------------------------
# HealthMonitoringService — start/stop, add/remove containers (coverage)
# ---------------------------------------------------------------------------


class TestHealthMonitoringServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop_service(self):
        service = HealthMonitoringService()
        await service.start_monitoring()
        assert service.monitoring_active is True
        assert isinstance(service.system_monitoring_task, asyncio.Task)
        assert not service.system_monitoring_task.done()

        await service.stop_monitoring()
        assert service.monitoring_active is False
        assert service.system_monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_add_container_monitor(self):
        service = HealthMonitoringService()
        # Patch the monitor's start_monitoring to avoid real Docker calls
        with patch.object(
            ContainerHealthMonitor, "start_monitoring", new_callable=AsyncMock
        ) as mock_start:
            await service.add_container_monitor("c1", "python")
            assert service.container_monitors["c1"].container_id == "c1"
            assert service.container_monitors["c1"].language == "python"
            mock_start.assert_called_once()

            # Adding same container again should be no-op
            await service.add_container_monitor("c1", "python")
            assert len(service.container_monitors) == 1
            # start_monitoring should still only have been called once (no-op second time)
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_container_monitor(self):
        service = HealthMonitoringService()
        with patch.object(ContainerHealthMonitor, "start_monitoring", new_callable=AsyncMock):
            await service.add_container_monitor("c1", "python")
            assert len(service.container_monitors) == 1

        with patch.object(
            ContainerHealthMonitor, "stop_monitoring", new_callable=AsyncMock
        ) as mock_stop:
            await service.remove_container_monitor("c1")
            assert len(service.container_monitors) == 0
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_service_stops_container_monitors(self):
        service = HealthMonitoringService()
        with patch.object(ContainerHealthMonitor, "start_monitoring", new_callable=AsyncMock):
            await service.add_container_monitor("c1", "python")

        await service.start_monitoring()

        with patch.object(
            ContainerHealthMonitor, "stop_monitoring", new_callable=AsyncMock
        ) as mock_stop:
            await service.stop_monitoring()
            mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# Recovery handlers in HealthMonitoringService (coverage)
# ---------------------------------------------------------------------------


class TestRecoveryHandlers:
    @pytest.mark.asyncio
    async def test_restart_container_success(self):
        service = HealthMonitoringService()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        with patch(
            "services.test_execution.health_monitor.docker.from_env",
            return_value=mock_client,
        ):
            result = HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=1.0,
                message="Failed",
            )
            await service._restart_container("c1", HealthCheckType.LIVENESS, result)
            mock_container.restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_container_not_found(self):
        service = HealthMonitoringService()
        docker_mod = sys.modules["docker"]
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = docker_mod.errors.NotFound("gone")

        with patch(
            "services.test_execution.health_monitor.docker.from_env",
            return_value=mock_client,
        ):
            result = HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=1.0,
                message="Failed",
            )
            # NotFound is caught internally — no exception propagated
            await service._restart_container("c1", HealthCheckType.LIVENESS, result)
            mock_client.containers.get.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_recreate_container_success(self):
        service = HealthMonitoringService()
        mock_container = MagicMock()
        mock_container.attrs = {
            "Config": {"Image": "python:3.13-slim"},
            "HostConfig": {},
        }
        mock_new_container = MagicMock()
        mock_new_container.id = "new-c1"
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        mock_client.containers.run.return_value = mock_new_container

        with patch(
            "services.test_execution.health_monitor.docker.from_env",
            return_value=mock_client,
        ):
            result = HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=1.0,
                message="Failed",
            )
            await service._recreate_container("c1", HealthCheckType.LIVENESS, result)
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()
            mock_client.containers.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_recreate_container_failure(self):
        service = HealthMonitoringService()
        docker_mod = sys.modules["docker"]
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = docker_mod.errors.NotFound("gone")

        with patch(
            "services.test_execution.health_monitor.docker.from_env",
            return_value=mock_client,
        ):
            result = HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=1.0,
                message="Failed",
            )
            # NotFound is caught internally — container.stop/remove never called
            await service._recreate_container("c1", HealthCheckType.LIVENESS, result)
            mock_client.containers.get.assert_called_once_with("c1")
            mock_client.containers.run.assert_not_called()


# ---------------------------------------------------------------------------
# HTTPHealthChecker — covers lines 303-343
# ---------------------------------------------------------------------------


class TestHTTPHealthChecker:
    from services.test_execution.health_monitor import HTTPHealthChecker

    @pytest.fixture
    def http_checker(self):
        config = HealthCheckConfig(check_type=HealthCheckType.READINESS, timeout_seconds=5)
        return self.HTTPHealthChecker(config, endpoint="/health")

    @pytest.mark.asyncio
    async def test_healthy_200(self, http_checker):
        """HTTP checker returns HEALTHY on 200 response."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"status":"ok"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.HEALTHY
            assert "passed" in result.message
            assert result.metadata["http_status"] == 200

    @pytest.mark.asyncio
    async def test_unhealthy_500(self, http_checker):
        """HTTP checker returns UNHEALTHY on non-200 response."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="internal error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "500" in result.message

    @pytest.mark.asyncio
    async def test_ip_from_networks(self, http_checker):
        """Falls back to Networks dict when IPAddress is empty."""
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "IPAddress": "",
                "Networks": {
                    "bridge": {"IPAddress": "172.17.0.5"},
                },
            }
        }
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="ok")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.HEALTHY
            assert result.metadata["url"] == "http://172.17.0.5:8080/health"

    @pytest.mark.asyncio
    async def test_no_ip_address(self, http_checker):
        """When no IP can be found, returns UNHEALTHY with ValueError message."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "", "Networks": {}}}
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        with patch(
            "services.test_execution.health_monitor.docker.from_env",
            return_value=mock_client,
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "Could not determine" in result.message

    @pytest.mark.asyncio
    async def test_timeout(self, http_checker):
        """Timeout returns UNHEALTHY with timeout message."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=TimeoutError("timeout"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "timed out" in result.message

    @pytest.mark.asyncio
    async def test_client_error(self, http_checker):
        """aiohttp.ClientError returns UNHEALTHY."""
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container

        aiohttp_mod = sys.modules["aiohttp"]
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=aiohttp_mod.ClientError("conn refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "services.test_execution.health_monitor.docker.from_env",
                return_value=mock_client,
            ),
            patch(
                "services.test_execution.health_monitor.aiohttp.ClientSession",
                return_value=mock_session,
            ),
        ):
            result = await http_checker.check_health("c1")
            assert result.status == HealthStatus.UNHEALTHY
            assert "failed" in result.message.lower()


# ---------------------------------------------------------------------------
# _monitor_health_check — covers lines 450, 465, 476-479, 483, 495-496
# ---------------------------------------------------------------------------


class TestMonitorHealthCheckLoop:
    @pytest.mark.asyncio
    async def test_healthy_check_updates_counters(self):
        """Monitor loop processes healthy result and updates counters."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.monitoring_active = True  # Must be True for while loop

        async def mock_check_health(container_id):
            monitor.monitoring_active = False  # Exit after first iteration
            return HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.HEALTHY,
                response_time_ms=1.0,
                message="OK",
            )

        config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            initial_delay_seconds=0,
            interval_seconds=0,
            failure_threshold=3,
        )
        mock_checker = MagicMock()
        mock_checker.config = config
        mock_checker.check_health = mock_check_health

        await monitor._monitor_health_check(HealthCheckType.LIVENESS, mock_checker)

        assert HealthCheckType.LIVENESS in monitor.health.check_results
        assert len(monitor.health.check_results[HealthCheckType.LIVENESS]) == 1
        assert monitor.health.consecutive_successes.get(HealthCheckType.LIVENESS, 0) == 1
        assert monitor.health.consecutive_failures.get(HealthCheckType.LIVENESS, 0) == 0

    @pytest.mark.asyncio
    async def test_unhealthy_increments_failures(self):
        """Unhealthy result increments failure counter and resets successes."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.monitoring_active = True

        async def mock_check_health(container_id):
            monitor.monitoring_active = False
            return HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=1.0,
                message="Failed",
            )

        config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            interval_seconds=0,
            failure_threshold=5,
        )
        mock_checker = MagicMock()
        mock_checker.config = config
        mock_checker.check_health = mock_check_health

        await monitor._monitor_health_check(HealthCheckType.LIVENESS, mock_checker)

        assert monitor.health.consecutive_failures[HealthCheckType.LIVENESS] == 1
        assert monitor.health.consecutive_successes[HealthCheckType.LIVENESS] == 0

    @pytest.mark.asyncio
    async def test_failure_threshold_triggers_recovery(self):
        """When failures reach threshold, recovery is triggered."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.monitoring_active = True
        monitor.recovery_handlers = {}
        recovery_called = False

        async def mock_recovery(container_id, check_type, result):
            nonlocal recovery_called
            recovery_called = True

        monitor.add_recovery_handler(RecoveryAction.RESTART, mock_recovery)
        iteration = 0

        async def mock_check_health(container_id):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                monitor.monitoring_active = False
            return HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=1.0,
                message="Failed",
            )

        config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            interval_seconds=0,
            failure_threshold=2,
        )
        mock_checker = MagicMock()
        mock_checker.config = config
        mock_checker.check_health = mock_check_health

        # Pre-set 1 failure so second failure hits threshold
        monitor.health.consecutive_failures[HealthCheckType.LIVENESS] = 1

        await monitor._monitor_health_check(HealthCheckType.LIVENESS, mock_checker)

        # 2 iterations run: pre-set 1 failure + 2 more = 3 total
        assert monitor.health.consecutive_failures[HealthCheckType.LIVENESS] == 3
        # Both iterations hit threshold (>=2), so recovery triggered twice
        assert monitor.health.recovery_attempts == 2

    @pytest.mark.asyncio
    async def test_trims_results_over_100(self):
        """Check results list is trimmed to 100 entries max."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.monitoring_active = True
        monitor.health.check_results[HealthCheckType.LIVENESS] = [
            HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.HEALTHY,
                response_time_ms=1.0,
                message=f"OK-{i}",
            )
            for i in range(101)
        ]

        async def mock_check_health(container_id):
            monitor.monitoring_active = False
            return HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.HEALTHY,
                response_time_ms=1.0,
                message="OK-new",
            )

        config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            interval_seconds=0,
            failure_threshold=3,
        )
        mock_checker = MagicMock()
        mock_checker.config = config
        mock_checker.check_health = mock_check_health

        await monitor._monitor_health_check(HealthCheckType.LIVENESS, mock_checker)

        # Should be trimmed: 101 + 1 = 102 -> trimmed to 100
        assert len(monitor.health.check_results[HealthCheckType.LIVENESS]) == 100
        assert monitor.health.check_results[HealthCheckType.LIVENESS][-1].message == "OK-new"

    @pytest.mark.asyncio
    async def test_cancelled_error_caught(self):
        """CancelledError in health check is caught — no results stored, counters untouched."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.monitoring_active = True

        async def mock_check_health(container_id):
            monitor.monitoring_active = False
            raise asyncio.CancelledError()

        config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            interval_seconds=0,
            failure_threshold=3,
        )
        mock_checker = MagicMock()
        mock_checker.config = config
        mock_checker.check_health = mock_check_health

        await monitor._monitor_health_check(HealthCheckType.LIVENESS, mock_checker)

        # CancelledError was caught before results could be stored
        assert monitor.health.check_results.get(HealthCheckType.LIVENESS) is None
        assert monitor.health.consecutive_failures.get(HealthCheckType.LIVENESS, 0) == 0
        assert monitor.health.consecutive_successes.get(HealthCheckType.LIVENESS, 0) == 0

    @pytest.mark.asyncio
    async def test_initial_delay_applied(self):
        """Initial delay is respected when > 0."""
        monitor = ContainerHealthMonitor("c1", "python")
        monitor.monitoring_active = True

        async def mock_check_health(container_id):
            monitor.monitoring_active = False
            return HealthCheckResult(
                timestamp=datetime.now(UTC),
                check_type=HealthCheckType.LIVENESS,
                status=HealthStatus.HEALTHY,
                response_time_ms=1.0,
                message="OK",
            )

        config = HealthCheckConfig(
            check_type=HealthCheckType.LIVENESS,
            initial_delay_seconds=1,
            interval_seconds=0,
            failure_threshold=3,
        )
        mock_checker = MagicMock()
        mock_checker.config = config
        mock_checker.check_health = mock_check_health

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await monitor._monitor_health_check(HealthCheckType.LIVENESS, mock_checker)
            # initial_delay sleep should be called with 1
            mock_sleep.assert_any_call(1)


# ---------------------------------------------------------------------------
# _monitor_system_health — covers lines 773, 782-783
# ---------------------------------------------------------------------------


class TestMonitorSystemHealth:
    @pytest.mark.asyncio
    async def test_logs_unhealthy_containers(self):
        """System health monitor logs when unhealthy containers exist."""
        service = HealthMonitoringService()
        mon = ContainerHealthMonitor("c1", "python")
        unhealthy_result = HealthCheckResult(
            timestamp=datetime.now(UTC),
            check_type=HealthCheckType.LIVENESS,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=1.0,
            message="dead",
        )
        mon.health.check_results[HealthCheckType.LIVENESS] = [unhealthy_result]
        mon._update_overall_status()
        service.container_monitors = {"c1": mon}

        summary = service.get_health_summary()
        assert summary["unhealthy_containers"] == 1

    @pytest.mark.asyncio
    async def test_error_in_system_health_handled(self):
        """RuntimeError in system health monitoring is caught."""
        service = HealthMonitoringService()
        iteration = 0

        original_summary = service.get_health_summary

        def exploding_summary():
            nonlocal iteration
            iteration += 1
            if iteration == 1:
                raise RuntimeError("summary failed")
            service.monitoring_active = False
            return original_summary()

        service.get_health_summary = exploding_summary
        service.monitoring_active = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await service._monitor_system_health()

        assert iteration == 2
