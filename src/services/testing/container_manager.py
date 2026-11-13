"""
Unified Container Manager - Manages test containers for all languages.

Integrates with:
- TestExecutionContainerRegistry (from test-orchestration)
- DockerService (existing)
- Provisioned containers (from GitHub)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from services.stubs import DockerService
from services.test_execution.test_execution_container_registry import (
    TestExecutionContainerRegistry,
)

logger = logging.getLogger(__name__)


class UnifiedContainerManager:
    """
    Unified container management for testing service.

    Handles:
    - Dynamic container creation
    - Provisioned container discovery
    - Container lifecycle management
    - Resource optimization
    """

    def __init__(self, docker_service: DockerService):
        """Initialize unified container manager."""
        self.docker_service = docker_service
        self.container_registry = TestExecutionContainerRegistry()
        self._container_cache: dict[str, Any] = {}

        logger.info("UnifiedContainerManager initialized")

    async def find_suitable_container(
        self,
        language: str,
        container_name: str | None = None,
        prefer_provisioned: bool = True,
    ) -> str | None:
        """
        Find a suitable container for test execution.

        Args:
            language: Programming language
            container_name: Specific container requested
            prefer_provisioned: Whether to prefer provisioned containers

        Returns:
            Container name if found, None otherwise
        """
        # 1. Check for specifically requested container
        if container_name:
            if await self._is_container_available(container_name):
                logger.info(f"Using requested container: {container_name}")
                return container_name
            else:
                logger.warning(f"Requested container not available: {container_name}")

        # 2. Check for language-appropriate provisioned containers
        if prefer_provisioned:
            provisioned = await self._find_provisioned_containers(language)
            if provisioned:
                logger.info(f"Found provisioned container for {language}: {provisioned[0]}")
                return provisioned[0]

        # 3. Check for existing test containers
        test_containers = await self._find_test_containers(language)
        if test_containers:
            logger.info(f"Found existing test container for {language}: {test_containers[0]}")
            return test_containers[0]

        # 4. No suitable container found
        logger.info(f"No suitable container found for {language}")
        return None

    async def ensure_container(
        self,
        language: str,
        container_name: str | None = None,
        create_if_missing: bool = True,
    ) -> str:
        """
        Ensure a container is available for testing.

        Args:
            language: Programming language
            container_name: Specific container name
            create_if_missing: Whether to create if not found

        Returns:
            Container name

        Raises:
            RuntimeError: If container cannot be ensured
        """
        # Try to find existing
        found = await self.find_suitable_container(
            language, container_name, prefer_provisioned=False
        )
        if found:
            return found

        # Create if allowed
        if not create_if_missing:
            raise RuntimeError(f"No container available for {language}")

        # Create new container
        logger.info(f"Creating new test container for {language}")
        container_spec = self.container_registry.get_container_spec(language)
        if not container_spec:
            raise RuntimeError(f"No container specification for {language}")

        # Generate container name
        if not container_name:
            from uuid import uuid7

            container_name = f"test-{language}-{uuid7().hex[:8]}"

        # Create through docker service
        await self._create_test_container(container_name, container_spec)
        return container_name

    async def list_containers(
        self, filter_type: str | None = None, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List all available containers.

        Args:
            filter_type: Filter by type (test, provisioned, all)
            tenant_id: If provided, only return containers belonging to this tenant.
                      If None, returns all containers (for admin use or legacy code).

        Returns:
            List of container information
        """
        containers = []

        # Get all containers from docker
        all_containers = await self.docker_service.list_containers(tenant_id=tenant_id)

        for container in all_containers:
            # Filter by tenant if tenant_id is provided
            if tenant_id is not None:
                container_tenant = (
                    container.get("tenant_id")
                    if isinstance(container, dict)
                    else getattr(container, "tenant_id", None)
                )
                # Only include containers matching tenant or legacy containers (no tenant)
                if container_tenant is not None and container_tenant != tenant_id:
                    continue

            # Support both dict and object-style containers
            if isinstance(container, dict):
                c_name = container.get("container_name", "")
                c_status = container.get("status", "unknown")
                c_endpoint = container.get("api_endpoint", "")
                c_internal_endpoint = container.get("internal_endpoint", c_endpoint)
                c_public_endpoint = container.get("public_endpoint", c_endpoint)
                c_created = container.get("created_at")
            else:
                c_name = getattr(container, "container_name", "")
                raw_status = getattr(container, "status", "unknown")
                c_status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
                c_endpoint = getattr(container, "api_endpoint", "")
                c_internal_endpoint = getattr(container, "internal_endpoint", c_endpoint)
                c_public_endpoint = getattr(container, "public_endpoint", c_endpoint)
                c_created = getattr(container, "created_at", None)

            container_info = {
                "name": c_name,
                "status": c_status,
                "type": self._determine_container_type(str(c_name)),
                "api_endpoint": c_endpoint,
                "internal_endpoint": c_internal_endpoint,
                "public_endpoint": c_public_endpoint,
                "created_at": c_created,
            }

            # Apply filter
            if filter_type and filter_type != "all":
                if container_info["type"] != filter_type:
                    continue

            containers.append(container_info)

        return containers

    async def get_container_info(
        self, container_name: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get detailed information about a container.

        Args:
            container_name: Container name
            tenant_id: If provided, validates tenant access before returning info.
                      Returns None if container belongs to different tenant.

        Returns:
            Container information or None
        """
        # Check cache first (only if no tenant filtering needed)
        if tenant_id is None and container_name in self._container_cache:
            cached = self._container_cache[container_name]
            # Cache for 5 minutes
            if (datetime.now(UTC) - cached["timestamp"]).seconds < 300:
                info = cached["info"]
                return info

        # Get from docker service
        container = await self.docker_service.find_container_by_name(container_name)
        if not container:
            return None

        # Validate tenant access if tenant_id is provided
        if tenant_id is not None:
            container_tenant = getattr(container, "tenant_id", None)
            if container_tenant is not None and container_tenant != tenant_id:
                return None

        # Get additional details
        info = {
            "name": container.container_name,
            "status": (
                container.status.value
                if hasattr(container.status, "value")
                else str(container.status)
            ),
            "type": self._determine_container_type(container.container_name),
            "language": self._detect_container_language(container),
            "api_endpoint": container.api_endpoint,
            "internal_endpoint": getattr(container, "internal_endpoint", container.api_endpoint),
            "public_endpoint": getattr(container, "public_endpoint", container.api_endpoint),
            "code_graph_id": container.code_graph_id,
            "repo_url": getattr(container, "repo_url", None),
            "created_at": getattr(container, "created_at", None),
        }

        # Cache result (only if no tenant filtering)
        if tenant_id is None:
            self._container_cache[container_name] = {
                "info": info,
                "timestamp": datetime.now(UTC),
            }

        return info

    async def cleanup_old_containers(self, max_age_hours: int = 24) -> int:
        """
        Clean up old test containers.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of containers cleaned up
        """
        cleaned = 0
        containers = await self.list_containers(filter_type="test")

        for container in containers:
            # Skip if no creation time
            if not container.get("created_at"):
                continue

            # Check age
            created = datetime.fromisoformat(container["created_at"])
            age_hours = (datetime.now(UTC) - created).total_seconds() / 3600

            if age_hours > max_age_hours:
                try:
                    await self.docker_service.remove_container(container["name"])
                    cleaned += 1
                    logger.info(f"Cleaned up old container: {container['name']}")
                except Exception as e:
                    logger.warning(f"Failed to clean up container {container['name']}: {e}")

        return cleaned

    # Private methods

    async def _is_container_available(self, container_name: str) -> bool:
        """Check if a container is available and running."""
        container = await self.docker_service.find_container_by_name(container_name)
        if not container:
            return False

        # Check status
        if hasattr(container.status, "value"):
            status_str = container.status.value
        else:
            status_str = container.status

        return str(status_str).lower() in ["running", "creating"]

    async def _find_provisioned_containers(self, language: str) -> list[str]:
        """Find provisioned containers for a language."""
        containers = await self.list_containers(filter_type="provisioned")
        suitable = []

        for container in containers:
            # Check if container supports the language
            if self._container_supports_language(container, language):
                suitable.append(container["name"])

        return suitable

    async def _find_test_containers(self, language: str, tenant_id: str | None = None) -> list[str]:
        """Find test containers for a language.

        Args:
            language: The programming language to find test containers for.
            tenant_id: Optional tenant ID to filter containers by tenant.
        """
        # Look for language-specific test containers
        pattern = f"{language}-test-executor"
        containers = await self.docker_service.list_containers(tenant_id=tenant_id)

        suitable = []
        for container in containers:
            c_name = (
                container.get("container_name", "")
                if isinstance(container, dict)
                else getattr(container, "container_name", "")
            )
            if pattern in str(c_name):
                suitable.append(str(c_name))

        return suitable

    async def _create_test_container(self, container_name: str, container_spec: Any) -> None:
        """Create a new test container using the docker service."""
        container_config = {
            "image": container_spec.image,
            "working_dir": getattr(container_spec, "working_dir", "/workspace"),
            "mem_limit": getattr(container_spec, "memory_limit", "512m"),
            "cpu_quota": getattr(container_spec, "cpu_quota", 50000),
            "network_disabled": True,
            "read_only": True,
            "user": getattr(container_spec, "user", "1001:1001"),
            "security_opt": getattr(container_spec, "security_opts", None)
            or ["no-new-privileges:true"],
            "tmpfs": {"/tmp": "noexec,nosuid,size=100m"},
            "cap_drop": ["ALL"],
            "labels": {
                "service": "unified_container_manager",
                "managed_by": "UnifiedContainerManager",
            },
            "environment": getattr(container_spec, "environment", {}),
            "name": container_name,
        }

        try:
            docker_id = await self.docker_service.create_test_container(container_config)
            await self.docker_service.start_container(docker_id)
            logger.info(
                "Created and started test container %s (docker_id=%s)",
                container_name,
                docker_id,
            )
        except Exception as e:
            logger.error("Failed to create test container %s: %s", container_name, e)
            raise RuntimeError(f"Container creation failed for {container_name}: {e}") from e

    def _determine_container_type(self, container_name: str) -> str:
        """Determine container type from name."""
        if "codegraph-" in container_name:
            return "provisioned"
        elif "-test-executor" in container_name or "test-" in container_name:
            return "test"
        else:
            return "unknown"

    def _detect_container_language(self, container: Any) -> str | None:
        """Detect language from container information."""
        # Check container name patterns
        name = container.container_name.lower()

        language_patterns = {
            "python": ["python", "py"],
            "javascript": ["javascript", "js", "node"],
            "typescript": ["typescript", "ts"],
            "go": ["go", "golang"],
            "rust": ["rust", "rs"],
            "java": ["java", "jvm"],
            "csharp": ["csharp", "cs", "dotnet"],
            "cpp": ["cpp", "c++", "cplusplus"],
        }

        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if pattern in name:
                    return language

        return None

    def _container_supports_language(self, container_info: dict[str, Any], language: str) -> bool:
        """Check if a container supports a specific language."""
        # Get container language
        container_lang = container_info.get("language")
        if not container_lang:
            return False

        # Direct match
        if container_lang == language:
            return True

        # Special cases
        if language == "nodejs" and container_lang == "javascript":
            return True

        return False
