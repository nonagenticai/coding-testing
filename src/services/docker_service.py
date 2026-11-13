"""Docker service stub - delegates to coding-containers microservice.

This module provides the DockerService interface expected by other services.
The actual implementation lives in coding-containers; this stub provides
the same interface with no-op or HTTP-forwarding implementations.
"""

import logging

from services.stubs import DockerService, get_docker_service

logger = logging.getLogger(__name__)

__all__ = ["DockerService", "get_docker_service", "create_docker_service"]


def create_docker_service() -> DockerService:
    """Create a DockerService instance."""
    return get_docker_service()
