"""Test framework logging manager stub."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LoggingManager:
    """Stub logging manager for test framework.

    Provides noop implementations - actual logging goes through Python's
    standard logging module.
    """

    _instance: "LoggingManager | None" = None

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "LoggingManager":
        """Get or create the LoggingManager singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, **kwargs: Any) -> None:
        """Configure logging (noop)."""
        pass

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name."""
        return logging.getLogger(name)

    def set_level(self, level: str) -> None:
        """Set logging level (noop - use standard logging config)."""
        pass

    def log_test_start(self, test_id: str, **kwargs: Any) -> None:
        """Log test start event."""
        logger.info(f"Test started: {test_id}")

    def log_test_end(self, test_id: str, success: bool = True, **kwargs: Any) -> None:
        """Log test end event."""
        logger.info(f"Test ended: {test_id} (success={success})")

    def log_test_error(self, test_id: str, error: str, **kwargs: Any) -> None:
        """Log test error event."""
        logger.error(f"Test error: {test_id}: {error}")

    def flush(self) -> None:
        """Flush log handlers (noop)."""
        pass

    def shutdown(self) -> None:
        """Shutdown logging manager (noop)."""
        pass
