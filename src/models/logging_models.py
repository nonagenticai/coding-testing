"""
Shared logging models for the application.

This module provides common logging-related models used across different
components of the application to avoid duplication.
"""

from __future__ import annotations

from enum import StrEnum


class LogLevel(StrEnum):
    """Standard logging levels with clear hierarchy."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_lowercase(cls, value: str) -> LogLevel:
        """Convert lowercase log level to enum value."""
        return cls(value.upper())

    def to_lowercase(self) -> str:
        """Convert enum value to lowercase string."""
        return self.value.lower()
