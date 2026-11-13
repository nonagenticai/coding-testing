"""Service type definitions."""

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Categories of errors for classification."""

    SYNTAX = "syntax"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    NETWORK = "network"
    PERMISSION = "permission"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    ASSERTION = "assertion"
    UNKNOWN = "unknown"
