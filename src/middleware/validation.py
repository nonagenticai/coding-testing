"""Request validation middleware for coding-testing (KAG standard).

Delegates to ``coding_client.api.validation``. Kept as a thin re-export so
existing ``from middleware.validation import ...`` call sites keep working.
"""

from __future__ import annotations

from coding_client.api.validation import (
    SecurityValidationException,
    ValidationMiddleware,
)

__all__ = ["SecurityValidationException", "ValidationMiddleware"]
