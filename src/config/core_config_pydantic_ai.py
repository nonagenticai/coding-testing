"""PydanticAI configuration stub."""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def load_pydantic_ai_config_from_env() -> dict[str, Any]:
    """Load PydanticAI configuration from environment variables."""
    return {
        "provider": os.getenv("PYDANTIC_AI_PROVIDER", "openai"),
        "model": os.getenv("PYDANTIC_AI_MODEL", "gpt-4"),
        "api_key": os.getenv("PYDANTIC_AI_API_KEY", ""),
        "base_url": os.getenv("PYDANTIC_AI_BASE_URL", ""),
        "max_retries": int(os.getenv("PYDANTIC_AI_MAX_RETRIES", "3")),
        "timeout_seconds": int(os.getenv("PYDANTIC_AI_TIMEOUT", "30")),
        "temperature": float(os.getenv("PYDANTIC_AI_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("PYDANTIC_AI_MAX_TOKENS", "2000")),
        "rate_limit_rpm": int(os.getenv("PYDANTIC_AI_RATE_LIMIT_RPM", "100")),
        "enabled": os.getenv("PYDANTIC_AI_ENABLED", "true").lower() == "true",
    }


def get_pydantic_ai_config() -> dict[str, Any]:
    """Get PydanticAI configuration."""
    return load_pydantic_ai_config_from_env()
