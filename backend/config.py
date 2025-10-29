import os
from enum import Enum

try:  # Optional: load .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - fallback, .env loading optional
    pass


class AIProviderType(str, Enum):
    """Supported AI provider backends."""

    MOCK = "mock"
    LEGACY = "legacy"
    REMOTE = "remote"
    GEMINI = "gemini"
    VERTEX = "vertex"


def get_ai_provider_type() -> AIProviderType:
    """Return the configured AI provider based on environment variables."""
    raw_value = os.getenv("COGNIVERSE_AI_PROVIDER", AIProviderType.MOCK.value).lower()
    try:
        return AIProviderType(raw_value)
    except ValueError:
        return AIProviderType.MOCK


def auto_advance_enabled() -> bool:
    """Return True if automatic turn advancement is enabled."""
    return os.getenv("COGNIVERSE_AUTO_ADVANCE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_auto_advance_interval() -> float:
    """Return the configured auto-advance interval in seconds (default 4s)."""
    try:
        return float(os.getenv("COGNIVERSE_AUTO_ADVANCE_INTERVAL", "4.0"))
    except ValueError:
        return 4.0
