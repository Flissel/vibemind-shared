"""VibeMind Shared — Multi-provider LLM client factory for the VibeMind OS ecosystem."""

from .llm_client import (
    get_client,
    get_client_sync,
    get_config,
    get_model,
    get_provider_info,
    get_temperature,
)

__version__ = "0.1.0"
__all__ = [
    "get_client",
    "get_client_sync",
    "get_config",
    "get_model",
    "get_provider_info",
    "get_temperature",
]
