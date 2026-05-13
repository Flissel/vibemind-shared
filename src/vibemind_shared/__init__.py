"""VibeMind Shared — Multi-provider LLM client factory for the VibeMind OS ecosystem."""

from .llm_client import (
    get_client,
    get_client_sync,
    get_config,
    get_model,
    get_provider_info,
    get_temperature,
    get_embedding_model,
    get_embedding_config,
    get_embedding_dim,
)
from .pricing import (
    get_pricing,
    estimate_cost,
    is_local,
    log_call,
    summarize_costs,
)
from .memory_loader import (
    load_memory,
    clear_memory_cache,
    MemoryLoadError,
)

__version__ = "0.4.0"
__all__ = [
    "get_client",
    "get_client_sync",
    "get_config",
    "get_model",
    "get_provider_info",
    "get_temperature",
    "get_embedding_model",
    "get_embedding_config",
    "get_embedding_dim",
    "get_pricing",
    "estimate_cost",
    "is_local",
    "log_call",
    "summarize_costs",
    "load_memory",
    "clear_memory_cache",
    "MemoryLoadError",
]
