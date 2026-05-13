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
from .memory_rag_sync import (
    sync_memory_to_rowboat,
    RowboatSyncError,
)
from .memory_writer import (
    propose_memory_diff,
    list_pending,
    rewrite_pending,
)
from .memory_qdrant_sync import (
    sync_memory_to_qdrant,
    search_memory,
    QdrantSyncError,
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
    "sync_memory_to_rowboat",
    "RowboatSyncError",
    "propose_memory_diff",
    "list_pending",
    "rewrite_pending",
    "sync_memory_to_qdrant",
    "search_memory",
    "QdrantSyncError",
]
