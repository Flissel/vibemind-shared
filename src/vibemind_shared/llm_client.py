"""
LLM Client Factory — Multi-Provider Support
================================================
Shared across the VibeMind OS ecosystem.

Config resolution order:
  1. $VIBEMIND_CONFIG_DIR/llm_config.yml
  2. ./llm_config.yml  (current working directory)
  3. Package default (minimal fallback)

Usage:
    from vibemind_shared import get_client, get_model, get_temperature

    client = get_client("red_team")
    model = get_model("red_team")
    temp = get_temperature("red_team")

    # Per-directory override:
    model = get_model("default", "poc_security_scanner")

    # Sync client:
    client = get_client_sync("issue_agent")
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Load .env from CWD (each repo keeps its own .env)
load_dotenv()


class OpenFangUnavailable(RuntimeError):
    """OpenFang exhausted its bounded retry window for an LLM request."""


_OPENFANG_TRANSIENT_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class AsyncOpenFangClient(AsyncOpenAI):
    """Async OpenAI-compatible client that fails closed after SDK retries."""

    async def request(self, *args, **kwargs):
        try:
            return await super().request(*args, **kwargs)
        except _OPENFANG_TRANSIENT_ERRORS as exc:
            raise OpenFangUnavailable(
                "OpenFang unreachable — LLM calls suspended"
            ) from exc


class OpenFangClient(OpenAI):
    """Sync OpenAI-compatible client that fails closed after SDK retries."""

    def request(self, *args, **kwargs):
        try:
            return super().request(*args, **kwargs)
        except _OPENFANG_TRANSIENT_ERRORS as exc:
            raise OpenFangUnavailable(
                "OpenFang unreachable — LLM calls suspended"
            ) from exc


def _find_config() -> Path:
    """Find llm_config.yml using resolution order."""
    # 1. Explicit env var
    env_dir = os.environ.get("VIBEMIND_CONFIG_DIR")
    if env_dir:
        p = Path(env_dir) / "llm_config.yml"
        if p.exists():
            return p

    # 2. Current working directory
    cwd = Path.cwd() / "llm_config.yml"
    if cwd.exists():
        return cwd

    # 3. Next to this file (for legacy installs where llm_config.yml sits beside llm_client.py)
    local = Path(__file__).parent / "llm_config.yml"
    if local.exists():
        return local

    raise FileNotFoundError(
        "llm_config.yml not found. Searched:\n"
        f"  1. $VIBEMIND_CONFIG_DIR={env_dir or '(not set)'}\n"
        f"  2. CWD: {cwd}\n"
        f"  3. Package dir: {local}\n"
        "Create llm_config.yml or set VIBEMIND_CONFIG_DIR."
    )


def _resolve_env(value):
    """Resolve ${ENV_VAR} references in config values."""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match):
        return os.environ.get(match.group(1), "")
    return pattern.sub(replacer, value)


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load and cache llm_config.yml."""
    config_path = _find_config()
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    """Get the full LLM configuration."""
    return _load_config()


def _resolve_role(role: str, directory: str = "") -> dict:
    """Resolve the config for a role, respecting overrides."""
    cfg = _load_config()
    result = dict(cfg.get("default", {}))

    if role != "default" and role in cfg.get("roles", {}):
        result.update(cfg["roles"][role])

    if directory:
        dir_name = directory.replace("\\", "/").rstrip("/").split("/")[-1]
        overrides = cfg.get("overrides", {}).get(dir_name, {})
        if role in overrides:
            result.update(overrides[role])
        elif "default" in overrides and role == "default":
            result.update(overrides["default"])
        elif "default" in overrides and role not in cfg.get("roles", {}):
            result.update(overrides["default"])

    return result


def _get_api_key(provider_name: str) -> str:
    cfg = _load_config()
    provider = cfg.get("providers", {}).get(provider_name, {})
    key_ref = provider.get("key_ref")
    if not key_ref:
        return ""
    keys = cfg.get("keys", {})
    raw = keys.get(key_ref, "")
    return _resolve_env(raw) if raw else ""


def _get_base_url(provider_name: str) -> str:
    cfg = _load_config()
    provider = cfg.get("providers", {}).get(provider_name, {})
    raw_base_url = provider.get("base_url")
    if not isinstance(raw_base_url, str) or not raw_base_url.strip():
        raise ValueError(f"provider {provider_name!r} requires a non-empty base_url")

    missing_env = [
        name
        for name in re.findall(r"\$\{(\w+)\}", raw_base_url)
        if not os.environ.get(name)
    ]
    if missing_env:
        raise ValueError(
            f"provider {provider_name!r} base_url requires environment variable(s): "
            + ", ".join(missing_env)
        )

    base_url = _resolve_env(raw_base_url).strip()
    parsed = urlparse(base_url)
    if (
        any(character.isspace() for character in base_url)
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
    ):
        raise ValueError(
            f"provider {provider_name!r} base_url must be an absolute HTTP(S) URL"
        )
    return base_url


def _get_provider_type(provider_name: str) -> str:
    cfg = _load_config()
    provider = cfg.get("providers", {}).get(provider_name, {})
    return provider.get("type", "openai")


def _get_provider_runtime(provider_name: str) -> dict:
    """Return validated transport options for a configured provider."""
    cfg = _load_config()
    provider = cfg.get("providers", {}).get(provider_name, {})
    fail_closed = provider.get("fail_closed") is True

    runtime = {"fail_closed": fail_closed}
    if "max_retries" in provider:
        max_retries = int(provider["max_retries"])
        if not 0 <= max_retries <= 10:
            raise ValueError("provider max_retries must be between 0 and 10")
        runtime["max_retries"] = max_retries
    if "timeout_seconds" in provider:
        timeout_seconds = float(provider["timeout_seconds"])
        if timeout_seconds <= 0:
            raise ValueError("provider timeout_seconds must be positive")
        runtime["timeout_seconds"] = timeout_seconds
    return runtime


def _openai_client_kwargs(provider_name: str) -> tuple[dict, dict]:
    """Build OpenAI SDK kwargs and return them with validated runtime metadata."""
    api_key = _get_api_key(provider_name)
    runtime = _get_provider_runtime(provider_name)
    kwargs = {
        "base_url": _get_base_url(provider_name),
        "api_key": api_key if api_key else "not-needed",
    }
    if "max_retries" in runtime:
        kwargs["max_retries"] = runtime["max_retries"]
    if "timeout_seconds" in runtime:
        kwargs["timeout"] = runtime["timeout_seconds"]
    return kwargs, runtime


def get_model(role: str = "default", directory: str = "") -> str:
    resolved = _resolve_role(role, directory)
    return resolved.get("model", "gpt-4.1")


def get_temperature(role: str = "default", directory: str = "") -> float:
    resolved = _resolve_role(role, directory)
    return float(resolved.get("temperature", 0))


def get_client(role: str = "default", directory: str = ""):
    """Get an async OpenAI-compatible client for a role."""
    resolved = _resolve_role(role, directory)
    provider_name = resolved.get("provider", "openai")
    provider_type = _get_provider_type(provider_name)

    if provider_type == "anthropic":
        try:
            import anthropic
            return anthropic.AsyncAnthropic(api_key=_get_api_key(provider_name))
        except ImportError:
            raise ImportError("pip install anthropic")

    kwargs, runtime = _openai_client_kwargs(provider_name)
    if provider_name == "openfang" and runtime["fail_closed"]:
        return AsyncOpenFangClient(**kwargs)
    return AsyncOpenAI(**kwargs)


def get_client_sync(role: str = "default", directory: str = ""):
    """Get a sync OpenAI-compatible client for a role."""
    resolved = _resolve_role(role, directory)
    provider_name = resolved.get("provider", "openai")
    provider_type = _get_provider_type(provider_name)

    if provider_type == "anthropic":
        try:
            import anthropic
            return anthropic.Anthropic(api_key=_get_api_key(provider_name))
        except ImportError:
            raise ImportError("pip install anthropic")

    return _get_sync_openai_compatible_client(provider_name)


def _get_sync_openai_compatible_client(provider_name: str) -> OpenAI:
    """Create the configured sync OpenAI-compatible transport for a provider."""
    if _get_provider_type(provider_name) not in {"openai", "ollama"}:
        raise ValueError(
            f"provider {provider_name!r} is not OpenAI-compatible"
        )

    kwargs, runtime = _openai_client_kwargs(provider_name)
    if provider_name == "openfang" and runtime["fail_closed"]:
        return OpenFangClient(**kwargs)
    return OpenAI(**kwargs)


def get_provider_info(role: str = "default", directory: str = "") -> dict:
    resolved = _resolve_role(role, directory)
    provider_name = resolved.get("provider", "openai")
    runtime = _get_provider_runtime(provider_name)
    return {
        "provider": provider_name,
        "model": resolved.get("model", "gpt-4.1"),
        "temperature": resolved.get("temperature", 0),
        "base_url": _get_base_url(provider_name),
        "type": _get_provider_type(provider_name),
        "fail_closed": runtime["fail_closed"],
        "max_retries": runtime.get("max_retries"),
        "timeout_seconds": runtime.get("timeout_seconds"),
    }


# =============================================================================
# Embedding Models — resolved from the `embeddings:` section of llm_config.yml
# =============================================================================

def _resolve_embedding_role(role: str = "default") -> dict:
    """Resolve an embedding role to its config dict."""
    cfg = _load_config()
    embeddings = cfg.get("embeddings", {})
    if not embeddings:
        # Fallback to a sensible default
        return {"driver": "sentence_transformers", "model": "all-MiniLM-L6-v2", "dim": 384}
    if role in embeddings:
        return dict(embeddings[role])
    if "default" in embeddings:
        return dict(embeddings["default"])
    # Last resort: pick the first entry
    first_key = next(iter(embeddings))
    return dict(embeddings[first_key])


def get_embedding_config(role: str = "default") -> dict:
    """Return the raw embedding config for a role (driver, model, dim, ...)."""
    return _resolve_embedding_role(role)


def get_embedding_model(role: str = "default", device: str = "auto"):
    """Return a loaded embedding model for the given role.

    Returns an object with `.encode(texts)` method, regardless of driver.
    Drivers:
      sentence_transformers → SentenceTransformer instance
      openai                → small wrapper that calls /embeddings endpoint
      ollama                → small wrapper that calls /v1/embeddings endpoint
    """
    resolved = _resolve_embedding_role(role)
    driver = resolved.get("driver", "sentence_transformers")
    model_name = resolved.get("model", "all-MiniLM-L6-v2")

    if driver == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        # Resolve device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return SentenceTransformer(model_name, device=device)

    if driver in ("openai", "ollama"):
        # Return a thin wrapper that exposes .encode(texts)
        provider_name = resolved.get("provider", driver)
        emb_client = _get_sync_openai_compatible_client(provider_name)

        class _EmbeddingWrapper:
            def __init__(self, client, model):
                self._c = client
                self._m = model

            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                resp = self._c.embeddings.create(model=self._m, input=texts)
                import numpy as np
                return np.array([d.embedding for d in resp.data], dtype=np.float32)

        return _EmbeddingWrapper(emb_client, model_name)

    raise ValueError(f"Unknown embedding driver: {driver}")


def get_embedding_dim(role: str = "default") -> int:
    """Return the embedding dimension for a role (from config, no model load)."""
    resolved = _resolve_embedding_role(role)
    return int(resolved.get("dim", 384))
