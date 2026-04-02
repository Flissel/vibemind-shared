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

import yaml
from dotenv import load_dotenv

# Load .env from CWD (each repo keeps its own .env)
load_dotenv()


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
    return provider.get("base_url", "https://api.openai.com/v1")


def _get_provider_type(provider_name: str) -> str:
    cfg = _load_config()
    provider = cfg.get("providers", {}).get(provider_name, {})
    return provider.get("type", "openai")


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

    from openai import AsyncOpenAI
    api_key = _get_api_key(provider_name)
    base_url = _get_base_url(provider_name)
    kwargs = {"base_url": base_url}
    kwargs["api_key"] = api_key if api_key else "not-needed"
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

    from openai import OpenAI
    api_key = _get_api_key(provider_name)
    base_url = _get_base_url(provider_name)
    kwargs = {"base_url": base_url}
    kwargs["api_key"] = api_key if api_key else "not-needed"
    return OpenAI(**kwargs)


def get_provider_info(role: str = "default", directory: str = "") -> dict:
    resolved = _resolve_role(role, directory)
    provider_name = resolved.get("provider", "openai")
    return {
        "provider": provider_name,
        "model": resolved.get("model", "gpt-4.1"),
        "temperature": resolved.get("temperature", 0),
        "base_url": _get_base_url(provider_name),
        "type": _get_provider_type(provider_name),
    }
