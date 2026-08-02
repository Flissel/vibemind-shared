"""
Smoke tests for vibemind_shared LLM factory.

These tests don't make real API calls — they verify the factory:
  - Loads llm_config.yml correctly
  - Resolves roles → provider config
  - Returns the right client class
  - Falls back to default for unknown roles
  - Handles overrides
  - Returns embedding configs

Run:
    cd vibemind-os && python -m pytest shared/tests/test_factory.py -v
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make vibemind_shared importable from the source tree
SHARED_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SHARED_SRC))


# A minimal config used by all tests
TEST_CONFIG = """
keys:
  openai: ${TEST_OPENAI_KEY}
  anthropic: ${TEST_ANTHROPIC_KEY}
  openfang: ${OPENFANG_API_KEY}
  ollama: null

providers:
  openai:
    type: openai
    base_url: https://api.openai.com/v1
    key_ref: openai
  anthropic:
    type: anthropic
    base_url: https://api.anthropic.com
    key_ref: anthropic
  openfang:
    type: openai
    base_url: ${OPENFANG_URL}/v1
    key_ref: openfang
    fail_closed: true
    max_retries: 3
    timeout_seconds: 8
  ollama:
    type: openai
    base_url: http://127.0.0.1:11434/v1
    key_ref: null
  ollama_compat:
    type: ollama
    base_url: http://127.0.0.1:11434/v1
    key_ref: null

default:
  provider: ollama
  model: qwen2.5:7b
  temperature: 0

roles:
  coding_planner:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.7
  fast_local:
    provider: ollama
    model: qwen2.5:3b
    temperature: 0
  voice_realtime:
    provider: openai
    model: gpt-4o-realtime-preview
    temperature: 0.8
  brain_planning:
    provider: openfang
    model: openfang:brain-orchestrator
    temperature: 0.3

overrides:
  the_brain:
    coding_planner:
      provider: openai
      model: gpt-4o

embeddings:
  default:
    driver: sentence_transformers
    model: all-MiniLM-L6-v2
    dim: 384
  openai_large:
    driver: openai
    provider: openai
    model: text-embedding-3-large
    dim: 3072
  openfang_large:
    driver: openai
    provider: openfang
    model: text-embedding-3-large
    dim: 3072
  ollama_small:
    driver: ollama
    provider: ollama_compat
    model: nomic-embed-text
    dim: 768
"""


@pytest.fixture(autouse=True)
def setup_test_config(tmp_path, monkeypatch):
    """Write a temp config file and point VIBEMIND_CONFIG_DIR at it."""
    config_path = tmp_path / "llm_config.yml"
    config_path.write_text(TEST_CONFIG, encoding="utf-8")
    monkeypatch.setenv("VIBEMIND_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-openai")
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENFANG_API_KEY", "openfang-test-token")
    monkeypatch.setenv("OPENFANG_URL", "http://127.0.0.1:4200")

    # Clear the lru_cache so each test gets a fresh load
    from vibemind_shared import llm_client
    llm_client._load_config.cache_clear()
    yield
    llm_client._load_config.cache_clear()


def test_imports():
    """All public symbols are importable."""
    from vibemind_shared import (
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
    assert callable(get_client)
    assert callable(get_embedding_model)


def test_load_config():
    from vibemind_shared import get_config
    cfg = get_config()
    assert "providers" in cfg
    assert "openai" in cfg["providers"]
    assert "anthropic" in cfg["providers"]
    assert "ollama" in cfg["providers"]


def test_default_role():
    from vibemind_shared import get_model, get_temperature, get_provider_info
    assert get_model("default") == "qwen2.5:7b"
    assert get_temperature("default") == 0
    info = get_provider_info("default")
    assert info["provider"] == "ollama"
    assert info["type"] == "openai"


def test_known_role():
    from vibemind_shared import get_model, get_provider_info
    assert get_model("coding_planner") == "claude-sonnet-4-5"
    info = get_provider_info("coding_planner")
    assert info["provider"] == "anthropic"
    assert info["type"] == "anthropic"


def test_unknown_role_falls_back_to_default():
    from vibemind_shared import get_model, get_provider_info
    # Unknown role should use default provider+model
    assert get_model("nonexistent_role_xyz") == "qwen2.5:7b"
    info = get_provider_info("nonexistent_role_xyz")
    assert info["provider"] == "ollama"


def test_directory_override():
    from vibemind_shared import get_model, get_provider_info
    # the_brain dir overrides coding_planner to openai/gpt-4o
    assert get_model("coding_planner", directory="the_brain") == "gpt-4o"
    info = get_provider_info("coding_planner", directory="the_brain")
    assert info["provider"] == "openai"


def test_override_does_not_affect_other_dirs():
    from vibemind_shared import get_model
    # Without directory, original config applies
    assert get_model("coding_planner") == "claude-sonnet-4-5"
    # Different directory, no override → original
    assert get_model("coding_planner", directory="other_dir") == "claude-sonnet-4-5"


def test_get_client_anthropic():
    """Anthropic role returns AsyncAnthropic instance."""
    pytest.importorskip("anthropic")
    from vibemind_shared import get_client
    client = get_client("coding_planner")
    # Should be an anthropic.AsyncAnthropic
    cls_name = client.__class__.__name__
    assert "Anthropic" in cls_name


def test_get_client_openai():
    """OpenAI/Ollama role returns AsyncOpenAI."""
    pytest.importorskip("openai")
    from vibemind_shared import get_client
    client = get_client("fast_local")  # ollama
    cls_name = client.__class__.__name__
    assert "OpenAI" in cls_name


def test_get_client_sync():
    pytest.importorskip("openai")
    from vibemind_shared import get_client_sync
    client = get_client_sync("voice_realtime")
    cls_name = client.__class__.__name__
    assert "OpenAI" in cls_name


def test_openfang_client_is_fail_closed_with_bounded_sdk_retries():
    pytest.importorskip("openai")
    from vibemind_shared import get_client, get_provider_info

    client = get_client("brain_planning")
    info = get_provider_info("brain_planning")

    assert client.__class__.__name__ == "AsyncOpenFangClient"
    assert client.max_retries == 3
    assert str(client.base_url) == "http://127.0.0.1:4200/v1/"
    assert info["fail_closed"] is True
    assert info["max_retries"] == 3
    assert info["timeout_seconds"] == 8.0


def test_openfang_base_url_resolves_environment_reference():
    """Provider endpoints expand env references before constructing clients."""
    from vibemind_shared import get_provider_info

    info = get_provider_info("brain_planning")

    assert info["base_url"] == "http://127.0.0.1:4200/v1"


def test_openfang_base_url_missing_environment_reference_fails_closed(monkeypatch):
    """An unresolved OpenFang endpoint must not fall back to a cloud URL."""
    from vibemind_shared import get_client

    monkeypatch.delenv("OPENFANG_URL")

    with pytest.raises(ValueError, match="OPENFANG_URL"):
        get_client("brain_planning")


def test_openfang_transient_failure_raises_explicit_hard_error(monkeypatch):
    asyncio = pytest.importorskip("asyncio")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from vibemind_shared import OpenFangUnavailable, get_client

    async def fail_after_sdk_retries(self, *args, **kwargs):
        raise openai.APIConnectionError(
            message="connection refused",
            request=httpx.Request("POST", "http://127.0.0.1:4200/v1/chat/completions"),
        )

    monkeypatch.setattr(openai.AsyncOpenAI, "request", fail_after_sdk_retries)
    client = get_client("brain_planning")

    with pytest.raises(
        OpenFangUnavailable,
        match="OpenFang unreachable — LLM calls suspended",
    ):
        asyncio.run(client.request(object, object()))


def test_openfang_sync_transient_failure_raises_explicit_hard_error(monkeypatch):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from vibemind_shared import OpenFangUnavailable, get_client_sync

    def fail_after_sdk_retries(self, *args, **kwargs):
        raise openai.APIConnectionError(
            message="connection refused",
            request=httpx.Request("POST", "http://127.0.0.1:4200/v1/chat/completions"),
        )

    monkeypatch.setattr(openai.OpenAI, "request", fail_after_sdk_retries)
    client = get_client_sync("brain_planning")

    with pytest.raises(
        OpenFangUnavailable,
        match="OpenFang unreachable — LLM calls suspended",
    ):
        client.request(object, object())


def test_declared_direct_provider_is_not_wrapped_as_openfang():
    pytest.importorskip("openai")
    from vibemind_shared import get_client_sync

    client = get_client_sync("voice_realtime")

    assert client.__class__.__name__ == "OpenAI"


def test_embedding_default():
    from vibemind_shared import get_embedding_config, get_embedding_dim
    cfg = get_embedding_config("default")
    assert cfg["model"] == "all-MiniLM-L6-v2"
    assert cfg["dim"] == 384
    assert get_embedding_dim("default") == 384


def test_embedding_openai_large():
    from vibemind_shared import get_embedding_config
    cfg = get_embedding_config("openai_large")
    assert cfg["driver"] == "openai"
    assert cfg["provider"] == "openai"
    assert cfg["dim"] == 3072


def test_embedding_openfang_uses_fail_closed_provider_client(monkeypatch):
    """OpenFang embeddings reuse the configured sync-client transport path."""
    pytest.importorskip("openai")
    from vibemind_shared import get_embedding_model
    from vibemind_shared import llm_client

    def unexpected_default_client(*args, **kwargs):
        raise AssertionError("embedding creation must not construct a default client")

    monkeypatch.setattr(llm_client, "get_client_sync", unexpected_default_client)

    model = get_embedding_model("openfang_large")

    assert model._c.__class__.__name__ == "OpenFangClient"
    assert model._c.max_retries == 3
    assert str(model._c.base_url) == "http://127.0.0.1:4200/v1/"


def test_embedding_openfang_transient_failure_raises_hard_error(monkeypatch):
    """Embedding requests exhaust OpenFang retries before reporting failure."""
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from vibemind_shared import OpenFangUnavailable, get_embedding_model

    def fail_after_sdk_retries(self, *args, **kwargs):
        raise openai.APIConnectionError(
            message="connection refused",
            request=httpx.Request("POST", "http://127.0.0.1:4200/v1/embeddings"),
        )

    monkeypatch.setattr(openai.OpenAI, "request", fail_after_sdk_retries)
    model = get_embedding_model("openfang_large")

    with pytest.raises(
        OpenFangUnavailable,
        match="OpenFang unreachable — LLM calls suspended",
    ):
        model._c.request(object, object())


def test_embedding_ollama_provider_type_remains_openai_compatible():
    """Ollama embedding roles retain the existing OpenAI SDK transport."""
    pytest.importorskip("openai")
    from vibemind_shared import get_embedding_model

    model = get_embedding_model("ollama_small")

    assert model._c.__class__.__name__ == "OpenAI"
    assert str(model._c.base_url) == "http://127.0.0.1:11434/v1/"


def test_embedding_unknown_role_falls_back():
    from vibemind_shared import get_embedding_config
    # Unknown role → default
    cfg = get_embedding_config("nonexistent_emb")
    assert cfg["model"] == "all-MiniLM-L6-v2"


def test_env_var_resolution():
    """Keys are resolved from env vars at lookup time."""
    from vibemind_shared.llm_client import _get_api_key
    assert _get_api_key("openai") == "sk-test-openai"
    assert _get_api_key("anthropic") == "sk-ant-test"
    # ollama has key_ref=null
    assert _get_api_key("ollama") == ""


def test_openfang_key_prefers_direct_environment_value_over_secret_file(monkeypatch, tmp_path):
    """A direct key remains compatible and wins over the file convention."""
    from vibemind_shared.llm_client import _get_api_key

    secret_file = tmp_path / "openfang-secret"
    secret_file.write_text("file-secret", encoding="utf-8")
    monkeypatch.setenv("OPENFANG_API_KEY", "direct-secret")
    monkeypatch.setenv("OPENFANG_API_KEY_FILE", str(secret_file))

    assert _get_api_key("openfang") == "direct-secret"


def test_placeholder_resolves_generic_named_secret_file_and_trims_whitespace(monkeypatch, tmp_path):
    """Any ${NAME} placeholder can opt into NAME_FILE without a path heuristic."""
    from vibemind_shared.llm_client import _resolve_env

    secret_file = tmp_path / "generic-secret"
    secret_file.write_text("  file-secret\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GENERIC_KEY", raising=False)
    monkeypatch.setenv("TEST_GENERIC_KEY_FILE", str(secret_file))

    assert _resolve_env("${TEST_GENERIC_KEY}") == "file-secret"


@pytest.mark.parametrize("invalid_file", ["missing", "directory", "empty", "unreadable"])
def test_placeholder_secret_file_errors_are_redacted(monkeypatch, tmp_path, invalid_file):
    """Invalid explicit secret files fail closed without exposing their location or value."""
    from vibemind_shared.llm_client import _resolve_env

    secret_file = tmp_path / "top-secret-value"
    if invalid_file == "directory":
        secret_file.mkdir()
    elif invalid_file == "empty":
        secret_file.write_text("\n\t", encoding="utf-8")
    elif invalid_file == "unreadable":
        secret_file.write_text("never-log-this", encoding="utf-8")

        def fail_read(self, *args, **kwargs):
            raise PermissionError("never-log-this")

        monkeypatch.setattr(Path, "read_text", fail_read)
    elif invalid_file == "missing":
        secret_file = tmp_path / "missing-secret"
    else:  # pragma: no cover - defensive guard for parametrization edits
        raise AssertionError(f"unexpected invalid file case: {invalid_file}")

    monkeypatch.delenv("OPENFANG_API_KEY", raising=False)
    monkeypatch.setenv("OPENFANG_API_KEY_FILE", str(secret_file))

    with pytest.raises(ValueError) as exc_info:
        _resolve_env("${OPENFANG_API_KEY}")

    message = str(exc_info.value)
    assert "OPENFANG_API_KEY" in message
    assert str(secret_file) not in message
    assert "never-log-this" not in message


def test_openfang_key_missing_direct_and_file_values_fails_closed(monkeypatch):
    """OpenFang must not reach the SDK's not-needed key fallback."""
    from vibemind_shared.llm_client import _get_api_key

    monkeypatch.delenv("OPENFANG_API_KEY", raising=False)
    monkeypatch.delenv("OPENFANG_API_KEY_FILE", raising=False)

    with pytest.raises(ValueError, match="OpenFang API key is not configured"):
        _get_api_key("openfang")


# =============================================================================
# Pricing / cost tracking tests
# =============================================================================

PRICING_FIXTURE = """
gpt-4o:
  provider: openai
  input: 2.50
  output: 10.00
gpt-4o-mini:
  provider: openai
  input: 0.15
  output: 0.60
claude-sonnet-4-5:
  provider: anthropic
  input: 3.00
  output: 15.00
"qwen2.5:7b":
  provider: ollama
  input: 0
  output: 0
"""


@pytest.fixture
def pricing_fixture(tmp_path, monkeypatch):
    """Write a temp pricing file alongside the test config."""
    pricing_path = tmp_path / "models_pricing.yml"
    pricing_path.write_text(PRICING_FIXTURE, encoding="utf-8")
    # tmp_path already pointed to by VIBEMIND_CONFIG_DIR via setup_test_config
    from vibemind_shared import pricing
    pricing._load_pricing.cache_clear()
    yield
    pricing._load_pricing.cache_clear()


def test_pricing_load(pricing_fixture):
    from vibemind_shared import get_pricing
    p = get_pricing("gpt-4o")
    assert p["input"] == 2.50
    assert p["output"] == 10.00
    assert p["provider"] == "openai"


def test_pricing_unknown_model(pricing_fixture):
    from vibemind_shared import get_pricing
    p = get_pricing("nonexistent-model-xyz")
    assert p == {}


def test_estimate_cost_basic(pricing_fixture):
    from vibemind_shared import estimate_cost
    # 1M input + 1M output of gpt-4o
    cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.50  # 2.50 + 10.00

    # 100k input + 50k output
    cost = estimate_cost("gpt-4o", 100_000, 50_000)
    assert abs(cost - (0.25 + 0.50)) < 1e-6


def test_estimate_cost_local_is_zero(pricing_fixture):
    from vibemind_shared import estimate_cost
    cost = estimate_cost("qwen2.5:7b", 1_000_000, 1_000_000)
    assert cost == 0.0


def test_estimate_cost_unknown_is_zero(pricing_fixture):
    from vibemind_shared import estimate_cost
    cost = estimate_cost("nonexistent-model", 1_000_000, 1_000_000)
    assert cost == 0.0


def test_is_local(pricing_fixture):
    from vibemind_shared import is_local
    assert is_local("qwen2.5:7b") is True
    assert is_local("gpt-4o") is False
    assert is_local("claude-sonnet-4-5") is False


def test_log_call_and_summarize(pricing_fixture, tmp_path):
    from vibemind_shared import log_call, summarize_costs

    log_path = str(tmp_path / "test_costs.jsonl")

    # Log 3 calls
    log_call("coding_planner", "claude-sonnet-4-5", 1000, 500, log_path=log_path)
    log_call("coding_planner", "claude-sonnet-4-5", 2000, 1000, log_path=log_path)
    log_call("voice_realtime", "gpt-4o", 500, 100, log_path=log_path)

    summary = summarize_costs(log_path=log_path)
    assert summary["total_calls"] == 3
    # claude-sonnet-4-5: (3000/1M * 3.00) + (1500/1M * 15.00) = 0.009 + 0.0225 = 0.0315
    # gpt-4o: (500/1M * 2.50) + (100/1M * 10.00) = 0.00125 + 0.001 = 0.00225
    expected_total = 0.0315 + 0.00225
    assert abs(summary["total_usd"] - expected_total) < 1e-4

    assert "coding_planner" in summary["by_role"]
    assert "voice_realtime" in summary["by_role"]
    assert summary["by_role"]["coding_planner"] > summary["by_role"]["voice_realtime"]


def test_log_call_local_is_free(pricing_fixture, tmp_path):
    from vibemind_shared import log_call

    log_path = str(tmp_path / "local_costs.jsonl")
    entry = log_call("local_default", "qwen2.5:7b", 100_000, 50_000, log_path=log_path)
    assert entry["cost_usd"] == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
