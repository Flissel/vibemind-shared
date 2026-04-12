"""
Cost tracking for vibemind-os LLM calls.

Loads models_pricing.yml and provides:
  - get_pricing(model)        → dict with input/output cost per 1M tokens
  - estimate_cost(model, in_tokens, out_tokens) → USD float
  - log_call(role, model, in_tokens, out_tokens) → append to cost log

Pricing file resolution (same order as llm_config.yml):
  1. $VIBEMIND_CONFIG_DIR/models_pricing.yml
  2. ./models_pricing.yml
  3. Package default (next to llm_client.py)
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml


def _find_pricing() -> Optional[Path]:
    env_dir = os.environ.get("VIBEMIND_CONFIG_DIR")
    if env_dir:
        p = Path(env_dir) / "models_pricing.yml"
        if p.exists():
            return p

    cwd = Path.cwd() / "models_pricing.yml"
    if cwd.exists():
        return cwd

    local = Path(__file__).parent / "models_pricing.yml"
    if local.exists():
        return local

    return None


@lru_cache(maxsize=1)
def _load_pricing() -> dict:
    """Load and cache models_pricing.yml. Returns {} if not found."""
    path = _find_pricing()
    if not path:
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_pricing(model: str) -> dict:
    """Return pricing dict for a model, or empty dict if unknown."""
    pricing = _load_pricing()
    return dict(pricing.get(model, {}))


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Estimate USD cost for a single call.

    Args:
        model: model id (e.g. "gpt-4o", "claude-sonnet-4-5")
        input_tokens: prompt token count
        output_tokens: completion token count

    Returns:
        Cost in USD as a float. Returns 0.0 for unknown or local models.
    """
    p = get_pricing(model)
    if not p:
        return 0.0
    in_per_m = float(p.get("input", 0))
    out_per_m = float(p.get("output", 0))
    return (input_tokens / 1_000_000) * in_per_m + (output_tokens / 1_000_000) * out_per_m


def is_local(model: str) -> bool:
    """Return True if the model is local (no cost)."""
    p = get_pricing(model)
    return p.get("provider") == "ollama" or (
        p.get("input", 0) == 0 and p.get("output", 0) == 0 and not p.get("embedding")
    )


def log_call(role: str, model: str, input_tokens: int, output_tokens: int = 0,
             log_path: Optional[str] = None) -> dict:
    """
    Append a call to the cost log (JSONL format).

    Returns the log entry dict (also written to disk).
    """
    import json
    import time

    entry = {
        "ts": time.time(),
        "role": role,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": estimate_cost(model, input_tokens, output_tokens),
    }

    log_file = log_path or os.environ.get(
        "VIBEMIND_COST_LOG",
        str(Path.cwd() / ".vibemind_cost.jsonl"),
    )
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging failures should never break the call

    return entry


def summarize_costs(log_path: Optional[str] = None) -> dict:
    """
    Read the cost log and return a summary by role and model.

    Returns:
        {
            "total_usd": 12.34,
            "total_calls": 100,
            "by_role": {"coding_planner": 5.20, ...},
            "by_model": {"claude-sonnet-4-5": 5.20, ...},
        }
    """
    import json
    from collections import defaultdict

    log_file = log_path or os.environ.get(
        "VIBEMIND_COST_LOG",
        str(Path.cwd() / ".vibemind_cost.jsonl"),
    )
    if not Path(log_file).exists():
        return {"total_usd": 0.0, "total_calls": 0, "by_role": {}, "by_model": {}}

    by_role: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    total = 0.0
    count = 0

    with open(log_file, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cost = float(entry.get("cost_usd", 0))
            by_role[entry.get("role", "?")] += cost
            by_model[entry.get("model", "?")] += cost
            total += cost
            count += 1

    return {
        "total_usd": round(total, 4),
        "total_calls": count,
        "by_role": {k: round(v, 4) for k, v in sorted(by_role.items(), key=lambda x: -x[1])},
        "by_model": {k: round(v, 4) for k, v in sorted(by_model.items(), key=lambda x: -x[1])},
    }
