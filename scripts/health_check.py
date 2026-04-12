"""
Health check for all configured LLM providers.

For each provider in llm_config.yml:
  1. Check if its API key env var is set
  2. Make a minimal request (list models / single token completion)
  3. Report online/offline + latency

Usage:
    python health_check.py [--config llm_config.yml] [--timeout 10]
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(2)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(2)


def find_config(start: Path) -> Path:
    for c in [start / "llm_config.yml", start / "llm_config.yml.example"]:
        if c.exists():
            return c
    raise FileNotFoundError("llm_config.yml not found")


def resolve_key(value):
    """Resolve ${VAR} → env var, or return as-is."""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def check_openai_compat(name: str, base_url: str, api_key: str, timeout: int) -> dict:
    """Check OpenAI-compatible /models endpoint."""
    url = base_url.rstrip("/") + "/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
    except requests.exceptions.ConnectionError as e:
        return {"status": "DOWN", "error": "connection refused", "ms": 0}
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "error": f"> {timeout}s", "ms": timeout * 1000}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:80], "ms": 0}

    if resp.status_code == 200:
        try:
            data = resp.json()
            count = len(data.get("data", []))
            return {"status": "OK", "models": count, "ms": int(latency_ms)}
        except Exception:
            return {"status": "OK", "models": "?", "ms": int(latency_ms)}

    if resp.status_code == 401:
        return {"status": "AUTH", "error": "401 unauthorized", "ms": int(latency_ms)}
    if resp.status_code == 403:
        return {"status": "AUTH", "error": "403 forbidden", "ms": int(latency_ms)}

    return {"status": "HTTP", "error": f"HTTP {resp.status_code}", "ms": int(latency_ms)}


def check_anthropic(name: str, api_key: str, timeout: int) -> dict:
    """Anthropic doesn't have /models — use a 1-token messages call."""
    if not api_key:
        return {"status": "NO_KEY", "error": "ANTHROPIC_API_KEY not set", "ms": 0}

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }

    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
    except requests.exceptions.ConnectionError:
        return {"status": "DOWN", "error": "connection refused", "ms": 0}
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "error": f"> {timeout}s", "ms": timeout * 1000}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:80], "ms": 0}

    if resp.status_code == 200:
        return {"status": "OK", "models": "claude family", "ms": int(latency_ms)}
    if resp.status_code == 401:
        return {"status": "AUTH", "error": "401 unauthorized", "ms": int(latency_ms)}
    if resp.status_code == 404:
        return {"status": "MODEL", "error": "haiku model unavailable", "ms": int(latency_ms)}

    return {"status": "HTTP", "error": f"HTTP {resp.status_code}", "ms": int(latency_ms)}


def main():
    parser = argparse.ArgumentParser(description="Health check LLM providers")
    parser.add_argument("--config", default="", help="Path to llm_config.yml")
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout (s)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else find_config(Path.cwd())
    print(f"Health-checking providers in: {config_path}\n")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    keys = cfg.get("keys", {})
    providers = cfg.get("providers", {})

    if not providers:
        print("ERROR: no providers in config")
        sys.exit(1)

    rows = []
    for name in sorted(providers.keys()):
        prov = providers[name]
        ptype = prov.get("type", "openai")
        base_url = prov.get("base_url", "")
        key_ref = prov.get("key_ref")
        api_key = resolve_key(keys.get(key_ref, "")) if key_ref else ""

        # Skip if cloud provider but no key
        if ptype == "openai" and key_ref and not api_key:
            rows.append((name, {
                "status": "NO_KEY",
                "error": f"${key_ref.upper()}_API_KEY not set",
                "ms": 0,
            }))
            continue

        if ptype == "anthropic":
            result = check_anthropic(name, api_key, args.timeout)
        else:
            result = check_openai_compat(name, base_url, api_key, args.timeout)

        rows.append((name, result))

    # Print table
    print(f"{'PROVIDER':<14} {'STATUS':<10} {'LATENCY':<10} INFO")
    print("-" * 70)
    ok_count = 0
    for name, result in rows:
        status = result.get("status", "?")
        ms = result.get("ms", 0)
        latency = f"{ms} ms" if ms else "-"
        info = result.get("error", "") or f"models: {result.get('models', '?')}"
        marker = "[OK]" if status == "OK" else "[--]"
        if status == "OK":
            ok_count += 1
        print(f"{name:<14} {marker} {status:<5} {latency:<10} {info}")

    print(f"\n{ok_count}/{len(rows)} providers reachable")
    if ok_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
