"""
Validate llm_config.yml — catches schema errors before they cause runtime failures.

Checks:
  1. YAML parses
  2. Required top-level keys exist (keys, providers, default, roles)
  3. Every role's `provider` exists in providers section
  4. Every provider's `key_ref` exists in keys section (or is null for local)
  5. Every override's `provider` (if specified) exists
  6. Every embedding role has a valid driver and model
  7. ${VAR} references in keys point to actual env vars (warn if missing)

Usage:
    python validate_config.py [--config llm_config.yml] [--strict]
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(2)

# Allowed driver types
ALLOWED_PROVIDER_TYPES = {"openai", "anthropic"}
ALLOWED_EMBEDDING_DRIVERS = {"sentence_transformers", "openai", "ollama"}

# 5 supported providers in our slim setup
SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "google", "ollama"}


def find_config(start: Path) -> Path:
    """Find llm_config.yml — same logic as vibemind_shared."""
    candidates = [
        start / "llm_config.yml",
        start / "llm_config.yml.example",  # for validation of the template
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No llm_config.yml in {start}")


def env_var_refs(value) -> list[str]:
    """Find all ${VAR} references in a string value."""
    if not isinstance(value, str):
        return []
    return re.findall(r"\$\{(\w+)\}", value)


def validate(cfg: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required top-level keys
    for key in ("keys", "providers", "default", "roles"):
        if key not in cfg:
            errors.append(f"missing top-level key: '{key}'")

    keys_section = cfg.get("keys", {})
    providers = cfg.get("providers", {})
    roles = cfg.get("roles", {})
    overrides = cfg.get("overrides", {}) or {}
    embeddings = cfg.get("embeddings", {}) or {}

    # 2. Provider definitions
    for prov_name, prov_cfg in providers.items():
        if not isinstance(prov_cfg, dict):
            errors.append(f"provider '{prov_name}': must be a dict")
            continue

        ptype = prov_cfg.get("type")
        if ptype not in ALLOWED_PROVIDER_TYPES:
            errors.append(
                f"provider '{prov_name}': type='{ptype}' not in {ALLOWED_PROVIDER_TYPES}"
            )

        if "base_url" not in prov_cfg:
            errors.append(f"provider '{prov_name}': missing 'base_url'")

        key_ref = prov_cfg.get("key_ref")
        if key_ref is not None:
            if key_ref not in keys_section:
                errors.append(
                    f"provider '{prov_name}': key_ref='{key_ref}' not in keys section"
                )

        # Warn if not in supported set
        if prov_name not in SUPPORTED_PROVIDERS:
            warnings.append(
                f"provider '{prov_name}': not in supported set {SUPPORTED_PROVIDERS} — "
                "add via overrides only"
            )

    # 3. ${VAR} references in keys
    for key_name, key_value in keys_section.items():
        for var in env_var_refs(key_value):
            if not os.environ.get(var):
                warnings.append(
                    f"keys.{key_name}: ${{{var}}} is not set in environment"
                )

    # 4. Default role
    default = cfg.get("default", {})
    if isinstance(default, dict):
        prov = default.get("provider")
        if prov and prov not in providers:
            errors.append(f"default.provider='{prov}' not in providers section")
        if "model" not in default:
            errors.append("default: missing 'model'")

    # 5. Roles
    if not isinstance(roles, dict):
        errors.append("'roles' must be a dict")
    else:
        for role_name, role_cfg in roles.items():
            if not isinstance(role_cfg, dict):
                errors.append(f"role '{role_name}': must be a dict")
                continue
            prov = role_cfg.get("provider")
            if prov and prov not in providers:
                errors.append(
                    f"role '{role_name}': provider='{prov}' not in providers section"
                )
            if "model" not in role_cfg:
                errors.append(f"role '{role_name}': missing 'model'")

    # 6. Overrides
    for dir_name, dir_overrides in overrides.items():
        if not isinstance(dir_overrides, dict):
            errors.append(f"overrides.{dir_name}: must be a dict")
            continue
        for role_name, ov_cfg in dir_overrides.items():
            if not isinstance(ov_cfg, dict):
                errors.append(
                    f"overrides.{dir_name}.{role_name}: must be a dict"
                )
                continue
            prov = ov_cfg.get("provider")
            if prov and prov not in providers:
                errors.append(
                    f"overrides.{dir_name}.{role_name}: "
                    f"provider='{prov}' not in providers section"
                )

    # 7. Embeddings
    for emb_name, emb_cfg in embeddings.items():
        if not isinstance(emb_cfg, dict):
            errors.append(f"embeddings.{emb_name}: must be a dict")
            continue
        driver = emb_cfg.get("driver")
        if driver not in ALLOWED_EMBEDDING_DRIVERS:
            errors.append(
                f"embeddings.{emb_name}: driver='{driver}' not in {ALLOWED_EMBEDDING_DRIVERS}"
            )
        if "model" not in emb_cfg:
            errors.append(f"embeddings.{emb_name}: missing 'model'")
        # If driver is openai/ollama, provider must reference a valid one
        if driver in ("openai", "ollama"):
            prov = emb_cfg.get("provider")
            if not prov:
                errors.append(
                    f"embeddings.{emb_name}: driver='{driver}' requires 'provider'"
                )
            elif prov not in providers:
                errors.append(
                    f"embeddings.{emb_name}: provider='{prov}' not in providers section"
                )

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate llm_config.yml")
    parser.add_argument("--config", default="", help="Path to config file")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = find_config(Path.cwd())

    print(f"Validating: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"\nYAML PARSE ERROR:\n  {e}")
        sys.exit(1)

    if cfg is None:
        print("\nERROR: config file is empty")
        sys.exit(1)

    errors, warnings = validate(cfg)

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  X{e}")
    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  !{w}")

    if not errors and not warnings:
        print("\nOK: config is valid")
        sys.exit(0)

    if errors:
        sys.exit(1)
    if args.strict and warnings:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
