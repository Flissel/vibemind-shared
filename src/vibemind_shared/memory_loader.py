"""
Memory Loader — profile-filtered persistent memory for VibeMind agents.

Reads markdown memory files from the user's Claude Code memory directory,
filters them by profile (defined in memory_profiles.yml), resolves [[name]]
links, and returns a formatted system-prompt block.

Hard-fails on missing dir, unknown profile, or YAML parse errors — no silent
fallback. Agents must run with valid memory context.

Usage:
    from vibemind_shared import load_memory, clear_memory_cache, MemoryLoadError

    block = load_memory("coding")                  # full block
    block = load_memory("minimal", max_tokens=500) # truncated
    clear_memory_cache()                           # after memory edits
"""

from __future__ import annotations

import fnmatch
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

MEMORY_DIR_DEFAULT = Path(r"C:\Users\User\.claude\projects\c--Users-User-myBrain\memory")
PROFILES_PATH = Path(__file__).parent / "memory_profiles.yml"
APPROX_CHARS_PER_TOKEN = 4
TYPE_PREFIX_MAP = {
    "user_": "user",
    "feedback_": "feedback",
    "project_": "project",
    "reference_": "reference",
}


class MemoryLoadError(RuntimeError):
    """Raised when memory cannot be loaded (missing dir, unknown profile, parse error)."""


def _memory_dir() -> Path:
    return Path(os.environ.get("VIBEMIND_MEMORY_DIR", str(MEMORY_DIR_DEFAULT)))


@lru_cache(maxsize=1)
def _load_profile_config() -> dict:
    if not PROFILES_PATH.exists():
        raise MemoryLoadError(f"profiles config not found: {PROFILES_PATH}")
    try:
        cfg = yaml.safe_load(PROFILES_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise MemoryLoadError(f"profiles config parse error: {e}") from e
    if not isinstance(cfg, dict) or "profiles" not in cfg:
        raise MemoryLoadError("profiles config missing 'profiles' key")
    return cfg


def _mtime_signature(d: Path) -> int:
    """Sum of mtime_ns across all .md files. Changes when any file is touched."""
    return sum(p.stat().st_mtime_ns for p in d.glob("*.md"))


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Missing or invalid frontmatter -> ({}, text)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, text[m.end():]


def _infer_type(name: str, fm: dict) -> str:
    """Determine memory type from frontmatter, falling back to filename prefix."""
    meta = fm.get("metadata") or {}
    t = meta.get("type") or fm.get("type")
    if t:
        return str(t)
    for prefix, inferred in TYPE_PREFIX_MAP.items():
        if name.startswith(prefix):
            return inferred
    return "unknown"


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _filter_files(profile_cfg: dict, files: dict[str, dict]) -> list[str]:
    """Return list of file stems matching the profile, in deterministic order."""
    allowed_types = set(profile_cfg.get("types") or [])
    include = profile_cfg.get("include") or ["*"]
    exclude = profile_cfg.get("exclude") or []
    out = []
    for name, info in files.items():
        if info["type"] not in allowed_types:
            continue
        if not _matches(name, include):
            continue
        if exclude and _matches(name, exclude):
            continue
        out.append(name)
    return sorted(out)


_LINK_RE = re.compile(r"\[\[([a-zA-Z0-9_\-]+)\]\]")


def _resolve_links(body: str, files: dict[str, dict], depth: int, seen: set[str]) -> str:
    """Inline [[name]] references one level deep. Cycle-safe via 'seen'."""
    if depth <= 0:
        return body

    def repl(m: re.Match) -> str:
        target = m.group(1)
        if target in seen or target not in files:
            return m.group(0)
        new_seen = seen | {target}
        inner = files[target]["body"].strip()
        return f"{m.group(0)}\n> linked from [[{target}]]:\n> " + inner.replace("\n", "\n> ") + "\n"

    return _LINK_RE.sub(repl, body)


def _truncate(text: str, budget_tokens: int) -> str:
    """Truncate from the tail when char count exceeds budget."""
    limit = budget_tokens * APPROX_CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    truncated = text[:limit].rstrip()
    return truncated + "\n\n<!-- [memory truncated to ~{} tokens] -->".format(budget_tokens)


def _read_all_files(d: Path) -> dict[str, dict]:
    """Read every *.md in memory dir (except MEMORY.md and _-prefixed). Return name -> info."""
    files: dict[str, dict] = {}
    for p in d.glob("*.md"):
        if p.name == "MEMORY.md" or p.name.startswith("_"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise MemoryLoadError(f"cannot read {p}: {e}") from e
        fm, body = _parse_frontmatter(text)
        files[p.stem] = {
            "path": p,
            "frontmatter": fm,
            "body": body,
            "type": _infer_type(p.stem, fm),
            "description": fm.get("description", ""),
        }
    return files


def _format_block(profile: str, selected: list[str], files: dict[str, dict], link_depth: int) -> str:
    """Render the final system-prompt block, grouped by type."""
    if not selected:
        return (
            f"<!-- vibemind-memory:profile={profile} -->\n"
            f"# Memory Context (profile: {profile})\n\n"
            f"(no matching memory files)\n"
            f"<!-- /vibemind-memory -->"
        )

    by_type: dict[str, list[str]] = {}
    for name in selected:
        by_type.setdefault(files[name]["type"], []).append(name)

    parts = [
        f"<!-- vibemind-memory:profile={profile} -->",
        f"# Memory Context (profile: {profile}, files: {len(selected)})",
        "",
        "This block contains durable knowledge about the user, their preferences, and ongoing projects.",
        "Use it to inform decisions; do not echo it back verbatim.",
    ]

    type_headers = {
        "user": "## User",
        "feedback": "## Feedback (rules from prior corrections)",
        "project": "## Projects",
        "reference": "## External references",
        "unknown": "## Other",
    }

    for tkey in ("user", "feedback", "project", "reference", "unknown"):
        if tkey not in by_type:
            continue
        parts.append("")
        parts.append(type_headers[tkey])
        for name in by_type[tkey]:
            info = files[name]
            body = _resolve_links(info["body"], files, link_depth, seen={name})
            desc = info["description"]
            header = f"### {name}"
            if desc:
                header += f" — {desc}"
            parts.append("")
            parts.append(header)
            parts.append(body.strip())

    parts.append("")
    parts.append("<!-- /vibemind-memory -->")
    return "\n".join(parts)


@lru_cache(maxsize=32)
def _load_memory_cached(profile: str, sig: int) -> str:
    """Cached load. Re-runs whenever any *.md mtime changes (sig differs)."""
    d = _memory_dir()
    cfg = _load_profile_config()
    profile_cfg = cfg["profiles"][profile]
    link_depth = int(cfg.get("link_depth", 1))

    files = _read_all_files(d)
    selected = _filter_files(profile_cfg, files)
    return _format_block(profile, selected, files, link_depth)


def load_memory(profile: str = "full", max_tokens: int | None = None) -> str:
    """
    Load profile-filtered memory as a system-prompt block. Hard-fails on errors.

    Args:
        profile: one of the keys defined in memory_profiles.yml.
        max_tokens: optional override for the profile's default token_budget.

    Returns:
        Formatted markdown block wrapped in <!-- vibemind-memory --> markers.

    Raises:
        MemoryLoadError: missing dir, unknown profile, parse error.
    """
    d = _memory_dir()
    if not d.is_dir():
        raise MemoryLoadError(f"memory dir not found: {d}")

    cfg = _load_profile_config()
    if profile not in cfg["profiles"]:
        raise MemoryLoadError(
            f"unknown profile: {profile!r}. Known: {sorted(cfg['profiles'].keys())}"
        )

    sig = _mtime_signature(d)
    block = _load_memory_cached(profile, sig)

    budget = max_tokens if max_tokens is not None else cfg.get("token_budget", {}).get(profile)
    if budget:
        block = _truncate(block, int(budget))
    return block


def clear_memory_cache() -> None:
    """Force re-read on next load_memory(). Call after memory file edits."""
    _load_memory_cached.cache_clear()
    _load_profile_config.cache_clear()
