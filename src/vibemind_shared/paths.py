"""
VibeMind cross-platform path resolution.

Single source of truth for every filesystem path the application needs. No
subsystem should ever hardcode `C:\\Users\\User\\...` or any absolute path —
import a helper from here instead.

Design:
  - repo_root() auto-detects the Vibemind_V1 checkout location. Override with
    the VIBEMIND_ROOT env var (set this in Docker / non-standard layouts).
  - Every helper returns a pathlib.Path (never an OS-specific string).
  - Every helper is individually overridable via its own env var, so a
    container or a custom layout can redirect any single path without
    touching code.
  - Dependency-light: only `os` + `pathlib`. Safe to import from anywhere
    (Brain, Voice, OpenFang scripts) with zero side effects.

Resolution order for repo_root():
  1. $VIBEMIND_ROOT          — explicit override (Docker, CI, custom layout)
  2. marker walk             — climb parents until a dir containing
                               `vibemind-os/` or `.git` is found
  3. cwd                     — last-resort fallback

Usage:
    from vibemind_shared.paths import openfang_agents_dir, downloads_dir
    agents = openfang_agents_dir()          # -> Path, cross-platform
    dl = downloads_dir() / "Klotski.json"   # -> ~/Downloads/Klotski.json
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "repo_root",
    "vibemind_os",
    "openfang_dir",
    "openfang_agents_dir",
    "issue_inbox",
    "rowboat_knowledge",
    "rowboat_home",
    "downloads_dir",
    "fungus_cache",
    "brain_dir",
    "shared_dir",
    "config_dir",
]


# ── core ─────────────────────────────────────────────────────────────────────

def _env_or(var: str, default: Path) -> Path:
    """Return the env-var path if `var` is set and non-empty, else `default`.

    Env values are normalized via _norm_env_path (absolute taken verbatim,
    relative resolved) so a Linux override survives a Windows test shell.
    """
    val = os.environ.get(var)
    if val and val.strip():
        return _norm_env_path(val)
    return default


def _norm_env_path(raw: str) -> Path:
    """Expand a path from an env var.

    An already-absolute value is taken verbatim; only a genuinely relative
    path is resolve()'d against cwd. "Absolute" is detected in a
    platform-agnostic way: pathlib.Path.is_absolute() is platform-specific
    (on Windows a POSIX `/opt/x` reports False and would get mis-anchored to
    `C:\\opt\\x`), so we ALSO treat a leading "/" or "\\" or a drive-letter
    prefix as absolute. This keeps a Linux VIBEMIND_ROOT correct even when
    the check happens to run under a Windows interpreter.
    """
    s = raw.strip()
    expanded = os.path.expanduser(s)
    looks_absolute = (
        expanded.startswith("/")
        or expanded.startswith("\\")
        or (len(expanded) >= 2 and expanded[1] == ":")  # C:\ , D:/ , ...
    )
    p = Path(expanded)
    if looks_absolute or p.is_absolute():
        return p
    return p.resolve()


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Locate the Vibemind_V1 checkout root.

    1. $VIBEMIND_ROOT override (highest priority — Docker / custom layouts).
    2. Fixed-relative anchor: this file is always at
       <root>/vibemind-os/shared/src/vibemind_shared/paths.py — so the root
       is exactly parents[4]. We sanity-check that parents[4]/vibemind-os
       exists; if the layout matches, use it. This is deterministic and
       immune to nested checkouts or stray `vibemind-os` dirs elsewhere.
    3. Marker walk fallback: nearest ancestor whose `vibemind-os/` child
       also contains `vibemind-os/shared` (distinguishes the real root from
       an unrelated dir that merely has a `vibemind-os` folder).
    4. cwd as last resort.
    """
    env = os.environ.get("VIBEMIND_ROOT")
    if env and env.strip():
        return _norm_env_path(env)

    here = Path(__file__).resolve()
    parents = here.parents

    # 2. Fixed-relative anchor (the normal case)
    if len(parents) > 4:
        candidate = parents[4]
        if (candidate / "vibemind-os" / "shared").is_dir():
            return candidate

    # 3. Marker walk — require vibemind-os/shared to avoid false positives
    for parent in parents:
        if (parent / "vibemind-os" / "shared").is_dir():
            return parent

    return Path.cwd().resolve()


# ── repo-relative directories ────────────────────────────────────────────────

def vibemind_os() -> Path:
    """The `vibemind-os/` directory (the OS monorepo inside the checkout)."""
    return _env_or("VIBEMIND_OS_DIR", repo_root() / "vibemind-os")


def brain_dir() -> Path:
    """The Brain root: vibemind-os/brain/the_brain/."""
    return _env_or("VIBEMIND_BRAIN_DIR", vibemind_os() / "brain" / "the_brain")


def shared_dir() -> Path:
    """The shared package root: vibemind-os/shared/."""
    return _env_or("VIBEMIND_SHARED_DIR", vibemind_os() / "shared")


def openfang_dir() -> Path:
    """The OpenFang submodule root: vibemind-os/openfang/."""
    return _env_or("OPENFANG_DIR", vibemind_os() / "openfang")


def openfang_agents_dir() -> Path:
    """OpenFang agent manifest directory: vibemind-os/openfang/agents/."""
    return _env_or("OPENFANG_AGENTS_DIR", openfang_dir() / "agents")


def fungus_cache() -> Path:
    """The la-fungus-search persistent index cache directory."""
    return _env_or(
        "FUNGUS_CACHE_DIR",
        vibemind_os() / "la-fungus-search" / ".fungus_cache",
    )


def issue_inbox() -> Path:
    """The issue-detector inbox markdown file."""
    return _env_or(
        "VIBEMIND_INBOX",
        vibemind_os() / "issue-detector" / "vibemind_inbox.md",
    )


def config_dir() -> Path:
    """Directory holding shared config files (llm_config.yml, models_pricing.yml).

    Honors VIBEMIND_CONFIG_DIR — the same env var pricing.py / llm_client.py
    already use for config-file resolution.
    """
    return _env_or("VIBEMIND_CONFIG_DIR", shared_dir() / "src" / "vibemind_shared")


# ── home-relative directories (user data, not in the repo) ──────────────────

def rowboat_home() -> Path:
    """The user's ~/.rowboat directory (markdown knowledge vault root)."""
    return _env_or("ROWBOAT_HOME", Path.home() / ".rowboat")


def rowboat_knowledge() -> Path:
    """The ~/.rowboat/knowledge directory (Brain reads user profile etc. here)."""
    return _env_or("ROWBOAT_KNOWLEDGE", rowboat_home() / "knowledge")


def downloads_dir() -> Path:
    """The user's Downloads directory — cross-platform (~/Downloads on all OSes)."""
    return _env_or("VIBEMIND_DOWNLOADS", Path.home() / "Downloads")


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick diagnostic: print every resolved path. Used by the verification
    # step — no hardcoded 'C:\\Users\\User' should appear unless that genuinely
    # is this machine's home/checkout.
    print(f"repo_root()           = {repo_root()}")
    print(f"vibemind_os()         = {vibemind_os()}")
    print(f"brain_dir()           = {brain_dir()}")
    print(f"shared_dir()          = {shared_dir()}")
    print(f"openfang_dir()        = {openfang_dir()}")
    print(f"openfang_agents_dir() = {openfang_agents_dir()}")
    print(f"fungus_cache()        = {fungus_cache()}")
    print(f"issue_inbox()         = {issue_inbox()}")
    print(f"config_dir()          = {config_dir()}")
    print(f"rowboat_home()        = {rowboat_home()}")
    print(f"rowboat_knowledge()   = {rowboat_knowledge()}")
    print(f"downloads_dir()       = {downloads_dir()}")
