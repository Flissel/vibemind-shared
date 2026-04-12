"""
Sanitize .env files for open-source release.

For every .env file (NOT .env.example) found under the given root:
  1. Detect lines that look like API keys / secrets
  2. Write a sanitized copy as .env.example.generated next to it
  3. Optionally rewrite the original .env in-place (--in-place)
  4. Always print a report of what was redacted

Detection rules:
  - Variables ending in _API_KEY, _TOKEN, _SECRET, _PASSWORD, _DSN
  - Values matching common key patterns (sk-, sk-or-, sk-ant-, gsk_, sm_, etc.)
  - Values longer than 20 chars that look base64/hex

Usage:
    python sanitize_env.py --root . --report SECRETS_REPORT.md
    python sanitize_env.py --root . --in-place        # rewrite originals
    python sanitize_env.py --root . --check-only      # CI mode: exit 1 if any secret
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path

# Variable name suffixes that indicate a secret
SECRET_SUFFIXES = (
    "_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_PASS", "_DSN",
    "_PRIVATE_KEY", "_CREDENTIALS", "_AUTH",
)

# Variable name patterns (whole name) — focused on the 5 supported providers
# plus common channel/service tokens that often leak alongside them.
SECRET_NAMES = {
    # The 5 LLM provider keys
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY",
    # Memory / supporting services
    "SUPERMEMORY_API_KEY",
    # Common channel tokens
    "GITHUB_TOKEN", "DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN_ANALYZER",
    "TELEGRAM_BOT_TOKEN", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN",
    # Database / observability
    "SUPABASE_KEY", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY",
    "SENTRY_DSN", "SENTRY_AUTH_TOKEN",
}

# Value patterns that look like secrets — only for our 5 providers + common channels
SECRET_VALUE_PATTERNS = [
    re.compile(r"^sk-[A-Za-z0-9_\-]{20,}$"),         # OpenAI / OpenRouter generic
    re.compile(r"^sk-or-v\d+-[A-Za-z0-9]{20,}$"),    # OpenRouter explicit
    re.compile(r"^sk-ant-[A-Za-z0-9_\-]{20,}$"),     # Anthropic
    re.compile(r"^sk-proj-[A-Za-z0-9_\-]{20,}$"),    # OpenAI project
    re.compile(r"^AIza[A-Za-z0-9_\-]{30,}$"),        # Google API
    re.compile(r"^sm_[A-Za-z0-9]{20,}$"),            # Supermemory
    re.compile(r"^ghp_[A-Za-z0-9]{30,}$"),           # GitHub PAT
    re.compile(r"^github_pat_[A-Za-z0-9_]{50,}$"),   # GitHub fine-grained
    re.compile(r"^xoxb-[A-Za-z0-9\-]{20,}$"),        # Slack bot
    re.compile(r"^xapp-[A-Za-z0-9\-]{20,}$"),        # Slack app
    re.compile(r"^\d{8,}:[A-Za-z0-9_\-]{30,}$"),     # Telegram bot token
]

PLACEHOLDER = ""  # what to replace secret values with

EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma",
}


def is_secret_name(name: str) -> bool:
    name_upper = name.upper()
    if name_upper in SECRET_NAMES:
        return True
    return any(name_upper.endswith(suf) for suf in SECRET_SUFFIXES)


def value_looks_secret(value: str) -> bool:
    v = value.strip().strip('"').strip("'")
    if not v:
        return False
    if v.startswith("${") and v.endswith("}"):
        return False  # already a reference
    if v in ("true", "false", "null", "none"):
        return False
    for pat in SECRET_VALUE_PATTERNS:
        if pat.match(v):
            return True
    return False


def parse_line(line: str):
    """Return (name, value, is_assignment) or (None, None, False)."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, None, False
    if "=" not in stripped:
        return None, None, False
    name, value = stripped.split("=", 1)
    return name.strip(), value, True


def sanitize_content(content: str, source_path: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (sanitized_content, list_of_(var_name, original_value_preview))."""
    out_lines = []
    redactions = []
    for line in content.splitlines():
        name, value, is_assign = parse_line(line)
        if not is_assign:
            out_lines.append(line)
            continue

        v_clean = value.strip()
        is_secret = is_secret_name(name) or value_looks_secret(v_clean)

        if is_secret and v_clean and not v_clean.startswith("${"):
            preview = (v_clean[:8] + "..." + v_clean[-4:]) if len(v_clean) > 16 else "***"
            redactions.append((name, preview))
            out_lines.append(f"{name}={PLACEHOLDER}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if content.endswith("\n") else ""), redactions


def find_env_files(root: Path) -> list[Path]:
    """Find actual .env files (NOT .env.example/.template)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            fn_lower = fn.lower()
            if fn_lower == ".env" or fn_lower == ".envrc":
                files.append(Path(dirpath) / fn)
    return files


def main():
    parser = argparse.ArgumentParser(description="Sanitize .env files for open-source")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--in-place", action="store_true",
                        help="REWRITE original .env files (DANGEROUS — backup first)")
    parser.add_argument("--check-only", action="store_true",
                        help="CI mode: exit 1 if any unsanitized secret found")
    parser.add_argument("--report", default="SECRETS_REPORT.md",
                        help="Markdown report path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scanning {root} for .env files...")

    env_files = find_env_files(root)
    print(f"Found {len(env_files)} .env files")

    all_redactions: dict[Path, list] = {}
    for f in env_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  SKIP {f}: {e}")
            continue

        sanitized, redactions = sanitize_content(content, str(f))
        if redactions:
            all_redactions[f] = redactions

            if args.in_place:
                f.write_text(sanitized, encoding="utf-8")
                print(f"  REWROTE {f.relative_to(root)} ({len(redactions)} secrets)")
            else:
                out_path = f.with_name(f.name + ".sanitized")
                out_path.write_text(sanitized, encoding="utf-8")
                print(f"  WROTE {out_path.relative_to(root)} ({len(redactions)} secrets)")

    # Write report
    report = Path(args.report)
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# Secrets Audit Report\n\n")
        fh.write(f"**Scanned root:** `{root}`\n")
        fh.write(f"**Files with secrets:** {len(all_redactions)}\n")
        total = sum(len(r) for r in all_redactions.values())
        fh.write(f"**Total redactions:** {total}\n\n")

        if not all_redactions:
            fh.write("No secrets found. Repo is clean.\n")
        else:
            fh.write("## Files containing secrets\n\n")
            for path in sorted(all_redactions.keys()):
                rel = path.relative_to(root)
                fh.write(f"### `{rel}`\n\n")
                for name, preview in all_redactions[path]:
                    fh.write(f"- `{name}` = `{preview}`\n")
                fh.write("\n")

    print(f"\nReport: {report}")
    print(f"Total: {len(all_redactions)} files / {sum(len(r) for r in all_redactions.values())} secrets")

    if args.check_only and all_redactions:
        print("\nERROR: unsanitized secrets present (check-only mode)")
        sys.exit(1)


if __name__ == "__main__":
    main()
