"""
Audit script: inventory all direct LLM client usage across vibemind-os.

Finds files that instantiate LLM clients directly (OpenAI, Anthropic, Ollama,
LangChain ChatX, etc.) instead of using vibemind_shared.get_client(role).

Output: a markdown report grouped by service, showing each file with the
specific provider/model it hardcodes.

Usage:
    python audit_llm_usage.py [--root <dir>] [--out <report.md>]
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Patterns that indicate a direct LLM client instantiation (= migration target)
PATTERNS = {
    # Python — OpenAI SDK
    "OpenAI()":              re.compile(r"\b(?:Async)?OpenAI\s*\("),
    "openai.OpenAI()":       re.compile(r"openai\.(?:Async)?OpenAI\s*\("),

    # Python — Anthropic SDK
    "Anthropic()":           re.compile(r"\b(?:Async)?Anthropic\s*\("),
    "anthropic.Anthropic":   re.compile(r"anthropic\.(?:Async)?Anthropic\s*\("),

    # Python — Ollama / Groq direct
    "ollama.Client":         re.compile(r"ollama\.Client\s*\("),
    "Groq()":                re.compile(r"\bGroq\s*\("),

    # Python — LangChain
    "ChatOpenAI()":          re.compile(r"\bChatOpenAI\s*\("),
    "ChatAnthropic()":       re.compile(r"\bChatAnthropic\s*\("),
    "ChatOllama()":          re.compile(r"\bChatOllama\s*\("),
    "ChatGroq()":            re.compile(r"\bChatGroq\s*\("),

    # Python — Autogen / OpenAIChatCompletionClient
    "OpenAIChatCompletionClient": re.compile(r"\bOpenAIChatCompletionClient\s*\("),

    # Hardcoded base URLs
    "openrouter URL":        re.compile(r"openrouter\.ai/api/v1"),
    "anthropic URL":         re.compile(r"api\.anthropic\.com"),
    "groq URL":              re.compile(r"api\.groq\.com"),
    "deepseek URL":          re.compile(r"api\.deepseek\.com"),

    # Hardcoded model strings (only common ones, to find direct usage)
    "gpt-4o literal":        re.compile(r"['\"]gpt-4o(?:-mini)?['\"]"),
    "claude-3 literal":      re.compile(r"['\"]claude-3-[a-z0-9-]+['\"]"),

    # TypeScript / JS
    "new OpenAI(":           re.compile(r"new\s+OpenAI\s*\("),
    "new Anthropic(":        re.compile(r"new\s+Anthropic\s*\("),
}

EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "target", ".next",
    ".fungus_cache", ".pytest_cache", "models", "dist", "build",
    "downloads", ".pitchdeck_chroma", ".playwright-mcp",
    "_archive", "old_tests", "test_data",
}

# Files that LEGITIMATELY instantiate LLM clients (the shared package itself)
ALLOWLIST = {
    "shared/src/vibemind_shared/llm_client.py",
    "vibemind_shared/llm_client.py",
}

CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx"}


def is_allowlisted(rel_path: str) -> bool:
    rel_path_norm = rel_path.replace("\\", "/")
    return any(rel_path_norm.endswith(a) for a in ALLOWLIST)


def already_uses_shared(content: str) -> bool:
    return (
        "from vibemind_shared" in content
        or "import vibemind_shared" in content
        or "vibemind_shared.get_client" in content
    )


def _match_in_string_literal(line: str, match_start: int) -> bool:
    """Check if a match position is inside a string literal on the same line.

    Counts unescaped quote characters before the match position. If the count
    is odd, the position is inside a string. Handles ', ", and triple quotes
    crudely by treating them as regular quotes.
    """
    prefix = line[:match_start]
    # Strip escaped quotes to avoid miscounting
    prefix_clean = prefix.replace("\\'", "").replace('\\"', "")
    single_count = prefix_clean.count("'")
    double_count = prefix_clean.count('"')
    return (single_count % 2 == 1) or (double_count % 2 == 1)


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    """Return list of (pattern_name, line_no, line_text) hits."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    try:
        return _scan_content(content)
    except Exception as e:
        print(f"  WARN: scan failed for {path}: {e}", file=sys.stderr)
        return []


def _scan_content(content: str) -> list[tuple[str, int, str]]:

    hits = []
    in_docstring = False
    docstring_marker = ""
    triple_double = chr(34) * 3  # """
    triple_single = chr(39) * 3  # '''
    for line_no, line in enumerate(content.splitlines(), 1):
        # Track triple-quote docstrings (skip pattern matches inside)
        for marker in [triple_double, triple_single]:
            occurrences = line.count(marker)
            if occurrences == 0:
                continue
            if not in_docstring:
                in_docstring = True
                docstring_marker = marker
                if occurrences >= 2:
                    in_docstring = False
                    docstring_marker = ""
            elif marker == docstring_marker:
                in_docstring = False
                docstring_marker = ""
        if in_docstring:
            continue

        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue

        for name, pattern in PATTERNS.items():
            m = pattern.search(line)
            if m and not _match_in_string_literal(line, m.start()):
                hits.append((name, line_no, line.strip()[:120]))
                break  # one hit per line is enough
    return hits


def walk_tree(root: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in CODE_EXTS:
                files.append(Path(dirpath) / fn)
    return files


def group_by_service(root: Path, files_with_hits: dict) -> dict:
    """Group files by top-level service directory."""
    grouped = defaultdict(list)
    for file_path, hits in files_with_hits.items():
        rel = Path(file_path).relative_to(root)
        parts = rel.parts
        service = parts[0] if parts else "(root)"
        grouped[service].append((rel, hits))
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Audit direct LLM client usage")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--out", default="LLM_AUDIT_REPORT.md", help="Output markdown file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scanning {root}...", file=sys.stderr)

    files = walk_tree(root)
    print(f"Found {len(files)} code files", file=sys.stderr)

    files_with_hits: dict = {}
    already_migrated: list[Path] = []

    for f in files:
        rel = str(f.relative_to(root))
        if is_allowlisted(rel):
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if already_uses_shared(content):
            already_migrated.append(f)
            continue
        hits = scan_file(f)
        if hits:
            files_with_hits[f] = hits

    grouped = group_by_service(root, files_with_hits)

    # Write report
    out = Path(args.out)
    with out.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write("# LLM Usage Audit Report\n\n")
        fh.write(f"**Scanned:** `{root}`\n")
        fh.write(f"**Files with direct LLM client usage:** {len(files_with_hits)}\n")
        fh.write(f"**Files already using vibemind_shared:** {len(already_migrated)}\n\n")
        fh.write("## Migration targets (grouped by service)\n\n")

        for service in sorted(grouped.keys()):
            entries = grouped[service]
            fh.write(f"### `{service}/` — {len(entries)} files\n\n")
            for rel, hits in sorted(entries, key=lambda x: str(x[0])):
                fh.write(f"- **`{rel}`**\n")
                # Group hits by pattern
                patterns_seen = defaultdict(list)
                for name, line_no, line in hits:
                    patterns_seen[name].append((line_no, line))
                for pname, lines in patterns_seen.items():
                    fh.write(f"  - `{pname}` ({len(lines)}x)\n")
                    if args.verbose:
                        for ln, text in lines[:3]:
                            fh.write(f"    - L{ln}: `{text}`\n")
            fh.write("\n")

        if already_migrated:
            fh.write("## Already migrated (uses `vibemind_shared`)\n\n")
            for f in sorted(already_migrated):
                rel = f.relative_to(root)
                fh.write(f"- `{rel}`\n")

    print(f"Report written: {out}")
    print(f"Migration targets: {len(files_with_hits)}")
    print(f"Already migrated: {len(already_migrated)}")


if __name__ == "__main__":
    main()
