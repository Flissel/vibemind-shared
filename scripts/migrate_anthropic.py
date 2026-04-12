"""
Automated LLM-config migration via Claude Code CLI (Max subscription).

Reads LLM_AUDIT_REPORT.md, sends each file as a patch-mode prompt to Claude,
applies the JSON patches, validates with py_compile, auto-retries failures.

Usage:
    # Dry run on one service
    python migrate_anthropic.py --service coding-engine --dry-run --limit 3

    # Apply all files in a service
    python migrate_anthropic.py --service coding-engine --apply

    # Run ALL services at once
    python migrate_anthropic.py --all --apply

    # Dry-run everything, auto-apply only syntax-clean results
    python migrate_anthropic.py --all --auto-apply
"""
from __future__ import annotations
import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent  # vibemind-os/
AUDIT_REPORT = ROOT / "LLM_AUDIT_REPORT.md"

# ---------------------------------------------------------------------------
# Role mapping — directory patterns → best-fit roles
# ---------------------------------------------------------------------------
ROLE_MAP = {
    "coding-engine/src/agents/":       "coding_executor",
    "coding-engine/src/api/":          "coding_executor",
    "coding-engine/src/mcp/":          "coding_planner",
    "coding-engine/src/tools/":        "coding_executor",
    "coding-engine/src/monitoring/":   "coding_executor",
    "coding-engine/src/autogen/":      "coding_planner",
    "coding-engine/src/services/":     "coding_executor",
    "coding-engine/src/engine/":       "coding_planner",
    "coding-engine/web-app/":          "coding_executor",
    "brain/":                          "brain_planning",
    "voice/":                          "voice_summarizer",
    "security/":                       "security_analyzer",
    "la-fungus-search/":               "fungus_summary",
    "spaces/autogen/":                 "coding_planner",
    "spaces/coding/":                  "coding_executor",
    "spaces/desktop/":                 "local_default",
    "spaces/ideas/":                   "local_default",
    "ops/":                            "local_default",
    "devops/":                         "local_default",
    "business/":                       "local_default",
}


def suggest_role(file_path: str) -> str:
    """Suggest a role based on file path."""
    fp = file_path.replace("\\", "/")
    for prefix, role in ROLE_MAP.items():
        if prefix in fp:
            return role
    return "local_default"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
MIGRATION_PROMPT = """You are migrating a Python file to use the vibemind_shared LLM factory.

Output ONLY a raw JSON array of text replacements. No markdown fences, no explanation, no commentary.

CHANGES TO MAKE:
1. `from openai import OpenAI`/`AsyncOpenAI` → `from vibemind_shared import get_client, get_model` (or get_client_sync for sync)
2. `from anthropic import Anthropic`/`AsyncAnthropic` → `from vibemind_shared import get_client_sync, get_model`
3. `import anthropic` then `anthropic.Anthropic(...)` → replace with get_client_sync. If `anthropic` is still used for exceptions (APIError etc), keep `from anthropic import APIError` separately.
4. `OpenAI(api_key=..., base_url=...)` → `get_client_sync("ROLE")` or `get_client("ROLE")` (match sync/async of original)
5. `Anthropic(api_key=...)` → `get_client_sync("ROLE")`
6. Hardcoded model strings like `"gpt-4o"`, `"claude-3-..."`, `"claude-haiku-..."` in `model=` args → `get_model("ROLE")`
7. If `import anthropic` or `import openai` was ONLY used for client construction and nothing else, remove the import entirely
8. If `from src.llm_config import ...` exists, replace with `from vibemind_shared import ...`
9. If `os.environ.get("..._API_KEY")` or `os.getenv("..._API_KEY")` is only used for the client constructor, remove it

CRITICAL RULES:
- Each "old" must be an EXACT substring of the file (copy-paste accurate, whitespace-perfect)
- For multi-line code: include ALL lines including the closing `)` or `]`
- Sync clients → get_client_sync("ROLE"). Async clients → get_client("ROLE")
- If `model=self.config.model` or `model=kwargs.get(...)` — LEAVE IT, it's already dynamic
- NEVER output markdown fences (```). NEVER add explanation text. ONLY the JSON array.

ROLE FOR THIS FILE: Use "{role}" as the role name (based on file directory).

Output format — raw JSON array, nothing else:
[{{"old": "text", "new": "replacement"}}]

If no changes needed: []

FILE: {file}

```
{content}
```
"""

RETRY_PROMPT = """Your previous output was not valid JSON. Output ONLY a raw JSON array, no markdown, no code fences, no explanation.

Previous file: {file}
Previous error: {error}

Try again — output ONLY the JSON array:
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_audit_report(report_path: Path, service: Optional[str] = None) -> list[Path]:
    if not report_path.exists():
        print(f"ERROR: {report_path} not found. Run audit_llm_usage.py first.")
        sys.exit(1)
    text = report_path.read_text(encoding="utf-8", errors="replace")
    files: list[Path] = []
    current_service: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"### `([^/]+)/`", line)
        if m:
            current_service = m.group(1)
            continue
        m = re.match(r"- \*\*`([^`]+)`\*\*", line)
        if m:
            if service and current_service != service:
                continue
            file_path = ROOT / m.group(1).replace("\\", "/")
            if file_path.exists() and file_path.suffix == ".py":
                files.append(file_path)
    return files


# ---------------------------------------------------------------------------
# Claude CLI
# ---------------------------------------------------------------------------
def call_claude(prompt: str) -> tuple[bool, str]:
    """Call Claude via CLI (Max subscription)."""
    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"
        result = subprocess.run(
            [claude_cmd, "--output-format", "text", "-p", "-"],
            input=prompt,
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace",
            env=env, shell=(sys.platform == "win32"),
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
        return False, f"CLI exit {result.returncode}: {result.stderr[:300]}"
    except FileNotFoundError:
        return False, "claude CLI not found"
    except subprocess.TimeoutExpired:
        return False, "timeout after 180s"


# ---------------------------------------------------------------------------
# Patch extraction + application
# ---------------------------------------------------------------------------
def extract_patches(output: str) -> Optional[list]:
    text = output.strip()
    # Strip code fences aggressively
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # Find JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        patches = json.loads(text[start:end + 1])
        if not isinstance(patches, list):
            return None
        for p in patches:
            if not isinstance(p, dict) or "old" not in p or "new" not in p:
                return None
        return patches
    except json.JSONDecodeError:
        return None


def apply_patches(original: str, patches: list) -> tuple[str, list[str]]:
    result = original
    failures = []
    for i, p in enumerate(patches):
        old, new = p["old"], p["new"]
        if old not in result:
            # Try with normalized line endings
            old_norm = old.replace("\r\n", "\n")
            if old_norm in result:
                result = result.replace(old_norm, new, 1)
                continue
            failures.append(f"patch {i}: old not found: {old[:60]!r}")
            continue
        result = result.replace(old, new, 1)
    return result, failures


def syntax_check(code: str) -> Optional[str]:
    tmp = tempfile.mktemp(suffix=".py")
    try:
        Path(tmp).write_text(code, encoding="utf-8")
        py_compile.compile(tmp, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return str(e)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------
def migrate_file(file_path: Path, dry_run: bool, auto_apply: bool, max_retries: int = 1) -> dict:
    rel = file_path.relative_to(ROOT)
    original = file_path.read_text(encoding="utf-8", errors="replace")

    # Skip if already migrated
    if "from vibemind_shared" in original or "import vibemind_shared" in original:
        return {"file": str(rel), "status": "ALREADY_MIGRATED", "elapsed": 0}

    role = suggest_role(str(rel))
    prompt = MIGRATION_PROMPT.format(file=str(rel), content=original, role=role)

    t0 = time.time()
    last_error = ""

    for attempt in range(1 + max_retries):
        if attempt > 0:
            prompt = RETRY_PROMPT.format(file=str(rel), error=last_error) + f"\n\n```\n{original}\n```"

        ok, output = call_claude(prompt)
        elapsed = time.time() - t0

        if not ok:
            return {"file": str(rel), "status": "API_ERROR", "error": output, "elapsed": elapsed}

        patches = extract_patches(output)
        if patches is None:
            last_error = f"Invalid JSON: {output[:100]}"
            if attempt < max_retries:
                continue  # retry
            return {"file": str(rel), "status": "PARSE_FAIL", "error": last_error, "elapsed": elapsed}

        if not patches:
            return {"file": str(rel), "status": "NO_CHANGES", "elapsed": elapsed}

        new_code, patch_failures = apply_patches(original, patches)

        if patch_failures:
            last_error = f"Patch failures: {'; '.join(patch_failures[:2])}"
            if attempt < max_retries:
                continue  # retry
            return {
                "file": str(rel), "status": "PATCH_FAIL",
                "error": last_error, "patches": len(patches), "elapsed": elapsed,
            }

        # Syntax check
        syn_err = syntax_check(new_code)
        if syn_err:
            last_error = f"Syntax error: {syn_err[:100]}"
            if attempt < max_retries:
                continue  # retry
            return {
                "file": str(rel), "status": "SYNTAX_ERROR",
                "error": last_error, "patches": len(patches), "elapsed": elapsed,
            }

        # Must contain vibemind_shared
        if "vibemind_shared" not in new_code:
            return {
                "file": str(rel), "status": "NO_IMPORT",
                "patches": len(patches), "elapsed": elapsed,
            }

        # Success! Decide whether to write
        should_write = (not dry_run) or auto_apply
        retried = " (retry)" if attempt > 0 else ""

        if should_write:
            backup = file_path.with_suffix(".py.bak")
            backup.write_text(original, encoding="utf-8")
            file_path.write_text(new_code, encoding="utf-8")
            return {
                "file": str(rel), "status": "APPLIED",
                "backup": str(backup), "patches": len(patches),
                "role": role, "elapsed": elapsed, "retried": attempt > 0,
            }
        else:
            diff_path = file_path.with_suffix(".py.migrated")
            diff_path.write_text(new_code, encoding="utf-8")
            return {
                "file": str(rel), "status": "DRY_RUN",
                "preview": str(diff_path), "patches": len(patches),
                "role": role, "elapsed": elapsed, "retried": attempt > 0,
            }

    return {"file": str(rel), "status": "MAX_RETRIES", "error": last_error, "elapsed": time.time() - t0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Migrate LLM usage via Claude (Max subscription)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--service", help="Migrate one service (e.g. coding-engine)")
    group.add_argument("--all", action="store_true", help="Migrate ALL services")
    parser.add_argument("--apply", action="store_true", help="Apply changes directly")
    parser.add_argument("--dry-run", action="store_true", help="Write .migrated preview files")
    parser.add_argument("--auto-apply", action="store_true",
                        help="Auto-apply only files that pass syntax check (safest batch mode)")
    parser.add_argument("--limit", type=int, default=0, help="Max files (0=all)")
    parser.add_argument("--retries", type=int, default=1, help="Max retries per file (default 1)")
    parser.add_argument("--report", default="migration_report.json")
    args = parser.parse_args()

    dry_run = not args.apply and not args.auto_apply
    auto_apply = args.auto_apply

    if args.all:
        files = parse_audit_report(AUDIT_REPORT, service=None)
        scope = "ALL services"
    else:
        files = parse_audit_report(AUDIT_REPORT, service=args.service)
        scope = args.service

    if not files:
        print(f"No Python files found for '{scope}' in {AUDIT_REPORT}")
        sys.exit(1)
    if args.limit > 0:
        files = files[:args.limit]

    mode = "AUTO-APPLY (syntax-gated)" if auto_apply else ("APPLY" if args.apply else "DRY-RUN")
    print(f"{'='*60}")
    print(f"LLM Config Migration — {scope}")
    print(f"{'='*60}")
    print(f"Files:   {len(files)}")
    print(f"Mode:    {mode}")
    print(f"Retries: {args.retries}")
    print(f"{'='*60}")
    print()

    results = []
    ok = fail = skip = 0
    t_start = time.time()

    for i, f in enumerate(files, 1):
        name = f.name
        print(f"[{i}/{len(files)}] {name:40s}", end=" ", flush=True)
        result = migrate_file(f, dry_run, auto_apply, max_retries=args.retries)
        results.append(result)

        status = result["status"]
        elapsed = result.get("elapsed", 0)
        patches = result.get("patches", 0)
        role = result.get("role", "")
        retried = " (retried)" if result.get("retried") else ""

        if status in ("APPLIED", "DRY_RUN"):
            ok += 1
            print(f"{status:10s} {patches} patches  {elapsed:5.1f}s  role={role}{retried}")
        elif status in ("ALREADY_MIGRATED", "NO_CHANGES"):
            skip += 1
            print(f"SKIP ({status})")
        else:
            fail += 1
            err = (result.get("error", "") or "")[:60]
            print(f"FAIL {status}: {err}")

    total_time = time.time() - t_start

    Path(args.report).write_text(json.dumps({
        "summary": {
            "scope": scope, "ok": ok, "fail": fail, "skip": skip,
            "total": len(files), "elapsed_s": round(total_time, 1),
        },
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"OK: {ok}  FAIL: {fail}  SKIP: {skip}  Total: {len(files)}")
    print(f"Time: {total_time/60:.1f} min")
    print(f"Report: {args.report}")

    if fail > 0:
        print(f"\nFailed files need manual migration via /llm-config-migration skill")

    # Post-migration: re-run audit
    if ok > 0 and (args.apply or auto_apply):
        print(f"\nRe-running audit to verify...")
        subprocess.run(
            [sys.executable, str(ROOT / "shared/scripts/audit_llm_usage.py"),
             "--root", str(ROOT), "--out", str(AUDIT_REPORT)],
            capture_output=True, timeout=300,
        )
        text = AUDIT_REPORT.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines()[:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
