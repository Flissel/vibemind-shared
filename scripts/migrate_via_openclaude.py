"""
Automated LLM-config migration via OpenClaude (local Ollama).

For each file in LLM_AUDIT_REPORT.md, build a migration prompt and send it to
OpenClaude (running headlessly with a local model). OpenClaude reads the file,
applies the migration, and writes the new version.

Why OpenClaude:
  - It runs on local Ollama → migrating 398 files costs nothing
  - It can read context (other files in the same module)
  - It already lives in vibemind-os/openclaude/
  - It's the OpenAI-compatible CLI fork, so we can swap models freely

Usage:
    # Dry run on one service (no writes)
    python migrate_via_openclaude.py --service brain --dry-run

    # Migrate all files in a service for real
    python migrate_via_openclaude.py --service brain --apply

    # Limit to N files (good for testing the prompt)
    python migrate_via_openclaude.py --service brain --apply --limit 3

    # Use a specific model (default: qwen2.5-coder:7b via Ollama)
    python migrate_via_openclaude.py --service brain --model deepseek-coder-v2

Requires:
    - openclaude installed (vibemind-os/openclaude/dist/cli.mjs)
    - bun runtime
    - Ollama running locally with the chosen model pulled
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent  # vibemind-os/
OPENCLAUDE_CLI = ROOT / "openclaude" / "dist" / "cli.mjs"
AUDIT_REPORT = ROOT / "LLM_AUDIT_REPORT.md"


# -----------------------------------------------------------------------------
# Migration prompt — sent to OpenClaude per file.
# Patch-mode: output JSON array of surgical {old, new} replacements.
# This avoids regenerating the full file and stays within small output budgets.
# -----------------------------------------------------------------------------
MIGRATION_PROMPT = """You are migrating a Python file to use the vibemind_shared LLM factory.

Your job: Output a JSON array of surgical text replacements needed to migrate this file. Do NOT output the full file.

REQUIRED CHANGES TO DETECT:
1. `from openai import OpenAI` / `AsyncOpenAI` → replace with `from vibemind_shared import get_client, get_model`
2. `from anthropic import Anthropic` / `AsyncAnthropic` → replace with `from vibemind_shared import get_client, get_model`
3. `OpenAI(api_key=...)` / `openai.OpenAI(...)` → replace with `get_client("ROLE_NAME")`
4. `Anthropic(api_key=...)` / `anthropic.Anthropic(...)` → replace with `get_client("ROLE_NAME")`
5. Hardcoded model strings like `"gpt-4o"`, `"claude-3-5-sonnet-..."` in `model=` args → replace with `get_model("ROLE_NAME")`

ROLE NAMES — pick the closest match for this file's purpose:
  brain_planning, brain_fast_reasoning, brain_communication
  coding_planner, coding_executor, coding_reviewer, coding_security_audit
  voice_realtime, voice_intent_classifier, voice_summarizer
  security_red_team, security_blue_team, security_judge
  email_personalizer, fungus_summary, rag_classifier
  local_default (generic fallback)

OUTPUT FORMAT — strict JSON array, nothing else:
[
  {{"old": "exact text to find", "new": "replacement text"}},
  {{"old": "another exact snippet", "new": "replacement"}}
]

RULES:
- Each `old` must be a unique, exact substring of the file (copy-paste accurate)
- Keep `old` short (1-5 lines) — just enough to be unique
- Include NO explanation, NO code fences, NO markdown — just the JSON array
- If the file has no direct LLM client instantiations, output `[]`
- Prefer multiple small patches over one large replacement

FILE PATH: {file}

FILE CONTENTS:
```python
{content}
```

Output the JSON array now:
"""


def parse_audit_report(report_path: Path, service: Optional[str] = None) -> list[Path]:
    """Extract file paths from LLM_AUDIT_REPORT.md, optionally filtered by service."""
    if not report_path.exists():
        print(f"ERROR: {report_path} not found. Run audit_llm_usage.py first.")
        sys.exit(1)

    text = report_path.read_text(encoding="utf-8", errors="replace")
    files: list[Path] = []
    current_service: Optional[str] = None

    for line in text.splitlines():
        # Service header: "### `brain/` — 10 files"
        m = re.match(r"### `([^/]+)/`", line)
        if m:
            current_service = m.group(1)
            continue

        # File entry: "- **`brain\the_brain\core\llm_data_collector.py`**"
        m = re.match(r"- \*\*`([^`]+)`\*\*", line)
        if m:
            if service and current_service != service:
                continue
            file_path = ROOT / m.group(1).replace("\\", "/")
            if file_path.exists() and file_path.suffix == ".py":
                files.append(file_path)

    return files


def call_openclaude(prompt: str, model: str, timeout: int = 600) -> tuple[bool, str]:
    """Call OpenClaude headlessly with a prompt. Returns (success, output)."""
    if not OPENCLAUDE_CLI.exists():
        return False, f"OpenClaude CLI not found at {OPENCLAUDE_CLI}"

    cmd = [
        "bun", "run", str(OPENCLAUDE_CLI),
        "--dangerously-skip-permissions",
        "--output-format", "text",
        "-p", prompt,
    ]

    env = os.environ.copy()
    env["CLAUDE_CODE_USE_OPENAI"] = "1"
    env["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    env["OPENAI_API_KEY"] = "ollama-no-key"
    env["OPENAI_MODEL"] = model

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except FileNotFoundError:
        return False, "bun runtime not found — install from https://bun.sh"

    if result.returncode != 0:
        return False, f"exit {result.returncode}: {result.stderr[:500]}"

    return True, result.stdout


def extract_json_patches(output: str) -> Optional[list]:
    """Extract JSON array of {old, new} patches from OpenClaude output."""
    text = output.strip()
    # Strip code fences if present
    fence = re.search(r"```(?:json)?\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Find the first [ and the last ] — JSON array delimiters
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        patches = json.loads(text[start:end + 1])
        if not isinstance(patches, list):
            return None
        # Validate each entry has old+new
        for p in patches:
            if not isinstance(p, dict) or "old" not in p or "new" not in p:
                return None
        return patches
    except json.JSONDecodeError:
        return None


def apply_patches(original: str, patches: list) -> tuple[str, list[str]]:
    """Apply list of {old, new} patches. Returns (new_content, failures)."""
    result = original
    failures = []
    for i, p in enumerate(patches):
        old = p["old"]
        new = p["new"]
        if old not in result:
            failures.append(f"patch {i}: 'old' text not found (first 60 chars: {old[:60]!r})")
            continue
        # Only replace first occurrence to avoid accidentally multi-replacing
        result = result.replace(old, new, 1)
    return result, failures


def migrate_file(file_path: Path, model: str, dry_run: bool, timeout: int) -> dict:
    """Migrate a single file via patch-mode. Returns result dict."""
    rel = file_path.relative_to(ROOT)
    print(f"  -> {rel}", flush=True)

    original = file_path.read_text(encoding="utf-8", errors="replace")
    prompt = MIGRATION_PROMPT.format(file=str(rel), content=original)

    t0 = time.time()
    ok, output = call_openclaude(prompt, model, timeout=timeout)
    elapsed = time.time() - t0

    if not ok:
        return {"file": str(rel), "status": "ERROR", "error": output, "elapsed": elapsed}

    patches = extract_json_patches(output)
    if patches is None:
        preview = output.strip()[:200].replace("\n", " ")
        return {
            "file": str(rel),
            "status": "PARSE_FAIL",
            "error": f"could not extract JSON patches. Output: {preview}",
            "elapsed": elapsed,
        }

    if not patches:
        return {
            "file": str(rel),
            "status": "NO_CHANGES",
            "error": "OpenClaude found no direct LLM client instantiations",
            "patches": 0,
            "elapsed": elapsed,
        }

    new_code, patch_failures = apply_patches(original, patches)

    if patch_failures:
        return {
            "file": str(rel),
            "status": "PATCH_FAIL",
            "error": f"{len(patch_failures)} patch(es) failed to apply",
            "failures": patch_failures,
            "patches": len(patches),
            "elapsed": elapsed,
        }

    # Sanity check: must now mention vibemind_shared
    if "vibemind_shared" not in new_code:
        return {
            "file": str(rel),
            "status": "NO_IMPORT",
            "error": "patches applied but vibemind_shared import still missing",
            "patches": len(patches),
            "elapsed": elapsed,
        }

    # Sanity check: length shouldn't change dramatically
    size_delta = abs(len(new_code) - len(original))
    if size_delta > len(original) * 0.2:
        return {
            "file": str(rel),
            "status": "SIZE_DELTA",
            "error": f"size changed by {size_delta} bytes ({size_delta/len(original)*100:.0f}%)",
            "patches": len(patches),
            "elapsed": elapsed,
        }

    if dry_run:
        diff_path = file_path.with_suffix(".py.migrated")
        diff_path.write_text(new_code, encoding="utf-8")
        return {
            "file": str(rel),
            "status": "DRY_RUN",
            "preview": str(diff_path),
            "patches": len(patches),
            "elapsed": elapsed,
        }

    # Backup original
    backup = file_path.with_suffix(".py.bak")
    backup.write_text(original, encoding="utf-8")
    file_path.write_text(new_code, encoding="utf-8")

    return {
        "file": str(rel),
        "status": "OK",
        "backup": str(backup),
        "patches": len(patches),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate LLM usage via OpenClaude")
    parser.add_argument("--service", required=True,
                        help="Top-level service to migrate (e.g. brain, coding-engine)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite files (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write .migrated files instead of modifying originals")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N files (0 = all)")
    parser.add_argument("--model", default="qwen2.5-coder:7b",
                        help="Model to use (default: qwen2.5-coder:7b)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-file timeout in seconds")
    parser.add_argument("--report", default="migration_report.json",
                        help="JSON output report")
    args = parser.parse_args()

    # Default to dry-run unless --apply is set
    dry_run = not args.apply or args.dry_run

    files = parse_audit_report(AUDIT_REPORT, service=args.service)
    if not files:
        print(f"No files found for service '{args.service}' in {AUDIT_REPORT}")
        sys.exit(1)

    if args.limit > 0:
        files = files[: args.limit]

    print(f"\nMigrating {len(files)} files in service '{args.service}'")
    print(f"Model: {args.model}")
    print(f"Mode: {'DRY-RUN (writes .migrated files)' if dry_run else 'APPLY (rewrites originals)'}")
    print(f"OpenClaude CLI: {OPENCLAUDE_CLI}")
    print()

    if not OPENCLAUDE_CLI.exists():
        print(f"ERROR: OpenClaude CLI not found.")
        print(f"Build it first: cd vibemind-os/openclaude && bun install && bun run build")
        sys.exit(1)

    results = []
    ok_count = 0
    fail_count = 0
    t_start = time.time()

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
        result = migrate_file(f, args.model, dry_run, args.timeout)
        results.append(result)
        if result["status"] in ("OK", "DRY_RUN"):
            ok_count += 1
            print(f"  {result['status']} ({result['elapsed']:.1f}s)", flush=True)
        else:
            fail_count += 1
            print(f"  FAIL: {result['status']} - {result.get('error', '')}", flush=True)

    total = time.time() - t_start

    # Write report
    Path(args.report).write_text(
        json.dumps({"summary": {"ok": ok_count, "fail": fail_count, "total": len(files), "elapsed": total},
                    "results": results}, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== Done in {total:.1f}s ===")
    print(f"OK:   {ok_count}")
    print(f"FAIL: {fail_count}")
    print(f"Report: {args.report}")
    if dry_run and ok_count > 0:
        print(f"\nPreview .migrated files were written next to the originals.")
        print(f"Review them, then re-run with --apply to rewrite for real.")


if __name__ == "__main__":
    main()
