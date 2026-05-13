"""
Memory Writer — append proposed memory diffs to a pending queue.

Agents discover learnings, corrections, or new project context and call
propose_memory_diff() to enqueue a candidate entry. Nothing mutates the
canonical MEMORY.md / *.md files — that happens only after the user reviews
via the /memory-review slash command.

The queue is an append-only JSONL at <memory_dir>/_pending.jsonl. Each line
is one entry with status="pending" initially; /memory-review rewrites it
with the resolved status (accepted/rejected/edited).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

MEMORY_DIR_DEFAULT = Path(r"C:\Users\User\.claude\projects\c--Users-User-myBrain\memory")
PENDING_FILENAME = "_pending.jsonl"
VALID_TYPES = ("user", "feedback", "project", "reference")


def _pending_path() -> Path:
    override = os.environ.get("VIBEMIND_PENDING_PATH")
    if override:
        return Path(override)
    base = Path(os.environ.get("VIBEMIND_MEMORY_DIR", str(MEMORY_DIR_DEFAULT)))
    return base / PENDING_FILENAME


def propose_memory_diff(
    content: str,
    memory_type: str,
    source_agent: str,
    target_file: str | None = None,
    reason: str = "",
    name: str | None = None,
) -> str:
    """
    Enqueue a candidate memory entry for user review.

    Args:
        content: the memory body to append/create.
        memory_type: one of user, feedback, project, reference.
        source_agent: identifier for the agent that produced this diff
            (e.g. "openclaude", "coordinator", "telegram-desktop").
        target_file: existing memory file stem to append to. None = create new.
        reason: why this is worth remembering (shown during review).
        name: slug for a new file (required if target_file is None and you want
            a deterministic filename; otherwise reviewer assigns).

    Returns:
        The entry id (8-char hex).

    Raises:
        ValueError: invalid memory_type.
        OSError: cannot write to the pending file.
    """
    if memory_type not in VALID_TYPES:
        raise ValueError(
            f"invalid memory_type: {memory_type!r}. Valid: {VALID_TYPES}"
        )
    if not source_agent:
        raise ValueError("source_agent is required")
    if not content.strip():
        raise ValueError("content cannot be empty")

    entry = {
        "id": uuid4().hex[:8],
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": memory_type,
        "source_agent": source_agent,
        "target_file": target_file,
        "name": name,
        "content": content,
        "reason": reason,
        "status": "pending",
    }
    p = _pending_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry["id"]


def list_pending() -> list[dict]:
    """Return all entries (any status). Empty list if file missing/empty."""
    p = _pending_path()
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def rewrite_pending(entries: list[dict]) -> None:
    """Atomically rewrite the pending file with given entries.

    Used by /memory-review after resolving statuses. Writes via temp+rename
    so a crash mid-write doesn't corrupt the queue.
    """
    p = _pending_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.replace(p)
