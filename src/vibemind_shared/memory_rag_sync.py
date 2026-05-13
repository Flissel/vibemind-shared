"""
Memory -> Rowboat sync.

One-way mirror from the markdown memory directory into a Rowboat data-source
named "vibemind-memory". Each *.md file (excluding MEMORY.md and _-prefixed
files) becomes a text-type DataSourceDoc. Rowboat indexes content into Qdrant
so agents can query it semantically via the memory_search MCP tool.

State is tracked in _sync_state.json next to the memory files: maps each
memory name -> {hash, doc_id}. On re-sync, unchanged files are skipped,
modified files are replaced (delete + create), removed files are deleted
from Rowboat.

Markdown remains the truth. This module never mutates the source files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import yaml

DATA_SOURCE_NAME = "vibemind-memory"
DATA_SOURCE_DESCRIPTION = "VibeMind persistent agent memory (auto-synced from markdown)."
SYNC_STATE_FILENAME = "_sync_state.json"
HTTP_TIMEOUT_SECONDS = 30

MEMORY_DIR_DEFAULT = Path(r"C:\Users\User\.claude\projects\c--Users-User-myBrain\memory")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class RowboatSyncError(RuntimeError):
    """Raised when sync cannot proceed (missing env, HTTP error, data-source mismatch)."""


def _memory_dir() -> Path:
    return Path(os.environ.get("VIBEMIND_MEMORY_DIR", str(MEMORY_DIR_DEFAULT)))


def _sync_state_path() -> Path:
    return _memory_dir() / SYNC_STATE_FILENAME


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RowboatSyncError(f"missing env var: {var}")
    return val


def _api(method: str, path: str, *, json_body: dict | None = None) -> Any:
    base = _require_env("ROWBOAT_URL")
    key = _require_env("ROWBOAT_API_KEY")
    headers = {"Authorization": f"Bearer {key}"}
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    r = requests.request(
        method,
        f"{base}{path}",
        headers=headers,
        json=json_body,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    if r.status_code >= 400:
        raise RowboatSyncError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
    if not r.text:
        return None
    try:
        return r.json()
    except ValueError as e:
        raise RowboatSyncError(f"{method} {path} -> non-JSON response: {r.text[:300]}") from e


def _project_id() -> str:
    return _require_env("ROWBOAT_PROJECT_ID")


def _ensure_data_source() -> str:
    """Find or create the vibemind-memory data-source. Returns its id."""
    result = _api("GET", f"/api/v1/{_project_id()}/data-sources")
    items = result.get("items", []) if isinstance(result, dict) else []
    for ds in items:
        if ds.get("name") == DATA_SOURCE_NAME:
            return ds["id"]
    created = _api(
        "POST",
        f"/api/v1/{_project_id()}/data-sources",
        json_body={
            "name": DATA_SOURCE_NAME,
            "description": DATA_SOURCE_DESCRIPTION,
            "type": "text",
        },
    )
    return created["id"]


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load_state() -> dict:
    p = _sync_state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"files": {}}
    return {"files": {}}


def _save_state(state: dict) -> None:
    _sync_state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
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


def _build_doc_payload(p: Path) -> dict:
    """Convert a markdown file into a Rowboat text-doc payload."""
    text = p.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    meta = fm.get("metadata") or {}
    memory_type = meta.get("type") or fm.get("type") or "unknown"
    description = fm.get("description") or ""

    # Embed metadata in the content so the RAG search can pick it up — Rowboat's
    # text doc schema doesn't have a separate metadata field at the doc level.
    header = (
        f"# {p.stem}\n"
        f"type: {memory_type}\n"
        f"source_file: {p.name}\n"
    )
    if description:
        header += f"description: {description}\n"
    header += "\n"

    return {
        "name": p.stem,
        "data": {"type": "text", "content": header + body},
    }


def _upload_doc(ds_id: str, p: Path) -> str:
    payload = _build_doc_payload(p)
    result = _api(
        "POST",
        f"/api/v1/{_project_id()}/data-sources/{ds_id}/docs",
        json_body={"docs": [payload]},
    )
    # Endpoint returns {ok: true, count: 1} — need to fetch the new doc id.
    # List docs and pick the most recently created with matching name.
    docs = _api("GET", f"/api/v1/{_project_id()}/data-sources/{ds_id}/docs")
    items = docs.get("items", []) if isinstance(docs, dict) else []
    candidates = [d for d in items if d.get("name") == p.stem and d.get("status") != "deleted"]
    if not candidates:
        raise RowboatSyncError(f"uploaded doc not found in list for {p.stem}")
    candidates.sort(key=lambda d: d.get("createdAt", ""), reverse=True)
    return candidates[0]["id"]


def _delete_doc(ds_id: str, doc_id: str) -> None:
    _api(
        "DELETE",
        f"/api/v1/{_project_id()}/data-sources/{ds_id}/docs/{doc_id}",
    )


def _list_current_files() -> dict[str, Path]:
    d = _memory_dir()
    return {
        p.stem: p
        for p in d.glob("*.md")
        if p.name != "MEMORY.md" and not p.name.startswith("_")
    }


def sync_memory_to_rowboat() -> dict:
    """
    Walk memory dir, diff against state, upload changed/new docs, delete removed.

    Skips gracefully if ROWBOAT_* env vars are missing — Rowboat is a best-effort
    secondary index; the canonical RAG backend is Qdrant via sync_memory_to_qdrant.

    Returns:
        dict with keys: added, updated, deleted, errors, data_source_id, skipped.
    """
    missing = [v for v in ("ROWBOAT_URL", "ROWBOAT_PROJECT_ID", "ROWBOAT_API_KEY")
               if not os.environ.get(v)]
    if missing:
        return {
            "added": 0, "updated": 0, "deleted": 0, "skipped": 0,
            "errors": [f"rowboat env not configured (missing: {', '.join(missing)}) — skipped"],
            "data_source_id": None,
        }
    ds_id = _ensure_data_source()
    state = _load_state()
    old_files: dict[str, dict] = state.get("files", {})
    current = _list_current_files()

    new_files: dict[str, dict] = {}
    added = updated = deleted = skipped = 0
    errors: list[str] = []

    for name, p in current.items():
        try:
            h = _file_hash(p)
        except OSError as e:
            errors.append(f"hash {name}: {e}")
            continue
        prev = old_files.get(name)
        if prev and prev.get("hash") == h:
            new_files[name] = prev
            skipped += 1
            continue
        try:
            if prev:
                _delete_doc(ds_id, prev["doc_id"])
                doc_id = _upload_doc(ds_id, p)
                updated += 1
            else:
                doc_id = _upload_doc(ds_id, p)
                added += 1
            new_files[name] = {"hash": h, "doc_id": doc_id}
        except Exception as e:
            errors.append(f"{name}: {e}")

    for name, prev in old_files.items():
        if name in current:
            continue
        try:
            _delete_doc(ds_id, prev["doc_id"])
            deleted += 1
        except Exception as e:
            errors.append(f"delete {name}: {e}")

    _save_state({"files": new_files})
    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
        "data_source_id": ds_id,
    }
