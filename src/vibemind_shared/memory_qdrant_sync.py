"""
Memory -> Qdrant direct sync.

Bypasses Rowboat's workflow engine entirely. Embeds each markdown file with
sentence-transformers (all-MiniLM-L6-v2 by default) and upserts into a
dedicated Qdrant collection. The memory_search MCP tool queries this
collection directly — no chat, no agents, no workflow.

State is shared with memory_rag_sync via _sync_state.json so /memory-review's
sync-step keeps both backends in sync.

Env vars:
- QDRANT_URL — default http://localhost:6730 (vibemind-qdrant)
- VIBEMIND_MEMORY_DIR — overrides default memory dir
- VIBEMIND_MEMORY_COLLECTION — overrides default "vibemind-memory" collection name
- VIBEMIND_EMBEDDING_MODEL — overrides default "all-MiniLM-L6-v2"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

DEFAULT_QDRANT_URL = "http://localhost:6730"
DEFAULT_COLLECTION = "vibemind-memory"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MEMORY_DIR_DEFAULT = Path(r"C:\Users\User\.claude\projects\c--Users-User-myBrain\memory")
QDRANT_STATE_FILENAME = "_qdrant_state.json"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class QdrantSyncError(RuntimeError):
    """Raised when sync cannot proceed."""


def _memory_dir() -> Path:
    return Path(os.environ.get("VIBEMIND_MEMORY_DIR", str(MEMORY_DIR_DEFAULT)))


def _qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)


def _collection_name() -> str:
    return os.environ.get("VIBEMIND_MEMORY_COLLECTION", DEFAULT_COLLECTION)


def _embedding_model_name() -> str:
    return os.environ.get("VIBEMIND_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _state_path() -> Path:
    return _memory_dir() / QDRANT_STATE_FILENAME


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


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"files": {}}
    return {"files": {}}


def _save_state(state: dict) -> None:
    _state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def _deterministic_id(name: str) -> str:
    """Stable Qdrant point id from memory entry name (UUIDv5)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vibemind-memory/{name}"))


_model_cache: Any = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(_embedding_model_name())
    return _model_cache


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=_qdrant_url())


def _ensure_collection(client, dim: int) -> None:
    """Create the collection if missing. Idempotent."""
    from qdrant_client.http.models import Distance, VectorParams
    existing = {c.name for c in client.get_collections().collections}
    if _collection_name() in existing:
        return
    client.create_collection(
        collection_name=_collection_name(),
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def _list_current_files() -> dict[str, Path]:
    d = _memory_dir()
    return {
        p.stem: p
        for p in d.glob("*.md")
        if p.name != "MEMORY.md" and not p.name.startswith("_")
    }


def _build_payload(p: Path) -> tuple[str, dict]:
    """Return (embedding-input-text, qdrant-payload)."""
    text = p.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    meta = fm.get("metadata") or {}
    memory_type = meta.get("type") or fm.get("type") or _infer_type_from_name(p.stem)
    description = fm.get("description") or ""

    # Embed the full body + description so search hits both content and intent.
    embed_text = f"{description}\n\n{body}".strip()
    payload = {
        "name": p.stem,
        "source_file": p.name,
        "memory_type": memory_type,
        "description": description,
        "body": body,
    }
    return embed_text, payload


def _infer_type_from_name(name: str) -> str:
    for prefix, t in (("user_", "user"), ("feedback_", "feedback"),
                      ("project_", "project"), ("reference_", "reference")):
        if name.startswith(prefix):
            return t
    return "unknown"


def sync_memory_to_qdrant() -> dict:
    """Walk memory dir, embed changed/new files, upsert into Qdrant. Returns summary."""
    from qdrant_client.http.models import PointStruct

    client = _get_qdrant_client()
    model = _get_model()
    # Sniff dim from a single encode (cheap; sentence-transformers caches model).
    dim = int(model.get_sentence_embedding_dimension())
    _ensure_collection(client, dim)

    state = _load_state()
    old_files: dict[str, dict] = state.get("files", {})
    current = _list_current_files()

    added = updated = deleted = skipped = 0
    errors: list[str] = []
    new_state: dict[str, dict] = {}

    to_embed: list[tuple[str, str, dict]] = []  # (name, embed_text, payload)

    for name, p in current.items():
        try:
            h = _file_hash(p)
        except OSError as e:
            errors.append(f"hash {name}: {e}")
            continue
        prev = old_files.get(name)
        if prev and prev.get("hash") == h:
            new_state[name] = prev
            skipped += 1
            continue
        embed_text, payload = _build_payload(p)
        to_embed.append((name, embed_text, payload))
        new_state[name] = {"hash": h, "point_id": _deterministic_id(name)}
        if prev:
            updated += 1
        else:
            added += 1

    # Batch-encode and upsert
    if to_embed:
        texts = [t for _, t, _ in to_embed]
        try:
            vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            errors.append(f"embed batch: {e}")
            vectors = []
        if len(vectors):
            points = [
                PointStruct(
                    id=_deterministic_id(name),
                    vector=vec.tolist(),
                    payload=payload,
                )
                for (name, _, payload), vec in zip(to_embed, vectors)
            ]
            try:
                client.upsert(collection_name=_collection_name(), points=points, wait=True)
            except Exception as e:
                errors.append(f"qdrant upsert: {e}")

    # Deletions
    for name, prev in old_files.items():
        if name in current:
            continue
        try:
            client.delete(
                collection_name=_collection_name(),
                points_selector={"points": [prev["point_id"]]},
                wait=True,
            )
            deleted += 1
        except Exception as e:
            errors.append(f"delete {name}: {e}")

    _save_state({"files": new_state})
    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
        "collection": _collection_name(),
        "embedding_model": _embedding_model_name(),
    }


def search_memory(query: str, top_k: int = 5) -> list[dict]:
    """Semantic query against the vibemind-memory collection.

    Returns list of {name, source_file, memory_type, description, score, snippet}.
    """
    client = _get_qdrant_client()
    model = _get_model()

    existing = {c.name for c in client.get_collections().collections}
    if _collection_name() not in existing:
        return []

    vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
    response = client.query_points(
        collection_name=_collection_name(),
        query=vec.tolist(),
        limit=max(1, min(int(top_k), 20)),
        with_payload=True,
    )
    out: list[dict] = []
    for h in response.points:
        p = h.payload or {}
        body = p.get("body", "") or ""
        snippet = body.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        out.append({
            "name": p.get("name"),
            "source_file": p.get("source_file"),
            "memory_type": p.get("memory_type"),
            "description": p.get("description"),
            "score": float(h.score),
            "snippet": snippet,
        })
    return out
