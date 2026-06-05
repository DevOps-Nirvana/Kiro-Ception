"""FastMCP server for Kiro Ception.

Uses a leader-follower pattern: the first instance becomes the leader
(holds embeddings in RAM, runs indexer, serves HTTP). Additional instances
become followers that forward requests to the leader.
"""

import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from .background_indexer import get_background_indexer
from .cache import EmbeddingCache
from .config import get_config
from .embeddings import get_embedding_backend
from .indexer import get_memory_limit
from .leader import get_instance_manager
from .models import Source

mcp = FastMCP("kiro-ception")

# --- Initialization ---
# Deferred: role assignment happens on first tool call to avoid
# blocking MCP server startup. The _ensure_initialized() function
# handles leader/follower setup.

_initialized = False


def _ensure_initialized():
    """Initialize instance manager on first use (deferred from startup)."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    manager = get_instance_manager()
    role = manager.initialize(search_handler=_handle_search_request)

    if role == "leader":
        # Start background indexer (only leader indexes)
        indexer = get_background_indexer()
        indexer.start()


def _handle_search_request(request: dict) -> dict:
    """Handle a search request (used by both leader directly and HTTP API)."""
    return _search(
        query=request.get("query", ""),
        workspace=request.get("workspace"),
        source=Source(request["source"]) if request.get("source") else None,
        after=request.get("after"),
        before=request.get("before"),
        context_size=request.get("context_size", 3),
        threshold=request.get("threshold", 0.2),
        max_results=request.get("max_results", 10),
        offset=request.get("offset", 0),
    )


# --- In-Memory Search Index (leader only) ---

_MIN_REFRESH_INTERVAL = 60  # seconds


class SearchIndex:
    """In-memory numpy matrix for fast vectorized cosine similarity search.

    Loaded from the SQLite cache. Refreshes at most once per minute to pick
    up new embeddings from the background indexer.
    """

    def __init__(self):
        self._embeddings: np.ndarray | None = None  # shape: (N, dims)
        self._metadata: list[dict] = []  # parallel to embeddings rows
        self._last_refresh: float = 0
        self._count: int = 0

    @property
    def message_count(self) -> int:
        return self._count

    def refresh_if_needed(self):
        """Reload from sqlite if more than 60 seconds since last refresh."""
        now = time.time()
        if now - self._last_refresh < _MIN_REFRESH_INTERVAL and self._embeddings is not None:
            return
        self._refresh()

    def _refresh(self):
        """Load all messages + embeddings from sqlite into numpy arrays."""
        indexer = get_background_indexer()
        cache = indexer.cache
        if cache is None:
            return

        # Load all messages with their text_hash
        cursor = cache.conn.execute(
            """SELECT uuid, session_id, workspace, timestamp, role, searchable_text,
                      message_index, source, text_hash
               FROM messages"""
        )
        rows = cursor.fetchall()

        if not rows:
            self._embeddings = None
            self._metadata = []
            self._count = 0
            self._last_refresh = time.time()
            return

        # Collect text hashes and load embeddings in batch
        text_hashes = [row[8] for row in rows]
        embeddings_dict = cache.get_embeddings_batch(text_hashes)

        # Build parallel arrays (only for messages that have embeddings)
        metadata = []
        embedding_list = []

        for row in rows:
            uuid, session_id, ws, ts, role, text, msg_idx, src, text_hash = row
            emb = embeddings_dict.get(text_hash)
            if emb is None:
                continue
            metadata.append({
                "uuid": uuid,
                "session_id": session_id,
                "workspace": ws,
                "timestamp": ts,
                "role": role,
                "content": text[:2000] + "..." if len(text) > 2000 else text,
                "message_index": msg_idx,
                "source": src,
            })
            embedding_list.append(emb)

        if embedding_list:
            self._embeddings = np.vstack(embedding_list)
            self._metadata = metadata
            self._count = len(metadata)
        else:
            self._embeddings = None
            self._metadata = []
            self._count = 0

        self._last_refresh = time.time()

    def search(
        self,
        query_embedding: np.ndarray,
        workspace: str | None = None,
        source: str | None = None,
        after_ts: float | None = None,
        before_ts: float | None = None,
        threshold: float = 0.2,
        max_results: int = 100,
    ) -> list[dict]:
        """Fast vectorized search against the in-memory matrix."""
        self.refresh_if_needed()

        if self._embeddings is None or self._count == 0:
            return []

        # Compute all similarities at once
        similarities = np.dot(self._embeddings, query_embedding)

        # Build candidate mask for filtering
        mask = similarities >= threshold

        # Apply filters
        if workspace or source or after_ts or before_ts:
            for i in range(self._count):
                if not mask[i]:
                    continue
                meta = self._metadata[i]
                if workspace and not meta["workspace"].startswith(workspace):
                    mask[i] = False
                elif source and meta["source"] != source:
                    mask[i] = False
                elif after_ts and meta["timestamp"] < after_ts:
                    mask[i] = False
                elif before_ts and meta["timestamp"] >= before_ts:
                    mask[i] = False

        # Get top results
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = similarities[valid_indices]
        top_order = np.argsort(valid_scores)[::-1][:max_results]

        results = []
        for idx in top_order:
            global_idx = valid_indices[idx]
            score = float(valid_scores[idx])
            meta = self._metadata[global_idx]
            results.append({**meta, "score": score})

        return results


# Global search index singleton (leader only)
_search_index: SearchIndex | None = None


def _get_search_index() -> SearchIndex:
    """Get or create the in-memory search index."""
    global _search_index
    if _search_index is None:
        _search_index = SearchIndex()
    _search_index.refresh_if_needed()
    return _search_index


def _get_current_workspace() -> str | None:
    """Get current workspace from environment or cwd."""
    for var in ("KIRO_PROJECT_DIR", "KIRO_WORKSPACE"):
        if val := os.environ.get(var):
            return val
    return os.environ.get("PWD") or str(Path.cwd())


def _search(
    query: str,
    workspace: str | None,
    source: Source | None,
    after: str | None,
    before: str | None,
    context_size: int,
    threshold: float,
    max_results: int,
    offset: int,
) -> dict:
    """Search implementation with leader/follower routing.

    Leader: searches directly against SQLite cache.
    Follower: forwards to leader via HTTP. On failure, attempts promotion.
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        # Follower: forward to leader
        follower = manager.follower_instance
        if follower:
            try:
                return follower.search({
                    "query": query,
                    "workspace": workspace,
                    "source": source.value if source else None,
                    "after": after,
                    "before": before,
                    "context_size": context_size,
                    "threshold": threshold,
                    "max_results": max_results,
                    "offset": offset,
                })
            except Exception:
                # Leader unreachable — attempt promotion
                if manager.promote_to_leader(_handle_search_request):
                    # We're now leader, start indexer
                    indexer = get_background_indexer()
                    indexer.start()
                    # Fall through to leader search below
                else:
                    return {
                        "results": [],
                        "query": query,
                        "total_matches": 0,
                        "error": "Leader unavailable and could not promote",
                        "hint": "Try again in a moment.",
                    }

    # Leader path: search directly
    return _leader_search(query, workspace, source, after, before,
                          context_size, threshold, max_results, offset)


def _leader_search(
    query: str,
    workspace: str | None,
    source: Source | None,
    after: str | None,
    before: str | None,
    context_size: int,
    threshold: float,
    max_results: int,
    offset: int,
) -> dict:
    """Direct search using in-memory numpy matrix (leader only)."""
    indexer = get_background_indexer()
    backend = indexer.backend

    if backend is None:
        try:
            backend = get_embedding_backend()
        except Exception as e:
            return {
                "results": [],
                "query": query,
                "total_matches": 0,
                "error": f"Backend not ready: {e}",
                "hint": "The embedding backend is still initializing. Try again in a moment.",
            }

    # Ensure in-memory index is loaded/refreshed
    search_index = _get_search_index()
    if search_index.message_count == 0:
        return {
            "results": [],
            "query": query,
            "total_matches": 0,
            "hint": "Index is empty. Indexing may still be in progress — try again shortly.",
        }

    # Parse date filters
    after_dt = _parse_date(after)
    before_dt = _parse_date(before)

    # Embed the query
    try:
        query_embedding = backend.encode_query(query)
    except Exception as e:
        return {
            "results": [],
            "query": query,
            "total_matches": 0,
            "error": f"Failed to embed query: {e}",
            "hint": "The embedding backend may be unavailable.",
        }

    # Fast vectorized search
    scored_results = search_index.search(
        query_embedding=query_embedding,
        workspace=workspace,
        source=source.value if source else None,
        after_ts=after_dt.timestamp() if after_dt else None,
        before_ts=before_dt.timestamp() if before_dt else None,
        threshold=threshold,
        max_results=(offset + max_results) * 3,  # Over-fetch for dedup
    )

    # Deduplicate overlapping context windows
    scored_results = _deduplicate(scored_results, context_size)

    total = len(scored_results)
    paginated = scored_results[offset:offset + max_results]

    # Build context windows for results
    cache = indexer.cache
    results = []
    for match in paginated:
        context = _get_context(cache, match["session_id"], match["message_index"], context_size, match["uuid"]) if cache else []
        results.append({
            "matched_message": {
                "role": match["role"],
                "content": match["content"],
                "timestamp": datetime.fromtimestamp(match["timestamp"]).isoformat(),
                "workspace": match["workspace"],
                "session_id": match["session_id"],
                "uuid": match["uuid"],
                "source": match["source"],
            },
            "score": round(match["score"], 4),
            "context": context,
        })

    has_more = offset + len(results) < total
    hint = _generate_hint(total, offset, len(results), max_results, has_more)

    return {
        "results": results,
        "query": query,
        "total_matches": total,
        "offset": offset,
        "has_more": has_more,
        "hint": hint,
    }


def _deduplicate(results: list[dict], context_size: int) -> list[dict]:
    """Deduplicate results with overlapping context windows."""
    if not results:
        return []

    from collections import defaultdict
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_session[r["session_id"]].append(r)

    dedup_distance = 2 * context_size
    deduplicated = []

    for session_results in by_session.values():
        session_results.sort(key=lambda x: x["message_index"])
        kept = []
        for r in session_results:
            if not kept:
                kept.append(r)
                continue
            if r["message_index"] - kept[-1]["message_index"] <= dedup_distance:
                if r["score"] > kept[-1]["score"]:
                    kept[-1] = r
            else:
                kept.append(r)
        deduplicated.extend(kept)

    deduplicated.sort(key=lambda x: x["score"], reverse=True)
    return deduplicated


def _get_context(cache: EmbeddingCache, session_id: str, message_index: int,
                 context_size: int, match_uuid: str) -> list[dict]:
    """Get context window around a matched message from the cache."""
    session_msgs = cache.get_session_messages(session_id)
    if not session_msgs:
        return []

    # Find the match position
    match_pos = None
    for i, msg in enumerate(session_msgs):
        if msg["uuid"] == match_uuid:
            match_pos = i
            break

    if match_pos is None:
        return []

    start = max(0, match_pos - context_size)
    end = min(len(session_msgs), match_pos + context_size + 1)

    context = []
    for msg in session_msgs[start:end]:
        text = msg["searchable_text"]
        context.append({
            "role": msg["role"],
            "content": text[:2000] + "..." if len(text) > 2000 else text,
            "timestamp": datetime.fromtimestamp(msg["timestamp"]).isoformat(),
            "is_match": msg["uuid"] == match_uuid,
        })
    return context


def _parse_date(value: str | None) -> datetime | None:
    """Parse ISO 8601 date string."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError:
        return None


def _generate_hint(total: int, offset: int, count: int, max_results: int, has_more: bool) -> str:
    if total == 0:
        return "No matches found. Try different search terms or lower the threshold."
    start, end = offset + 1, offset + count
    if has_more:
        return f"Showing {start}-{end} of {total}. Use offset: {offset + max_results} for more."
    if start == 1:
        return f"Showing all {total} matches."
    return f"Showing {start}-{end} of {total} (final page)."


# --- MCP Tools ---


@mcp.tool()
def search_project_history(
    query: str,
    after: str | None = None,
    before: str | None = None,
    context_size: int = 3,
    threshold: float = 0.2,
    max_results: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search conversation history for the CURRENT WORKSPACE only.

    Use this to find workspace-specific context: past decisions, implementation
    details, bugs discussed, architecture choices in this codebase.

    Args:
        query: Keywords or sentence describing what to find
        after: Filter to messages on/after this date (ISO 8601: "2025-01-15")
        before: Filter to messages before this date (ISO 8601)
        context_size: Messages to include before AND after each match (default: 3)
        threshold: Minimum similarity 0-1 (default: 0.2)
        max_results: Maximum results to return (default: 10)
        offset: Skip results for pagination (default: 0)

    Returns:
        Search results with matched messages, scores, context, and pagination info
    """
    return _search(
        query=query,
        workspace=_get_current_workspace(),
        source=None,
        after=after,
        before=before,
        context_size=context_size,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
    )


@mcp.tool()
def search_global_history(
    query: str,
    after: str | None = None,
    before: str | None = None,
    context_size: int = 3,
    threshold: float = 0.2,
    max_results: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search conversation history across ALL WORKSPACES.

    Use this to find cross-project knowledge: user preferences, coding patterns,
    common solutions, and insights from all previous work.

    Args:
        query: Keywords or sentence describing what to find
        after: Filter to messages on/after this date (ISO 8601: "2025-01-15")
        before: Filter to messages before this date (ISO 8601)
        context_size: Messages to include before AND after each match (default: 3)
        threshold: Minimum similarity 0-1 (default: 0.2)
        max_results: Maximum results to return (default: 10)
        offset: Skip results for pagination (default: 0)

    Returns:
        Search results with matched messages, scores, workspace, context, pagination
    """
    return _search(
        query=query,
        workspace=None,
        source=None,
        after=after,
        before=before,
        context_size=context_size,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
    )


@mcp.tool()
def search_cli_history(
    query: str,
    after: str | None = None,
    before: str | None = None,
    context_size: int = 3,
    threshold: float = 0.2,
    max_results: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search Kiro CLI conversation history only.

    Args:
        query: Keywords or sentence describing what to find
        after: Filter to messages on/after this date (ISO 8601)
        before: Filter to messages before this date (ISO 8601)
        context_size: Messages before AND after each match (default: 3)
        threshold: Minimum similarity 0-1 (default: 0.2)
        max_results: Maximum results (default: 10)
        offset: Skip results for pagination (default: 0)

    Returns:
        Search results from CLI conversations only
    """
    return _search(
        query=query,
        workspace=None,
        source=Source.CLI,
        after=after,
        before=before,
        context_size=context_size,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
    )


@mcp.tool()
def search_ide_history(
    query: str,
    after: str | None = None,
    before: str | None = None,
    context_size: int = 3,
    threshold: float = 0.2,
    max_results: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search Kiro IDE conversation history only.

    Args:
        query: Keywords or sentence describing what to find
        after: Filter to messages on/after this date (ISO 8601)
        before: Filter to messages before this date (ISO 8601)
        context_size: Messages before AND after each match (default: 3)
        threshold: Minimum similarity 0-1 (default: 0.2)
        max_results: Maximum results (default: 10)
        offset: Skip results for pagination (default: 0)

    Returns:
        Search results from IDE conversations only
    """
    return _search(
        query=query,
        workspace=None,
        source=Source.IDE,
        after=after,
        before=before,
        context_size=context_size,
        threshold=threshold,
        max_results=max_results,
        offset=offset,
    )


@mcp.tool()
def get_indexing_status() -> dict:
    """
    Get the current status of the background indexer.

    Returns progress, rate, ETA, errors, and whether indexing is active.
    Use this to check if the search index is up-to-date or still building.
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        follower = manager.follower_instance
        if follower:
            try:
                return follower.get_status()
            except Exception:
                return {"error": "Leader unavailable", "state": "unknown"}

    indexer = get_background_indexer()
    return indexer.status.to_dict()


@mcp.tool()
def force_reindex() -> dict:
    """
    Trigger a full reindex of all conversation history.

    Clears the session state cache so ALL sessions are re-read from disk
    and re-processed. Existing embeddings are preserved (only truly new text
    gets re-embedded). Use this after changing filter rules or if the index
    seems corrupted.

    This is the heavy option — prefer rescan_now for picking up new messages.

    Returns:
        Confirmation that full reindex was triggered
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        follower = manager.follower_instance
        if follower:
            try:
                return follower.trigger_reindex()
            except Exception:
                return {"error": "Leader unavailable"}

    indexer = get_background_indexer()
    indexer.trigger_reindex()
    return {
        "status": "force_reindex_triggered",
        "message": "Full reindex started — all sessions will be re-read. Existing embeddings preserved. Use get_indexing_status to monitor.",
    }


@mcp.tool()
def rescan_now() -> dict:
    """
    Trigger an immediate rescan for new or changed conversations.

    Checks all session files for mtime changes and indexes any new/modified
    sessions. Much faster than force_reindex since unchanged files are skipped.
    Use this to pick up recent conversations without waiting for the periodic rescan.

    Returns:
        Confirmation that rescan was triggered
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        follower = manager.follower_instance
        if follower:
            try:
                return follower.trigger_rescan()
            except Exception:
                return {"error": "Leader unavailable"}

    indexer = get_background_indexer()
    indexer.trigger_rescan()
    return {
        "status": "rescan_triggered",
        "message": "Immediate rescan started — checking for new/changed sessions. Use get_indexing_status to monitor.",
    }


@mcp.tool()
def get_config() -> dict:
    """
    Get the current computed configuration for Kiro Ception.

    Shows the effective configuration including defaults, user overrides,
    and computed values like the embedding backend fingerprint.

    Returns:
        Full configuration details
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        follower = manager.follower_instance
        if follower:
            try:
                return follower.get_config()
            except Exception:
                pass  # Fall through to local config

    config = get_config()
    indexer = get_background_indexer()

    return {
        "sources": {
            "cli": {
                "enabled": config.cli.enabled,
                "database_path": str(config.cli.database_path) if config.cli.database_path else None,
            },
            "ide": {
                "enabled": config.ide.enabled,
                "patterns": config.ide.patterns,
            },
        },
        "embedding": {
            "backend": config.embedding.backend,
            "model": config.embedding.model,
            "api_base": config.embedding.api_base or None,
            "dimensions": config.embedding.dimensions,
            "batch_size": config.embedding.batch_size,
            "cache_dir": config.embedding.cache_dir,
            "fingerprint": indexer.backend.fingerprint() if indexer.backend else "not initialized",
        },
        "search": {
            "default_threshold": config.search.default_threshold,
            "default_max_results": config.search.default_max_results,
            "default_context_window": config.search.default_context_window,
        },
        "memory": {
            "fraction": config.memory.fraction,
            "limit_mb": config.memory.limit_mb,
            "effective_limit_mb": round(get_memory_limit() / 1024 / 1024) if get_memory_limit() > 0 else "unlimited",
        },
        "indexing": {
            "throttle_ms": config.indexing.throttle_ms,
            "rescan_interval_minutes": config.indexing.rescan_interval_minutes,
        },
        "server": {
            "leader_port": config.server.leader_port,
        },
        "cache": {
            "database_path": str(indexer.cache.db_path) if indexer.cache else "not initialized",
            "embedding_count": indexer.cache.embedding_count if indexer.cache else 0,
            "message_count": indexer.cache.message_count if indexer.cache else 0,
            "indexed_sessions": indexer.cache.indexed_session_count if indexer.cache else 0,
        },
    }


@mcp.tool()
def get_instance_role() -> dict:
    """
    Get the role of this MCP server instance (leader or follower).

    The leader holds embeddings in RAM and runs the background indexer.
    Followers forward requests to the leader via localhost HTTP.
    Useful for debugging multi-window scenarios.

    Returns:
        Role information including PID and port details
    """
    _ensure_initialized()
    manager = get_instance_manager()
    return manager.get_role_info()


@mcp.tool()
def reload_config() -> dict:
    """
    Reload configuration from disk and apply safe changes immediately.

    Re-reads ~/.config/kiro-ception/config.toml and applies changes.
    Safe changes (throttle_ms, batch_size, search defaults) take effect immediately.
    Breaking changes (model, backend, dimensions) are detected but require
    force_reindex to take effect.

    Returns:
        List of changes detected, what was applied, and any warnings
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        follower = manager.follower_instance
        if follower:
            try:
                return follower.reload_config()
            except Exception:
                return {"error": "Leader unavailable"}

    from .config import reload_config as _reload, diff_configs

    old_config, new_config = _reload()
    changes = diff_configs(old_config, new_config)

    if not changes:
        return {
            "status": "no_changes",
            "message": "Configuration file has not changed.",
        }

    # Categorize changes
    safe_changes = [c for c in changes if c["impact"] == "safe"]
    breaking_changes = [c for c in changes if c["impact"] == "requires_reindex"]

    applied = []
    deferred = []
    warnings = []

    # Apply safe changes to the background indexer
    if safe_changes:
        applied = [c["key"] for c in safe_changes]

    # Flag breaking changes
    if breaking_changes:
        for c in breaking_changes:
            deferred.append(f"{c['key']} — requires force_reindex to take effect")
            warnings.append(
                f"{c['key']} changed: {c['old']} → {c['new']}. "
                f"This requires a full reindex. Run force_reindex to rebuild with the new settings."
            )

    return {
        "status": "config_reloaded",
        "changes": changes,
        "applied": applied,
        "deferred": deferred,
        "warnings": warnings,
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
