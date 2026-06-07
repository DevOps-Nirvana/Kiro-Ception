"""FastMCP server for Kiro Ception — MCP tool definitions and entrypoint.

Uses a leader-follower pattern: the first instance becomes the leader
(holds embeddings in RAM, runs indexer, serves HTTP). Additional instances
become followers that forward requests to the leader.
"""

import os
import platform
import time as _time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .background_indexer import get_background_indexer
from .config import get_config as _get_config
from .config import expand_path
from .coordination import get_instance_manager
from .memory import get_memory_limit
from .models import Source
from .search import SearchIndex, get_search_index, handle_search_request, leader_search, search

mcp = FastMCP("kiro-ception")

# Track process start time for uptime reporting
_process_start_time = _time.time()

# --- Initialization ---

_initialized = False


def _initialize():
    """Initialize instance manager: elect leader, start indexer, load search index.

    By default this runs eagerly at startup (in main()). If server.deferred_init
    is True in config, this is deferred to the first tool call instead.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    manager = get_instance_manager()
    role = manager.initialize(search_handler=handle_search_request)

    if role == "leader":
        # Start background indexer (only leader indexes)
        indexer = get_background_indexer()
        indexer.start()
        # Eagerly load the search index from existing cache so the first
        # search doesn't hit an empty matrix (if prior data exists in SQLite)
        get_search_index()

    # Start heartbeat thread (leader writes heartbeat, follower checks liveness)
    manager.start_heartbeat()


def _ensure_initialized():
    """Ensure initialization has run (called at the top of each tool).

    With eager init (default), this is a no-op. With deferred_init=True,
    this triggers initialization on first tool call.
    """
    if not _initialized:
        _initialize()


def _get_current_workspace() -> str | None:
    """Get current workspace path.

    Priority:
    1. Config file: search.workspace_dir (if set)
    2. Environment: KIRO_WORKSPACE (set by Kiro IDE/CLI)
    3. Current working directory (PWD or cwd)
    """
    config = _get_config()
    if config.search.workspace_dir:
        return str(expand_path(config.search.workspace_dir))
    if val := os.environ.get("KIRO_WORKSPACE"):
        return val
    return os.environ.get("PWD") or str(Path.cwd())


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
    _ensure_initialized()
    return search(
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
    source: str = "all",
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
        source: Filter by conversation source. Options: "all" (default), "cli", "ide".
                Only set to "cli" or "ide" if the user explicitly asks to search
                only CLI or only IDE conversations. Otherwise leave as "all".

    Returns:
        Search results with matched messages, scores, workspace, context, pagination
    """
    _ensure_initialized()
    source_filter = None
    if source == "cli":
        source_filter = Source.CLI
    elif source == "ide":
        source_filter = Source.IDE

    return search(
        query=query,
        workspace=None,
        source=source_filter,
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
    Get the current status of the background indexer and search readiness.

    Returns progress, rate, ETA, errors, and whether indexing is active.
    Also reports whether the search index is loaded and ready for queries
    (it can be ready from a prior cache while new indexing is in progress).
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
    status = indexer.status.to_dict()

    # Add search readiness info from the in-memory index
    search_index = get_search_index()
    status["search_ready"] = search_index.message_count > 0
    status["search_message_count"] = search_index.message_count

    # Add database size on disk
    if indexer.cache:
        try:
            from pathlib import Path
            db_path = Path(indexer.cache.db_path) if not isinstance(indexer.cache.db_path, Path) else indexer.cache.db_path
            db_size = db_path.stat().st_size if db_path.exists() else 0
            status["db_size_mb"] = round(db_size / (1024 * 1024), 2)
        except (OSError, TypeError):
            status["db_size_mb"] = None
    else:
        status["db_size_mb"] = None

    # Schema and FTS status
    if indexer.cache:
        from .migrations import get_schema_version
        status["schema_version"] = get_schema_version(indexer.cache.conn)
        try:
            indexer.cache.conn.execute("SELECT 1 FROM messages_fts LIMIT 0")
            status["fts_enabled"] = True
        except Exception:
            status["fts_enabled"] = False
    else:
        status["schema_version"] = None
        status["fts_enabled"] = False

    # Process uptime
    status["uptime_seconds"] = round(_time.time() - _process_start_time, 1)

    # Process memory usage
    import resource
    mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports in bytes, Linux in kilobytes
    if platform.system() == "Darwin":
        mem_mb = mem_bytes / (1024 * 1024)
    else:
        mem_mb = mem_bytes / 1024
    status["memory_used_mb"] = round(mem_mb, 1)
    memory_limit = get_memory_limit()
    if memory_limit > 0:
        status["memory_limit_mb"] = round(memory_limit / (1024 * 1024))
        status["memory_used_percent"] = round(mem_mb / (memory_limit / (1024 * 1024)) * 100, 1)
    else:
        status["memory_limit_mb"] = None
        status["memory_used_percent"] = None

    return status


@mcp.tool()
def rescan(full: bool = False) -> dict:
    """
    Trigger a rescan for new or changed conversations.

    By default, checks all session files for mtime changes and indexes only
    new/modified sessions. This is fast since unchanged files are skipped.

    Set full=True to ignore mtime caching and re-read all sessions from disk.
    This is useful after changing message filter rules or if the index seems
    stale. Existing embeddings are preserved — only truly new text gets
    re-embedded, so even a full rescan is relatively quick.

    Args:
        full: If True, clear mtime cache and re-read all sessions. If False
              (default), only process files whose mtime has changed.

    Returns:
        Confirmation that rescan was triggered
    """
    _ensure_initialized()
    manager = get_instance_manager()

    if not manager.is_leader:
        follower = manager.follower_instance
        if follower:
            try:
                if full:
                    return follower.trigger_reindex()
                return follower.trigger_rescan()
            except Exception:
                return {"error": "Leader unavailable"}

    indexer = get_background_indexer()
    if full:
        indexer.trigger_reindex()
        return {
            "status": "full_rescan_triggered",
            "message": "Full rescan started — all sessions will be re-read (mtime cache cleared). Existing embeddings preserved. Use get_indexing_status to monitor.",
        }
    else:
        indexer.trigger_rescan()
        return {
            "status": "rescan_triggered",
            "message": "Rescan started — checking for new/changed sessions. Use get_indexing_status to monitor.",
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

    config = _get_config()
    indexer = get_background_indexer()

    from .config import CONFIG_FILE

    return {
        "instance": manager.get_role_info(),
        "version": {
            "kiro_ception": __import__("kiro_ception").__version__,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
        },
        "paths": {
            "config_file": str(CONFIG_FILE),
            "config_exists": CONFIG_FILE.exists(),
            "cache_dir": str(config.embedding.cache_path),
            "cache_database": str(indexer.cache.db_path) if indexer.cache else "not initialized",
            "leader_lock": str(config.embedding.cache_path / "leader.lock"),
            "leader_info": str(config.embedding.cache_path / "leader.json"),
        },
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
            "listen_address": f"127.0.0.1:{manager.leader_instance.port}"
            if manager.is_leader and manager.leader_instance
            else f"127.0.0.1:{manager.follower_instance.leader_port}"
            if not manager.is_leader and manager.follower_instance and manager.follower_instance.leader_port
            else None,
        },
        "peers": {
            "enabled": config.peers.enabled,
            "nodes": config.peers.nodes,
            "secret_configured": bool(config.peers.secret),
            "timeout_seconds": config.peers.timeout_seconds,
        },
        "cache": {
            "embedding_count": indexer.cache.embedding_count if indexer.cache else 0,
            "message_count": indexer.cache.message_count if indexer.cache else 0,
            "indexed_sessions": indexer.cache.indexed_session_count if indexer.cache else 0,
        },
    }


@mcp.tool()
def reload_config() -> dict:
    """
    Reload configuration from disk and apply safe changes immediately.

    Re-reads ~/.config/kiro-ception/config.toml and applies changes.
    Safe changes (throttle_ms, batch_size, search defaults) take effect immediately.
    Breaking changes (model, backend, dimensions) are detected but require
    rescan(full=True) to take effect.

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
            deferred.append(f"{c['key']} — requires rescan(full=True) to take effect")
            warnings.append(
                f"{c['key']} changed: {c['old']} → {c['new']}. "
                f"This requires a full rescan. Run rescan(full=True) to rebuild with the new settings."
            )

    return {
        "status": "config_reloaded",
        "changes": changes,
        "applied": applied,
        "deferred": deferred,
        "warnings": warnings,
    }


# --- Conditional tools (registered at import time based on config) ---

_startup_config = _get_config()
if _startup_config.peers.enabled and _startup_config.peers.debug_tool_enabled:

    @mcp.tool()
    def search_peer_history(
        query: str,
        after: str | None = None,
        before: str | None = None,
        context_size: int = 3,
        threshold: float = 0.2,
        max_results: int = 10,
        offset: int = 0,
        source: str = "all",
    ) -> dict:
        """
        Search conversation history on REMOTE PEERS ONLY (excludes local results).

        Debug/advanced tool for querying peer machines directly. Only available
        when peers are enabled and peers.debug_tool_enabled = true in config.

        Args:
            query: Keywords or sentence describing what to find
            after: Filter to messages on/after this date (ISO 8601)
            before: Filter to messages before this date (ISO 8601)
            context_size: Messages to include before AND after each match (default: 3)
            threshold: Minimum similarity 0-1 (default: 0.2)
            max_results: Maximum results to return (default: 10)
            offset: Skip results for pagination (default: 0)
            source: Filter by source — "all" (default), "cli", or "ide"

        Returns:
            Search results from peer machines only (local index is not searched)
        """
        _ensure_initialized()

        from .peers import fan_out_search

        source_filter = None
        if source == "cli":
            source_filter = Source.CLI
        elif source == "ide":
            source_filter = Source.IDE

        peer_request = {
            "query": query,
            "workspace": None,
            "source": source_filter.value if source_filter else None,
            "after": after,
            "before": before,
            "context_size": context_size,
            "threshold": threshold,
            "max_results": max_results,
            "offset": offset,
        }

        peer_responses = fan_out_search(peer_request)
        if not peer_responses:
            return {
                "results": [],
                "query": query,
                "total_matches": 0,
                "hint": "No peer responses received. Check peer connectivity with get_config.",
            }

        # Merge all peer responses
        all_results = []
        for resp in peer_responses:
            all_results.extend(resp.get("results", []))

        # Sort by score descending and paginate
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        total = len(all_results)
        paginated = all_results[offset:offset + max_results]
        has_more = offset + len(paginated) < total

        return {
            "results": paginated,
            "query": query,
            "total_matches": total,
            "offset": offset,
            "has_more": has_more,
            "peers_responded": len(peer_responses),
            "hint": f"Showing {offset + 1}-{offset + len(paginated)} of {total} from {len(peer_responses)} peer(s)." if paginated else "No matches from peers.",
        }


def main():
    """Run the MCP server."""
    config = _get_config()
    if not config.server.deferred_init:
        _initialize()
    mcp.run()


if __name__ == "__main__":
    main()
