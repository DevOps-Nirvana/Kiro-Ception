# Architecture Guide

## System Design

This is a Kiro MCP Power (kiro-ception) that provides semantic search across conversation history. It runs as a background service launched by each Kiro IDE window.

### Core Principles

- **Non-blocking**: The MCP server responds immediately. All heavy work (indexing, embedding) happens in background threads.
- **Leader-follower**: Only one process holds the search index in RAM. Additional processes proxy to it via localhost HTTP (port 19742 by default).
- **Streaming**: Never load all conversations into memory at once. Process one session at a time, discard text after embedding.
- **Crash-safe**: SQLite with WAL mode. Every embedding and session state update is committed atomically. Ctrl+C loses at most one in-flight message.
- **Incremental**: File mtime tracking skips unchanged sessions. Text hash deduplication avoids re-embedding identical content.
- **Eager cold-start**: Leader loads the SearchIndex from existing SQLite cache immediately on init, before the first search arrives.

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `server.py` | MCP tool definitions, search routing (leader/follower), in-memory numpy SearchIndex |
| `search_utils.py` | Pure post-processing: deduplication, context windows, pagination, date parsing |
| `background_indexer.py` | Background thread: discovers sessions, embeds messages, periodic rescan |
| `leader.py` | File-lock based leader election, HTTP server for followers, failover |
| `peers.py` | Cross-machine federation: fan-out search, result merging, encrypted HTTP |
| `peer_crypto.py` | Argon2id key derivation, AES-256-GCM encrypt/decrypt |
| `cache.py` | SQLite cache: embeddings, message metadata, session state, execution index |
| `embeddings.py` | Backend abstraction: sentence-transformers or OpenAI-compatible API |
| `ide_loader.py` | Loads IDE conversations (legacy .chat + workspace-sessions + execution logs) |
| `cli_loader.py` | Loads CLI conversations from SQLite database |
| `loader.py` | Unified entry point combining CLI + IDE loaders |
| `config.py` | TOML config loading, dataclasses, hot-reload support |
| `indexer.py` | Memory limit utilities (get_memory_limit, select_sessions_within_limit) |
| `models.py` | Pydantic data models (IndexedMessage, SessionInfo, SearchResult, etc.) |

### Data Flow

1. Session discovery: stat files, compare mtimes against session_state table
2. Message extraction: parse JSON, filter boilerplate, replace code blocks with placeholders
3. Embedding: hash text → check cache → call backend if not cached → store in sqlite
4. Search: numpy dot product against in-memory matrix → filter by workspace/source/date → deduplicate → build context windows → paginate

### Search Path (Read)

```
MCP tool call → _search() → _leader_search()
  → SearchIndex.search() (numpy cosine similarity)
  → deduplicate_results() (search_utils.py)
  → format_search_response() (search_utils.py)
    → build_context_window() for each match
    → generate_hint() for pagination
```

### Indexing Path (Write)

```
BackgroundIndexer._run() → _index_loop()
  → list_all_sessions() (loader.py)
  → select_sessions_within_limit() (indexer.py)
  → For each changed session:
    → load_session_messages() → embed → cache.put_embeddings_batch()
  → SearchIndex picks up new data on next refresh (60s)
```

### Cold-Start Behavior

On leader initialization:
1. BackgroundIndexer starts in background thread
2. SearchIndex._refresh() is called eagerly from _ensure_initialized()
3. If SQLite has data from prior run: matrix loads in <1s, first search works immediately
4. If no prior data: SearchIndex detects "loading" state (embeddings exist but metadata not yet populated) and returns informative "still loading" response instead of empty results
5. The 60-second refresh throttle only activates after the first successful load

### Key Design Decisions

- **SQLite over pickle**: Atomic per-row writes, no full-file rewrites, concurrent reader support
- **Execution logs for assistant responses**: Kiro IDE stores user messages in session files but assistant responses in separate execution log files (actionType="say")
- **Code block placeholders**: `[code:python]` preserves language signal without embedding thousands of code tokens
- **60-second matrix refresh**: Balances search freshness vs memory churn
- **10-minute rescan interval**: Picks up new conversations without hammering the filesystem
- **No separate CLI indexer**: Background indexing in the MCP server handles everything; `rescan(full=True)` tool covers manual rebuilds
- **search_utils.py extraction**: Pure functions for post-processing (dedup, pagination, context) are separated from server.py for independent unit testing
- **Peer federation via HTTP fan-out**: Each machine maintains its own index. Peers are queried in parallel and results are merged by score. No shared state, no sync conflicts.
- **Optional AES-256-GCM encryption for peers**: Key derived via Argon2id (memory-hard KDF). Both peers derive the same key from the same passphrase independently — no key exchange protocol needed.
