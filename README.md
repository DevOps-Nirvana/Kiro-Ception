# Kiro Ception

Ever told Kiro "like we discussed yesterday" only to realize... it has no idea?

**Kiro Ception** gives Kiro the memory it's missing.

## The Problem

1. **Sessions Are Isolated**: Each Kiro session starts fresh. Yesterday's architecture discussion? Gone.
2. **Projects Don't Share Knowledge**: Your preferences (testing style, package managers, patterns) aren't remembered across projects.
3. **CLI and IDE Are Separate**: Conversations in Kiro CLI don't connect to Kiro IDE.

**Kiro Ception indexes every Kiro conversation and provides semantic search.** Find discussions by *meaning*, not just keywords.

## Quickstart

### As a Kiro Power (Recommended, IDE only)

1. In Kiro IDE: Powers panel → **Add power from GitHub**
2. Enter: `https://github.com/farleyfarley/kiro-ception`
3. The power activates automatically when you mention "recall", "remember", "remind me", "what did we", or reference past conversations

### Manual MCP Setup (CLI and IDE)

Add to `~/.kiro/settings/mcp.json` (this config is shared by both CLI and IDE):

```json
{
  "mcpServers": {
    "kiro-ception": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/farleyfarley/kiro-ception", "kiro-ception"]
    }
  }
}
```

**Restart Kiro CLI/IDE** after adding. MCP servers are only loaded at startup.

### Verify Installation

In Kiro CLI: `/mcp` should list `kiro-ception`

In Kiro IDE: Check the MCP Servers panel

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Kiro CLI & IDE                                  │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
│  │  CLI: SQLite DB          │    │  IDE: .chat + workspace-sessions │   │
│  │  ~/Library/App Support/  │    │  ~/Library/App Support/Kiro/     │   │
│  │  kiro-cli/data.sqlite3   │    │  User/globalStorage/...          │   │
│  └────────────┬─────────────┘    └─────────────┬────────────────────┘   │
│               └────────────────┬───────────────┘                        │
│                                ▼                                        │
│                    ┌───────────────────────┐                            │
│                    │   Unified Loader      │                            │
│                    └───────────┬───────────┘                            │
└────────────────────────────────┼────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Kiro Ception                                  │
│                                                                         │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────────────────────┐  │
│  │ Background       │  │ In-Memory       │  │ MCP Server            │  │
│  │ Indexer (thread) │─▶│ Search Index    │◀─│ 6 tools               │  │
│  └────────┬─────────┘  │ (numpy matrix)  │  └───────────────────────┘  │
│           │             └─────────────────┘                             │
│           ▼                                                             │
│  ┌──────────────────┐  ┌─────────────────────────────────────────────┐  │
│  │ Embedding        │  │ SQLite Cache (embeddings + metadata +       │  │
│  │ Backend          │  │  session state + execution index)           │  │
│  └──────────────────┘  └─────────────────────────────────────────────┘  │
│           │                                                             │
│   ┌───────┴───────┐                                                    │
│   ▼               ▼                                                     │
│  sentence-    OpenAI-compatible                                         │
│  transformers (Ollama/LM Studio/OpenAI)                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Highlights

- **Non-blocking startup**: Server is immediately queryable. Indexing runs in a background thread.
- **Leader-follower**: Multiple Kiro windows share one leader process (holds RAM index). Followers proxy requests via localhost HTTP. Automatic failover if leader dies.
- **Streaming indexer**: Processes sessions one at a time — never loads all conversations into memory.
- **SQLite cache**: Embeddings, metadata, session state, and execution index all in one atomic database. No more pickle blobs.
- **Incremental**: Only new/changed files are processed. File mtime tracking skips unchanged sessions entirely.

### The Index: Making Search Fast

On startup, Kiro Ception:

1. **Loads** existing embeddings from SQLite cache into a numpy matrix (instant)
2. **Starts** background indexer thread to discover and embed new messages
3. **Serves** searches immediately against whatever is already indexed

The background indexer:
- Discovers sessions from CLI (SQLite) and IDE (legacy .chat + workspace-sessions format)
- Loads execution logs for full assistant responses
- Embeds messages using the configured backend
- Stores results atomically in SQLite (survives Ctrl+C)
- Refreshes the in-memory search matrix every 60 seconds
- **Rescans for new/changed sessions every 10 minutes** (configurable)

## Features

- **Semantic Search**: Find by meaning, not just keywords
- **Dual Source**: Searches both CLI and IDE conversations (legacy + current format)
- **Pluggable Embedding Backends**: Use local sentence-transformers (default) or any OpenAI-compatible API (Ollama, LM Studio, OpenAI, etc.)
- **Background Indexing**: Non-blocking — search works immediately, new results appear as indexing progresses
- **Leader-Follower**: Multiple Kiro windows share one index without duplicating RAM
- **Cross-Machine Peering**: Federate search across computers with optional AES-256-GCM encryption
- **Context Windows**: See surrounding messages for each match
- **Date Filtering**: Filter by time range (ISO 8601)
- **Incremental Indexing**: File-level mtime tracking — only processes changed sessions
- **Automatic Cache Invalidation**: Changing model or backend creates a new cache automatically
- **Throttling**: Configurable `throttle_ms` to reduce GPU load during active use
- **Crash-Safe**: SQLite WAL mode with per-row writes — never loses progress

## MCP Tools

### Search Tools

| Tool | Scope | Use Case |
|------|-------|----------|
| `search_project_history` | Current workspace | Bugs, decisions in *this* codebase |
| `search_global_history` | All workspaces | Preferences, patterns across *all* work. Optional `source` param: `"all"` (default), `"cli"`, or `"ide"` |

### Management Tools

| Tool | Use Case |
|------|----------|
| `get_indexing_status` | Check indexing progress, rate, ETA, errors |
| `rescan` | Pick up new/changed conversations. Use `full=True` to re-read all sessions |
| `reload_config` | Re-read config file and apply safe changes without restart |
| `get_config` | Show effective configuration, file paths, instance role, cache stats |

### Parameters

All search tools accept:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | required | Keywords or sentence to search |
| `after` | none | Filter to messages on/after this date (inclusive). ISO 8601 format. |
| `before` | none | Filter to messages before this date (exclusive). ISO 8601 format. |
| `context_size` | 3 | Messages before AND after each match |
| `threshold` | 0.2 | Minimum similarity (0-1, higher = stricter) |
| `max_results` | 10 | Maximum results to return |
| `offset` | 0 | Skip results (for pagination) |

Additionally, `search_global_history` accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | `"all"` | Filter by source: `"all"`, `"cli"`, or `"ide"`. Only set to `"cli"` or `"ide"` if the user explicitly asks. |

### Workspace Detection

`search_project_history` scopes results to the current workspace. The workspace is determined by (in priority order):

1. `search.workspace_dir` in config (if set)
2. `KIRO_WORKSPACE` environment variable (set automatically by Kiro IDE/CLI)
3. Current working directory

### Date Filtering Examples

```python
# Messages from a specific day
search_project_history(query="auth bug", after="2026-06-01", before="2026-06-02")

# Messages from the past week
search_project_history(query="refactoring", after="2026-05-28")

# Messages in June
search_project_history(query="database", after="2026-06-01", before="2026-07-01")
```

### Response Structure

```json
{
  "results": [
    {
      "matched_message": {
        "role": "assistant",
        "content": "To fix the authentication bug...",
        "timestamp": "2026-06-01T10:30:00",
        "workspace": "/Users/dev/myproject",
        "session_id": "abc123",
        "uuid": "msg-456",
        "source": "ide"
      },
      "score": 0.8542,
      "context": [
        {"role": "user", "content": "How do I fix this auth bug?", "timestamp": "...", "is_match": false},
        {"role": "assistant", "content": "To fix the authentication bug...", "timestamp": "...", "is_match": true}
      ]
    }
  ],
  "query": "authentication bug fix",
  "total_matches": 25,
  "offset": 0,
  "has_more": true,
  "hint": "Showing 1-10 of 25 matches. Use offset: 10 for more."
}
```

## Usage Examples

Just ask naturally:

```
"How did we fix that auth bug?"
"Remind me what we implemented for incidentfox"
"What's my usual approach to error handling?"
"What did we decide about the database schema last week?"
"Where did we leave off with the payment integration?"
```

Or use tools directly:

```python
search_project_history(query="authentication bug fix")
search_global_history(query="React component patterns")
search_global_history(query="deployment", source="ide", after="2026-06-01")
```

## Configuration

Create `~/.config/kiro-ception/config.toml` to customize:

```toml
[sources.cli]
enabled = true
paths = [
    "~/Library/Application Support/kiro-cli/data.sqlite3",
    "~/.local/share/kiro-cli/data.sqlite3",
    "~/AppData/Roaming/kiro-cli/data.sqlite3",
]

[sources.ide]
enabled = true
patterns = [
    "~/Library/Application Support/Kiro/User/globalStorage/kiro.kiroagent/*/*.chat",
    "~/.config/Kiro/User/globalStorage/kiro.kiroagent/*/*.chat",
    "~/AppData/Roaming/Kiro/User/globalStorage/kiro.kiroagent/*/*.chat",
]

[embedding]
# Backend: "sentence-transformers" (default) or "openai-compatible"
backend = "sentence-transformers"
model = "all-MiniLM-L6-v2"
cache_dir = "~/.cache/kiro-ception"
batch_size = 16  # Messages per embedding request

[search]
default_threshold = 0.2
default_max_results = 10
default_context_window = 3
# workspace_dir = "~/projects/my-app"  # Override workspace for search_project_history

[memory]
fraction = 0.33  # Use 1/3 of RAM
# limit_mb = 512  # Or set explicit limit

[indexing]
throttle_ms = 0  # Sleep between batches (0 = full speed, 500+ = gentle)
rescan_interval_minutes = 10  # Auto-rescan for new messages (0 = disabled)

[server]
leader_port = 19742  # Localhost port for leader-follower communication

[peers]
# Cross-machine federation (disabled by default)
enabled = false
nodes = ["other-machine:19742"]  # Peers to search
secret = "shared-passphrase"     # Encrypt traffic (optional, omit for plaintext)
timeout_seconds = 5
```

### Embedding Backends

Kiro Ception supports two embedding backends:

#### sentence-transformers (default)

Uses the [sentence-transformers](https://www.sbert.net/) library to run embedding models locally in-process. No external service needed — the model is downloaded once from Hugging Face and cached.

```toml
[embedding]
backend = "sentence-transformers"
model = "all-MiniLM-L6-v2"  # 22M params, 384 dims, fast on CPU
```

#### openai-compatible (Ollama, LM Studio, OpenAI, etc.)

Connects to any service exposing an OpenAI-compatible `/v1/embeddings` endpoint. This enables GPU-accelerated models via Ollama or LM Studio locally, or hosted providers like OpenAI remotely.

| Setting | Description | Example |
|---------|-------------|---------|
| `backend` | Must be `"openai-compatible"` | |
| `model` | Model name to pass in the API request | `"qwen3-embedding:4b"` |
| `api_base` | Base URL of the API (without `/embeddings`) | `"http://localhost:11434/v1"` |
| `api_key` | API key (optional for local, required for hosted) | `"sk-..."` |
| `dimensions` | Output embedding dimensions (optional, omit for model default) | `1024` |
| `batch_size` | Messages per API request (1 for large models, 16+ for small) | `1` |

**Ollama example** (local GPU, no API key needed):

```toml
[embedding]
backend = "openai-compatible"
model = "qwen3-embedding:4b"
api_base = "http://localhost:11434/v1"
dimensions = 1024
batch_size = 1
```

Setup: `ollama pull qwen3-embedding:4b`

**LM Studio example** (local GPU):

```toml
[embedding]
backend = "openai-compatible"
model = "nomic-embed-text-v1.5"
api_base = "http://localhost:1234/v1"
```

**OpenAI example** (remote, requires API key):

```toml
[embedding]
backend = "openai-compatible"
model = "text-embedding-3-large"
api_base = "https://api.openai.com/v1"
api_key = "sk-..."
dimensions = 1024
```

#### Switching backends or models

When you change the backend, model, or dimensions in config, Kiro Ception automatically detects the change and creates a new cache. Existing embeddings in the old cache are preserved (in case you switch back).

No manual intervention is needed — just update the config and restart.

#### Which model should I use?

| Model | Backend | Dims | Speed | Quality | Notes |
|-------|---------|------|-------|---------|-------|
| `all-MiniLM-L6-v2` | sentence-transformers | 384 | Fast (CPU) | Good | Default, no setup needed |
| `nomic-embed-text` | Ollama | 768 | Fast (GPU) | Better | 137M params, 8K context |
| `mxbai-embed-large` | Ollama | 1024 | Medium | Better | 335M params |
| `qwen3-embedding:4b` | Ollama | up to 4096 | Medium | Very good | 4B params, 40K context |
| `qwen3-embedding:8b` | Ollama | up to 4096 | Slower | Best | 8B params, 32K context, #1 MTEB |

If you have a GPU and Ollama installed, `qwen3-embedding:4b` with `dimensions = 1024` is a great balance of quality and speed.

## Multi-Instance (Leader-Follower)

When multiple Kiro windows are open, each spawns its own MCP server process. To avoid duplicating RAM and indexing work:

- **First instance** becomes the **leader**: holds the numpy search matrix in RAM, runs the background indexer, listens on `localhost:19742`
- **Additional instances** become **followers**: forward all requests to the leader via HTTP, use zero RAM for embeddings
- **Automatic failover**: if the leader dies, the next request from a follower promotes it to leader

Use `get_config` to check the current instance role and leader port.

## Cross-Machine Peering

You can federate search across multiple computers. Each machine maintains its own independent index and serves its own conversations. When peering is enabled, search queries are fanned out to remote peers in parallel and results are merged.

### Setup

On **both machines**, add to `~/.config/kiro-ception/config.toml`:

```toml
[peers]
enabled = true
nodes = ["other-machine:19742"]  # The other machine's address
secret = "our-shared-passphrase"  # Same on both (optional, for encryption)
timeout_seconds = 5
```

Machine A's config points to Machine B, and vice versa.

### Networking

Peers communicate via HTTP on the leader port (default 19742). The machines need to be able to reach each other on that port. Options:

- **Tailscale** (recommended): Zero-config VPN. Use Tailscale hostnames (e.g., `workpc.tail1234.ts.net:19742`). No firewall rules needed.
- **Same LAN**: Use local IPs (e.g., `192.168.1.50:19742`). May need firewall rules.
- **WireGuard/VPN**: Any VPN that gives both machines reachable IPs.

### Encryption

When `secret` is set, all peer traffic is encrypted with AES-256-GCM. The key is derived from the passphrase using Argon2id (memory-hard, brute-force resistant). Both peers must use the same passphrase.

- **No secret**: plaintext HTTP (fine on Tailscale or trusted LAN)
- **With secret**: encrypted payloads, peers with wrong/missing secret get 401 rejected

### Behavior

- Search queries are fanned out to all configured peers in parallel
- Results are merged with local results, deduplicated by message UUID, sorted by score
- If a peer is unreachable or times out, local results still return (graceful degradation)
- Each machine indexes only its own local conversation files
- Peers don't need the same embedding model — results are matched by score regardless of how they were embedded

## Memory Management

Kiro Ception limits in-memory index size to prevent excessive memory usage. By default, it uses 1/3 of physical RAM. When the limit is reached, the oldest sessions are excluded from the index (newest sessions are kept).

Configure via `~/.config/kiro-ception/config.toml`:

```toml
[memory]
fraction = 0.33    # Use 1/3 of RAM (default)
# limit_mb = 2048  # Or set an explicit limit in MB
# limit_mb = 0     # Set to 0 to disable the memory limit entirely
```

## Troubleshooting

### Search returns no results or "still loading"

On first startup (or after clearing the cache), indexing takes ~35 minutes for a large history. Use `get_indexing_status` to check progress. Search works immediately for already-indexed content.

If you see "still loading" with an embedding count, the data exists but hasn't been loaded into the search matrix yet — retry in a few seconds.

### Stale results (missing recent conversations)

The periodic rescan runs every 10 minutes. To pick up conversations immediately, use `rescan`. If sessions are still missing, use `rescan(full=True)` to re-read all files ignoring the mtime cache.

### "Failed to embed query" error

The embedding backend (Ollama/LM Studio/OpenAI) is unreachable or crashed. Check:
- Ollama: `ollama ps` (should show a running model)
- If the model was unloaded: `ollama pull <model-name>` to reload it
- If using a remote API: check network connectivity and API key validity

Search will automatically recover once the backend is back — no restart needed.

### "Leader unavailable and could not promote"

This means all leader-follower communication failed. Usually happens if:
- The leader process crashed but left a stale lock file
- Port 19742 is blocked by a firewall or another process

Fix: restart Kiro. The stale lock is auto-cleaned on the next startup. If the port is in use by another process, set a different port in config:
```toml
[server]
leader_port = 19743
```

### High error count during indexing

Check `get_indexing_status` — the `last_error` field shows what went wrong. Common causes:
- Ollama running out of memory on very long messages (they're retried individually, then skipped)
- Corrupt session files on disk (skipped, other sessions continue)
- Network timeouts to remote embedding API

Errors don't stop indexing — the indexer skips problematic messages and continues. Use `get_indexing_status` to see `messages_skipped` count.

### Multiple Kiro windows behaving oddly

Only one window is the "leader" (holds RAM index, runs indexer). Others are "followers" that proxy to it. Use `get_config` to see the `instance` section showing which role the current window has.

If a window is stuck as a follower with a dead leader: close all Kiro windows, delete `~/.cache/kiro-ception/leader.lock`, and reopen.

### Peer search not returning remote results

Check `get_config` — the `peers` section shows if peering is enabled and what nodes are configured. Common issues:
- Peer machine is offline or Kiro isn't running on it
- Firewall blocking port 19742 (try `curl http://peer-address:19742/health`)
- Mismatched `secret` between machines (one encrypted, other not → 401 rejection)
- Peer not yet indexed (just started, still building its index)

Peers that timeout or fail are silently skipped — local results always return regardless.

### Resetting the cache (nuclear option)

If the database is corrupt, search returns garbage, or you've changed embedding dimensions and want a clean start:

```bash
rm -rf ~/.cache/kiro-ception/
```

Then restart Kiro. The index will rebuild from scratch (all conversations are re-read and re-embedded). This takes ~35 minutes for 4000+ sessions.

### Finding cache/config paths

Use `get_config` — the `paths` section shows the exact file locations for your system.

## Project Structure

```
kiro-ception/
├── POWER.md                      # Kiro Power manifest + steering
├── mcp.json                      # MCP server config for Power
├── config.default.toml           # Default configuration
├── src/kiro_ception/
│   ├── server.py                 # FastMCP server, tool definitions, search
│   ├── background_indexer.py     # Background thread indexing + status
│   ├── search_utils.py           # Pure search post-processing functions
│   ├── peers.py                  # Cross-machine federation (fan-out, merge)
│   ├── peer_crypto.py            # Argon2id + AES-256-GCM encryption
│   ├── leader.py                 # Leader-follower coordination
│   ├── cache.py                  # SQLite-backed embedding + metadata cache
│   ├── embeddings.py             # Embedding backend abstraction
│   ├── indexer.py                # Memory limits, session selection
│   ├── loader.py                 # Unified loader (CLI + IDE)
│   ├── cli_loader.py             # SQLite parsing for CLI
│   ├── ide_loader.py             # IDE .chat + workspace-sessions + execution logs
│   ├── config.py                 # Configuration management
│   └── models.py                 # Pydantic data models
├── tests/                        # Comprehensive test suite (270 tests)
├── pyproject.toml
└── LICENSE
```

## Technical Details

| Component | Technology |
|-----------|------------|
| Embedding (default) | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via sentence-transformers |
| Embedding (optional) | Any OpenAI-compatible API (Ollama, LM Studio, OpenAI, etc.) |
| Vector search | Cosine similarity via NumPy dot product (in-memory matrix) |
| Cache/storage | SQLite with WAL mode (atomic, crash-safe) |
| Inter-process | Leader-follower via localhost HTTP + file lock |
| Peer encryption | AES-256-GCM with Argon2id key derivation (via `cryptography` + `argon2-cffi`) |
| MCP framework | [FastMCP](https://github.com/jlowin/fastmcp) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## License

[MIT License](LICENSE)

Some logic and the concept was inspired by [kiro-total-recall](https://github.com/danilop/kiro-total-recall) by Danilo Poccia.
