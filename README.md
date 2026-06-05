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
│  │ Indexer (thread) │─▶│ Search Index    │◀─│ 8 tools               │  │
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
| `search_global_history` | All workspaces | Preferences, patterns across *all* work |
| `search_cli_history` | CLI only | Kiro CLI conversations |
| `search_ide_history` | IDE only | Kiro IDE conversations |

### Management Tools

| Tool | Use Case |
|------|----------|
| `get_indexing_status` | Check indexing progress, rate, ETA, errors |
| `rescan_now` | Trigger immediate rescan for new/changed conversations |
| `force_reindex` | Clear session state and re-read ALL files (heavy, use sparingly) |
| `reload_config` | Re-read config file and apply safe changes without restart |
| `get_config` | Show effective configuration (model, backend, cache stats) |
| `get_instance_role` | Show if this instance is leader or follower (debugging) |

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
search_ide_history(query="deployment", after="2026-06-01")
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

[memory]
fraction = 0.33  # Use 1/3 of RAM
# limit_mb = 512  # Or set explicit limit

[indexing]
throttle_ms = 0  # Sleep between batches (0 = full speed, 500+ = gentle)
rescan_interval_minutes = 10  # Auto-rescan for new messages (0 = disabled)

[server]
leader_port = 19742  # Localhost port for leader-follower communication
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

Use `get_instance_role` to check which role the current instance has.

## CLI Index Builder

For initial indexing or manual rebuilds, use the standalone CLI tool:

```bash
# Normal run (skips unchanged sessions)
uv run kiro-ception-index

# Force full rebuild
uv run kiro-ception-index --force
```

The CLI provides detailed progress reporting with per-session logging, timing, and error handling.

## Memory Management

Kiro Ception limits in-memory index size to prevent excessive memory usage. By default, it uses 1/3 of physical RAM. When the limit is reached, the oldest sessions are excluded from the index (newest sessions are kept).

| Variable | Description | Default |
|----------|-------------|---------|
| `KIRO_RECALL_MEMORY_LIMIT_MB` | Override memory limit in MB | 1/3 of RAM |
| `KIRO_RECALL_NO_MEMORY_LIMIT` | Set to any value to disable limit | - |

## Project Structure

```
kiro-ception/
├── POWER.md                      # Kiro Power manifest + steering
├── mcp.json                      # MCP server config for Power
├── config.default.toml           # Default configuration
├── src/kiro_ception/
│   ├── server.py                 # FastMCP server, tool definitions, search
│   ├── background_indexer.py     # Background thread indexing + status
│   ├── leader.py                 # Leader-follower coordination
│   ├── cache.py                  # SQLite-backed embedding + metadata cache
│   ├── embeddings.py             # Embedding backend abstraction
│   ├── indexer.py                # Memory limits, session selection
│   ├── loader.py                 # Unified loader (CLI + IDE)
│   ├── cli_loader.py             # SQLite parsing for CLI
│   ├── ide_loader.py             # IDE .chat + workspace-sessions + execution logs
│   ├── cli.py                    # CLI index builder tool
│   ├── config.py                 # Configuration management
│   └── models.py                 # Pydantic data models
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
| MCP framework | [FastMCP](https://github.com/jlowin/fastmcp) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## License

[MIT License](LICENSE)
