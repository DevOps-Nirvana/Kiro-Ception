# refactor: Separate MCP server into proxy + engine subprocess

## Summary

Replaces the in-process engine-follower coordination pattern with a two-process architecture:

- **MCP proxy** (`server.py`): Thin FastMCP stdio server. Fast startup, no heavy dependencies loaded. Forwards all tool calls to the engine via HTTP.
- **Engine process** (`engine_main.py`): Standalone subprocess that owns the file lock, HTTP API, background indexer, and in-memory search index.

The MCP proxy spawns the engine on first use and reconnects/respawns as needed. Code fingerprinting detects source changes -- when you `git pull && uv sync`, the next MCP tool call kills the stale engine and respawns with the new code automatically. No manual restart required.

## Motivation

The old single-process model had several pain points:

1. **Startup latency**: Loading sentence-transformers/scipy on the MCP process main thread added 3-5s to every Kiro window open.
2. **Windows Loader Lock**: Native DLL imports from background threads deadlocked against the Windows Loader Lock when multiple threads existed.
3. **Stale code**: After updating, users had to manually restart Kiro or toggle the MCP server to pick up changes.
4. **Follower complexity**: The InstanceManager/EngineInstance/FollowerInstance pattern was ~500 lines of coordination logic with zombie handling, promotion races, and heartbeat threads.

## What Changed

### New files

| File | Purpose |
|------|---------|
| `src/kiro_ception/engine_main.py` | Standalone engine process (HTTP server, indexer, search) |
| `src/kiro_ception/engine_client.py` | Client module: spawn, discover, health-check, and HTTP-forward to engine |
| `src/kiro_ception/dashboard.py` | HTML dashboard extracted from coordination.py |
| `tests/test_engine_client.py` | Unit tests for spawn/discovery/fingerprint/lifecycle |
| `TODO.md` | Tech debt tracking |

### Deleted files

| File | Reason |
|------|--------|
| `src/kiro_ception/coordination.py` | Replaced entirely by engine_main.py + engine_client.py |
| `tests/test_leader.py` | Tested old InstanceManager/EngineInstance/FollowerInstance |
| `tests/test_leader_http.py` | Tested old in-process ThreadingHTTPServer |

### Modified files

| File | Change |
|------|--------|
| `src/kiro_ception/server.py` | Gutted to thin proxy -- all tool calls forward via EngineClient |
| `src/kiro_ception/search.py` | Removed dead follower routing; added workspace decode fallback for legacy indexes |
| `src/kiro_ception/ide_loader.py` | Fixed base64 workspace decode bug (Kiro uses `_` as padding); added `_decode_workspace_dir_name` helper; applied to both legacy and workspace-sessions loaders |
| `src/kiro_ception/cache.py` | FTS search workspace filter matches both decoded and base64-encoded forms |
| `src/kiro_ception/config.py` | Added `ServerConfig` fields (engine_port, heartbeat, log_file) |
| `config.default.toml` | Added `[server]` section documentation |
| `tests/test_tools.py` | Updated to mock EngineClient instead of InstanceManager |
| `tests/test_edge_cases.py` | Replaced FollowerFailover tests with EngineClient port-refresh tests |
| `tests/test_configurable_db_path.py` | Removed coordination.py imports; tests path derivation directly |
| `tests/test_ide_loader.py` | Added tests for `_decode_workspace_dir_name` |
| `tests/test_remaining_gaps.py` | Updated for new architecture |
| `README.md` | Updated architecture description, troubleshooting, data locations |

## Key Design Decisions

### Engine spawn and PID tracking

- `spawn_engine()` does NOT track the Popen PID (on Windows, the venv python.exe shim creates an intermediate process with a different PID than the real engine)
- Instead, polls `engine.json` for a fresh engine (using `started_at` timestamp guard to distinguish from stale leftovers)
- Engine writes `parent_pid` (via `os.getppid()`) to engine.json so the proxy can determine if it spawned the engine or another client did
- Cross-platform: proxy checks `spawned_pid in (pid, parent_pid)` -- matches on `pid` for Unix (no shim) and `parent_pid` for Windows (shim sits between)

### Workspace decode fix

- Kiro IDE encodes workspace paths as URL-safe base64 directory names, using `_` as padding (replacing standard `=`)
- The old decode logic treated ALL `_` as base64 content (substitute for `/`), including the trailing padding characters, producing garbage bytes that failed UTF-8 decode
- Fix: strip trailing `_` before substitution and padding
- Search filter handles both old (base64) and new (decoded) workspace values in the DB without requiring a migration

## Architecture

```
MCP Proxy (stdio)              HTTP (127.0.0.1:19742)              Engine Process
  server.py            ---------------------------------------->     engine_main.py
  engine_client.py     <----------------------------------------     background indexer
  Fast startup                  JSON responses                       search index
  No heavy imports                                                   SQLite cache
                                                                     HTTP API + dashboard
```

### Key behaviors

- **Fingerprint-based reload**: Client computes MD5 of all `*.py` in the package. On `/health`, engine echoes its startup fingerprint. Mismatch = kill + respawn.
- **PID registry**: Engine tracks `X-Follower-PID` headers from all MCP proxies. When all registered PIDs die, engine self-terminates (no orphan processes).
- **Race-safe spawn**: If two proxies try to spawn simultaneously, both poll engine.json. The first engine to acquire the lock writes engine.json; both proxies find it and connect. The loser's spawn hits the lock, prints "already running", and exits.
- **Heartbeat**: Engine refreshes engine.json timestamp periodically so stale info files from crashed engines are detectable.
- **Parent PID**: Engine records `os.getppid()` in engine.json. Proxy matches against its Popen PID to log whether it spawned the engine or found an existing one.

## Testing

710 tests pass (including property-based tests via Hypothesis):

```
$ uv run pytest tests/ -q
710 passed in 56s
```

## Breaking Changes

None for end users. The engine process is an implementation detail -- users interact only with MCP tools, which have the same names and parameters as before.

## How to Test

```bash
git checkout refactor/process-separation
uv sync
uv run pytest tests/ -q        # All 710 tests pass
uv run kiro-ception            # MCP server starts, spawns engine on first tool call
```

To verify fingerprint reload: make a trivial edit to any `src/kiro_ception/*.py` file, then call any MCP tool -- the engine will be killed and respawned with the new code.
