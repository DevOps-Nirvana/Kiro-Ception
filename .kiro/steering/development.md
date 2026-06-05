# Development Guide

## Language & Runtime

- Python 3.11+ (uses `|` union syntax, `tomllib`, `match` patterns)
- Package manager: `uv` (not pip, not poetry)
- Build system: hatchling
- Sync/install: `uv sync` (installs into `.venv`)
- Run commands: `uv run <command>`

## Code Style

- Formatter/linter: `ruff` (configured in pyproject.toml)
- Line length: 100 characters
- Type hints on all function signatures
- Dataclasses for configuration, Pydantic for data models
- Use `from __future__ import annotations` is NOT needed (3.11+ native)

## Key Conventions

- All print statements during indexing use `flush=True` for immediate visibility
- Log prefix: `[kiro-ception]` for all runtime log messages
- Background threads are daemon threads (auto-die with main process)
- SQLite connections use `check_same_thread=False` for cross-thread access
- File locks use the `filelock` library (cross-platform)
- Config uses `@lru_cache(maxsize=1)` with explicit `cache_clear()` for hot-reload

## Package & Entrypoints

- Package name: `kiro-ception`
- Python module: `kiro_ception`
- Single entrypoint: `kiro-ception` (runs the MCP server)
- No separate CLI indexer — background indexing is built into the MCP server

## Configuration

- User config: `~/.config/kiro-ception/config.toml`
- Default config: `config.default.toml` in repo root
- Cache directory: `~/.cache/kiro-ception/`
- All paths support `~` expansion via `Path.expanduser()`

## Testing Changes

After modifying code:
```bash
uv sync                              # Rebuild package in venv
uv run kiro-ception                  # Test MCP server starts (Ctrl+C to exit)
```

To test with the installed Power, reinstall it from the Kiro Powers panel (or restart Kiro to reload the MCP server).

## Adding New MCP Tools

1. Define the function with `@mcp.tool()` decorator in `server.py`
2. Add `_ensure_initialized()` at the top of the function
3. Handle leader/follower routing if the tool needs indexer/cache access
4. Add corresponding HTTP endpoint in `leader.py` if followers need it
5. Add follower method in `FollowerInstance` class
6. Update POWER.md tool table and trigger scenarios
7. Update README.md management tools table

## Adding New Config Options

1. Add field to the appropriate dataclass in `config.py`
2. Add to `diff_configs()` safe_keys or reindex_keys list
3. Add to `config.default.toml` with comment
4. Add to `get_config` tool response in `server.py`
5. Update README Configuration section

## Message Filtering

Filtering is in `ide_loader.py` via `_SKIP_PREFIXES` and `_should_skip_message()`.
Messages are skipped if they start with system prompt boilerplate:
- `# System Prompt`, `<identity>`, `<system>`, `<instructions>`
- `Follow these instructions for user requests related to spec tasks`
- `## Included Rules`, `<steering-reminder>`
- `CONTEXT TRANSFER:`
- `You are operating in a workspace`

Code blocks are replaced with `[code:language]` placeholders via `_replace_code_blocks()`.

## Embedding Cache

The cache DB is namespaced by backend fingerprint:
`~/.cache/kiro-ception/cache_<hash>.db`

Changing model/backend/dimensions = different hash = different DB file.
Old cache files are preserved for rollback.

## Project Structure

```
src/kiro_ception/
├── server.py              # MCP tools + SearchIndex (read path)
├── background_indexer.py  # Background thread (write path)
├── search_utils.py        # Pure search post-processing functions (testable)
├── cache.py               # SQLite-backed storage
├── config.py              # TOML config loading
├── embeddings.py          # Embedding backend abstraction
├── ide_loader.py          # IDE conversation loader
├── cli_loader.py          # CLI conversation loader
├── loader.py              # Unified loader facade
├── leader.py              # Leader-follower coordination
├── indexer.py             # Memory limit utilities
└── models.py              # Pydantic data models
```
