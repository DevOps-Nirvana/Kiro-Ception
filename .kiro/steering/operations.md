# Operations Guide

## Embedding Backend

This project uses Ollama with `qwen3-embedding:4b` at 1024 dimensions.
The user's config is at `~/.config/kiro-ception/config.toml`:

```toml
[embedding]
backend = "openai-compatible"
model = "qwen3-embedding:4b"
api_base = "http://localhost:11434/v1"
dimensions = 1024
batch_size = 1
```

Ollama must be running locally for embedding to work. If embedding fails,
check that Ollama is up and the model is pulled (`ollama pull qwen3-embedding:4b`).

## Performance Characteristics

- **First startup** after fresh install: ~35 minutes to index 4300+ sessions
- **Subsequent startups**: <2 seconds (all sessions tracked via mtime)
- **Search latency**: <10ms (numpy dot product against in-memory matrix)
- **Matrix refresh**: every 60 seconds (picks up newly embedded messages)
- **Cold-start**: Eager load from existing SQLite cache on leader init (<1s if prior data exists)
- **Periodic rescan**: every 10 minutes (checks for new/changed session files)
- **Embedding rate**: ~3-5 messages/second with qwen3-embedding:4b

## Cache Location

All persistent data lives in `~/.cache/kiro-ception/`:
- `cache_<hash>.db` — SQLite database (embeddings, metadata, session state, exec index)
- `leader.lock` — File lock for leader election
- `leader.json` — Leader port/PID info for followers

## Troubleshooting

### "Backend not ready" or "still loading" response
- On fresh startup, the index eagerly loads from SQLite cache
- If embeddings exist but metadata hasn't populated yet, the response says "still loading" with embedding count — retry in a few seconds
- The 60-second refresh throttle does NOT apply until the first successful load

### Embedding errors / timeouts
- Check Ollama is running: `ollama ps`
- Very long messages (>50K chars) may timeout — they're processed individually with fallback
- Context overflow errors are logged and the message is skipped

### Stale results (missing recent conversations)
- Periodic rescan runs every 10 minutes
- Use `rescan_now` tool for immediate pickup
- Check `get_indexing_status` to see if indexing is active

### Config changes not taking effect
- Use `reload_config` tool (applies safe changes immediately)
- Model/backend/dimensions changes require `force_reindex`
- The rescan interval is re-read on each loop iteration

### Multiple windows / leader-follower issues
- Use `get_instance_role` to check which process is leader
- If leader dies, next request from a follower auto-promotes it
- Leader lock file: `~/.cache/kiro-ception/leader.lock`
- If lock is stale (process dead), it's auto-cleaned on next startup

## Index Health Checks

Via MCP tools:
- `get_indexing_status` — see state, progress, errors, rate
- `get_config` — verify embedding count, cache path, session count
- `get_instance_role` — confirm leader is active and on expected port

## Forcing a Clean Slate

If something is fundamentally broken:
```bash
rm -rf ~/.cache/kiro-ception/
```
Then restart Kiro — it will rebuild everything from scratch.
