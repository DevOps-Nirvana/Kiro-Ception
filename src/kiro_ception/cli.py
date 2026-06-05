"""CLI tool to build/rebuild the embedding index with progress reporting.

Streams sessions one at a time to avoid loading all message text into memory.
Includes single-message fallback on batch failure and context overflow handling.
Uses file-level mtime tracking to skip unchanged sessions entirely.
Now backed by SQLite cache for atomic, incremental persistence.
"""

import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

from .cache import EmbeddingCache
from .config import get_config
from .embeddings import get_embedding_backend
from .indexer import get_memory_limit, select_sessions_within_limit
from .loader import list_all_sessions, load_session_messages
from .models import Source


def _ts():
    """Current timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str):
    """Print a timestamped log line, flushed immediately."""
    print(f"[{_ts()}] {msg}", flush=True)


def _preview(text: str, max_len: int = 80) -> str:
    """Get a short preview of text for logging."""
    clean = text.replace("\n", " ").strip()
    if len(clean) > max_len:
        return clean[:max_len] + "..."
    return clean


def _is_context_overflow_error(error: Exception) -> bool:
    """Detect if an error is likely a context window overflow."""
    msg = str(error).lower()
    return any(keyword in msg for keyword in [
        "context length",
        "too long",
        "maximum context",
        "token limit",
        "exceeds",
        "413",
        "content too large",
    ])


def _embed_single_fallback(backend, text: str, text_hash: str, cache: EmbeddingCache) -> bool:
    """Try to embed a single message. Returns True on success, False on skip."""
    try:
        embedding = backend.encode([text])
        cache.put_embedding(text_hash, embedding[0])
        return True
    except Exception as e:
        if _is_context_overflow_error(e):
            char_count = len(text)
            word_count = len(text.split())
            log(f"      SKIP (context overflow): {char_count} chars, {word_count} words — {_preview(text, 60)}")
            log(f"      Error: {e}")
        else:
            log(f"      SKIP (error): {_preview(text, 60)}")
            log(f"      Error: {e}")
        return False


def _get_session_mtime(session) -> float:
    """Get the mtime for a session (used for change detection)."""
    if session.modified:
        return session.modified.timestamp()
    if session.created:
        return session.created.timestamp()
    return 0.0


def main():
    """Build the embedding index, streaming one session at a time."""
    # Parse --force flag
    force_rebuild = "--force" in sys.argv or "-f" in sys.argv

    print("=" * 60, flush=True)
    print("  Kiro Ception — Index Builder (streaming)", flush=True)
    print("=" * 60, flush=True)
    if force_rebuild:
        print("  ** FORCE REBUILD — ignoring file-level cache **", flush=True)
    print(flush=True)

    # Show config
    config = get_config()
    log(f"Backend:    {config.embedding.backend}")
    log(f"Model:      {config.embedding.model}")
    if config.embedding.backend == "openai-compatible":
        log(f"API base:   {config.embedding.api_base}")
        if config.embedding.dimensions:
            log(f"Dimensions: {config.embedding.dimensions}")
    log(f"Cache dir:  {config.embedding.cache_dir}")
    print(flush=True)

    # Initialize backend
    log("Initializing embedding backend...")
    t_init = time.time()
    backend = get_embedding_backend()
    log(f"Backend created in {time.time() - t_init:.2f}s")

    log("Probing model for embedding dimensions...")
    t_probe = time.time()
    dims = backend.dimensions()
    log(f"Dimensions confirmed: {dims} (probe took {time.time() - t_probe:.2f}s)")
    print(flush=True)

    # Open SQLite cache
    backend_fingerprint = backend.fingerprint()
    cache = EmbeddingCache(backend_fingerprint)
    log(f"Cache database: {cache.db_path}")
    log(f"Cache: {cache.embedding_count} embeddings, {cache.indexed_session_count} indexed sessions")
    print(flush=True)

    # Discover sessions
    log("Discovering conversation sessions...")
    t_disc = time.time()
    all_sessions = list_all_sessions()
    cli_sessions = [s for s in all_sessions if s.source == Source.CLI]
    ide_sessions = [s for s in all_sessions if s.source == Source.IDE]
    log(f"Found {len(all_sessions)} sessions (CLI: {len(cli_sessions)}, IDE: {len(ide_sessions)}) in {time.time() - t_disc:.2f}s")

    # Apply memory limit
    memory_limit = get_memory_limit()
    if memory_limit > 0:
        log(f"Memory limit: {memory_limit / 1024 / 1024:.0f} MB")
    else:
        log("Memory limit: disabled")
    selected, excluded = select_sessions_within_limit(all_sessions, memory_limit)
    if excluded:
        log(f"Excluding {len(excluded)} oldest sessions to stay within memory limit")
    log(f"Will index {len(selected)} sessions")
    print(flush=True)

    if force_rebuild:
        cache.clear_session_state()
        log("Session state cleared (force rebuild)")
        print(flush=True)

    # Stream through sessions one at a time
    batch_size = config.embedding.batch_size
    total_messages = 0
    total_cached = 0
    total_embedded = 0
    total_skipped = 0
    total_unchanged = 0
    total_sessions_processed = 0
    total_sessions = len(selected)
    errors = 0

    log(f"Streaming {total_sessions} sessions (batch size: {batch_size})")
    print(flush=True)

    t0 = time.time()

    for session_idx, session in enumerate(selected):
        # Check if session has changed since last index
        current_mtime = _get_session_mtime(session)
        stored_mtime = cache.get_session_mtime(session.session_id)

        if not force_rebuild and stored_mtime is not None and current_mtime == stored_mtime:
            total_unchanged += 1
            total_sessions_processed += 1
            continue

        # Load messages for this session only
        try:
            messages = load_session_messages(session)
        except Exception as e:
            log(f"  ERROR loading session {session.session_id}: {e}")
            errors += 1
            continue

        if not messages:
            cache.update_session_state(session.session_id, current_mtime)
            total_sessions_processed += 1
            continue

        # Session-level info
        session_chars = sum(len(m.searchable_text) for m in messages)
        session_words = sum(len(m.searchable_text.split()) for m in messages)
        first_msg = messages[0]
        first_preview = _preview(first_msg.searchable_text, 70)

        log(
            f"  session {session_idx + 1}/{total_sessions} | "
            f"id: {session.session_id[:12]}... | "
            f"{len(messages)} msgs, {session_chars:,} chars, {session_words:,} words | "
            f"src: {session.source.value}"
        )
        log(f"    first msg ({first_msg.role}): {first_preview}")

        # Hash and check cache for this session's messages
        texts_to_embed = []
        hashes_to_embed = []
        message_metadata = []

        for msg in messages:
            text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
            total_messages += 1

            # Store metadata for all messages
            message_metadata.append((
                msg.uuid, msg.session_id, msg.workspace,
                msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                msg.message_index, msg.source.value, text_hash,
            ))

            if cache.has_embedding(text_hash):
                total_cached += 1
            else:
                texts_to_embed.append(msg.searchable_text)
                hashes_to_embed.append(text_hash)

        # Store message metadata (always, even if embeddings are cached)
        cache.delete_session_messages(session.session_id)
        cache.put_messages_batch(message_metadata)

        if not texts_to_embed:
            log(f"    all {len(messages)} messages cached, skipping")
            cache.update_session_state(session.session_id, current_mtime)
            total_sessions_processed += 1
            continue

        log(f"    to embed: {len(texts_to_embed)} messages (cached: {len(messages) - len(texts_to_embed)})")

        # Embed uncached messages in batches
        session_embedded = 0
        session_skipped = 0

        for batch_start in range(0, len(texts_to_embed), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            batch_hashes = hashes_to_embed[batch_start:batch_end]

            t_batch = time.time()
            try:
                batch_embeddings = backend.encode(batch_texts)

                # Store embeddings atomically
                items = [(h, batch_embeddings[i]) for i, h in enumerate(batch_hashes)]
                cache.put_embeddings_batch(items)

                session_embedded += len(batch_texts)
                total_embedded += len(batch_texts)

            except Exception as e:
                # Batch failed — fall back to single-message embedding
                batch_time = time.time() - t_batch
                log(f"    BATCH FAILED ({len(batch_texts)} msgs, {batch_time:.1f}s): {e}")
                log(f"    Falling back to single-message embedding...")

                for i, (text, h) in enumerate(zip(batch_texts, batch_hashes)):
                    char_count = len(text)
                    word_count = len(text.split())
                    log(f"      trying msg {batch_start + i + 1}: {char_count:,} chars, {word_count:,} words")

                    if _embed_single_fallback(backend, text, h, cache):
                        session_embedded += 1
                        total_embedded += 1
                    else:
                        session_skipped += 1
                        total_skipped += 1

            batch_time = time.time() - t_batch
            elapsed = time.time() - t0
            rate = total_embedded / elapsed if elapsed > 0 and total_embedded > 0 else 0

            log(
                f"    batch done: {batch_time:.1f}s | "
                f"session progress: {session_embedded}/{len(texts_to_embed)} | "
                f"total embedded: {total_embedded} | "
                f"rate: {rate:.1f} msg/s | "
                f"elapsed: {elapsed:.0f}s"
            )

        if session_skipped > 0:
            log(f"    session complete: {session_embedded} embedded, {session_skipped} skipped")

        # Mark session as indexed
        cache.update_session_state(session.session_id, current_mtime)
        total_sessions_processed += 1

        # Every 50 sessions with work, print a global summary
        if (total_sessions_processed) % 50 == 0 and total_embedded > 0:
            elapsed = time.time() - t0
            print(flush=True)
            log(
                f"  === PROGRESS: {session_idx + 1}/{total_sessions} sessions | "
                f"{total_unchanged} unchanged | "
                f"{total_embedded} embedded | "
                f"{total_cached} cached | "
                f"{total_skipped} skipped | "
                f"{elapsed:.0f}s elapsed ==="
            )
            print(flush=True)

    print(flush=True)
    elapsed_total = time.time() - t0

    # Close cache
    cache.close()

    print(flush=True)
    log("=" * 50)
    log("Index build complete.")
    log(f"  Sessions processed: {total_sessions_processed}/{total_sessions}")
    log(f"  Unchanged (skipped):{total_unchanged}")
    log(f"  Total messages:     {total_messages}")
    log(f"  Already cached:     {total_cached}")
    log(f"  Newly embedded:     {total_embedded}")
    log(f"  Skipped (errors):   {total_skipped}")
    log(f"  Load/embed errors:  {errors}")
    log(f"  Total time:         {elapsed_total:.1f}s")
    if total_embedded > 0:
        log(f"  Embedding rate:     {total_embedded / elapsed_total:.1f} msg/s")
    log(f"  Cache database:     {_get_cache_db_path(backend_fingerprint)}")
    log("=" * 50)


def _get_cache_db_path(fingerprint: str) -> Path:
    """Helper to show the cache path in final summary."""
    config = get_config()
    fp_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
    return config.embedding.cache_path / f"cache_{fp_hash}.db"


if __name__ == "__main__":
    main()
