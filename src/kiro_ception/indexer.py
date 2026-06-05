"""Semantic embedding index for conversation search."""

import contextlib
import functools
import hashlib
import logging
import os
import pickle
import platform
import subprocess
from datetime import datetime

import numpy as np
from filelock import FileLock

from .config import (
    BYTES_PER_MESSAGE,
    DEFAULT_MEMORY_FRACTION,
    MEMORY_LIMIT_DISABLED_ENV,
    MEMORY_LIMIT_ENV,
    get_config,
)
from .embeddings import EmbeddingBackend, get_embedding_backend
from .loader import list_all_sessions, load_messages_for_sessions
from .models import IndexedMessage, SessionInfo, Source

logger = logging.getLogger(__name__)


def get_physical_memory() -> int:
    """Get physical memory in bytes."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            pass
    return 0


def get_memory_limit() -> int:
    """Get memory limit in bytes for the index."""
    if os.environ.get(MEMORY_LIMIT_DISABLED_ENV):
        return 0

    override = os.environ.get(MEMORY_LIMIT_ENV)
    if override:
        try:
            return int(override) * 1024 * 1024
        except ValueError:
            logger.warning(f"Invalid {MEMORY_LIMIT_ENV} value: {override}")

    config = get_config()
    if config.memory.limit_mb:
        return config.memory.limit_mb * 1024 * 1024

    physical = get_physical_memory()
    fraction = config.memory.fraction or DEFAULT_MEMORY_FRACTION
    return int(physical * fraction) if physical else 0


def select_sessions_within_limit(
    sessions: list[SessionInfo], memory_limit_bytes: int
) -> tuple[list[SessionInfo], list[SessionInfo]]:
    """Select newest sessions that fit within memory limit."""
    if memory_limit_bytes <= 0:
        return sessions, []

    sorted_sessions = sorted(sessions, key=lambda s: s.timestamp_fallback, reverse=True)
    selected, excluded = [], []
    current_bytes = 0

    # Estimate ~10 messages per session if message_count not set
    for session in sorted_sessions:
        msg_count = session.message_count if session.message_count > 0 else 10
        estimated = msg_count * BYTES_PER_MESSAGE
        if current_bytes + estimated <= memory_limit_bytes:
            selected.append(session)
            current_bytes += estimated
        else:
            excluded.append(session)

    return selected, excluded


def _get_sessions_fingerprint() -> str:
    """Get fingerprint of all sessions to detect changes."""
    sessions = list_all_sessions()
    parts = [f"{s.session_id}:{s.timestamp_fallback.timestamp()}" for s in sessions]
    return hashlib.md5("\n".join(sorted(parts)).encode()).hexdigest()


class ConversationIndex:
    """Index of conversation messages with semantic embeddings."""

    def __init__(self):
        self._backend: EmbeddingBackend | None = None
        self._backend_fingerprint: str = ""
        self._messages: list[IndexedMessage] = []
        self._embeddings: np.ndarray | None = None
        self._text_hashes: list[str] = []
        self._sessions_fingerprint: str = ""
        self._excluded_session_count: int = 0
        self._timestamp_order: np.ndarray | None = None
        self._sorted_timestamps: np.ndarray | None = None

    @property
    def backend(self) -> EmbeddingBackend:
        """Get or create the embedding backend."""
        if self._backend is None:
            self._backend = get_embedding_backend()
            self._backend_fingerprint = self._backend.fingerprint()
        return self._backend

    @property
    def model(self):
        """Legacy property for backward compatibility. Returns the backend."""
        return self.backend

    def _get_embedding_dim(self) -> int:
        """Get embedding dimensions from backend."""
        return self.backend.dimensions()

    def _compute_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cache_file(self):
        """Get the cache file path, namespaced by backend fingerprint.

        The cache filename includes a hash of the backend config so that
        changing the model/backend/dimensions automatically invalidates the cache.
        """
        config = get_config()
        fingerprint = self.backend.fingerprint()
        fp_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
        return config.embedding.cache_path / f"embeddings_{fp_hash}.pkl"

    def _get_lock_file(self):
        """Get the lock file path corresponding to the current cache file."""
        cache_file = self._get_cache_file()
        return cache_file.with_suffix(".lock")

    def _load_cache(self) -> dict[str, np.ndarray]:
        """Load cached embeddings for the current backend configuration."""
        cache_file = self._get_cache_file()
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    return {}
                # Validate that cached embeddings have the right dimensions
                if data:
                    sample = next(iter(data.values()))
                    expected_dim = self._get_embedding_dim()
                    if sample.shape[0] != expected_dim:
                        print(
                            f"[kiro-ception] Cache dimension mismatch "
                            f"(cached={sample.shape[0]}, expected={expected_dim}). "
                            f"Discarding cache."
                        )
                        return {}
                return data
        except (OSError, pickle.PickleError, EOFError):
            return {}

    def _save_cache(self, new_embeddings: dict[str, np.ndarray]):
        """Save embeddings atomically with file locking."""
        config = get_config()
        cache_path = config.embedding.cache_path
        cache_file = self._get_cache_file()
        lock_file = self._get_lock_file()

        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            with FileLock(lock_file):
                existing = self._load_cache()
                merged = {**existing, **new_embeddings}
                temp_file = cache_file.with_suffix(f".tmp.{os.getpid()}")
                try:
                    with open(temp_file, "wb") as f:
                        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temp_file, cache_file)
                except OSError:
                    with contextlib.suppress(OSError):
                        temp_file.unlink()
        except OSError:
            pass

    def _build_metadata_indices(self):
        """Build sorted indices for efficient date filtering."""
        if not self._messages:
            self._timestamp_order = np.array([], dtype=np.int64)
            self._sorted_timestamps = np.array([], dtype=np.float64)
            return

        timestamps = np.array([m.timestamp.timestamp() for m in self._messages])
        self._timestamp_order = np.argsort(timestamps)
        self._sorted_timestamps = timestamps[self._timestamp_order]

    def needs_rebuild(self) -> bool:
        """Check if index needs rebuild (new sessions or backend config changed)."""
        if self._backend and self._backend.fingerprint() != self._backend_fingerprint:
            print("[kiro-ception] Backend configuration changed. Full re-index required.")
            return True
        return _get_sessions_fingerprint() != self._sessions_fingerprint

    def build_index(self):
        """Build or update the search index."""
        self._sessions_fingerprint = _get_sessions_fingerprint()
        self._backend_fingerprint = self.backend.fingerprint()

        all_sessions = list_all_sessions()
        memory_limit = get_memory_limit()
        selected, excluded = select_sessions_within_limit(all_sessions, memory_limit)
        self._excluded_session_count = len(excluded)

        if excluded:
            logger.warning(
                f"Memory limit reached: excluding {len(excluded)} oldest sessions"
            )

        # Count sessions by source
        cli_sessions = [s for s in selected if s.source == Source.CLI]
        ide_sessions = [s for s in selected if s.source == Source.IDE]
        print(f"[kiro-ception] Loading messages: {len(cli_sessions)} CLI sessions, {len(ide_sessions)} IDE sessions...")

        self._messages = load_messages_for_sessions(selected)

        if not self._messages:
            print("[kiro-ception] No messages found.")
            self._embeddings = np.array([])
            self._text_hashes = []
            return

        # Count messages by source
        cli_msgs = sum(1 for m in self._messages if m.source == Source.CLI)
        ide_msgs = sum(1 for m in self._messages if m.source == Source.IDE)
        print(f"[kiro-ception] Loaded {len(self._messages)} messages (CLI: {cli_msgs}, IDE: {ide_msgs})")

        embedding_dim = self._get_embedding_dim()
        cache = self._load_cache()
        self._text_hashes = []
        texts_to_embed, indices_to_embed = [], []

        for i, msg in enumerate(self._messages):
            text_hash = self._compute_text_hash(msg.searchable_text)
            self._text_hashes.append(text_hash)
            if text_hash not in cache:
                texts_to_embed.append(msg.searchable_text)
                indices_to_embed.append(i)

        self._embeddings = np.zeros((len(self._messages), embedding_dim), dtype=np.float32)

        # Load cached embeddings
        cached_count = 0
        for i, text_hash in enumerate(self._text_hashes):
            if text_hash in cache:
                self._embeddings[i] = cache[text_hash]
                cached_count += 1

        if cached_count > 0:
            print(f"[kiro-ception] Loaded {cached_count} cached embeddings.")

        new_cache = {}
        if texts_to_embed:
            print(f"[kiro-ception] Embedding {len(texts_to_embed)} new messages...")
            batch_size = 100
            for batch_start in range(0, len(texts_to_embed), batch_size):
                batch_end = min(batch_start + batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[batch_start:batch_end]
                batch_indices = indices_to_embed[batch_start:batch_end]

                batch_embeddings = self.backend.encode(batch_texts)

                for i, idx in enumerate(batch_indices):
                    self._embeddings[idx] = batch_embeddings[i]
                    new_cache[self._text_hashes[idx]] = batch_embeddings[i]

                if batch_end < len(texts_to_embed):
                    print(f"[kiro-ception] Embedded {batch_end}/{len(texts_to_embed)} messages...")

            print(f"[kiro-ception] Embedding complete. Saving cache...")

        if new_cache:
            self._save_cache(new_cache)
            print(f"[kiro-ception] Cache saved ({len(new_cache)} new embeddings).")

        self._build_metadata_indices()
        print(f"[kiro-ception] Index ready.")

    def ensure_index(self):
        """Ensure index is built and up-to-date."""
        if self._embeddings is None or self.needs_rebuild():
            self.build_index()

    def search(
        self,
        query: str,
        workspace: str | None = None,
        source: Source | None = None,
        threshold: float = 0.3,
        max_results: int = 100,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[tuple[IndexedMessage, float]]:
        """Search for messages matching the query."""
        self.ensure_index()

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Date filtering with binary search
        if after is not None or before is not None:
            after_ts = after.timestamp() if after else float("-inf")
            before_ts = before.timestamp() if before else float("inf")
            start_idx = np.searchsorted(self._sorted_timestamps, after_ts, side="left")
            end_idx = np.searchsorted(self._sorted_timestamps, before_ts, side="left")
            candidate_indices = self._timestamp_order[start_idx:end_idx]
        else:
            candidate_indices = np.arange(len(self._messages))

        if len(candidate_indices) == 0:
            return []

        query_embedding = self.backend.encode_query(query)

        candidate_embeddings = self._embeddings[candidate_indices]
        similarities = np.dot(candidate_embeddings, query_embedding)
        sorted_local_indices = np.argsort(similarities)[::-1]

        results = []
        for local_idx in sorted_local_indices:
            score = float(similarities[local_idx])
            if score < threshold:
                break

            global_idx = candidate_indices[local_idx]
            msg = self._messages[global_idx]

            # Apply filters
            if workspace and not msg.workspace.startswith(workspace):
                continue
            if source and msg.source != source:
                continue

            results.append((msg, score))
            if len(results) >= max_results:
                break

        return results

    def get_messages_by_session(self, session_id: str) -> list[IndexedMessage]:
        """Get all messages for a session, sorted by index."""
        return sorted(
            [m for m in self._messages if m.session_id == session_id],
            key=lambda m: m.message_index,
        )

    def get_context_window(
        self, message: IndexedMessage, context_size: int = 3
    ) -> list[IndexedMessage]:
        """Get messages around a matched message."""
        session_messages = self.get_messages_by_session(message.session_id)
        msg_idx = next(
            (i for i, m in enumerate(session_messages) if m.uuid == message.uuid),
            None,
        )
        if msg_idx is None:
            return [message]

        start = max(0, msg_idx - context_size)
        end = min(len(session_messages), msg_idx + context_size + 1)
        return session_messages[start:end]

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def excluded_session_count(self) -> int:
        return self._excluded_session_count


@functools.lru_cache(maxsize=1)
def get_index() -> ConversationIndex:
    """Get or create the global conversation index."""
    return ConversationIndex()
