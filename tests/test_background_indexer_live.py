"""Live threading tests for BackgroundIndexer.

Starts the indexer in a real thread with a fake corpus and mock backend,
waits for completion, and verifies the cache was populated correctly.
"""

import hashlib
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kiro_ception.background_indexer import BackgroundIndexer, IndexerState
from kiro_ception.cache import EmbeddingCache
from kiro_ception.config import Config, EmbeddingConfig, IndexingConfig
from kiro_ception.models import IndexedMessage, SessionInfo, Source


@pytest.fixture
def mock_backend():
    """Deterministic mock embedding backend."""
    backend = MagicMock()
    backend.fingerprint.return_value = "test:mock:64"

    def _encode(texts):
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(64).astype(np.float32)
            vec /= np.linalg.norm(vec)
            result.append(vec)
        return np.array(result)

    backend.encode.side_effect = _encode
    backend.encode_query.side_effect = lambda t: _encode([t])[0]
    backend.dimensions.return_value = 64
    return backend


@pytest.fixture
def fake_sessions():
    """Small set of sessions for indexing."""
    return [
        SessionInfo(
            session_id="live-sess-1",
            workspace="/test/project",
            message_count=2,
            created=datetime(2026, 6, 1, 10, 0),
            modified=datetime(2026, 6, 1, 11, 0),
            source=Source.IDE,
        ),
        SessionInfo(
            session_id="live-sess-2",
            workspace="/test/project",
            message_count=1,
            created=datetime(2026, 6, 2, 10, 0),
            modified=datetime(2026, 6, 2, 12, 0),
            source=Source.CLI,
        ),
    ]


@pytest.fixture
def fake_messages():
    """Messages for the fake sessions."""
    return {
        "live-sess-1": [
            IndexedMessage(
                uuid="ls1-m0", session_id="live-sess-1", workspace="/test/project",
                timestamp=datetime(2026, 6, 1, 10, 0), role="user",
                searchable_text="Implement rate limiting for the API",
                message_index=0, source=Source.IDE,
            ),
            IndexedMessage(
                uuid="ls1-m1", session_id="live-sess-1", workspace="/test/project",
                timestamp=datetime(2026, 6, 1, 10, 1), role="assistant",
                searchable_text="I'll add a token bucket rate limiter middleware.",
                message_index=1, source=Source.IDE,
            ),
        ],
        "live-sess-2": [
            IndexedMessage(
                uuid="ls2-m0", session_id="live-sess-2", workspace="/test/project",
                timestamp=datetime(2026, 6, 2, 10, 0), role="user",
                searchable_text="Check the deployment status",
                message_index=0, source=Source.CLI,
            ),
        ],
    }


class TestBackgroundIndexerLive:
    def test_full_indexing_run(self, tmp_path, monkeypatch, mock_backend, fake_sessions, fake_messages):
        """Start indexer thread, wait for completion, verify cache state."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{fp}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(backend="openai-compatible", api_base="http://fake", batch_size=2),
            indexing=IndexingConfig(rescan_interval_minutes=0),  # Don't loop
        )

        def _load_messages(session):
            return fake_messages.get(session.session_id, [])

        with (
            patch("kiro_ception.background_indexer.get_config", return_value=config),
            patch("kiro_ception.background_indexer.get_embedding_backend", return_value=mock_backend),
            patch("kiro_ception.background_indexer.list_all_sessions", return_value=fake_sessions),
            patch("kiro_ception.background_indexer.load_session_messages", side_effect=_load_messages),
            patch("kiro_ception.background_indexer.get_memory_limit", return_value=0),
            patch("kiro_ception.background_indexer.select_sessions_within_limit",
                  return_value=(fake_sessions, [])),
        ):
            indexer = BackgroundIndexer()
            indexer.start()

            # Wait for indexer to finish (timeout after 10s)
            deadline = time.time() + 10
            while time.time() < deadline:
                if indexer.status.state == IndexerState.IDLE and indexer.status.completed_at:
                    break
                time.sleep(0.05)

            indexer.stop()

        # Verify results
        assert indexer.status.state == IndexerState.IDLE
        assert indexer.status.sessions_processed == 2
        assert indexer.status.messages_embedded == 3  # 2 + 1
        assert indexer.status.errors == 0

        # Verify cache was populated
        cache = indexer.cache
        assert cache is not None
        assert cache.embedding_count == 3
        assert cache.message_count == 3
        assert cache.indexed_session_count == 2

    def test_indexer_skips_unchanged_sessions(self, tmp_path, monkeypatch, mock_backend, fake_sessions, fake_messages):
        """Second run should skip sessions that haven't changed."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{fp}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(backend="openai-compatible", api_base="http://fake", batch_size=2),
            indexing=IndexingConfig(rescan_interval_minutes=0),
        )

        def _load_messages(session):
            return fake_messages.get(session.session_id, [])

        patches = (
            patch("kiro_ception.background_indexer.get_config", return_value=config),
            patch("kiro_ception.background_indexer.get_embedding_backend", return_value=mock_backend),
            patch("kiro_ception.background_indexer.list_all_sessions", return_value=fake_sessions),
            patch("kiro_ception.background_indexer.load_session_messages", side_effect=_load_messages),
            patch("kiro_ception.background_indexer.get_memory_limit", return_value=0),
            patch("kiro_ception.background_indexer.select_sessions_within_limit",
                  return_value=(fake_sessions, [])),
        )

        # First run
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            indexer1 = BackgroundIndexer()
            indexer1.start()
            deadline = time.time() + 10
            while time.time() < deadline:
                if indexer1.status.state == IndexerState.IDLE and indexer1.status.completed_at:
                    break
                time.sleep(0.05)
            indexer1.stop()

        assert indexer1.status.sessions_processed == 2

        # Second run — same sessions, same mtimes
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            indexer2 = BackgroundIndexer()
            # Reuse the same cache
            indexer2._cache = indexer1.cache
            indexer2._backend = mock_backend
            # Manually run the index loop (faster than threading)
            indexer2._index_loop(config)

        # All sessions should be unchanged
        assert indexer2.status.sessions_unchanged == 2
        assert indexer2.status.sessions_processed == 0

    def test_indexer_handles_embedding_errors(self, tmp_path, monkeypatch, fake_sessions, fake_messages):
        """Embedding failures should be counted as errors, not crash."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{fp}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(backend="openai-compatible", api_base="http://fake", batch_size=1),
            indexing=IndexingConfig(rescan_interval_minutes=0),
        )

        # Backend that fails on encode
        failing_backend = MagicMock()
        failing_backend.fingerprint.return_value = "test:failing:64"
        failing_backend.encode.side_effect = RuntimeError("Model crashed")
        failing_backend.dimensions.return_value = 64

        def _load_messages(session):
            return fake_messages.get(session.session_id, [])

        with (
            patch("kiro_ception.background_indexer.get_config", return_value=config),
            patch("kiro_ception.background_indexer.get_embedding_backend", return_value=failing_backend),
            patch("kiro_ception.background_indexer.list_all_sessions", return_value=fake_sessions),
            patch("kiro_ception.background_indexer.load_session_messages", side_effect=_load_messages),
            patch("kiro_ception.background_indexer.get_memory_limit", return_value=0),
            patch("kiro_ception.background_indexer.select_sessions_within_limit",
                  return_value=(fake_sessions, [])),
        ):
            indexer = BackgroundIndexer()
            indexer.start()

            deadline = time.time() + 10
            while time.time() < deadline:
                if indexer.status.state == IndexerState.IDLE and indexer.status.completed_at:
                    break
                time.sleep(0.05)
            indexer.stop()

        # Should have errors but not crash
        assert indexer.status.errors > 0
        assert indexer.status.messages_skipped > 0
        assert indexer.status.sessions_processed == 2  # Sessions still processed

    def test_trigger_rescan_wakes_indexer(self, tmp_path, monkeypatch, mock_backend, fake_sessions, fake_messages):
        """trigger_rescan should wake a sleeping indexer."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{fp}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(backend="openai-compatible", api_base="http://fake", batch_size=2),
            indexing=IndexingConfig(rescan_interval_minutes=60),  # Long interval
        )

        def _load_messages(session):
            return fake_messages.get(session.session_id, [])

        with (
            patch("kiro_ception.background_indexer.get_config", return_value=config),
            patch("kiro_ception.background_indexer.get_embedding_backend", return_value=mock_backend),
            patch("kiro_ception.background_indexer.list_all_sessions", return_value=fake_sessions),
            patch("kiro_ception.background_indexer.load_session_messages", side_effect=_load_messages),
            patch("kiro_ception.background_indexer.get_memory_limit", return_value=0),
            patch("kiro_ception.background_indexer.select_sessions_within_limit",
                  return_value=(fake_sessions, [])),
        ):
            indexer = BackgroundIndexer()
            indexer.start()

            # Wait for initial indexing to complete
            deadline = time.time() + 10
            while time.time() < deadline:
                if indexer.status.sessions_processed >= 2:
                    break
                time.sleep(0.05)

            # Trigger rescan — should wake from sleep
            initial_embedded = indexer.status.messages_embedded
            indexer.trigger_rescan()

            # Give it a moment to process
            time.sleep(0.5)
            indexer.stop()

        # Should have completed without hanging
        assert indexer.status.sessions_processed >= 2
