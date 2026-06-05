"""Integration tests for cache.py — SQLite-backed embedding and metadata storage.

Uses a real temporary SQLite database per test (via tmp_path fixture).
Validates actual read/write behavior, not mocked interactions.
"""

import time

import numpy as np
import pytest

from kiro_ception.cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Create an EmbeddingCache pointing at a temp directory."""
    monkeypatch.setattr(
        "kiro_ception.cache._get_cache_db_path",
        lambda fp: tmp_path / f"cache_{fp}.db",
    )
    c = EmbeddingCache("test-fingerprint")
    # Force connection/table creation
    _ = c.conn
    yield c
    c.close()


# --- Embedding operations ---


class TestEmbeddingOperations:
    def test_put_and_get_embedding(self, cache):
        emb = np.random.randn(1024).astype(np.float32)
        cache.put_embedding("hash1", emb)

        result = cache.get_embedding("hash1")
        assert result is not None
        np.testing.assert_array_almost_equal(result, emb)

    def test_get_nonexistent_embedding_returns_none(self, cache):
        assert cache.get_embedding("nonexistent") is None

    def test_has_embedding(self, cache):
        emb = np.random.randn(384).astype(np.float32)
        assert cache.has_embedding("hash1") is False
        cache.put_embedding("hash1", emb)
        assert cache.has_embedding("hash1") is True

    def test_put_embeddings_batch(self, cache):
        items = [
            (f"hash_{i}", np.random.randn(1024).astype(np.float32))
            for i in range(10)
        ]
        cache.put_embeddings_batch(items)

        assert cache.embedding_count == 10
        for text_hash, expected_emb in items:
            result = cache.get_embedding(text_hash)
            np.testing.assert_array_almost_equal(result, expected_emb)

    def test_get_embeddings_batch(self, cache):
        items = [
            (f"hash_{i}", np.random.randn(512).astype(np.float32))
            for i in range(5)
        ]
        cache.put_embeddings_batch(items)

        hashes = [h for h, _ in items]
        result = cache.get_embeddings_batch(hashes)

        assert len(result) == 5
        for text_hash, expected_emb in items:
            np.testing.assert_array_almost_equal(result[text_hash], expected_emb)

    def test_get_embeddings_batch_with_missing(self, cache):
        emb = np.random.randn(256).astype(np.float32)
        cache.put_embedding("exists", emb)

        result = cache.get_embeddings_batch(["exists", "missing"])
        assert "exists" in result
        assert "missing" not in result

    def test_get_embeddings_batch_empty_input(self, cache):
        assert cache.get_embeddings_batch([]) == {}

    def test_get_all_embedding_hashes(self, cache):
        items = [(f"hash_{i}", np.random.randn(128).astype(np.float32)) for i in range(3)]
        cache.put_embeddings_batch(items)

        hashes = cache.get_all_embedding_hashes()
        assert hashes == {"hash_0", "hash_1", "hash_2"}

    def test_embedding_count(self, cache):
        assert cache.embedding_count == 0
        cache.put_embedding("h1", np.random.randn(64).astype(np.float32))
        assert cache.embedding_count == 1
        cache.put_embedding("h2", np.random.randn(64).astype(np.float32))
        assert cache.embedding_count == 2

    def test_put_embedding_upsert(self, cache):
        """Putting the same hash twice should overwrite (not duplicate)."""
        emb1 = np.ones(128, dtype=np.float32)
        emb2 = np.zeros(128, dtype=np.float32)

        cache.put_embedding("hash1", emb1)
        cache.put_embedding("hash1", emb2)

        assert cache.embedding_count == 1
        result = cache.get_embedding("hash1")
        np.testing.assert_array_equal(result, emb2)

    def test_preserves_dimensions(self, cache):
        """Embeddings with different dimensions are stored/retrieved correctly."""
        emb_small = np.random.randn(128).astype(np.float32)
        emb_large = np.random.randn(2048).astype(np.float32)

        cache.put_embedding("small", emb_small)
        cache.put_embedding("large", emb_large)

        assert cache.get_embedding("small").shape == (128,)
        assert cache.get_embedding("large").shape == (2048,)


# --- Message operations ---


class TestMessageOperations:
    def _make_message_tuple(self, uuid="msg-1", session_id="sess-1", workspace="/test",
                            timestamp=None, role="user", text="hello world",
                            message_index=0, source="ide", text_hash="hash1"):
        if timestamp is None:
            timestamp = time.time()
        return (uuid, session_id, workspace, timestamp, role, text, message_index, source, text_hash)

    def test_put_and_get_messages(self, cache):
        msg = self._make_message_tuple()
        cache.put_message(*msg)
        cache.conn.commit()

        messages = cache.get_session_messages("sess-1")
        assert len(messages) == 1
        assert messages[0]["uuid"] == "msg-1"
        assert messages[0]["role"] == "user"
        assert messages[0]["searchable_text"] == "hello world"

    def test_put_messages_batch(self, cache):
        msgs = [
            self._make_message_tuple(uuid=f"msg-{i}", message_index=i, text=f"text {i}")
            for i in range(5)
        ]
        cache.put_messages_batch(msgs)

        messages = cache.get_session_messages("sess-1")
        assert len(messages) == 5
        # Should be sorted by message_index
        for i, msg in enumerate(messages):
            assert msg["message_index"] == i

    def test_get_session_messages_sorted_by_index(self, cache):
        # Insert out of order
        msgs = [
            self._make_message_tuple(uuid="msg-3", message_index=3),
            self._make_message_tuple(uuid="msg-1", message_index=1),
            self._make_message_tuple(uuid="msg-2", message_index=2),
        ]
        cache.put_messages_batch(msgs)

        messages = cache.get_session_messages("sess-1")
        indices = [m["message_index"] for m in messages]
        assert indices == [1, 2, 3]

    def test_get_session_messages_empty(self, cache):
        assert cache.get_session_messages("nonexistent") == []

    def test_delete_session_messages(self, cache):
        msgs = [self._make_message_tuple(uuid=f"msg-{i}", message_index=i) for i in range(3)]
        cache.put_messages_batch(msgs)
        assert cache.message_count == 3

        cache.delete_session_messages("sess-1")
        cache.conn.commit()
        assert cache.message_count == 0
        assert cache.get_session_messages("sess-1") == []

    def test_message_count(self, cache):
        assert cache.message_count == 0
        msgs = [
            self._make_message_tuple(uuid="m1", session_id="s1", message_index=0),
            self._make_message_tuple(uuid="m2", session_id="s2", message_index=0),
        ]
        cache.put_messages_batch(msgs)
        assert cache.message_count == 2

    def test_messages_filtered_by_session(self, cache):
        msgs = [
            self._make_message_tuple(uuid="m1", session_id="s1", message_index=0),
            self._make_message_tuple(uuid="m2", session_id="s2", message_index=0),
            self._make_message_tuple(uuid="m3", session_id="s1", message_index=1),
        ]
        cache.put_messages_batch(msgs)

        s1_msgs = cache.get_session_messages("s1")
        assert len(s1_msgs) == 2
        s2_msgs = cache.get_session_messages("s2")
        assert len(s2_msgs) == 1


# --- Session state operations ---


class TestSessionState:
    def test_get_session_mtime_not_tracked(self, cache):
        assert cache.get_session_mtime("unknown-session") is None

    def test_update_and_get_session_state(self, cache):
        mtime = 1717614000.0
        cache.update_session_state("sess-1", mtime)

        result = cache.get_session_mtime("sess-1")
        assert result == mtime

    def test_update_session_state_upsert(self, cache):
        cache.update_session_state("sess-1", 100.0)
        cache.update_session_state("sess-1", 200.0)

        assert cache.get_session_mtime("sess-1") == 200.0

    def test_clear_session_state(self, cache):
        cache.update_session_state("sess-1", 100.0)
        cache.update_session_state("sess-2", 200.0)
        assert cache.indexed_session_count == 2

        cache.clear_session_state()
        assert cache.indexed_session_count == 0
        assert cache.get_session_mtime("sess-1") is None

    def test_indexed_session_count(self, cache):
        assert cache.indexed_session_count == 0
        cache.update_session_state("s1", 100.0)
        cache.update_session_state("s2", 200.0)
        assert cache.indexed_session_count == 2

    def test_session_state_scoped_by_fingerprint(self, tmp_path, monkeypatch):
        """Sessions from a different fingerprint shouldn't count."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{fp}.db",
        )
        cache1 = EmbeddingCache("fingerprint-A")
        cache1.update_session_state("sess-1", 100.0)
        assert cache1.indexed_session_count == 1

        # A different fingerprint on same DB won't see the session
        # (because indexed_session_count filters by fingerprint)
        # But they use different DB files due to _get_cache_db_path
        cache2 = EmbeddingCache("fingerprint-B")
        assert cache2.indexed_session_count == 0

        cache1.close()
        cache2.close()


# --- Execution index operations ---


class TestExecutionIndex:
    def test_put_and_get_exec_files(self, cache):
        cache.put_exec_file("/path/to/exec1.json", "sess-1", 1000.0)
        cache.put_exec_file("/path/to/exec2.json", "sess-1", 2000.0)
        cache.conn.commit()

        files = cache.get_exec_files_for_session("sess-1")
        assert set(files) == {"/path/to/exec1.json", "/path/to/exec2.json"}

    def test_put_exec_files_batch(self, cache):
        items = [
            ("/exec/a.json", "sess-1", 100.0),
            ("/exec/b.json", "sess-1", 200.0),
            ("/exec/c.json", "sess-2", 300.0),
        ]
        cache.put_exec_files_batch(items)

        assert cache.get_exec_index_count() == 3
        assert len(cache.get_exec_files_for_session("sess-1")) == 2
        assert len(cache.get_exec_files_for_session("sess-2")) == 1

    def test_get_all_exec_file_mtimes(self, cache):
        items = [
            ("/exec/a.json", "sess-1", 100.0),
            ("/exec/b.json", "sess-2", 200.0),
        ]
        cache.put_exec_files_batch(items)

        mtimes = cache.get_all_exec_file_mtimes()
        assert mtimes == {"/exec/a.json": 100.0, "/exec/b.json": 200.0}

    def test_get_exec_files_empty_session(self, cache):
        assert cache.get_exec_files_for_session("unknown") == []

    def test_exec_file_upsert(self, cache):
        cache.put_exec_file("/exec/a.json", "sess-1", 100.0)
        cache.put_exec_file("/exec/a.json", "sess-1", 200.0)
        cache.conn.commit()

        assert cache.get_exec_index_count() == 1
        mtimes = cache.get_all_exec_file_mtimes()
        assert mtimes["/exec/a.json"] == 200.0
