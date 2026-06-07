"""Tests for schema migrations and hybrid FTS search."""

import sqlite3
import time
from unittest.mock import patch

import numpy as np
import pytest

from kiro_ception.cache import EmbeddingCache
from kiro_ception.migrations import (
    CURRENT_SCHEMA_VERSION,
    _execute_with_retry,
    _migrate_v1_to_v2,
    _migrate_v2_to_v3,
    get_schema_version,
    run_migrations,
)


class TestMigrationFramework:
    """Test the migration versioning and execution framework."""

    def test_fresh_db_gets_current_version(self, tmp_path, monkeypatch):
        """A brand new database should end up at CURRENT_SCHEMA_VERSION."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))
        cache = EmbeddingCache("test-fresh")
        _ = cache.conn  # Trigger init
        version = get_schema_version(cache.conn)
        assert version == CURRENT_SCHEMA_VERSION
        cache.close()

    def test_v1_db_migrates_to_current(self, tmp_path, monkeypatch):
        """A pre-migration database (v1) should be migrated to current."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))

        # Create a v1-style database manually (no schema_version, no FTS)
        db_path = tmp_path / "cache_test.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE messages (
                uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                source TEXT NOT NULL,
                text_hash TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE session_state (
                session_id TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                fingerprint TEXT NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE TABLE execution_index (
                exec_file_path TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_mtime REAL NOT NULL
            );
        """)
        conn.execute("INSERT INTO meta (key, value) VALUES ('fingerprint', 'test')")
        conn.commit()

        # Verify it's detected as v1
        assert get_schema_version(conn) == 1

        # Run migrations
        run_migrations(conn)

        # Should now be at current version
        assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION

        # FTS table should exist
        conn.execute("SELECT 1 FROM messages_fts LIMIT 0")
        conn.close()

    def test_get_schema_version_no_meta_table(self, tmp_path):
        """A DB without a meta table should return version 0."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        assert get_schema_version(conn) == 0
        conn.close()

    def test_get_schema_version_meta_without_key(self, tmp_path):
        """A DB with meta table but no schema_version key should return 1."""
        db_path = tmp_path / "v1.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta (key, value) VALUES ('fingerprint', 'test')")
        conn.commit()
        assert get_schema_version(conn) == 1
        conn.close()

    def test_migrations_are_idempotent(self, tmp_path, monkeypatch):
        """Running migrations multiple times should be safe."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))
        cache = EmbeddingCache("test-idempotent")
        _ = cache.conn

        # Run migrations again — should not error
        run_migrations(cache.conn)
        run_migrations(cache.conn)

        assert get_schema_version(cache.conn) == CURRENT_SCHEMA_VERSION
        cache.close()


class TestFTSMigrationWithData:
    """Test that the v1→v2 migration populates FTS from existing messages."""

    def test_existing_messages_are_indexed_in_fts(self, tmp_path, monkeypatch):
        """Messages that exist before migration should be searchable via FTS."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))

        # Create a v1 database with messages
        db_path = tmp_path / "cache_abc123.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE messages (
                uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                source TEXT NOT NULL,
                text_hash TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE session_state (
                session_id TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                fingerprint TEXT NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE TABLE execution_index (
                exec_file_path TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_mtime REAL NOT NULL
            );
        """)
        conn.execute("INSERT INTO meta (key, value) VALUES ('fingerprint', 'test')")
        # Insert test messages
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("u1", "s1", "/ws", time.time(), "user", "Deploy to kubernetes with helm", 0, "ide", "h1"),
                ("u2", "s1", "/ws", time.time(), "assistant", "Use helm install command", 1, "ide", "h2"),
                ("u3", "s2", "/ws", time.time(), "user", "Fix the authentication bug", 0, "ide", "h3"),
            ],
        )
        conn.commit()

        # Run migration
        run_migrations(conn)

        # Verify FTS search works on pre-existing messages
        cursor = conn.execute("""
            SELECT m.uuid FROM messages_fts
            JOIN messages m ON messages_fts.rowid = m.rowid
            WHERE messages_fts MATCH 'kubernetes'
        """)
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0][0] == "u1"

        cursor = conn.execute("""
            SELECT m.uuid FROM messages_fts
            JOIN messages m ON messages_fts.rowid = m.rowid
            WHERE messages_fts MATCH 'authentication'
        """)
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0][0] == "u3"
        conn.close()


class TestV2ToV3Migration:
    """Test the v2→v3 migration that adds content_tier and tool_name columns."""

    def _create_v2_db(self, tmp_path):
        """Create a v2 database (has FTS, no content_tier/tool_name)."""
        db_path = tmp_path / "cache_v2test.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE messages (
                uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                source TEXT NOT NULL,
                text_hash TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE session_state (
                session_id TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                fingerprint TEXT NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE TABLE execution_index (
                exec_file_path TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_mtime REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                searchable_text,
                content='messages',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );
        """)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', '2')"
        )
        conn.commit()
        return conn

    def test_v2_db_migrates_to_v3(self, tmp_path):
        """A v2 database should be migrated to v3 with new columns."""
        conn = self._create_v2_db(tmp_path)
        assert get_schema_version(conn) == 2

        run_migrations(conn)

        assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION

        # Verify content_tier column exists with correct default
        conn.execute(
            "INSERT INTO messages (uuid, session_id, workspace, timestamp, role, "
            "searchable_text, message_index, source, text_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("u1", "s1", "/ws", time.time(), "user", "test text", 0, "ide", "h1"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT content_tier, tool_name FROM messages WHERE uuid = 'u1'"
        ).fetchone()
        assert row[0] == "conversation"  # DEFAULT value
        assert row[1] is None  # NULL for tool_name
        conn.close()

    def test_content_tier_column_added(self, tmp_path):
        """Migration adds content_tier column with NOT NULL DEFAULT 'conversation'."""
        conn = self._create_v2_db(tmp_path)

        # Insert a message before migration
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("pre", "s1", "/ws", time.time(), "user", "existing msg", 0, "ide", "h0"),
        )
        conn.commit()

        _migrate_v2_to_v3(conn)

        # Existing messages should default to 'conversation'
        row = conn.execute(
            "SELECT content_tier FROM messages WHERE uuid = 'pre'"
        ).fetchone()
        assert row[0] == "conversation"
        conn.close()

    def test_tool_name_column_added(self, tmp_path):
        """Migration adds nullable tool_name column."""
        conn = self._create_v2_db(tmp_path)
        _migrate_v2_to_v3(conn)

        # Insert with explicit tool_name
        conn.execute(
            "INSERT INTO messages (uuid, session_id, workspace, timestamp, role, "
            "searchable_text, message_index, source, text_hash, content_tier, tool_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("t1", "s1", "/ws", time.time(), "assistant", "summary", 1, "ide", "ht",
             "tool_context", "readFile"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT tool_name FROM messages WHERE uuid = 't1'"
        ).fetchone()
        assert row[0] == "readFile"
        conn.close()

    def test_content_tier_index_created(self, tmp_path):
        """Migration creates idx_messages_content_tier index."""
        conn = self._create_v2_db(tmp_path)
        _migrate_v2_to_v3(conn)

        # Verify the index exists
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_content_tier'"
        ).fetchall()
        assert len(indexes) == 1
        conn.close()

    def test_migration_handles_column_already_exists(self, tmp_path):
        """Running migration twice does not error on duplicate columns."""
        conn = self._create_v2_db(tmp_path)

        # Run migration twice — second should be graceful
        _migrate_v2_to_v3(conn)
        _migrate_v2_to_v3(conn)  # Should not raise

        # Verify columns still work
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages'"
        ).fetchone()
        assert "content_tier" in row[0]
        assert "tool_name" in row[0]
        conn.close()

    def test_migration_handles_locked_database_with_retry(self, tmp_path):
        """Migration retries on database lock with exponential backoff."""
        call_count = [0]

        class LockingConnection:
            """Wrapper that simulates a locked database on first attempt."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args, **kwargs):
                if "ALTER TABLE" in sql and "content_tier" in sql and call_count[0] < 1:
                    call_count[0] += 1
                    raise sqlite3.OperationalError("database is locked")
                return self._conn.execute(sql, *args, **kwargs)

            def commit(self):
                return self._conn.commit()

        conn = self._create_v2_db(tmp_path)
        wrapper = LockingConnection(conn)

        with patch("kiro_ception.migrations.time.sleep") as mock_sleep:
            _execute_with_retry(
                wrapper,
                "ALTER TABLE messages ADD COLUMN content_tier TEXT NOT NULL DEFAULT 'conversation'",
                "content_tier column",
            )
            # Should have retried once with 1s delay (2^0 = 1)
            mock_sleep.assert_called_once_with(1)

        # Verify the column was added on retry
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages'"
        ).fetchone()
        assert "content_tier" in row[0]
        conn.close()

    def test_migration_raises_after_max_lock_retries(self, tmp_path):
        """Migration raises after exhausting lock retry attempts."""

        class AlwaysLockedConnection:
            """Wrapper that always raises 'database is locked'."""

            def execute(self, sql, *args, **kwargs):
                raise sqlite3.OperationalError("database is locked")

        wrapper = AlwaysLockedConnection()

        with patch("kiro_ception.migrations.time.sleep"):
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                _execute_with_retry(
                    wrapper,
                    "ALTER TABLE messages ADD COLUMN content_tier TEXT NOT NULL DEFAULT 'conversation'",
                    "content_tier column",
                )

    def test_existing_messages_get_default_tier(self, tmp_path):
        """Existing messages in a v2 DB default to 'conversation' tier after migration."""
        conn = self._create_v2_db(tmp_path)

        # Insert messages before migration
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("m1", "s1", "/ws", time.time(), "user", "Hello", 0, "ide", "h1"),
                ("m2", "s1", "/ws", time.time(), "assistant", "Hi back", 1, "ide", "h2"),
            ],
        )
        conn.commit()

        _migrate_v2_to_v3(conn)

        rows = conn.execute(
            "SELECT uuid, content_tier, tool_name FROM messages ORDER BY uuid"
        ).fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row[1] == "conversation"
            assert row[2] is None
        conn.close()


class TestFTSSearch:
    """Test the FTS search functionality via EmbeddingCache."""

    @pytest.fixture
    def cache_with_data(self, tmp_path, monkeypatch):
        """Create a cache with test messages for FTS searching."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))
        cache = EmbeddingCache("test-fts")
        now = time.time()
        messages = [
            ("u1", "s1", "/workspace/project-a", now - 100, "user",
             "How do I deploy to kubernetes with helm charts?", 0, "ide", "h1"),
            ("u2", "s1", "/workspace/project-a", now - 90, "assistant",
             "You can use helm install to deploy your chart to the cluster", 1, "ide", "h2"),
            ("u3", "s2", "/workspace/project-a", now - 50, "user",
             "Fix the authentication bug in the login flow", 0, "ide", "h3"),
            ("u4", "s2", "/workspace/project-a", now - 40, "assistant",
             "The JWT token validation was missing the expiry check", 1, "ide", "h4"),
            ("u5", "s3", "/workspace/project-b", now - 20, "user",
             "Set up the CI pipeline for automated testing", 0, "cli", "h5"),
            ("u6", "s3", "/workspace/project-b", now - 10, "user",
             "Configure GitHub Actions workflow with pytest", 1, "cli", "h6"),
        ]
        cache.put_messages_batch(messages)
        yield cache
        cache.close()

    def test_basic_fts_search(self, cache_with_data):
        """FTS finds messages matching keywords."""
        results = cache_with_data.fts_search("kubernetes helm")
        assert len(results) >= 1
        uuids = [r["uuid"] for r in results]
        assert "u1" in uuids

    def test_fts_search_stemming(self, cache_with_data):
        """FTS with porter stemming finds word variants."""
        results = cache_with_data.fts_search("deploying")
        uuids = [r["uuid"] for r in results]
        # Porter stemming: "deploying" → "deploy", should match "deploy"
        assert "u1" in uuids or "u2" in uuids

    def test_fts_search_workspace_filter(self, cache_with_data):
        """FTS respects workspace filter."""
        results = cache_with_data.fts_search("pipeline", workspace="/workspace/project-b")
        assert len(results) >= 1
        for r in results:
            assert r["workspace"].startswith("/workspace/project-b")

    def test_fts_search_workspace_excludes_other(self, cache_with_data):
        """FTS workspace filter excludes results from other workspaces."""
        results = cache_with_data.fts_search("kubernetes", workspace="/workspace/project-b")
        assert len(results) == 0

    def test_fts_search_source_filter(self, cache_with_data):
        """FTS respects source filter."""
        results = cache_with_data.fts_search("pipeline", source="cli")
        assert len(results) >= 1
        for r in results:
            assert r["source"] == "cli"

    def test_fts_search_date_filter(self, cache_with_data):
        """FTS respects timestamp filters."""
        now = time.time()
        # Only very recent messages (last 30 seconds)
        results = cache_with_data.fts_search("pipeline", after_ts=now - 30)
        uuids = [r["uuid"] for r in results]
        assert "u5" in uuids  # 20 seconds ago

    def test_fts_search_no_results(self, cache_with_data):
        """FTS returns empty list for non-matching query."""
        results = cache_with_data.fts_search("xyznonexistentterm")
        assert results == []

    def test_fts_search_score_normalization(self, cache_with_data):
        """FTS scores are normalized between 0 and 1."""
        results = cache_with_data.fts_search("authentication")
        assert len(results) >= 1
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_fts_search_returns_correct_fields(self, cache_with_data):
        """FTS results contain all expected fields."""
        results = cache_with_data.fts_search("helm")
        assert len(results) >= 1
        r = results[0]
        assert "uuid" in r
        assert "session_id" in r
        assert "workspace" in r
        assert "timestamp" in r
        assert "role" in r
        assert "content" in r
        assert "message_index" in r
        assert "source" in r
        assert "score" in r

    def test_fts_triggers_keep_sync_on_insert(self, tmp_path, monkeypatch):
        """New messages inserted after migration are searchable via FTS."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))
        cache = EmbeddingCache("test-triggers")

        # Insert a message after init (FTS triggers should fire)
        cache.put_messages_batch([
            ("new-uuid", "sess", "/ws", time.time(), "user",
             "Deploying microservices to ECS Fargate", 0, "ide", "hx"),
        ])

        results = cache.fts_search("microservices Fargate")
        assert len(results) == 1
        assert results[0]["uuid"] == "new-uuid"
        cache.close()

    def test_fts_triggers_keep_sync_on_delete(self, tmp_path, monkeypatch):
        """Deleted messages are removed from FTS index."""
        monkeypatch.setattr("kiro_ception.cache.get_config", lambda: _fake_config(tmp_path))
        cache = EmbeddingCache("test-delete")

        cache.put_messages_batch([
            ("del-uuid", "sess-del", "/ws", time.time(), "user",
             "This message will be deleted soon", 0, "ide", "hdel"),
        ])
        assert len(cache.fts_search("deleted soon")) == 1

        # Delete via session
        cache.delete_session_messages("sess-del")
        assert len(cache.fts_search("deleted soon")) == 0
        cache.close()


class TestHybridSearchMerge:
    """Test the hybrid result merging logic."""

    def test_vector_only_results(self):
        """When FTS returns nothing, vector results are scaled by vector weight."""
        from kiro_ception.search import _merge_hybrid_results, _VECTOR_WEIGHT

        vector_results = [
            {"uuid": "a", "score": 0.8, "session_id": "s1", "message_index": 0},
            {"uuid": "b", "score": 0.6, "session_id": "s1", "message_index": 5},
        ]
        merged = _merge_hybrid_results(vector_results, [])
        # With no FTS results, should return vector results unchanged
        assert merged == vector_results

    def test_fts_only_results(self):
        """FTS-only results get FTS weight applied."""
        from kiro_ception.search import _merge_hybrid_results, _FTS_WEIGHT

        fts_results = [
            {"uuid": "c", "score": 0.9, "session_id": "s2", "message_index": 0},
        ]
        merged = _merge_hybrid_results([], fts_results)
        assert len(merged) == 1
        assert merged[0]["uuid"] == "c"
        assert merged[0]["score"] == pytest.approx(0.9 * _FTS_WEIGHT)

    def test_overlapping_results_get_boosted(self):
        """Results found by both methods get combined score."""
        from kiro_ception.search import _merge_hybrid_results, _VECTOR_WEIGHT, _FTS_WEIGHT

        vector_results = [
            {"uuid": "x", "score": 0.7, "session_id": "s1", "message_index": 0},
        ]
        fts_results = [
            {"uuid": "x", "score": 0.8, "session_id": "s1", "message_index": 0},
        ]
        merged = _merge_hybrid_results(vector_results, fts_results)
        assert len(merged) == 1
        expected_score = (0.7 * _VECTOR_WEIGHT) + (0.8 * _FTS_WEIGHT)
        assert merged[0]["score"] == pytest.approx(expected_score)

    def test_mixed_results_sorted_by_score(self):
        """Merged results are sorted by combined score descending."""
        from kiro_ception.search import _merge_hybrid_results

        vector_results = [
            {"uuid": "low", "score": 0.3, "session_id": "s1", "message_index": 0},
            {"uuid": "high", "score": 0.9, "session_id": "s1", "message_index": 5},
        ]
        fts_results = [
            {"uuid": "fts-only", "score": 1.0, "session_id": "s2", "message_index": 0},
        ]
        merged = _merge_hybrid_results(vector_results, fts_results)
        scores = [r["score"] for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_by_uuid(self):
        """Same UUID from both sources produces one result, not two."""
        from kiro_ception.search import _merge_hybrid_results

        vector_results = [
            {"uuid": "same", "score": 0.5, "session_id": "s1", "message_index": 0},
        ]
        fts_results = [
            {"uuid": "same", "score": 0.5, "session_id": "s1", "message_index": 0},
        ]
        merged = _merge_hybrid_results(vector_results, fts_results)
        assert len(merged) == 1


# --- Test helpers ---

def _fake_config(tmp_path):
    """Create a fake config that points cache_path to tmp_path."""
    from kiro_ception.config import Config, EmbeddingConfig

    config = Config()
    config.embedding = EmbeddingConfig(cache_dir=str(tmp_path))
    return config


class TestRecencyBoost:
    """Test the recency boost logic."""

    def test_compute_halflife_basic(self):
        """Halflife is calculated so oldest message gets floor multiplier."""
        from kiro_ception.search import _compute_halflife
        import time

        now = time.time()
        oldest = now - (180 * 86400)  # 180 days ago
        floor = 0.85

        halflife = _compute_halflife(oldest, floor)
        assert halflife > 0

        # Verify: applying decay at max_age gives us the floor
        import math
        max_age = now - oldest
        decay = math.exp(-0.693 * max_age / halflife)
        assert decay == pytest.approx(floor, abs=0.001)

    def test_compute_halflife_longer_history(self):
        """Longer history produces larger halflife (gentler decay)."""
        from kiro_ception.search import _compute_halflife
        import time

        now = time.time()
        floor = 0.85

        halflife_6mo = _compute_halflife(now - 180 * 86400, floor)
        halflife_1yr = _compute_halflife(now - 365 * 86400, floor)

        assert halflife_1yr > halflife_6mo

    def test_compute_halflife_floor_1_disables(self):
        """Floor of 1.0 returns 0 (disabled)."""
        from kiro_ception.search import _compute_halflife
        import time

        assert _compute_halflife(time.time() - 86400, 1.0) == 0

    def test_compute_halflife_no_history_returns_zero(self):
        """Zero or future oldest timestamp returns 0."""
        from kiro_ception.search import _compute_halflife
        import time

        assert _compute_halflife(0, 0.85) == 0
        assert _compute_halflife(time.time() + 100, 0.85) == 0

    def test_apply_recency_boost_recent_beats_old(self, monkeypatch):
        """Recent messages score higher than equally-scored older messages."""
        from kiro_ception.search import _apply_recency_boost
        import time

        monkeypatch.setattr("kiro_ception.search.get_config", lambda: _make_search_config(0.85))

        now = time.time()
        oldest = now - (180 * 86400)

        results = [
            {"uuid": "old", "score": 0.8, "timestamp": now - 90 * 86400,
             "session_id": "s1", "message_index": 0},
            {"uuid": "new", "score": 0.8, "timestamp": now - 1 * 86400,
             "session_id": "s2", "message_index": 0},
        ]

        boosted = _apply_recency_boost(results, oldest)

        # The recent one should now rank higher
        assert boosted[0]["uuid"] == "new"
        assert boosted[1]["uuid"] == "old"
        # Both should have scores < 0.8 (multiplied by decay < 1.0)
        assert boosted[0]["score"] < 0.8
        assert boosted[1]["score"] < 0.8
        # Recent should be closer to original score
        assert boosted[0]["score"] > boosted[1]["score"]

    def test_apply_recency_boost_today_near_original(self, monkeypatch):
        """A message from today gets a multiplier very close to 1.0."""
        from kiro_ception.search import _apply_recency_boost
        import time

        monkeypatch.setattr("kiro_ception.search.get_config", lambda: _make_search_config(0.85))

        now = time.time()
        oldest = now - (180 * 86400)

        results = [
            {"uuid": "today", "score": 0.9, "timestamp": now - 60,
             "session_id": "s1", "message_index": 0},
        ]

        boosted = _apply_recency_boost(results, oldest)
        # Should be very close to original (decay of ~1 minute is negligible)
        assert boosted[0]["score"] == pytest.approx(0.9, abs=0.01)

    def test_apply_recency_boost_oldest_gets_floor(self, monkeypatch):
        """The oldest message gets approximately floor × original score."""
        from kiro_ception.search import _apply_recency_boost
        import time

        monkeypatch.setattr("kiro_ception.search.get_config", lambda: _make_search_config(0.85))

        now = time.time()
        oldest = now - (180 * 86400)

        results = [
            {"uuid": "oldest", "score": 1.0, "timestamp": oldest,
             "session_id": "s1", "message_index": 0},
        ]

        boosted = _apply_recency_boost(results, oldest)
        # Should be approximately 0.85 (1.0 * floor)
        assert boosted[0]["score"] == pytest.approx(0.85, abs=0.02)

    def test_apply_recency_boost_disabled_when_floor_is_1(self, monkeypatch):
        """No boost applied when floor is 1.0."""
        from kiro_ception.search import _apply_recency_boost
        import time

        monkeypatch.setattr("kiro_ception.search.get_config", lambda: _make_search_config(1.0))

        now = time.time()
        results = [
            {"uuid": "a", "score": 0.8, "timestamp": now - 90 * 86400,
             "session_id": "s1", "message_index": 0},
        ]

        boosted = _apply_recency_boost(results, now - 180 * 86400)
        assert boosted[0]["score"] == 0.8  # Unchanged

    def test_apply_recency_boost_empty_results(self, monkeypatch):
        """Empty results list returns empty."""
        from kiro_ception.search import _apply_recency_boost

        monkeypatch.setattr("kiro_ception.search.get_config", lambda: _make_search_config(0.85))

        assert _apply_recency_boost([], 1000.0) == []

    def test_apply_recency_boost_no_oldest_timestamp(self, monkeypatch):
        """If oldest_timestamp is 0, no boost is applied."""
        from kiro_ception.search import _apply_recency_boost
        import time

        monkeypatch.setattr("kiro_ception.search.get_config", lambda: _make_search_config(0.85))

        now = time.time()
        results = [
            {"uuid": "a", "score": 0.8, "timestamp": now - 86400,
             "session_id": "s1", "message_index": 0},
        ]

        boosted = _apply_recency_boost(results, 0)
        assert boosted[0]["score"] == 0.8  # Unchanged


def _make_search_config(recency_floor: float):
    """Create a config with a specific recency_floor for testing."""
    from kiro_ception.config import Config, SearchConfig

    config = Config()
    config.search = SearchConfig(recency_floor=recency_floor)
    return config
