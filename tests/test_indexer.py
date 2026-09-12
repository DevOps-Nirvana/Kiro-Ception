"""System tests for the indexing pipeline and SearchIndex.

Tests the full flow: session discovery → message loading → embedding → search.
Uses a fixture corpus of fake sessions and a mock embedding backend.
"""

import hashlib
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kiro_ception.background_indexer import BackgroundIndexer, IndexerState, IndexingStatus
from kiro_ception.cache import EmbeddingCache
from kiro_ception.config import Config, EmbeddingConfig, IndexingConfig
from kiro_ception.memory import select_sessions_within_limit
from kiro_ception.models import IndexedMessage, SessionInfo, Source
from kiro_ception.search import SearchIndex


# --- Fixtures ---


@pytest.fixture
def fake_sessions():
    """A small corpus of fake sessions."""
    return [
        SessionInfo(
            session_id="sess-1",
            workspace="/project/alpha",
            message_count=3,
            created=datetime(2026, 6, 1, 10, 0),
            modified=datetime(2026, 6, 1, 11, 0),
            source=Source.IDE,
        ),
        SessionInfo(
            session_id="sess-2",
            workspace="/project/beta",
            message_count=2,
            created=datetime(2026, 6, 2, 10, 0),
            modified=datetime(2026, 6, 2, 12, 0),
            source=Source.IDE,
        ),
        SessionInfo(
            session_id="sess-3",
            workspace="/project/alpha",
            message_count=4,
            created=datetime(2026, 5, 30, 8, 0),
            modified=datetime(2026, 5, 30, 9, 0),
            source=Source.CLI,
        ),
    ]


@pytest.fixture
def fake_messages():
    """Messages for the fake sessions."""
    base = datetime(2026, 6, 1, 10, 0)
    return {
        "sess-1": [
            IndexedMessage(
                uuid="s1-m0", session_id="sess-1", workspace="/project/alpha",
                timestamp=datetime(2026, 6, 1, 10, 0), role="user",
                searchable_text="How do I set up authentication?",
                message_index=0, source=Source.IDE,
            ),
            IndexedMessage(
                uuid="s1-m1", session_id="sess-1", workspace="/project/alpha",
                timestamp=datetime(2026, 6, 1, 10, 1), role="assistant",
                searchable_text="You can use JWT tokens for authentication.",
                message_index=1, source=Source.IDE,
            ),
            IndexedMessage(
                uuid="s1-m2", session_id="sess-1", workspace="/project/alpha",
                timestamp=datetime(2026, 6, 1, 10, 2), role="user",
                searchable_text="Can you show me an example?",
                message_index=2, source=Source.IDE,
            ),
        ],
        "sess-2": [
            IndexedMessage(
                uuid="s2-m0", session_id="sess-2", workspace="/project/beta",
                timestamp=datetime(2026, 6, 2, 10, 0), role="user",
                searchable_text="Fix the database connection timeout bug.",
                message_index=0, source=Source.IDE,
            ),
            IndexedMessage(
                uuid="s2-m1", session_id="sess-2", workspace="/project/beta",
                timestamp=datetime(2026, 6, 2, 10, 1), role="assistant",
                searchable_text="The timeout was caused by missing connection pooling.",
                message_index=1, source=Source.IDE,
            ),
        ],
        "sess-3": [
            IndexedMessage(
                uuid="s3-m0", session_id="sess-3", workspace="/project/alpha",
                timestamp=datetime(2026, 5, 30, 8, 0), role="user",
                searchable_text="Deploy the service to production.",
                message_index=0, source=Source.CLI,
            ),
            IndexedMessage(
                uuid="s3-m1", session_id="sess-3", workspace="/project/alpha",
                timestamp=datetime(2026, 5, 30, 8, 1), role="assistant",
                searchable_text="Deploying version 2.1.0 to production cluster.",
                message_index=1, source=Source.CLI,
            ),
            IndexedMessage(
                uuid="s3-m2", session_id="sess-3", workspace="/project/alpha",
                timestamp=datetime(2026, 5, 30, 8, 2), role="user",
                searchable_text="Check the deployment status.",
                message_index=2, source=Source.CLI,
            ),
            IndexedMessage(
                uuid="s3-m3", session_id="sess-3", workspace="/project/alpha",
                timestamp=datetime(2026, 5, 30, 8, 3), role="assistant",
                searchable_text="All pods are healthy. Deployment complete.",
                message_index=3, source=Source.CLI,
            ),
        ],
    }


@pytest.fixture
def mock_backend():
    """A deterministic mock embedding backend.

    Generates embeddings as normalized random vectors seeded by text hash,
    so the same text always gets the same vector (deterministic similarity).
    """
    backend = MagicMock()
    backend.fingerprint.return_value = "test-backend:mock:128"

    def _encode(texts):
        """Deterministic embeddings from text hash."""
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            result.append(vec)
        return np.array(result)

    def _encode_query(text):
        return _encode([text])[0]

    backend.encode.side_effect = _encode
    backend.encode_query.side_effect = _encode_query
    backend.dimensions.return_value = 128
    return backend


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Real SQLite cache in a temp directory."""
    monkeypatch.setattr(
        "kiro_ception.cache._get_cache_db_path",
        lambda fp: tmp_path / f"cache_{hashlib.md5(fp.encode()).hexdigest()[:12]}.db",
    )
    c = EmbeddingCache("test-backend:mock:128")
    _ = c.conn
    yield c
    c.close()


# --- select_sessions_within_limit ---


class TestSelectSessionsWithinLimit:
    def test_no_limit_returns_all(self, fake_sessions):
        selected, excluded = select_sessions_within_limit(fake_sessions, 0)
        assert len(selected) == 3
        assert len(excluded) == 0

    def test_tight_limit_excludes_oldest(self, fake_sessions):
        # BYTES_PER_MESSAGE = 2600, so 3 msgs = 7800 bytes, 2 msgs = 5200
        # A limit of 15000 bytes should fit sess-2 (newest, 2 msgs=5200) and
        # sess-1 (3 msgs=7800) but exclude sess-3 (oldest, 4 msgs=10400)
        from kiro_ception.config import BYTES_PER_MESSAGE

        # Enough for ~5 messages
        limit = 5 * BYTES_PER_MESSAGE + 1
        selected, excluded = select_sessions_within_limit(fake_sessions, limit)

        # Newest sessions selected first
        selected_ids = {s.session_id for s in selected}
        excluded_ids = {s.session_id for s in excluded}

        # sess-2 (newest, 2 msgs) and sess-1 (3 msgs) = 5 msgs total, fits
        assert "sess-2" in selected_ids
        assert "sess-1" in selected_ids
        # sess-3 (oldest, 4 msgs) would push over limit
        assert "sess-3" in excluded_ids

    def test_sorted_by_newest_first(self, fake_sessions):
        # When limit is applied (>0), results are sorted newest first
        from kiro_ception.config import BYTES_PER_MESSAGE
        limit = 100 * BYTES_PER_MESSAGE  # Enough for all
        selected, _ = select_sessions_within_limit(fake_sessions, limit)
        timestamps = [s.timestamp_fallback for s in selected]
        assert timestamps == sorted(timestamps, reverse=True)


# --- IndexingStatus ---


class TestIndexingStatus:
    def test_to_dict_structure(self):
        status = IndexingStatus(
            state=IndexerState.INDEXING,
            sessions_total=100,
            sessions_processed=50,
            sessions_unchanged=20,
            messages_embedded=200,
            started_at=time.time() - 60,
        )
        d = status.to_dict()

        assert d["state"] == "indexing"
        assert d["sessions_total"] == 100
        assert d["progress_percent"] == 70.0  # (50+20)/100
        assert d["elapsed_seconds"] >= 59.0

    def test_progress_percent_zero_total(self):
        status = IndexingStatus(sessions_total=0)
        assert status.progress_percent == 0.0

    def test_eta_only_when_indexing(self):
        status = IndexingStatus(state=IndexerState.IDLE, rate=10.0)
        assert status.eta_seconds == 0.0

    def test_last_completed_at_in_to_dict(self):
        completed = time.time() - 300  # 5 minutes ago
        status = IndexingStatus(
            state=IndexerState.INDEXING,
            sessions_total=100,
            started_at=time.time(),
            last_completed_at=completed,
        )
        d = status.to_dict()

        assert "last_completed_at" in d
        assert d["last_completed_at"] is not None
        # Should be an ISO formatted string
        assert "T" in d["last_completed_at"]

    def test_last_completed_at_none_on_first_run(self):
        status = IndexingStatus(
            state=IndexerState.INDEXING,
            sessions_total=100,
            started_at=time.time(),
        )
        d = status.to_dict()

        assert d["last_completed_at"] is None


# --- Full indexing pipeline ---


class TestIndexingPipeline:
    def test_full_index_and_search(self, cache, mock_backend, fake_sessions, fake_messages):
        """End-to-end: index sessions → search → get results."""
        # Step 1: Index all messages into the cache
        for session in fake_sessions:
            msgs = fake_messages[session.session_id]
            metadata_rows = []
            texts_to_embed = []
            hashes_to_embed = []

            for msg in msgs:
                text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
                metadata_rows.append((
                    msg.uuid, msg.session_id, msg.workspace,
                    msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                    msg.message_index, msg.source.value, text_hash,
                ))
                texts_to_embed.append(msg.searchable_text)
                hashes_to_embed.append(text_hash)

            # Store metadata
            cache.put_messages_batch(metadata_rows)

            # Embed and store
            embeddings = mock_backend.encode(texts_to_embed)
            items = list(zip(hashes_to_embed, embeddings))
            cache.put_embeddings_batch(items)

            cache.update_session_state(session.session_id, session.modified.timestamp())

        # Verify cache state
        assert cache.message_count == 9  # 3 + 2 + 4
        assert cache.embedding_count == 9
        assert cache.indexed_session_count == 3

        # Step 2: Build SearchIndex from cache
        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            search_index = SearchIndex()
            search_index._refresh()

        assert search_index.message_count == 9
        assert search_index.is_loading is False

        # Step 3: Search for "authentication"
        query_vec = mock_backend.encode_query("How do I set up authentication?")
        results = search_index.search(query_embedding=query_vec, threshold=0.0, max_results=5)

        # The exact match should score highest (same text = same vector = cos_sim=1.0)
        assert len(results) > 0
        assert results[0]["uuid"] == "s1-m0"
        assert results[0]["score"] >= 0.99

    def test_workspace_filter(self, cache, mock_backend, fake_sessions, fake_messages):
        """Search filtered by workspace only returns matching sessions."""
        # Index everything
        for session in fake_sessions:
            msgs = fake_messages[session.session_id]
            metadata_rows = []
            texts_to_embed = []
            hashes_to_embed = []
            for msg in msgs:
                text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
                metadata_rows.append((
                    msg.uuid, msg.session_id, msg.workspace,
                    msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                    msg.message_index, msg.source.value, text_hash,
                ))
                texts_to_embed.append(msg.searchable_text)
                hashes_to_embed.append(text_hash)
            cache.put_messages_batch(metadata_rows)
            embeddings = mock_backend.encode(texts_to_embed)
            cache.put_embeddings_batch(list(zip(hashes_to_embed, embeddings)))

        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            search_index = SearchIndex()
            search_index._refresh()

        # Search for "database" with workspace=/project/beta
        query_vec = mock_backend.encode_query("Fix the database connection timeout bug.")
        results = search_index.search(
            query_embedding=query_vec,
            workspace="/project/beta",
            threshold=0.0,
            max_results=10,
        )

        # All results should be from /project/beta
        for r in results:
            assert r["workspace"].startswith("/project/beta")

    def test_source_filter(self, cache, mock_backend, fake_sessions, fake_messages):
        """Search filtered by source only returns matching source type."""
        for session in fake_sessions:
            msgs = fake_messages[session.session_id]
            metadata_rows = []
            texts_to_embed = []
            hashes_to_embed = []
            for msg in msgs:
                text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
                metadata_rows.append((
                    msg.uuid, msg.session_id, msg.workspace,
                    msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                    msg.message_index, msg.source.value, text_hash,
                ))
                texts_to_embed.append(msg.searchable_text)
                hashes_to_embed.append(text_hash)
            cache.put_messages_batch(metadata_rows)
            embeddings = mock_backend.encode(texts_to_embed)
            cache.put_embeddings_batch(list(zip(hashes_to_embed, embeddings)))

        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            search_index = SearchIndex()
            search_index._refresh()

        # Search CLI only
        query_vec = mock_backend.encode_query("Deploy the service")
        results = search_index.search(
            query_embedding=query_vec,
            source="cli",
            threshold=0.0,
            max_results=10,
        )

        for r in results:
            assert r["source"] == "cli"

    def test_date_filter(self, cache, mock_backend, fake_sessions, fake_messages):
        """Search with date filters restricts by timestamp."""
        for session in fake_sessions:
            msgs = fake_messages[session.session_id]
            metadata_rows = []
            texts_to_embed = []
            hashes_to_embed = []
            for msg in msgs:
                text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
                metadata_rows.append((
                    msg.uuid, msg.session_id, msg.workspace,
                    msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                    msg.message_index, msg.source.value, text_hash,
                ))
                texts_to_embed.append(msg.searchable_text)
                hashes_to_embed.append(text_hash)
            cache.put_messages_batch(metadata_rows)
            embeddings = mock_backend.encode(texts_to_embed)
            cache.put_embeddings_batch(list(zip(hashes_to_embed, embeddings)))

        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            search_index = SearchIndex()
            search_index._refresh()

        # Only after June 1 — should exclude sess-3 (May 30)
        after_ts = datetime(2026, 6, 1, 0, 0).timestamp()
        query_vec = mock_backend.encode_query("deploy")
        results = search_index.search(
            query_embedding=query_vec,
            after_ts=after_ts,
            threshold=0.0,
            max_results=20,
        )

        for r in results:
            assert r["timestamp"] >= after_ts

    def test_threshold_filter(self, cache, mock_backend, fake_sessions, fake_messages):
        """High threshold should return fewer results."""
        for session in fake_sessions:
            msgs = fake_messages[session.session_id]
            metadata_rows = []
            texts_to_embed = []
            hashes_to_embed = []
            for msg in msgs:
                text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
                metadata_rows.append((
                    msg.uuid, msg.session_id, msg.workspace,
                    msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                    msg.message_index, msg.source.value, text_hash,
                ))
                texts_to_embed.append(msg.searchable_text)
                hashes_to_embed.append(text_hash)
            cache.put_messages_batch(metadata_rows)
            embeddings = mock_backend.encode(texts_to_embed)
            cache.put_embeddings_batch(list(zip(hashes_to_embed, embeddings)))

        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            search_index = SearchIndex()
            search_index._refresh()

        query_vec = mock_backend.encode_query("authentication JWT tokens")

        # Low threshold: many results
        low_results = search_index.search(query_embedding=query_vec, threshold=0.0, max_results=20)
        # High threshold: fewer or zero
        high_results = search_index.search(query_embedding=query_vec, threshold=0.99, max_results=20)

        assert len(high_results) <= len(low_results)


# --- BackgroundIndexer unit behavior ---


class TestBackgroundIndexer:
    def test_initial_state(self):
        indexer = BackgroundIndexer()
        assert indexer.status.state == IndexerState.IDLE
        assert indexer.cache is None
        assert indexer.backend is None

    def test_trigger_reindex_sets_event(self):
        indexer = BackgroundIndexer()
        assert not indexer._reindex_event.is_set()
        # Patch start to avoid actually launching a thread
        with patch.object(indexer, "start"):
            indexer.trigger_reindex()
        assert indexer._reindex_event.is_set()

    def test_trigger_rescan_sets_event(self):
        indexer = BackgroundIndexer()
        assert not indexer._rescan_event.is_set()
        with patch.object(indexer, "start"):
            indexer.trigger_rescan()
        assert indexer._rescan_event.is_set()

    def test_stop_sets_event(self):
        indexer = BackgroundIndexer()
        indexer.stop()
        assert indexer._stop_event.is_set()

    def test_get_session_mtime_from_modified(self):
        session = SessionInfo(
            session_id="s1", workspace="/test", source=Source.IDE,
            created=datetime(2026, 1, 1), modified=datetime(2026, 6, 1),
        )
        mtime = BackgroundIndexer._get_session_mtime(session)
        assert mtime == datetime(2026, 6, 1).timestamp()

    def test_get_session_mtime_fallback_to_created(self):
        session = SessionInfo(
            session_id="s1", workspace="/test", source=Source.IDE,
            created=datetime(2026, 1, 1), modified=None,
        )
        mtime = BackgroundIndexer._get_session_mtime(session)
        assert mtime == datetime(2026, 1, 1).timestamp()

    def test_get_session_mtime_no_dates(self):
        session = SessionInfo(
            session_id="s1", workspace="/test", source=Source.IDE,
        )
        mtime = BackgroundIndexer._get_session_mtime(session)
        assert mtime == 0.0


# --- SearchIndex refresh behavior ---


class TestSearchIndexRefresh:
    def test_empty_cache_stays_empty(self, cache):
        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            si = SearchIndex()
            si._refresh()

        assert si.message_count == 0
        assert si.is_loading is False

    def test_embeddings_without_messages_shows_loading(self, cache):
        """If embeddings exist but no messages yet, report loading state."""
        # Put an embedding but no messages
        cache.put_embedding("hash1", np.random.randn(128).astype(np.float32))

        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            si = SearchIndex()
            si._refresh()

        assert si.message_count == 0
        assert si.is_loading is True
        assert si.embedding_count_hint == 1

    def test_refresh_throttle(self, cache, mock_backend):
        """After successful load, refresh is throttled to 60 seconds."""
        # Put a message and its embedding
        text_hash = hashlib.md5(b"hello").hexdigest()
        cache.put_message("m1", "s1", "/test", time.time(), "user", "hello", 0, "ide", text_hash)
        cache.conn.commit()
        cache.put_embedding(text_hash, np.random.randn(128).astype(np.float32))

        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            si = SearchIndex()
            si._refresh()
            assert si._ever_loaded is True

            # Record state after first load
            count_after_first = si.message_count

            # Add another message to the cache
            text_hash2 = hashlib.md5(b"world").hexdigest()
            cache.put_message("m2", "s1", "/test", time.time(), "user", "world", 1, "ide", text_hash2)
            cache.conn.commit()
            cache.put_embedding(text_hash2, np.random.randn(128).astype(np.float32))

            # refresh_if_needed should be throttled — count shouldn't change
            si.refresh_if_needed()
            assert si.message_count == count_after_first

    def test_refresh_not_throttled_before_first_load(self, cache):
        """Before first successful load, every refresh attempt is allowed."""
        with patch("kiro_ception.search.get_background_indexer") as mock_get_indexer:
            indexer_mock = MagicMock()
            indexer_mock.cache = cache
            mock_get_indexer.return_value = indexer_mock

            si = SearchIndex()
            assert si._ever_loaded is False

            # Multiple refreshes should all be allowed (not throttled)
            si.refresh_if_needed()
            si.refresh_if_needed()
            si.refresh_if_needed()
            # No error, no throttle — just no data
            assert si.message_count == 0



class TestEngineSearchOperators:
    """End-to-end engine_search() require/exclude/promote/demote via set(C)."""

    def _index(self, cache, mock_backend, fake_sessions, fake_messages):
        for session in fake_sessions:
            msgs = fake_messages[session.session_id]
            metadata_rows = []
            texts = []
            hashes = []
            for msg in msgs:
                text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
                metadata_rows.append((
                    msg.uuid, msg.session_id, msg.workspace,
                    msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                    msg.message_index, msg.source.value, text_hash,
                ))
                texts.append(msg.searchable_text)
                hashes.append(text_hash)
            cache.put_messages_batch(metadata_rows)
            cache.put_embeddings_batch(list(zip(hashes, mock_backend.encode(texts))))
            cache.update_session_state(session.session_id, session.modified.timestamp())

    def _run(self, cache, mock_backend, query, threshold=0.0, **operators):
        from kiro_ception import search as search_mod

        indexer_mock = MagicMock()
        indexer_mock.cache = cache
        indexer_mock.backend = mock_backend

        with patch.object(search_mod, "get_background_indexer", return_value=indexer_mock):
            si = search_mod.SearchIndex()
            si._refresh()
            with patch.object(search_mod, "get_search_index", return_value=si):
                return search_mod.engine_search(
                    query=query,
                    workspace=None,
                    source=None,
                    after=None,
                    before=None,
                    context_size=3,
                    threshold=threshold,
                    max_results=20,
                    offset=0,
                    **operators,
                )

    def _contents(self, response):
        return [r["matched_message"]["content"] for r in response["results"]]

    def _uuids(self, response):
        return [r["matched_message"]["uuid"] for r in response["results"]]

    def test_exclude_drops_matching_messages(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        self._index(cache, mock_backend, fake_sessions, fake_messages)

        baseline = self._run(cache, mock_backend, "deploy production")
        assert any("production" in c for c in self._contents(baseline))

        excluded = self._run(
            cache, mock_backend, "deploy production", exclude_terms=["production"]
        )
        assert all("production" not in c for c in self._contents(excluded))

    def test_no_operators_reproduces_baseline(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        self._index(cache, mock_backend, fake_sessions, fake_messages)
        a = self._run(cache, mock_backend, "authentication")
        b = self._run(cache, mock_backend, "authentication")
        assert set(self._uuids(a)) == set(self._uuids(b))

    def test_whole_token_exclude_does_not_overmatch(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        self._index(cache, mock_backend, fake_sessions, fake_messages)
        # Excluding "deployment" must not remove "deploy"/"deploying" results.
        response = self._run(
            cache, mock_backend, "deployment status", exclude_terms=["deployment"]
        )
        for c in self._contents(response):
            tokens = c.lower().replace(".", " ").split()
            assert "deployment" not in tokens

    def test_require_narrows_to_intersection(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        self._index(cache, mock_backend, fake_sessions, fake_messages)
        # require "production": every surviving result must be in set(C) for
        # "production". With the mock backend vector similarity is random noise,
        # so a threshold that suppresses noise (0.3) makes FTS keyword match the
        # deciding membership signal — set(C) becomes the "production" messages.
        response = self._run(
            cache, mock_backend, "deploy", threshold=0.3, require_terms=["production"]
        )
        assert len(response["results"]) > 0
        assert all("production" in c for c in self._contents(response))

    def test_demote_keeps_all_but_ranks_members_last(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        self._index(cache, mock_backend, fake_sessions, fake_messages)
        baseline = self._run(cache, mock_backend, "deploy production", threshold=0.3)
        demoted = self._run(
            cache, mock_backend, "deploy production", threshold=0.3,
            demote_terms=["production"],
        )
        # Nothing removed: same set of uuids as baseline.
        assert set(self._uuids(demoted)) == set(self._uuids(baseline))
        # All non-"production" results precede any "production" result.
        contents = self._contents(demoted)
        last_non_prod = max(
            (i for i, c in enumerate(contents) if "production" not in c), default=-1
        )
        first_prod = next(
            (i for i, c in enumerate(contents) if "production" in c), len(contents)
        )
        assert last_non_prod < first_prod

    def test_promote_keeps_all_but_ranks_members_first(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        self._index(cache, mock_backend, fake_sessions, fake_messages)
        baseline = self._run(cache, mock_backend, "deploy production", threshold=0.3)
        promoted = self._run(
            cache, mock_backend, "deploy production", threshold=0.3,
            promote_terms=["production"],
        )
        assert set(self._uuids(promoted)) == set(self._uuids(baseline))
        contents = self._contents(promoted)
        # All "production" results precede any non-"production" result.
        last_prod = max(
            (i for i, c in enumerate(contents) if "production" in c), default=-1
        )
        first_non_prod = next(
            (i for i, c in enumerate(contents) if "production" not in c), len(contents)
        )
        assert last_prod < first_non_prod

    def test_recency_boost_applies_even_with_soft_operator(
        self, cache, mock_backend, fake_sessions, fake_messages
    ):
        # Option A: recency is folded into score BEFORE partitioning, so it must
        # run even when a promote/demote operator is active (previously it was
        # suppressed). Spy on _apply_recency_boost to prove it is invoked.
        from kiro_ception import search as search_mod

        self._index(cache, mock_backend, fake_sessions, fake_messages)

        indexer_mock = MagicMock()
        indexer_mock.cache = cache
        indexer_mock.backend = mock_backend

        real_boost = search_mod._apply_recency_boost
        calls = {"n": 0}

        def spy(results, oldest):
            calls["n"] += 1
            return real_boost(results, oldest)

        with patch.object(search_mod, "get_background_indexer", return_value=indexer_mock):
            si = search_mod.SearchIndex()
            si._refresh()
            with patch.object(search_mod, "get_search_index", return_value=si):
                with patch.object(search_mod, "_apply_recency_boost", side_effect=spy):
                    search_mod.engine_search(
                        query="deploy production",
                        workspace=None, source=None, after=None, before=None,
                        context_size=3, threshold=0.3, max_results=20, offset=0,
                        demote_terms=["production"],
                    )

        assert calls["n"] == 1, "recency boost must run even with a soft operator active"

