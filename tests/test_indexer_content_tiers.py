"""Tests that verify the indexer correctly persists content_tier and tool_name.

Ensures that when sessions containing tool actions are indexed (especially via
full reindex), the two-tier content model fields are written to the database —
not silently dropped as 'conversation'/None.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kiro_ception.background_indexer import BackgroundIndexer, IndexerState
from kiro_ception.config import Config, EmbeddingConfig, IndexingConfig, ToolSummariesConfig
from kiro_ception.models import ContentTier, IndexedMessage, SessionInfo, Source


@pytest.fixture
def mock_backend():
    """Deterministic mock embedding backend producing 64-dim vectors."""
    backend = MagicMock()
    backend.fingerprint.return_value = "test:tier-test:64"

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
def execution_log_dir(tmp_path):
    """Create a fake execution log directory with tool actions."""
    exec_dir = tmp_path / "executions"
    exec_dir.mkdir()

    # Execution log with a mix of tool actions and a say action
    exec_log = {
        "executionId": "exec-tier-001",
        "workflowType": "act",
        "status": "completed",
        "startTime": "2026-06-10T12:00:00.000Z",
        "input": {
            "data": {
                "userPrompt": "Refactor the database connection pool to use async initialization"
            }
        },
        "chatSessionId": "tier-session-1",
        "actions": [
            {
                "type": "action",
                "actionId": "tier-action-read",
                "actionState": "completed",
                "actionType": "readFile",
                "input": {"path": "/src/db/pool.ts"},
                "output": {"filePath": "/src/db/pool.ts", "content": "pool code..."},
                "emittedAt": 1718020001000,
            },
            {
                "type": "action",
                "actionId": "tier-action-grep",
                "actionState": "completed",
                "actionType": "grep_search",
                "input": {"query": "ConnectionPool"},
                "output": {"results": [{"file": "x.ts", "line": 1, "text": "class ConnectionPool"}]},
                "emittedAt": 1718020002000,
            },
            {
                "type": "action",
                "actionId": "tier-action-write",
                "actionState": "completed",
                "actionType": "fs_write",
                "input": {"path": "/src/db/pool.ts"},
                "output": {"filePath": "/src/db/pool.ts"},
                "emittedAt": 1718020003000,
            },
            {
                "type": "action",
                "actionId": "tier-action-say",
                "actionState": "completed",
                "actionType": "say",
                "input": {},
                "output": {
                    "message": "I've refactored the database connection pool to use async initialization with proper error handling and retry logic."
                },
                "emittedAt": 1718020004000,
            },
        ],
    }

    exec_file = exec_dir / "exec-tier-001.json"
    exec_file.write_text(json.dumps(exec_log))
    return exec_dir


@pytest.fixture
def session_with_tools(execution_log_dir):
    """A session that has both conversation messages and tool context via execution logs."""
    return SessionInfo(
        session_id="tier-session-1",
        workspace="/test/tier-project",
        message_count=5,
        created=datetime(2026, 6, 10, 12, 0),
        modified=datetime(2026, 6, 10, 12, 5),
        source=Source.IDE,
    )


def _make_messages_with_tiers(session_id: str) -> list[IndexedMessage]:
    """Simulate what the updated loader produces: messages with proper content_tier."""
    return [
        # User prompt extracted from execution log
        IndexedMessage(
            uuid=f"{session_id}-prompt-exec-tier-001",
            session_id=session_id,
            workspace="/test/tier-project",
            timestamp=datetime(2026, 6, 10, 12, 0),
            role="user",
            searchable_text="Refactor the database connection pool to use async initialization",
            message_index=0,
            source=Source.IDE,
            content_tier=ContentTier.CONVERSATION,
        ),
        # Tool context: readFile
        IndexedMessage(
            uuid="tier-action-read",
            session_id=session_id,
            workspace="/test/tier-project",
            timestamp=datetime(2026, 6, 10, 12, 0, 1),
            role="assistant",
            searchable_text="[readFile] /src/db/pool.ts → completed",
            message_index=1,
            source=Source.IDE,
            content_tier=ContentTier.TOOL_CONTEXT,
            tool_name="readFile",
        ),
        # Tool context: grep_search
        IndexedMessage(
            uuid="tier-action-grep",
            session_id=session_id,
            workspace="/test/tier-project",
            timestamp=datetime(2026, 6, 10, 12, 0, 2),
            role="assistant",
            searchable_text='[grep_search] query="ConnectionPool" (1 results) → completed',
            message_index=2,
            source=Source.IDE,
            content_tier=ContentTier.TOOL_CONTEXT,
            tool_name="grep_search",
        ),
        # Tool context: fs_write
        IndexedMessage(
            uuid="tier-action-write",
            session_id=session_id,
            workspace="/test/tier-project",
            timestamp=datetime(2026, 6, 10, 12, 0, 3),
            role="assistant",
            searchable_text="[fs_write] /src/db/pool.ts → completed",
            message_index=3,
            source=Source.IDE,
            content_tier=ContentTier.TOOL_CONTEXT,
            tool_name="fs_write",
        ),
        # Assistant say response (conversation tier)
        IndexedMessage(
            uuid="tier-action-say",
            session_id=session_id,
            workspace="/test/tier-project",
            timestamp=datetime(2026, 6, 10, 12, 0, 4),
            role="assistant",
            searchable_text="I've refactored the database connection pool to use async initialization with proper error handling and retry logic.",
            message_index=4,
            source=Source.IDE,
            content_tier=ContentTier.CONVERSATION,
        ),
    ]


class TestIndexerContentTierPersistence:
    """Verify the indexer writes content_tier and tool_name to the database."""

    def test_full_reindex_persists_content_tier_and_tool_name(
        self, tmp_path, monkeypatch, mock_backend, session_with_tools
    ):
        """After a full reindex, the cache must contain both conversation and
        tool_context rows with correct tool_name values."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{hashlib.md5(fp.encode()).hexdigest()[:12]}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(
                backend="openai-compatible", api_base="http://fake", batch_size=10
            ),
            indexing=IndexingConfig(rescan_interval_minutes=0),
            tool_summaries=ToolSummariesConfig(
                excluded_tools=[], max_summary_length=800, include_meaningful_output=True
            ),
        )

        messages = _make_messages_with_tiers(session_with_tools.session_id)

        def _load_messages(session):
            if session.session_id == session_with_tools.session_id:
                return messages
            return []

        sessions = [session_with_tools]

        with (
            patch("kiro_ception.background_indexer.get_config", return_value=config),
            patch("kiro_ception.background_indexer.get_embedding_backend", return_value=mock_backend),
            patch("kiro_ception.background_indexer.list_all_sessions", return_value=sessions),
            patch("kiro_ception.background_indexer.load_session_messages", side_effect=_load_messages),
            patch("kiro_ception.background_indexer.get_memory_limit", return_value=0),
            patch("kiro_ception.background_indexer.select_sessions_within_limit",
                  return_value=(sessions, [])),
        ):
            indexer = BackgroundIndexer()
            indexer.start()

            deadline = time.time() + 10
            while time.time() < deadline:
                if indexer.status.state == IndexerState.IDLE and indexer.status.completed_at:
                    break
                time.sleep(0.05)
            indexer.stop()

        # Verify indexing completed
        assert indexer.status.state == IndexerState.IDLE
        assert indexer.status.sessions_processed == 1
        assert indexer.status.messages_embedded == 5

        # Now query the cache directly to verify content_tier and tool_name
        cache = indexer.cache
        stored_messages = cache.get_session_messages(session_with_tools.session_id)

        assert len(stored_messages) == 5

        # Check content_tier distribution
        conversation_msgs = [m for m in stored_messages if m["content_tier"] == "conversation"]
        tool_context_msgs = [m for m in stored_messages if m["content_tier"] == "tool_context"]

        assert len(conversation_msgs) == 2, (
            f"Expected 2 conversation messages, got {len(conversation_msgs)}. "
            f"Tiers found: {[m['content_tier'] for m in stored_messages]}"
        )
        assert len(tool_context_msgs) == 3, (
            f"Expected 3 tool_context messages, got {len(tool_context_msgs)}. "
            f"Tiers found: {[m['content_tier'] for m in stored_messages]}"
        )

        # Verify tool_name is populated for tool_context messages
        for msg in tool_context_msgs:
            assert msg["tool_name"] is not None, (
                f"tool_name should be set for tool_context message: {msg['uuid']}"
            )

        tool_names = {m["tool_name"] for m in tool_context_msgs}
        assert tool_names == {"readFile", "grep_search", "fs_write"}

        # Verify tool_name is NOT set for conversation messages
        for msg in conversation_msgs:
            assert msg["tool_name"] is None, (
                f"tool_name should be None for conversation message: {msg['uuid']}"
            )

    def test_reindex_after_upgrade_replaces_flat_conversation_with_tiers(
        self, tmp_path, monkeypatch, mock_backend, session_with_tools
    ):
        """Simulates upgrading from old code (everything is 'conversation')
        to new code (proper tiers). After full reindex, tool_context rows
        must appear where previously all were 'conversation'."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{hashlib.md5(fp.encode()).hexdigest()[:12]}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(
                backend="openai-compatible", api_base="http://fake", batch_size=10
            ),
            indexing=IndexingConfig(rescan_interval_minutes=0),
            tool_summaries=ToolSummariesConfig(
                excluded_tools=[], max_summary_length=800, include_meaningful_output=True
            ),
        )

        messages = _make_messages_with_tiers(session_with_tools.session_id)
        sessions = [session_with_tools]

        # --- Step 1: Manually populate the cache as if the OLD indexer ran ---
        # (all messages stored as 'conversation' tier with no tool_name)
        from kiro_ception.cache import EmbeddingCache

        cache = EmbeddingCache(mock_backend.fingerprint())

        for msg in messages:
            text_hash = hashlib.md5(msg.searchable_text.encode()).hexdigest()
            # Store everything as 'conversation' with no tool_name (old behavior)
            cache.put_message(
                msg.uuid, msg.session_id, msg.workspace,
                msg.timestamp.timestamp(), msg.role, msg.searchable_text,
                msg.message_index, msg.source.value, text_hash,
                "conversation", None,
            )
            # Also store embeddings so they're cached
            emb = mock_backend.encode([msg.searchable_text])
            cache.put_embedding(text_hash, emb[0])

        # Mark session as indexed
        cache.update_session_state(
            session_with_tools.session_id,
            session_with_tools.modified.timestamp(),
        )

        # Verify pre-condition: everything is 'conversation'
        stored = cache.get_session_messages(session_with_tools.session_id)
        assert len(stored) == 5
        assert all(m["content_tier"] == "conversation" for m in stored)
        assert all(m["tool_name"] is None for m in stored)

        # --- Step 2: Full reindex with proper tier-aware loader ---
        import kiro_ception.background_indexer as bi_module

        monkeypatch.setattr(bi_module, "list_all_sessions", lambda: sessions)
        monkeypatch.setattr(bi_module, "select_sessions_within_limit", lambda s, m: (sessions, []))
        monkeypatch.setattr(bi_module, "load_session_messages", lambda s: messages)

        indexer = BackgroundIndexer()
        indexer._cache = cache
        indexer._backend = mock_backend

        # Clear session state to force re-processing (simulates rescan full=True)
        cache.clear_session_state()
        indexer._index_pass(config)

        # --- Step 3: Verify tiers are now correctly persisted ---
        stored_after = cache.get_session_messages(session_with_tools.session_id)
        assert len(stored_after) == 5

        conversation_msgs = [m for m in stored_after if m["content_tier"] == "conversation"]
        tool_context_msgs = [m for m in stored_after if m["content_tier"] == "tool_context"]

        assert len(conversation_msgs) == 2, (
            f"After reindex, expected 2 conversation messages but got {len(conversation_msgs)}"
        )
        assert len(tool_context_msgs) == 3, (
            f"After reindex, expected 3 tool_context messages but got {len(tool_context_msgs)}"
        )

        # Verify tool_name is populated
        tool_names = {m["tool_name"] for m in tool_context_msgs}
        assert tool_names == {"readFile", "grep_search", "fs_write"}

    def test_fts_search_respects_content_tier_filter(
        self, tmp_path, monkeypatch, mock_backend, session_with_tools
    ):
        """FTS search with include_tool_context=False must exclude tool_context rows;
        with include_tool_context=True must include them."""
        monkeypatch.setattr(
            "kiro_ception.cache._get_cache_db_path",
            lambda fp: tmp_path / f"cache_{hashlib.md5(fp.encode()).hexdigest()[:12]}.db",
        )

        config = Config(
            embedding=EmbeddingConfig(
                backend="openai-compatible", api_base="http://fake", batch_size=10
            ),
            indexing=IndexingConfig(rescan_interval_minutes=0),
            tool_summaries=ToolSummariesConfig(
                excluded_tools=[], max_summary_length=800, include_meaningful_output=True
            ),
        )

        messages = _make_messages_with_tiers(session_with_tools.session_id)
        sessions = [session_with_tools]

        def _load_messages(session):
            if session.session_id == session_with_tools.session_id:
                return messages
            return []

        with (
            patch("kiro_ception.background_indexer.get_config", return_value=config),
            patch("kiro_ception.background_indexer.get_embedding_backend", return_value=mock_backend),
            patch("kiro_ception.background_indexer.list_all_sessions", return_value=sessions),
            patch("kiro_ception.background_indexer.load_session_messages", side_effect=_load_messages),
            patch("kiro_ception.background_indexer.get_memory_limit", return_value=0),
            patch("kiro_ception.background_indexer.select_sessions_within_limit",
                  return_value=(sessions, [])),
        ):
            indexer = BackgroundIndexer()
            indexer.start()

            deadline = time.time() + 10
            while time.time() < deadline:
                if indexer.status.state == IndexerState.IDLE and indexer.status.completed_at:
                    break
                time.sleep(0.05)
            indexer.stop()

        cache = indexer.cache

        # Search for "readFile" which only appears in tool_context messages
        # With include_tool_context=False: should find nothing
        results_no_tools = cache.fts_search(
            query="readFile",
            include_tool_context=False,
        )
        assert len(results_no_tools) == 0, (
            "FTS without include_tool_context should not return tool_context messages"
        )

        # With include_tool_context=True: should find the readFile summary
        results_with_tools = cache.fts_search(
            query="readFile",
            include_tool_context=True,
        )
        assert len(results_with_tools) >= 1, (
            "FTS with include_tool_context=True should return tool_context messages"
        )
        assert any(r.get("content_tier") == "tool_context" for r in results_with_tools)

        # Search for "refactor" which appears in conversation messages
        # Both modes should return it
        results_conv_only = cache.fts_search(query="refactor", include_tool_context=False)
        results_conv_all = cache.fts_search(query="refactor", include_tool_context=True)
        assert len(results_conv_only) >= 1
        assert len(results_conv_all) >= 1
