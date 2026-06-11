"""Unit tests for MCP server interface changes — include_tool_context parameter.

Tests that:
- search_project_history accepts and passes include_tool_context
- search_global_history accepts and passes include_tool_context
- Backward compatibility: omitting include_tool_context works identically to unmodified behavior
- Context messages include content_tier field

Requirements: 9.1, 9.2
"""

import time

from unittest.mock import MagicMock, patch

import pytest

from kiro_ception.search_utils import build_context_window


# --- Shared fixtures ---


@pytest.fixture
def mock_leader_setup():
    """Set up mocks for a leader instance."""
    manager = MagicMock()
    manager.is_leader = True

    indexer = MagicMock()
    indexer.backend = MagicMock()
    indexer.backend.fingerprint.return_value = "openai-compatible:localhost:qwen3:1024"
    indexer.cache = MagicMock()
    indexer.cache.db_path = "/tmp/cache.db"

    return manager, indexer


# --- search_project_history: include_tool_context ---


class TestSearchProjectHistoryIncludeToolContext:
    """Tests that search_project_history accepts and passes include_tool_context."""

    def test_include_tool_context_true_passed_to_search(self, mock_leader_setup):
        """include_tool_context=True is forwarded to the search layer."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server._get_current_workspace", return_value="/my/project"),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_project_history

            search_project_history(query="test", include_tool_context=True)

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["include_tool_context"] is True

    def test_include_tool_context_false_passed_to_search(self, mock_leader_setup):
        """include_tool_context=False is forwarded to the search layer."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server._get_current_workspace", return_value="/my/project"),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_project_history

            search_project_history(query="test", include_tool_context=False)

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["include_tool_context"] is False

    def test_omitting_include_tool_context_defaults_false(self, mock_leader_setup):
        """Omitting include_tool_context defaults to False (backward compatible)."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server._get_current_workspace", return_value="/my/project"),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_project_history

            # Call without include_tool_context
            search_project_history(query="test")

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["include_tool_context"] is False


# --- search_global_history: include_tool_context ---


class TestSearchGlobalHistoryIncludeToolContext:
    """Tests that search_global_history accepts and passes include_tool_context."""

    def test_include_tool_context_true_passed_to_search(self, mock_leader_setup):
        """include_tool_context=True is forwarded to the search layer."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history

            search_global_history(query="test", include_tool_context=True)

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["include_tool_context"] is True

    def test_include_tool_context_false_passed_to_search(self, mock_leader_setup):
        """include_tool_context=False is forwarded to the search layer."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history

            search_global_history(query="test", include_tool_context=False)

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["include_tool_context"] is False

    def test_omitting_include_tool_context_defaults_false(self, mock_leader_setup):
        """Omitting include_tool_context defaults to False (backward compatible)."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "test", "total_matches": 0}
            from kiro_ception.server import search_global_history

            # Call without include_tool_context
            search_global_history(query="test")

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["include_tool_context"] is False


# --- Backward compatibility ---


class TestBackwardCompatibility:
    """Omitting include_tool_context produces identical behavior to unmodified Kiro Ception."""

    def test_project_search_same_params_as_before(self, mock_leader_setup):
        """search_project_history without include_tool_context passes all original params unchanged."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server._get_current_workspace", return_value="/workspace"),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "hello", "total_matches": 0}
            from kiro_ception.server import search_project_history

            search_project_history(
                query="hello",
                after="2025-01-01",
                before="2025-12-31",
                context_size=5,
                threshold=0.3,
                max_results=20,
                offset=10,
            )

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["query"] == "hello"
            assert call_kwargs["workspace"] == "/workspace"
            assert call_kwargs["after"] == "2025-01-01"
            assert call_kwargs["before"] == "2025-12-31"
            assert call_kwargs["context_size"] == 5
            assert call_kwargs["threshold"] == 0.3
            assert call_kwargs["max_results"] == 20
            assert call_kwargs["offset"] == 10
            # Default: include_tool_context is False
            assert call_kwargs["include_tool_context"] is False

    def test_global_search_same_params_as_before(self, mock_leader_setup):
        """search_global_history without include_tool_context passes all original params unchanged."""
        manager, indexer = mock_leader_setup

        with (
            patch("kiro_ception.server.get_instance_manager", return_value=manager),
            patch("kiro_ception.server.get_background_indexer", return_value=indexer),
            patch("kiro_ception.server._initialized", True),
            patch("kiro_ception.server.search") as mock_search,
        ):
            mock_search.return_value = {"results": [], "query": "world", "total_matches": 0}
            from kiro_ception.server import search_global_history
            from kiro_ception.models import Source

            search_global_history(
                query="world",
                after="2025-06-01",
                before="2025-06-30",
                context_size=2,
                threshold=0.1,
                max_results=5,
                offset=0,
                source="cli",
            )

            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["query"] == "world"
            assert call_kwargs["workspace"] is None
            assert call_kwargs["source"] == Source.CLI
            assert call_kwargs["after"] == "2025-06-01"
            assert call_kwargs["before"] == "2025-06-30"
            assert call_kwargs["context_size"] == 2
            assert call_kwargs["threshold"] == 0.1
            assert call_kwargs["max_results"] == 5
            assert call_kwargs["offset"] == 0
            # Default: include_tool_context is False
            assert call_kwargs["include_tool_context"] is False


# --- Context messages include content_tier field ---


class TestContextMessagesContentTier:
    """Tests that context messages returned by the search layer include content_tier."""

    def _make_session_messages(self):
        """Create a set of session messages with mixed content tiers."""
        base_ts = time.time()
        return [
            {
                "uuid": "msg-0",
                "role": "user",
                "searchable_text": "How do I deploy to production?",
                "timestamp": base_ts,
                "message_index": 0,
                "content_tier": "conversation",
            },
            {
                "uuid": "msg-1",
                "role": "assistant",
                "searchable_text": "[grep_search] deploy → completed",
                "timestamp": base_ts + 10,
                "message_index": 1,
                "content_tier": "tool_context",
            },
            {
                "uuid": "msg-2",
                "role": "assistant",
                "searchable_text": "[execute_pwsh] npm run deploy → completed",
                "timestamp": base_ts + 20,
                "message_index": 2,
                "content_tier": "tool_context",
            },
            {
                "uuid": "msg-3",
                "role": "assistant",
                "searchable_text": "I've deployed the application to production.",
                "timestamp": base_ts + 30,
                "message_index": 3,
                "content_tier": "conversation",
            },
        ]

    def test_context_window_includes_content_tier_field(self):
        """Every context message must include a content_tier field."""
        messages = self._make_session_messages()
        context = build_context_window(messages, "msg-0", context_size=3)

        assert len(context) == 4
        for msg in context:
            assert "content_tier" in msg

    def test_context_window_conversation_tier_correct(self):
        """Conversation messages have content_tier='conversation'."""
        messages = self._make_session_messages()
        context = build_context_window(messages, "msg-0", context_size=3)

        # msg-0 (user) and msg-3 (assistant say) are conversation
        assert context[0]["content_tier"] == "conversation"
        assert context[3]["content_tier"] == "conversation"

    def test_context_window_tool_context_tier_correct(self):
        """Tool context messages have content_tier='tool_context'."""
        messages = self._make_session_messages()
        context = build_context_window(messages, "msg-0", context_size=3)

        # msg-1 and msg-2 are tool_context
        assert context[1]["content_tier"] == "tool_context"
        assert context[2]["content_tier"] == "tool_context"

    def test_context_window_defaults_to_conversation_when_missing(self):
        """Messages without content_tier field default to 'conversation'."""
        base_ts = time.time()
        messages = [
            {
                "uuid": "msg-0",
                "role": "user",
                "searchable_text": "hello",
                "timestamp": base_ts,
                "message_index": 0,
                # No content_tier field — should default to "conversation"
            },
        ]
        context = build_context_window(messages, "msg-0", context_size=0)

        assert len(context) == 1
        assert context[0]["content_tier"] == "conversation"

    def test_context_window_preserves_tier_order(self):
        """Context messages are interleaved in message_index order regardless of tier."""
        messages = self._make_session_messages()
        context = build_context_window(messages, "msg-0", context_size=3)

        # Messages should be in order: conversation, tool_context, tool_context, conversation
        expected_tiers = ["conversation", "tool_context", "tool_context", "conversation"]
        actual_tiers = [msg["content_tier"] for msg in context]
        assert actual_tiers == expected_tiers
