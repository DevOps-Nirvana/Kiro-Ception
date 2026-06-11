"""Unit tests for search_utils.py — all pure functions, no I/O dependencies."""

from datetime import datetime

import pytest

from kiro_ception.search_utils import (
    build_context_window,
    deduplicate_results,
    format_search_response,
    generate_hint,
    parse_date,
    truncate_content,
)


# --- parse_date ---


class TestParseDate:
    def test_none_input(self):
        assert parse_date(None) is None

    def test_iso_date(self):
        result = parse_date("2026-06-01")
        assert result == datetime(2026, 6, 1)

    def test_iso_datetime(self):
        result = parse_date("2026-06-01T14:30:00")
        assert result == datetime(2026, 6, 1, 14, 30, 0)

    def test_iso_with_z_suffix(self):
        result = parse_date("2026-06-01T14:30:00Z")
        # Z is converted and tzinfo stripped
        assert result == datetime(2026, 6, 1, 14, 30, 0)
        assert result.tzinfo is None

    def test_iso_with_timezone_offset(self):
        result = parse_date("2026-06-01T14:30:00+05:00")
        # timezone is stripped
        assert result == datetime(2026, 6, 1, 14, 30, 0)
        assert result.tzinfo is None

    def test_invalid_string(self):
        assert parse_date("not-a-date") is None

    def test_empty_string(self):
        assert parse_date("") is None

    def test_partial_date(self):
        # "2026-06" is valid ISO 8601 for Python's fromisoformat in 3.11+
        result = parse_date("2026-06")
        # Should either parse or return None without crashing
        assert result is None or isinstance(result, datetime)


# --- truncate_content ---


class TestTruncateContent:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert truncate_content(text) == text

    def test_exact_limit_unchanged(self):
        text = "x" * 2000
        assert truncate_content(text, max_length=2000) == text

    def test_over_limit_truncated(self):
        text = "x" * 2001
        result = truncate_content(text, max_length=2000)
        assert len(result) == 2003  # 2000 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        text = "abcdefghij"
        result = truncate_content(text, max_length=5)
        assert result == "abcde..."

    def test_empty_string(self):
        assert truncate_content("") == ""


# --- deduplicate_results ---


class TestDeduplicateResults:
    def test_empty_list(self):
        assert deduplicate_results([], context_size=3) == []

    def test_single_result_unchanged(self):
        results = [{"session_id": "s1", "message_index": 0, "score": 0.9}]
        deduped = deduplicate_results(results, context_size=3)
        assert len(deduped) == 1

    def test_non_overlapping_same_session(self):
        # Two results far apart in same session — both kept
        results = [
            {"session_id": "s1", "message_index": 0, "score": 0.8},
            {"session_id": "s1", "message_index": 100, "score": 0.7},
        ]
        deduped = deduplicate_results(results, context_size=3)
        assert len(deduped) == 2

    def test_overlapping_keeps_higher_score(self):
        # Two results within 2*context_size — higher score wins
        results = [
            {"session_id": "s1", "message_index": 5, "score": 0.6},
            {"session_id": "s1", "message_index": 8, "score": 0.9},
        ]
        deduped = deduplicate_results(results, context_size=3)
        assert len(deduped) == 1
        assert deduped[0]["score"] == 0.9

    def test_different_sessions_independent(self):
        # Same indices but different sessions — both kept
        results = [
            {"session_id": "s1", "message_index": 5, "score": 0.8},
            {"session_id": "s2", "message_index": 5, "score": 0.7},
        ]
        deduped = deduplicate_results(results, context_size=3)
        assert len(deduped) == 2

    def test_results_sorted_by_score_descending(self):
        results = [
            {"session_id": "s1", "message_index": 0, "score": 0.5},
            {"session_id": "s2", "message_index": 0, "score": 0.9},
            {"session_id": "s3", "message_index": 0, "score": 0.7},
        ]
        deduped = deduplicate_results(results, context_size=3)
        scores = [r["score"] for r in deduped]
        assert scores == sorted(scores, reverse=True)

    def test_boundary_distance_equal_to_dedup_distance(self):
        # Exactly at 2*context_size — should still be considered overlapping
        results = [
            {"session_id": "s1", "message_index": 0, "score": 0.8},
            {"session_id": "s1", "message_index": 6, "score": 0.7},
        ]
        deduped = deduplicate_results(results, context_size=3)
        # distance = 6, dedup_distance = 6 → overlapping (<=), keep higher score
        assert len(deduped) == 1
        assert deduped[0]["score"] == 0.8

    def test_boundary_distance_one_beyond_dedup_distance(self):
        # One beyond 2*context_size — not overlapping
        results = [
            {"session_id": "s1", "message_index": 0, "score": 0.8},
            {"session_id": "s1", "message_index": 7, "score": 0.7},
        ]
        deduped = deduplicate_results(results, context_size=3)
        assert len(deduped) == 2


# --- generate_hint ---


class TestGenerateHint:
    def test_zero_results(self):
        hint = generate_hint(total=0, offset=0, count=0, max_results=10, has_more=False)
        assert "No matches found" in hint

    def test_all_results_shown(self):
        hint = generate_hint(total=5, offset=0, count=5, max_results=10, has_more=False)
        assert "Showing all 5 matches" in hint

    def test_has_more(self):
        hint = generate_hint(total=25, offset=0, count=10, max_results=10, has_more=True)
        assert "1-10 of 25" in hint
        assert "offset: 10" in hint

    def test_final_page(self):
        hint = generate_hint(total=25, offset=20, count=5, max_results=10, has_more=False)
        assert "21-25 of 25" in hint
        assert "final page" in hint


# --- build_context_window ---


class TestBuildContextWindow:
    def _make_messages(self, count: int) -> list[dict]:
        """Helper to create a list of session messages."""
        import time
        base_ts = time.time()
        return [
            {
                "uuid": f"msg-{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "searchable_text": f"Message content {i}",
                "timestamp": base_ts + i * 60,
            }
            for i in range(count)
        ]

    def test_empty_session(self):
        assert build_context_window([], "msg-0", context_size=3) == []

    def test_match_not_found(self):
        messages = self._make_messages(5)
        assert build_context_window(messages, "nonexistent", context_size=3) == []

    def test_match_in_middle(self):
        messages = self._make_messages(10)
        context = build_context_window(messages, "msg-5", context_size=2)
        # Should include msg-3, msg-4, msg-5, msg-6, msg-7
        assert len(context) == 5
        assert context[2]["is_match"] is True
        assert all(not c["is_match"] for c in context[:2])
        assert all(not c["is_match"] for c in context[3:])

    def test_match_at_start(self):
        messages = self._make_messages(10)
        context = build_context_window(messages, "msg-0", context_size=3)
        # Can only go forward, not backward
        assert len(context) == 4  # msg-0 through msg-3
        assert context[0]["is_match"] is True

    def test_match_at_end(self):
        messages = self._make_messages(10)
        context = build_context_window(messages, "msg-9", context_size=3)
        # Can only go backward, not forward
        assert len(context) == 4  # msg-6 through msg-9
        assert context[-1]["is_match"] is True

    def test_context_size_zero(self):
        messages = self._make_messages(10)
        context = build_context_window(messages, "msg-5", context_size=0)
        # Only the match itself
        assert len(context) == 1
        assert context[0]["is_match"] is True

    def test_long_content_truncated(self):
        import time
        messages = [{
            "uuid": "long-msg",
            "role": "assistant",
            "searchable_text": "x" * 3000,
            "timestamp": time.time(),
        }]
        context = build_context_window(messages, "long-msg", context_size=0)
        assert len(context[0]["content"]) == 2003  # 2000 + "..."


# --- format_search_response ---


class TestFormatSearchResponse:
    def test_empty_results(self):
        response = format_search_response(
            scored_results=[],
            query="test",
            offset=0,
            max_results=10,
            context_size=3,
            get_session_messages=lambda sid: [],
        )
        assert response["total_matches"] == 0
        assert response["results"] == []
        assert response["has_more"] is False
        assert "No matches found" in response["hint"]

    def test_pagination_offset(self):
        import time
        results = [
            {
                "uuid": f"msg-{i}",
                "session_id": f"s{i}",
                "workspace": "/test",
                "timestamp": time.time(),
                "role": "user",
                "content": f"content {i}",
                "message_index": i,
                "source": "ide",
                "score": 0.9 - i * 0.1,
            }
            for i in range(5)
        ]

        response = format_search_response(
            scored_results=results,
            query="test",
            offset=2,
            max_results=2,
            context_size=0,
            get_session_messages=lambda sid: [],
        )
        assert response["total_matches"] == 5
        assert len(response["results"]) == 2
        assert response["offset"] == 2
        assert response["has_more"] is True

    def test_has_more_false_on_last_page(self):
        import time
        results = [
            {
                "uuid": "msg-0",
                "session_id": "s0",
                "workspace": "/test",
                "timestamp": time.time(),
                "role": "user",
                "content": "content",
                "message_index": 0,
                "source": "cli",
                "score": 0.9,
            }
        ]

        response = format_search_response(
            scored_results=results,
            query="test",
            offset=0,
            max_results=10,
            context_size=0,
            get_session_messages=lambda sid: [],
        )
        assert response["has_more"] is False


# --- build_context_window with content_tier ---


class TestBuildContextWindowContentTier:
    """Tests for content_tier support in build_context_window."""

    def _make_messages_with_tiers(self, count: int, tool_indices: list[int] | None = None) -> list[dict]:
        """Helper to create messages with content_tier field.

        Args:
            count: Number of messages to create.
            tool_indices: Indices that should be tool_context. Others are conversation.
        """
        import time
        base_ts = time.time()
        tool_indices = tool_indices or []
        return [
            {
                "uuid": f"msg-{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "searchable_text": f"Message content {i}",
                "timestamp": base_ts + i * 60,
                "message_index": i,
                "content_tier": "tool_context" if i in tool_indices else "conversation",
            }
            for i in range(count)
        ]

    def test_content_tier_included_in_output(self):
        """Each context message includes a content_tier field."""
        messages = self._make_messages_with_tiers(5)
        context = build_context_window(messages, "msg-2", context_size=2)
        assert len(context) == 5
        for msg in context:
            assert "content_tier" in msg
            assert msg["content_tier"] == "conversation"

    def test_content_tier_defaults_to_conversation(self):
        """Messages without content_tier field default to 'conversation'."""
        import time
        messages = [
            {
                "uuid": "msg-0",
                "role": "user",
                "searchable_text": "Hello",
                "timestamp": time.time(),
                "message_index": 0,
                # No content_tier field
            }
        ]
        context = build_context_window(messages, "msg-0", context_size=0)
        assert context[0]["content_tier"] == "conversation"

    def test_tool_context_messages_interleaved(self):
        """Both tiers interleaved by message_index order."""
        messages = self._make_messages_with_tiers(7, tool_indices=[1, 3, 5])
        context = build_context_window(messages, "msg-3", context_size=3)
        # Window: msg-0 through msg-6 (all 7)
        assert len(context) == 7
        # Check interleaving: indices 1, 3, 5 are tool_context
        assert context[0]["content_tier"] == "conversation"
        assert context[1]["content_tier"] == "tool_context"
        assert context[2]["content_tier"] == "conversation"
        assert context[3]["content_tier"] == "tool_context"
        assert context[4]["content_tier"] == "conversation"
        assert context[5]["content_tier"] == "tool_context"
        assert context[6]["content_tier"] == "conversation"

    def test_both_tiers_count_toward_context_size(self):
        """context_size counts both conversation and tool_context messages."""
        messages = self._make_messages_with_tiers(10, tool_indices=[2, 3, 4, 5, 6, 7])
        # Match at index 5, context_size=2 → indices 3, 4, 5, 6, 7
        context = build_context_window(messages, "msg-5", context_size=2)
        assert len(context) == 5

    def test_overflow_logic_not_triggered_at_20(self):
        """Exactly 20 messages should NOT trigger overflow logic."""
        # Need exactly 20 messages in the window
        messages = self._make_messages_with_tiers(20, tool_indices=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        # context_size=10, match in middle → up to 21 messages but capped by list size
        context = build_context_window(messages, "msg-10", context_size=10)
        # With 20 messages, match at 10, context_size=10: start=0, end=20 → 20 messages
        assert len(context) == 20
        # All tool_context should still be present
        tool_count = sum(1 for m in context if m["content_tier"] == "tool_context")
        assert tool_count == 10

    def test_overflow_logic_triggered_above_20(self):
        """Window > 20 messages drops oldest tool_context first."""
        # Create 25 messages with some as tool_context
        messages = self._make_messages_with_tiers(25, tool_indices=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
        # Match at 12, context_size=12 → window would be indices 0-24 (25 messages)
        context = build_context_window(messages, "msg-12", context_size=12)
        # Should be trimmed to 20
        assert len(context) <= 20
        # All conversation messages should be retained
        conv_count = sum(1 for m in context if m["content_tier"] == "conversation")
        # Original conversation messages in 0-24: indices 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 = 13
        assert conv_count == 13
        # The match should always be present
        match_msgs = [m for m in context if m["is_match"]]
        assert len(match_msgs) == 1

    def test_overflow_drops_oldest_tool_context(self):
        """Overflow drops oldest tool_context messages first (lowest indices)."""
        import time
        base_ts = time.time()
        # Create 22 messages: 12 conversation, 10 tool_context
        messages = []
        for i in range(22):
            tier = "tool_context" if i < 10 else "conversation"
            messages.append({
                "uuid": f"msg-{i}",
                "role": "assistant",
                "searchable_text": f"Content {i}",
                "timestamp": base_ts + i * 60,
                "message_index": i,
                "content_tier": tier,
            })
        # Match at index 11 (conversation), context_size=11 → indices 0-21 (22 messages)
        context = build_context_window(messages, "msg-11", context_size=11)
        # Should trim to 20 by dropping 2 oldest tool_context (msg-0, msg-1)
        assert len(context) == 20
        # All 12 conversation messages retained
        conv_count = sum(1 for m in context if m["content_tier"] == "conversation")
        assert conv_count == 12
        # 8 tool_context remaining (10 - 2 dropped)
        tool_count = sum(1 for m in context if m["content_tier"] == "tool_context")
        assert tool_count == 8

    def test_overflow_retains_all_conversation(self):
        """Even if window is large, all conversation messages are kept."""
        import time
        base_ts = time.time()
        # Create 30 messages: 15 conversation interspersed with 15 tool_context
        messages = []
        for i in range(30):
            tier = "tool_context" if i % 2 == 1 else "conversation"
            messages.append({
                "uuid": f"msg-{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "searchable_text": f"Content {i}",
                "timestamp": base_ts + i * 60,
                "message_index": i,
                "content_tier": tier,
            })
        context = build_context_window(messages, "msg-14", context_size=15)
        # Window: 0-29 (30 messages), overflow triggered → trim to 20
        assert len(context) <= 20
        # All 15 conversation messages should be retained
        conv_count = sum(1 for m in context if m["content_tier"] == "conversation")
        assert conv_count == 15

    def test_overflow_with_match_as_tool_context(self):
        """Matched message is never dropped, even if it's tool_context."""
        import time
        base_ts = time.time()
        # Create 22 messages, all tool_context except the match
        messages = []
        for i in range(22):
            messages.append({
                "uuid": f"msg-{i}",
                "role": "assistant",
                "searchable_text": f"Content {i}",
                "timestamp": base_ts + i * 60,
                "message_index": i,
                "content_tier": "tool_context",
            })
        # Match at index 11 which is tool_context
        context = build_context_window(messages, "msg-11", context_size=11)
        # Should trim to 20 but keep the matched message
        assert len(context) <= 20
        match_msgs = [m for m in context if m["is_match"]]
        assert len(match_msgs) == 1

    def test_messages_ordered_by_message_index_after_overflow(self):
        """After overflow logic, messages remain in message_index order."""
        import time
        base_ts = time.time()
        messages = []
        for i in range(22):
            tier = "tool_context" if i < 10 else "conversation"
            messages.append({
                "uuid": f"msg-{i}",
                "role": "assistant",
                "searchable_text": f"Content {i}",
                "timestamp": base_ts + i * 60,
                "message_index": i,
                "content_tier": tier,
            })
        context = build_context_window(messages, "msg-11", context_size=11)
        # Verify the timestamps are in increasing order (proxy for message_index order)
        timestamps = [m["timestamp"] for m in context]
        assert timestamps == sorted(timestamps)
