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
