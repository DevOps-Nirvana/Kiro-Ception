"""Tests for message index assignment and ordering in ide_loader.py.

Tests the _process_execution_logs_for_session function which implements:
- Sequential message_index: user prompt → tool summaries → assistant say responses
- Base offset execution_order * 100 for cross-execution ordering
- Executions ordered by startTime for monotonically increasing indices
- Fallback to session modified timestamp for invalid timestamps
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro_ception.config import Config
from kiro_ception.ide_loader import (
    _parse_execution_start_time,
    _process_execution_logs_for_session,
)
from kiro_ception.models import ContentTier, Source


class TestParseExecutionStartTime:
    """Tests for _parse_execution_start_time helper."""

    def test_parses_valid_millisecond_timestamp(self, tmp_path):
        exec_file = tmp_path / "exec.json"
        exec_file.write_text(json.dumps({"startTime": 1717614000000}), encoding="utf-8")
        fallback = datetime(2020, 1, 1)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result.year == 2024
        assert result != fallback

    def test_parses_valid_iso_timestamp(self, tmp_path):
        exec_file = tmp_path / "exec.json"
        exec_file.write_text(
            json.dumps({"startTime": "2025-03-15T10:30:00Z"}), encoding="utf-8"
        )
        fallback = datetime(2020, 1, 1)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result.year == 2025
        assert result.month == 3

    def test_falls_back_for_missing_start_time(self, tmp_path):
        exec_file = tmp_path / "exec.json"
        exec_file.write_text(json.dumps({"actions": []}), encoding="utf-8")
        fallback = datetime(2020, 6, 15, 12, 0)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result == fallback

    def test_falls_back_for_invalid_start_time(self, tmp_path):
        exec_file = tmp_path / "exec.json"
        exec_file.write_text(json.dumps({"startTime": "not-a-date"}), encoding="utf-8")
        fallback = datetime(2020, 6, 15, 12, 0)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result == fallback

    def test_falls_back_for_null_start_time(self, tmp_path):
        exec_file = tmp_path / "exec.json"
        exec_file.write_text(json.dumps({"startTime": None}), encoding="utf-8")
        fallback = datetime(2020, 6, 15, 12, 0)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result == fallback

    def test_falls_back_for_malformed_json(self, tmp_path):
        exec_file = tmp_path / "exec.json"
        exec_file.write_text("not valid json {{{", encoding="utf-8")
        fallback = datetime(2020, 6, 15, 12, 0)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result == fallback

    def test_falls_back_for_nonexistent_file(self, tmp_path):
        exec_file = tmp_path / "nonexistent.json"
        fallback = datetime(2020, 6, 15, 12, 0)
        result = _parse_execution_start_time(exec_file, fallback)
        assert result == fallback


def _make_exec_log(
    user_prompt: str = "Hello world",
    actions: list[dict] | None = None,
    start_time: int | str | None = None,
    session_id: str = "session-1",
) -> dict:
    """Helper to create a well-formed execution log dict."""
    data: dict = {
        "chatSessionId": session_id,
        "input": {"data": {"userPrompt": user_prompt}},
        "actions": actions or [],
    }
    if start_time is not None:
        data["startTime"] = start_time
    return data


class TestProcessExecutionLogsForSession:
    """Tests for _process_execution_logs_for_session."""

    def _write_exec_files(self, tmp_path: Path, logs: list[dict]) -> list[str]:
        """Write multiple execution log files and return their paths."""
        paths = []
        for i, log_data in enumerate(logs):
            exec_file = tmp_path / f"exec_{i}.json"
            exec_file.write_text(json.dumps(log_data), encoding="utf-8")
            paths.append(str(exec_file))
        return paths

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_single_execution_ordering(self, mock_index, tmp_path):
        """User prompt → tool summaries → assistant say responses within one execution."""
        log = _make_exec_log(
            user_prompt="Fix the bug",
            start_time=1717614000000,
            actions=[
                {
                    "actionType": "readFile",
                    "actionId": "tool-1",
                    "actionState": "completed",
                    "input": {"path": "/src/main.ts"},
                    "output": {"content": "file content"},
                },
                {
                    "actionType": "grep_search",
                    "actionId": "tool-2",
                    "actionState": "completed",
                    "input": {"query": "TODO"},
                    "output": {"results": [{"file": "a.ts", "line": 1}]},
                },
                {
                    "actionType": "say",
                    "actionId": "say-1",
                    "actionState": "completed",
                    "input": {},
                    "output": {"message": "I found the issue and fixed it."},
                },
            ],
        )
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        # Should have: 1 user prompt + 2 tool summaries + 1 assistant response = 4
        assert len(messages) == 4

        user_msgs = [m for m in messages if m.role == "user"]
        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]
        asst_msgs = [m for m in messages if m.role == "assistant" and m.content_tier == ContentTier.CONVERSATION]

        assert len(user_msgs) == 1
        assert len(tool_msgs) == 2
        assert len(asst_msgs) == 1

        # User prompt index < all tool indices < assistant indices
        assert user_msgs[0].message_index < tool_msgs[0].message_index
        assert tool_msgs[-1].message_index < asst_msgs[0].message_index

        # Base offset should be 0 (first execution)
        assert user_msgs[0].message_index == 0
        assert tool_msgs[0].message_index == 1
        assert tool_msgs[1].message_index == 2
        assert asst_msgs[0].message_index == 3

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_multiple_executions_ordered_by_start_time(self, mock_index, tmp_path):
        """Executions sorted by startTime, indices monotonically increasing."""
        # Second execution has EARLIER startTime (should come first after sort)
        log_later = _make_exec_log(
            user_prompt="Second question",
            start_time=1717614002000,  # Later
            actions=[
                {
                    "actionType": "say",
                    "actionId": "say-2",
                    "actionState": "completed",
                    "input": {},
                    "output": {"message": "Second answer."},
                },
            ],
        )
        log_earlier = _make_exec_log(
            user_prompt="First question",
            start_time=1717614001000,  # Earlier
            actions=[
                {
                    "actionType": "say",
                    "actionId": "say-1",
                    "actionState": "completed",
                    "input": {},
                    "output": {"message": "First answer."},
                },
            ],
        )

        # Write them in reverse chronological order to test sorting
        paths = self._write_exec_files(tmp_path, [log_later, log_earlier])
        mock_index.return_value = {"session-1": paths}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        # Should have 4 messages: 2 user prompts + 2 assistant responses
        assert len(messages) == 4

        # Find messages by content
        first_prompt = next(m for m in messages if "First question" in m.searchable_text)
        second_prompt = next(m for m in messages if "Second question" in m.searchable_text)
        first_answer = next(m for m in messages if "First answer" in m.searchable_text)
        second_answer = next(m for m in messages if "Second answer" in m.searchable_text)

        # First execution (earlier startTime) → base = 0 * 100 = 0
        assert first_prompt.message_index == 0
        assert first_answer.message_index == 1

        # Second execution (later startTime) → base = 1 * 100 = 100
        assert second_prompt.message_index == 100
        assert second_answer.message_index == 101

        # Monotonically increasing across executions
        assert first_prompt.message_index < second_prompt.message_index
        assert first_answer.message_index < second_prompt.message_index

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_fallback_timestamp_for_invalid_start_time(self, mock_index, tmp_path):
        """Falls back to session modified timestamp when startTime is invalid."""
        log = _make_exec_log(
            user_prompt="Some question",
            start_time="invalid-not-a-date",
            actions=[],
        )
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        fallback = datetime(2025, 6, 15, 12, 0)
        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=fallback,
            config=config,
        )

        assert len(messages) == 1
        # The message's timestamp should be the fallback
        assert messages[0].timestamp == fallback

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_empty_session_returns_empty(self, mock_index, tmp_path):
        """Returns empty list when no execution logs exist for the session."""
        mock_index.return_value = {}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        assert messages == []

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_execution_with_no_user_prompt_still_indexes_tools_and_responses(
        self, mock_index, tmp_path
    ):
        """Even without a user prompt, tool and assistant messages are indexed."""
        log = {
            "chatSessionId": "session-1",
            "input": {"data": {}},  # No userPrompt
            "startTime": 1717614000000,
            "actions": [
                {
                    "actionType": "readFile",
                    "actionId": "tool-1",
                    "actionState": "completed",
                    "input": {"path": "/src/main.ts"},
                    "output": {"content": "file content"},
                },
                {
                    "actionType": "say",
                    "actionId": "say-1",
                    "actionState": "completed",
                    "input": {},
                    "output": {"message": "Done."},
                },
            ],
        }
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        # No user prompt, but tool + assistant = 2 messages
        assert len(messages) == 2
        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]
        asst_msgs = [m for m in messages if m.role == "assistant" and m.content_tier == ContentTier.CONVERSATION]
        assert len(tool_msgs) == 1
        assert len(asst_msgs) == 1

        # Tool index should still be 1 (base + 1, since base+0 is reserved for prompt)
        assert tool_msgs[0].message_index == 1
        assert asst_msgs[0].message_index == 2

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_excluded_tools_are_skipped(self, mock_index, tmp_path):
        """Tools in the excluded_tools list do not generate messages."""
        log = _make_exec_log(
            user_prompt="Check something",
            start_time=1717614000000,
            actions=[
                {
                    "actionType": "readFile",
                    "actionId": "tool-1",
                    "actionState": "completed",
                    "input": {"path": "/src/main.ts"},
                    "output": {"content": "file content"},
                },
                {
                    "actionType": "list_directory",
                    "actionId": "tool-2",
                    "actionState": "completed",
                    "input": {"path": "/src"},
                    "output": {"entries": ["a.ts", "b.ts"]},
                },
            ],
        )
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        config = Config()
        config.tool_summaries.excluded_tools = ["list_directory"]
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_name == "readFile"

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_three_executions_correct_base_offsets(self, mock_index, tmp_path):
        """Three executions get base offsets 0, 100, 200."""
        logs = [
            _make_exec_log(user_prompt="Q1", start_time=1717614001000, actions=[]),
            _make_exec_log(user_prompt="Q2", start_time=1717614002000, actions=[]),
            _make_exec_log(user_prompt="Q3", start_time=1717614003000, actions=[]),
        ]
        paths = self._write_exec_files(tmp_path, logs)
        mock_index.return_value = {"session-1": paths}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        assert len(messages) == 3
        indices = sorted([m.message_index for m in messages])
        assert indices == [0, 100, 200]

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_malformed_json_execution_skipped(self, mock_index, tmp_path):
        """Malformed execution log files are skipped gracefully."""
        # Write one valid, one malformed
        valid_log = _make_exec_log(user_prompt="Valid prompt", start_time=1717614000000)
        valid_file = tmp_path / "exec_valid.json"
        valid_file.write_text(json.dumps(valid_log), encoding="utf-8")

        bad_file = tmp_path / "exec_bad.json"
        bad_file.write_text("{{not json", encoding="utf-8")

        mock_index.return_value = {"session-1": [str(valid_file), str(bad_file)]}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        # Only the valid execution should produce messages
        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 1
        assert "Valid prompt" in user_msgs[0].searchable_text

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_content_tier_assignment(self, mock_index, tmp_path):
        """Verifies content_tier is properly set for all message types."""
        log = _make_exec_log(
            user_prompt="My question",
            start_time=1717614000000,
            actions=[
                {
                    "actionType": "readFile",
                    "actionId": "tool-1",
                    "actionState": "completed",
                    "input": {"path": "/src/main.ts"},
                    "output": {"content": "file"},
                },
                {
                    "actionType": "say",
                    "actionId": "say-1",
                    "actionState": "completed",
                    "input": {},
                    "output": {"message": "Here's the answer."},
                },
            ],
        )
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        user_msg = next(m for m in messages if m.role == "user")
        tool_msg = next(m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT)
        asst_msg = next(
            m for m in messages
            if m.role == "assistant" and m.content_tier == ContentTier.CONVERSATION
        )

        assert user_msg.content_tier == ContentTier.CONVERSATION
        assert tool_msg.content_tier == ContentTier.TOOL_CONTEXT
        assert tool_msg.role == "assistant"
        assert asst_msg.content_tier == ContentTier.CONVERSATION

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_tool_summaries_follow_action_array_order(self, mock_index, tmp_path):
        """Tool summary indices follow the order of actions in the array."""
        log = _make_exec_log(
            user_prompt="Do stuff",
            start_time=1717614000000,
            actions=[
                {
                    "actionType": "readFile",
                    "actionId": "tool-read",
                    "actionState": "completed",
                    "input": {"path": "/first.ts"},
                    "output": {"content": "content"},
                },
                {
                    "actionType": "grep_search",
                    "actionId": "tool-grep",
                    "actionState": "completed",
                    "input": {"query": "pattern"},
                    "output": {"results": []},
                },
                {
                    "actionType": "execute_pwsh",
                    "actionId": "tool-exec",
                    "actionState": "completed",
                    "input": {"command": "npm test"},
                    "output": {"exitCode": 0, "stdout": "ok"},
                },
            ],
        )
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]
        assert len(tool_msgs) == 3

        # Order should match action array: readFile, grep_search, execute_pwsh
        assert tool_msgs[0].tool_name == "readFile"
        assert tool_msgs[1].tool_name == "grep_search"
        assert tool_msgs[2].tool_name == "execute_pwsh"

        # Indices are sequential
        assert tool_msgs[0].message_index == 1
        assert tool_msgs[1].message_index == 2
        assert tool_msgs[2].message_index == 3

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_nonexistent_files_skipped(self, mock_index, tmp_path):
        """Execution files that don't exist are silently skipped."""
        mock_index.return_value = {"session-1": [str(tmp_path / "missing.json")]}

        config = Config()
        messages = _process_execution_logs_for_session(
            session_id="session-1",
            workspace="/workspace",
            fallback_timestamp=datetime(2026, 1, 1),
            config=config,
        )

        assert messages == []

    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_uses_default_config_when_none(self, mock_index, tmp_path):
        """When config is None, uses get_config() defaults."""
        log = _make_exec_log(user_prompt="Hello", start_time=1717614000000, actions=[])
        paths = self._write_exec_files(tmp_path, [log])
        mock_index.return_value = {"session-1": paths}

        # Call without config parameter
        with patch("kiro_ception.ide_loader.get_config", return_value=Config()):
            messages = _process_execution_logs_for_session(
                session_id="session-1",
                workspace="/workspace",
                fallback_timestamp=datetime(2026, 1, 1),
            )

        assert len(messages) == 1
        assert messages[0].searchable_text == "Hello"
