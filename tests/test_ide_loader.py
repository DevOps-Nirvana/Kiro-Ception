"""Integration tests for ide_loader.py — message filtering and text processing.

Tests the pure/semi-pure functions: _should_skip_message, _replace_code_blocks,
_extract_text_from_content, _parse_timestamp.
"""

from datetime import datetime

import pytest

from kiro_ception.ide_loader import (
    _extract_text_from_content,
    _parse_timestamp,
    _replace_code_blocks,
    _should_skip_message,
)


# --- _should_skip_message ---


class TestShouldSkipMessage:
    def test_normal_user_message_not_skipped(self):
        assert _should_skip_message("user", "How do I fix this bug?") is False

    def test_normal_assistant_message_not_skipped(self):
        assert _should_skip_message("assistant", "# System Prompt content") is False

    def test_system_prompt_prefix_skipped(self):
        assert _should_skip_message("user", "# System Prompt\nYou are an AI assistant") is True

    def test_identity_tag_skipped(self):
        assert _should_skip_message("user", "<identity>\nYou are Kiro") is True

    def test_system_tag_skipped(self):
        assert _should_skip_message("user", "<system>\nFollow these rules") is True

    def test_instructions_tag_skipped(self):
        assert _should_skip_message("user", "<instructions>\nDo this") is True

    def test_included_rules_skipped(self):
        assert _should_skip_message("user", "## Included Rules (global)") is True

    def test_steering_reminder_skipped(self):
        assert _should_skip_message("user", "<steering-reminder>Remember to...") is True

    def test_spec_task_instructions_skipped(self):
        assert _should_skip_message(
            "user",
            "Follow these instructions for user requests related to spec tasks and do...",
        ) is True

    def test_workspace_context_skipped(self):
        assert _should_skip_message(
            "user",
            "You are operating in a workspace with files and folders...",
        ) is True

    def test_context_transfer_skipped(self):
        assert _should_skip_message("user", "CONTEXT TRANSFER: from previous session") is True

    def test_assistant_role_never_skipped(self):
        """Assistant messages are never skipped regardless of content."""
        assert _should_skip_message("assistant", "# System Prompt") is False
        assert _should_skip_message("assistant", "<identity>") is False
        assert _should_skip_message("assistant", "## Included Rules") is False

    def test_prefix_must_be_at_start(self):
        """Only messages starting with the prefix are skipped."""
        assert _should_skip_message("user", "I said # System Prompt") is False
        assert _should_skip_message("user", "look at <identity> tag") is False


# --- _replace_code_blocks ---


class TestReplaceCodeBlocks:
    def test_no_code_blocks(self):
        text = "This is plain text without any code."
        assert _replace_code_blocks(text) == text

    def test_python_code_block(self):
        text = "Here's code:\n```python\ndef foo():\n    pass\n```\nDone."
        result = _replace_code_blocks(text)
        assert result == "Here's code:\n[code:python]\nDone."

    def test_javascript_code_block(self):
        text = "```javascript\nconsole.log('hi');\n```"
        result = _replace_code_blocks(text)
        assert result == "[code:javascript]"

    def test_no_language_tag(self):
        text = "```\nsome code\n```"
        result = _replace_code_blocks(text)
        assert result == "[code]"

    def test_multiple_code_blocks(self):
        text = "First:\n```python\nx=1\n```\nSecond:\n```bash\necho hi\n```\nEnd."
        result = _replace_code_blocks(text)
        assert "[code:python]" in result
        assert "[code:bash]" in result
        assert "x=1" not in result
        assert "echo hi" not in result

    def test_multiline_code_block(self):
        text = "```python\nclass Foo:\n    def bar(self):\n        return 42\n\n    def baz(self):\n        pass\n```"
        result = _replace_code_blocks(text)
        assert result == "[code:python]"
        assert "class Foo" not in result

    def test_preserves_surrounding_text(self):
        text = "Before code.\n```yaml\nkey: value\n```\nAfter code."
        result = _replace_code_blocks(text)
        assert "Before code." in result
        assert "After code." in result
        assert "key: value" not in result

    def test_language_tag_case_insensitive(self):
        text = "```Python\nfoo()\n```"
        result = _replace_code_blocks(text)
        assert result == "[code:python]"


# --- _extract_text_from_content ---


class TestExtractTextFromContent:
    def test_none_returns_empty(self):
        assert _extract_text_from_content(None) == ""

    def test_string_passthrough(self):
        assert _extract_text_from_content("hello world") == "hello world"

    def test_dict_with_text_key(self):
        assert _extract_text_from_content({"text": "from dict"}) == "from dict"

    def test_dict_without_text_key(self):
        assert _extract_text_from_content({"other": "value"}) == ""

    def test_list_of_text_items(self):
        content = [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ]
        result = _extract_text_from_content(content)
        assert "part one" in result
        assert "part two" in result

    def test_list_of_strings(self):
        content = ["line one", "line two"]
        result = _extract_text_from_content(content)
        assert "line one" in result
        assert "line two" in result

    def test_list_mixed_types(self):
        content = [
            {"type": "text", "text": "from dict"},
            "plain string",
        ]
        result = _extract_text_from_content(content)
        assert "from dict" in result
        assert "plain string" in result

    def test_empty_list(self):
        assert _extract_text_from_content([]) == ""


# --- _parse_timestamp ---


class TestParseTimestamp:
    def test_none_returns_none(self):
        assert _parse_timestamp(None) is None

    def test_integer_milliseconds(self):
        # 1717614000000 ms = 2024-06-05 some time
        result = _parse_timestamp(1717614000000)
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_iso_string(self):
        result = _parse_timestamp("2026-06-01T14:30:00")
        assert result == datetime(2026, 6, 1, 14, 30, 0)

    def test_iso_string_with_z(self):
        result = _parse_timestamp("2026-06-01T14:30:00Z")
        assert result is not None
        assert result.tzinfo is None  # tzinfo stripped

    def test_invalid_string(self):
        assert _parse_timestamp("not-a-date") is None

    def test_zero_timestamp(self):
        result = _parse_timestamp(0)
        assert isinstance(result, datetime)


# --- _extract_user_prompt_from_execution_log ---

import json
import tempfile
from pathlib import Path

from kiro_ception.ide_loader import _extract_user_prompt_from_execution_log
from kiro_ception.models import ContentTier


class TestExtractUserPromptFromExecutionLog:
    """Tests for extracting user prompts from execution log files."""

    def _write_exec_log(self, tmp_path: Path, data: dict) -> Path:
        """Helper to write execution log JSON to a temp file."""
        exec_file = tmp_path / "test_exec.json"
        exec_file.write_text(json.dumps(data), encoding="utf-8")
        return exec_file

    def test_extracts_user_prompt(self, tmp_path):
        """Extracts input.data.userPrompt as authoritative user message."""
        data = {
            "input": {"data": {"userPrompt": "How do I fix this bug?"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is not None
        assert result.searchable_text == "How do I fix this bug?"
        assert result.role == "user"
        assert result.content_tier == ContentTier.CONVERSATION

    def test_applies_code_block_replacement(self, tmp_path):
        """Applies _replace_code_blocks to extracted prompt."""
        data = {
            "input": {"data": {"userPrompt": "Fix this:\n```python\ndef foo():\n    pass\n```"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is not None
        assert "[code:python]" in result.searchable_text
        assert "def foo():" not in result.searchable_text

    def test_applies_should_skip_filter(self, tmp_path):
        """Filters out system prompts via _should_skip_message."""
        data = {
            "input": {"data": {"userPrompt": "# System Prompt\nYou are an AI assistant"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_returns_none_for_absent_user_prompt(self, tmp_path):
        """Returns None when input.data.userPrompt is absent (fallback case)."""
        data = {
            "input": {"data": {"otherField": "value"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_returns_none_for_null_user_prompt(self, tmp_path):
        """Returns None when input.data.userPrompt is null."""
        data = {
            "input": {"data": {"userPrompt": None}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_returns_none_for_empty_user_prompt(self, tmp_path):
        """Returns None when input.data.userPrompt is empty string."""
        data = {
            "input": {"data": {"userPrompt": ""}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_returns_none_for_whitespace_only_prompt(self, tmp_path):
        """Returns None when input.data.userPrompt is just whitespace."""
        data = {
            "input": {"data": {"userPrompt": "   \n\t  "}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_handles_malformed_json(self, tmp_path):
        """Logs warning and returns None for malformed JSON."""
        exec_file = tmp_path / "bad.json"
        exec_file.write_text("not valid json {{{", encoding="utf-8")
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_handles_missing_input_field(self, tmp_path):
        """Returns None gracefully when 'input' field is missing."""
        data = {"actions": []}
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_handles_input_not_dict(self, tmp_path):
        """Returns None when 'input' is not a dict."""
        data = {"input": "not a dict", "actions": []}
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_handles_data_field_not_dict(self, tmp_path):
        """Returns None when 'input.data' is not a dict."""
        data = {"input": {"data": "string"}, "actions": []}
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_uses_start_time_for_timestamp(self, tmp_path):
        """Uses startTime from execution log as message timestamp."""
        data = {
            "input": {"data": {"userPrompt": "hello"}},
            "startTime": 1717614000000,  # 2024-06-05
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is not None
        assert result.timestamp.year == 2024

    def test_falls_back_to_provided_timestamp(self, tmp_path):
        """Uses fallback_timestamp when startTime is absent."""
        data = {
            "input": {"data": {"userPrompt": "hello"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        fallback = datetime(2025, 3, 15, 10, 30)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", fallback
        )
        assert result is not None
        assert result.timestamp == fallback

    def test_generates_stable_uuid(self, tmp_path):
        """Generates a deterministic UUID based on session_id and file stem."""
        data = {
            "input": {"data": {"userPrompt": "hello"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is not None
        assert result.uuid == f"session-1-prompt-{exec_file.stem}"

    def test_handles_nonexistent_file(self, tmp_path):
        """Returns None for a file that doesn't exist."""
        exec_file = tmp_path / "nonexistent.json"
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_user_prompt_not_string(self, tmp_path):
        """Returns None when userPrompt is not a string (e.g., integer)."""
        data = {
            "input": {"data": {"userPrompt": 12345}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is None

    def test_source_is_ide(self, tmp_path):
        """Extracted messages have source=IDE."""
        data = {
            "input": {"data": {"userPrompt": "hello"}},
            "actions": [],
        }
        exec_file = self._write_exec_log(tmp_path, data)
        result = _extract_user_prompt_from_execution_log(
            exec_file, "session-1", "/workspace", datetime(2026, 1, 1)
        )
        assert result is not None
        from kiro_ception.models import Source
        assert result.source == Source.IDE
