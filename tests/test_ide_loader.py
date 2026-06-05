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
