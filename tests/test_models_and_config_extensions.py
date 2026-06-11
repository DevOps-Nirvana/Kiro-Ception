"""Unit tests for model extensions (ContentTier, IndexedMessage) and config parsing (ToolSummariesConfig).

Requirements: 6.1, 8.1
"""

from datetime import datetime

import pytest

from kiro_ception.config import Config, ToolSummariesConfig
from kiro_ception.models import ContentTier, IndexedMessage, Source


# --- ContentTier enum ---


class TestContentTier:
    """Tests for ContentTier enum serialization and deserialization."""

    def test_conversation_value(self):
        assert ContentTier.CONVERSATION == "conversation"
        assert ContentTier.CONVERSATION.value == "conversation"

    def test_tool_context_value(self):
        assert ContentTier.TOOL_CONTEXT == "tool_context"
        assert ContentTier.TOOL_CONTEXT.value == "tool_context"

    def test_from_string_conversation(self):
        tier = ContentTier("conversation")
        assert tier is ContentTier.CONVERSATION

    def test_from_string_tool_context(self):
        tier = ContentTier("tool_context")
        assert tier is ContentTier.TOOL_CONTEXT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ContentTier("invalid")

    def test_value_equals_string(self):
        """ContentTier values compare equal to plain strings."""
        assert ContentTier.CONVERSATION == "conversation"
        assert ContentTier.TOOL_CONTEXT == "tool_context"

    def test_is_string_subclass(self):
        """ContentTier is a str enum, so instances are also strings."""
        assert isinstance(ContentTier.CONVERSATION, str)
        assert isinstance(ContentTier.TOOL_CONTEXT, str)


# --- IndexedMessage defaults ---


class TestIndexedMessageDefaults:
    """Tests for IndexedMessage default field values."""

    def _make_message(self, **overrides) -> IndexedMessage:
        """Helper to create an IndexedMessage with minimal required fields."""
        defaults = {
            "uuid": "test-uuid",
            "session_id": "session-1",
            "workspace": "/test/workspace",
            "timestamp": datetime(2025, 1, 15, 10, 0, 0),
            "role": "user",
            "searchable_text": "hello world",
        }
        defaults.update(overrides)
        return IndexedMessage(**defaults)

    def test_content_tier_defaults_to_conversation(self):
        msg = self._make_message()
        assert msg.content_tier == ContentTier.CONVERSATION
        assert msg.content_tier.value == "conversation"

    def test_tool_name_defaults_to_none(self):
        msg = self._make_message()
        assert msg.tool_name is None

    def test_content_tier_can_be_set_to_tool_context(self):
        msg = self._make_message(content_tier=ContentTier.TOOL_CONTEXT)
        assert msg.content_tier == ContentTier.TOOL_CONTEXT

    def test_content_tier_accepts_string_value(self):
        msg = self._make_message(content_tier="tool_context")
        assert msg.content_tier == ContentTier.TOOL_CONTEXT

    def test_tool_name_can_be_set(self):
        msg = self._make_message(tool_name="readFile")
        assert msg.tool_name == "readFile"

    def test_serialization_includes_content_tier(self):
        msg = self._make_message(content_tier=ContentTier.TOOL_CONTEXT, tool_name="grep_search")
        data = msg.model_dump()
        assert data["content_tier"] == "tool_context"
        assert data["tool_name"] == "grep_search"

    def test_serialization_defaults(self):
        msg = self._make_message()
        data = msg.model_dump()
        assert data["content_tier"] == "conversation"
        assert data["tool_name"] is None

    def test_deserialization_from_dict(self):
        data = {
            "uuid": "test-uuid",
            "session_id": "session-1",
            "workspace": "/test/workspace",
            "timestamp": "2025-01-15T10:00:00",
            "role": "assistant",
            "searchable_text": "[readFile] /src/app.ts → completed",
            "content_tier": "tool_context",
            "tool_name": "readFile",
        }
        msg = IndexedMessage(**data)
        assert msg.content_tier == ContentTier.TOOL_CONTEXT
        assert msg.tool_name == "readFile"

    def test_message_index_default(self):
        msg = self._make_message()
        assert msg.message_index == 0

    def test_source_default(self):
        msg = self._make_message()
        assert msg.source == Source.CLI


# --- ToolSummariesConfig defaults and custom values ---


class TestToolSummariesConfig:
    """Tests for ToolSummariesConfig dataclass defaults and custom values."""

    def test_default_excluded_tools_is_empty(self):
        config = ToolSummariesConfig()
        assert config.excluded_tools == []

    def test_default_max_summary_length(self):
        config = ToolSummariesConfig()
        assert config.max_summary_length == 800

    def test_default_include_meaningful_output(self):
        config = ToolSummariesConfig()
        assert config.include_meaningful_output is True

    def test_custom_excluded_tools(self):
        config = ToolSummariesConfig(excluded_tools=["readFile", "list_directory"])
        assert config.excluded_tools == ["readFile", "list_directory"]

    def test_custom_max_summary_length(self):
        config = ToolSummariesConfig(max_summary_length=500)
        assert config.max_summary_length == 500

    def test_custom_include_meaningful_output_false(self):
        config = ToolSummariesConfig(include_meaningful_output=False)
        assert config.include_meaningful_output is False

    def test_excluded_tools_list_is_independent(self):
        """Each instance gets its own list (no shared mutable default)."""
        config1 = ToolSummariesConfig()
        config2 = ToolSummariesConfig()
        config1.excluded_tools.append("readFile")
        assert config2.excluded_tools == []


# --- Config loading with missing [tool_summaries] section ---


class TestConfigToolSummariesParsing:
    """Tests for Config.from_dict handling of the [tool_summaries] section."""

    def test_missing_tool_summaries_section_uses_defaults(self):
        """When [tool_summaries] is absent from config dict, defaults are used."""
        data = {
            "embedding": {"model": "all-MiniLM-L6-v2"},
        }
        config = Config.from_dict(data)
        assert config.tool_summaries.excluded_tools == []
        assert config.tool_summaries.max_summary_length == 800
        assert config.tool_summaries.include_meaningful_output is True

    def test_empty_dict_uses_tool_summaries_defaults(self):
        config = Config.from_dict({})
        assert config.tool_summaries.excluded_tools == []
        assert config.tool_summaries.max_summary_length == 800
        assert config.tool_summaries.include_meaningful_output is True

    def test_tool_summaries_section_parsed(self):
        data = {
            "tool_summaries": {
                "excluded_tools": ["readFile", "fs_write"],
                "max_summary_length": 500,
                "include_meaningful_output": False,
            }
        }
        config = Config.from_dict(data)
        assert config.tool_summaries.excluded_tools == ["readFile", "fs_write"]
        assert config.tool_summaries.max_summary_length == 500
        assert config.tool_summaries.include_meaningful_output is False

    def test_partial_tool_summaries_section(self):
        """Partial [tool_summaries] merges with defaults for unspecified fields."""
        data = {
            "tool_summaries": {
                "max_summary_length": 1200,
            }
        }
        config = Config.from_dict(data)
        assert config.tool_summaries.excluded_tools == []  # default
        assert config.tool_summaries.max_summary_length == 1200  # custom
        assert config.tool_summaries.include_meaningful_output is True  # default

    def test_tool_summaries_in_diff_configs(self):
        """diff_configs detects changes in tool_summaries settings."""
        from kiro_ception.config import diff_configs

        old = Config()
        new = Config(tool_summaries=ToolSummariesConfig(
            excluded_tools=["readFile"],
            max_summary_length=500,
        ))
        changes = diff_configs(old, new)
        keys = {c["key"] for c in changes}
        assert "tool_summaries.excluded_tools" in keys
        assert "tool_summaries.max_summary_length" in keys

    def test_tool_summaries_diff_is_safe_impact(self):
        """Tool summaries config changes are safe (hot-reloadable)."""
        from kiro_ception.config import diff_configs

        old = Config()
        new = Config(tool_summaries=ToolSummariesConfig(include_meaningful_output=False))
        changes = diff_configs(old, new)
        for c in changes:
            assert c["impact"] == "safe"
