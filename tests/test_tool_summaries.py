"""Unit tests for tool_summaries module."""

import json

import pytest

from kiro_ception.tool_summaries import (
    ToolSummaryConfig,
    _describe_action,
    _extract_meaningful_output,
    _extract_outcome,
    generate_tool_summary,
)


class TestToolSummaryConfig:
    """Tests for ToolSummaryConfig defaults and custom values."""

    def test_defaults(self):
        config = ToolSummaryConfig()
        assert config.excluded_tools == []
        assert config.max_summary_length == 800
        assert config.include_meaningful_output is True

    def test_custom_values(self):
        config = ToolSummaryConfig(
            excluded_tools=["readFile", "list_directory"],
            max_summary_length=500,
            include_meaningful_output=False,
        )
        assert config.excluded_tools == ["readFile", "list_directory"]
        assert config.max_summary_length == 500
        assert config.include_meaningful_output is False


class TestGenerateToolSummary:
    """Tests for generate_tool_summary function."""

    def test_returns_none_for_say_actions(self):
        action = {"actionType": "say", "actionState": "completed", "output": {"message": "Hello"}}
        config = ToolSummaryConfig()
        assert generate_tool_summary(action, config) is None

    def test_returns_none_for_excluded_tools(self):
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/src/file.ts"},
            "output": {"content": "file contents"},
        }
        config = ToolSummaryConfig(excluded_tools=["readFile"])
        assert generate_tool_summary(action, config) is None

    def test_basic_format(self):
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/src/server.ts"},
            "output": {"filePath": "/src/server.ts", "content": "..."},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert result == "[readFile] /src/server.ts → completed"

    def test_format_with_arrow(self):
        """Verify the format uses → (unicode arrow)."""
        action = {
            "actionType": "fs_write",
            "actionState": "completed",
            "input": {"path": "/src/db.ts"},
            "output": {},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert "→" in result
        assert result.startswith("[fs_write]")
        assert result.endswith("→ completed")

    def test_enforces_max_summary_length(self):
        # Create an action with a very long path to force truncation
        long_path = "/very/long/" + "a" * 900 + "/file.ts"
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": long_path},
            "output": {},
        }
        config = ToolSummaryConfig(max_summary_length=200)
        result = generate_tool_summary(action, config)
        assert len(result) <= 200
        assert result.endswith("...")

    def test_includes_meaningful_output_when_configured(self):
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "npm run test"},
            "output": {
                "exitCode": 1,
                "stderr": "Error: test suite failed with 3 assertions",
            },
        }
        config = ToolSummaryConfig(include_meaningful_output=True)
        result = generate_tool_summary(action, config)
        assert "test suite failed" in result

    def test_excludes_meaningful_output_when_disabled(self):
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "npm run test"},
            "output": {
                "exitCode": 1,
                "stderr": "Error: test suite failed with 3 assertions",
            },
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert "test suite failed" not in result


    def test_indicator_text_when_no_meaningful_output(self):
        """Requirement 8.6: include indicator text when include_meaningful_output=true
        but no meaningful output is found."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/src/server.ts"},
            "output": {"filePath": "/src/server.ts", "content": "just normal content"},
        }
        config = ToolSummaryConfig(include_meaningful_output=True)
        result = generate_tool_summary(action, config)
        assert "(no meaningful output)" in result

    def test_no_indicator_when_meaningful_output_disabled(self):
        """When include_meaningful_output=false, no indicator text appears."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/src/server.ts"},
            "output": {"filePath": "/src/server.ts", "content": "just normal content"},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert "(no meaningful output)" not in result

    def test_no_indicator_when_meaningful_output_present(self):
        """When meaningful output exists, no indicator text — actual output is used."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "failed",
            "input": {"command": "npm test"},
            "output": {"exitCode": 1, "stderr": "FAILED: 2 tests failed"},
        }
        config = ToolSummaryConfig(include_meaningful_output=True)
        result = generate_tool_summary(action, config)
        assert "(no meaningful output)" not in result
        assert "FAILED: 2 tests failed" in result

    def test_failed_action_includes_error_in_outcome(self):
        """Requirement 2.9: failed actions include error message truncated to 200 chars."""
        error_msg = "Cannot find module '@/utils' from 'src/components/App.tsx'"
        action = {
            "actionType": "execute_pwsh",
            "actionState": "failed",
            "input": {"command": "npm run build"},
            "output": {"error": error_msg, "exitCode": 1},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert f"failed: {error_msg}" in result

    def test_errored_action_includes_truncated_error(self):
        """Requirement 2.9: error message truncated to 200 chars."""
        long_error = "x" * 300
        action = {
            "actionType": "readFile",
            "actionState": "error",
            "input": {"path": "/src/file.ts"},
            "output": {"error": long_error},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert "error: " in result
        # The error portion should be at most 200 chars
        arrow_idx = result.index("→ ")
        outcome_part = result[arrow_idx + 2:]
        error_text = outcome_part[len("error: "):]
        assert len(error_text) <= 200
        assert error_text.endswith("...")

    def test_serialization_error_returns_empty_description(self):
        """Handle serialization errors gracefully — empty description on failure."""
        # Create an action with non-serializable input for an unrecognized type
        class Unserializable:
            pass

        action = {
            "actionType": "custom_tool",
            "actionState": "completed",
            "input": Unserializable(),
            "output": {},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        # Should not crash, should return a valid summary with empty description
        assert result is not None
        assert "[custom_tool]" in result
        assert "→ completed" in result

    def test_unicode_safe_truncation(self):
        """Character-safe string slicing — no mid-codepoint cuts.
        Python 3 str handles this natively but verify emoji sequences survive."""
        # Use emoji that are multi-codepoint (family emoji with ZWJ)
        emoji_path = "/src/" + "🏠" * 200 + "/file.ts"
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": emoji_path},
            "output": {},
        }
        config = ToolSummaryConfig(max_summary_length=100, include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert len(result) <= 100
        # Verify result is valid unicode (would raise if broken)
        result.encode("utf-8")

    def test_meaningful_output_with_include_flag_true(self):
        """Requirement 2.7: meaningful output appended when flag is true."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "npm test"},
            "output": {
                "exitCode": 0,
                "stdout": "Test Suites: 3 passed, 3 total\nTests: 15 passed, 15 total",
            },
        }
        config = ToolSummaryConfig(include_meaningful_output=True)
        result = generate_tool_summary(action, config)
        # "passed" is a meaningful indicator, so output should be included
        assert "Test Suites: 3 passed" in result

    def test_meaningful_output_omitted_with_include_flag_false(self):
        """Requirement 8.5: no output excerpts when flag is false."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "npm test"},
            "output": {
                "exitCode": 0,
                "stdout": "Test Suites: 3 passed, 3 total\nTests: 15 passed, 15 total",
            },
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        # Should NOT contain output excerpts or indicator text
        assert "Test Suites" not in result
        assert "(no meaningful output)" not in result
        # Should only have tool name, description, and outcome
        assert result == "[execute_pwsh] npm test (exit 0) → completed"


class TestDescribeAction:
    """Tests for _describe_action per action type."""

    def test_read_file_shows_path_only(self):
        action = {
            "input": {"path": "/src/server.ts"},
            "output": {"filePath": "/src/server.ts", "content": "full file contents here"},
        }
        result = _describe_action("readFile", action)
        assert result == "/src/server.ts"
        assert "full file contents" not in result

    def test_read_file_uses_output_filepath_as_fallback(self):
        action = {"input": {}, "output": {"filePath": "/src/fallback.ts"}}
        result = _describe_action("readFile", action)
        assert result == "/src/fallback.ts"

    def test_read_file_unknown_path(self):
        action = {"input": {}, "output": {}}
        result = _describe_action("readFile", action)
        assert result == "(unknown path)"

    def test_write_file(self):
        action = {"input": {"path": "/src/db.ts"}, "output": {}}
        result = _describe_action("writeFile", action)
        assert result == "/src/db.ts"

    def test_fs_write(self):
        action = {"input": {"path": "/src/config.ts"}, "output": {}}
        result = _describe_action("fs_write", action)
        assert result == "/src/config.ts"

    def test_grep_search_with_results(self):
        action = {
            "input": {"query": "TODO"},
            "output": {"results": [1, 2, 3]},
        }
        result = _describe_action("grep_search", action)
        assert result == 'query="TODO" (3 results)'

    def test_grep_search_without_count(self):
        action = {"input": {"query": "findme"}, "output": {}}
        result = _describe_action("grep_search", action)
        assert result == 'query="findme"'

    def test_file_search(self):
        action = {
            "input": {"query": "config\\.ts"},
            "output": {"matches": ["a.ts", "b.ts"]},
        }
        result = _describe_action("file_search", action)
        assert result == 'query="config\\.ts" (2 results)'

    def test_execute_pwsh_with_exit_code(self):
        action = {
            "input": {"command": "npm run build"},
            "output": {"exitCode": 0},
        }
        result = _describe_action("execute_pwsh", action)
        assert result == "npm run build (exit 0)"

    def test_execute_pwsh_long_command_truncated(self):
        long_cmd = "a" * 150
        action = {"input": {"command": long_cmd}, "output": {"exitCode": 0}}
        result = _describe_action("execute_pwsh", action)
        # Command should be truncated to 100 chars (97 + ...)
        assert len(result.split(" (exit")[0]) <= 100

    def test_invoke_sub_agent(self):
        action = {
            "input": {"name": "context-gatherer", "prompt": "Analyze the authentication flow"},
            "output": {},
        }
        result = _describe_action("invoke_sub_agent", action)
        assert result == 'context-gatherer: "Analyze the authentication flow"'

    def test_invoke_sub_agent_long_prompt_truncated(self):
        long_prompt = "x" * 200
        action = {
            "input": {"name": "context-gatherer", "prompt": long_prompt},
            "output": {},
        }
        result = _describe_action("invoke_sub_agent", action)
        # Prompt should be truncated to 100 chars (97 + ...)
        assert len(result) <= len('context-gatherer: "') + 100 + 1  # closing quote

    def test_unrecognized_type_serializes_input(self):
        action = {
            "input": {"custom_field": "value", "other": 42},
            "output": {},
        }
        result = _describe_action("unknown_tool", action)
        assert "custom_field" in result
        assert "value" in result

    def test_unrecognized_type_truncates_long_input(self):
        action = {
            "input": {"data": "x" * 200},
            "output": {},
        }
        result = _describe_action("unknown_tool", action)
        assert len(result) <= 100


class TestExtractOutcome:
    """Tests for _extract_outcome function."""

    def test_completed(self):
        action = {"actionState": "completed", "output": {}}
        assert _extract_outcome(action) == "completed"

    def test_failed_without_message(self):
        action = {"actionState": "failed", "output": {}}
        assert _extract_outcome(action) == "failed"

    def test_failed_with_error_message(self):
        action = {"actionState": "failed", "output": {"error": "file not found"}}
        assert _extract_outcome(action) == "failed: file not found"

    def test_error_with_message(self):
        action = {"actionState": "error", "output": {"error": "timeout exceeded"}}
        assert _extract_outcome(action) == "error: timeout exceeded"

    def test_error_truncates_long_message(self):
        long_error = "E" * 300
        action = {"actionState": "error", "output": {"error": long_error}}
        result = _extract_outcome(action)
        assert result.startswith("error: ")
        # Total error msg within 200 chars (197 + ...)
        error_part = result[len("error: "):]
        assert len(error_part) <= 200

    def test_unknown_state_defaults_to_completed(self):
        action = {"actionState": "running", "output": {}}
        assert _extract_outcome(action) == "completed"


class TestExtractMeaningfulOutput:
    """Tests for _extract_meaningful_output function."""

    def test_empty_output(self):
        action = {"output": {}}
        assert _extract_meaningful_output(action) == ""

    def test_no_output_field(self):
        action = {}
        assert _extract_meaningful_output(action) == ""

    def test_extracts_error(self):
        action = {"output": {"error": "TypeError: cannot read property 'x' of undefined"}}
        result = _extract_meaningful_output(action)
        assert "TypeError" in result

    def test_extracts_stderr(self):
        action = {"output": {"stderr": "npm ERR! missing script: test"}}
        result = _extract_meaningful_output(action)
        assert "npm ERR" in result

    def test_extracts_meaningful_stdout(self):
        action = {"output": {"stdout": "FAILED: 3 tests failed, 5 passed"}}
        result = _extract_meaningful_output(action)
        assert "FAILED" in result

    def test_ignores_non_meaningful_stdout(self):
        action = {"output": {"stdout": "this is just normal output without indicators"}}
        result = _extract_meaningful_output(action)
        assert result == ""

    def test_truncates_to_500_chars(self):
        long_error = "Error: " + "x" * 600
        action = {"output": {"error": long_error}}
        result = _extract_meaningful_output(action)
        assert len(result) <= 500
        assert result.endswith("...")
