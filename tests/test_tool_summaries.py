"""Unit tests for tool_summaries module."""

import json

import pytest

from kiro_ception.tool_summaries import (
    ToolSummaryConfig,
    _describe_action,
    _extract_meaningful_output,
    _extract_outcome,
    _count_search_results,
    _get_error_message,
    _looks_meaningful,
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


# --- Boundary and edge case tests (from mutation testing) ---


class TestGenerateToolSummaryBoundaries:
    """Precise boundary tests for generate_tool_summary."""

    def test_missing_action_type_key_produces_valid_summary(self):
        """When actionType is completely absent, summary should still work."""
        action = {
            "actionState": "completed",
            "input": {"path": "/foo.txt"},
            "output": {"filePath": "/foo.txt"},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert result is not None
        assert result.startswith("[")
        assert "→" in result

    def test_missing_action_type_uses_empty_string_in_brackets(self):
        """The actionType in brackets should be empty string, not 'None'."""
        action = {"actionState": "completed", "input": {}, "output": {}}
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert result is not None
        assert result.startswith("[] ")
        assert "None" not in result

    def test_summary_truncated_to_exact_max_length(self):
        """When summary exceeds max_summary_length, result length == max_summary_length."""
        long_path = "/" + "x" * 900
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": long_path},
            "output": {"filePath": long_path},
        }
        config = ToolSummaryConfig(max_summary_length=100, include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert result is not None
        assert len(result) == 100
        assert result.endswith("...")

    def test_summary_at_exact_max_not_truncated(self):
        """A summary exactly at max_summary_length should NOT have '...' appended."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/short.txt"},
            "output": {},
        }
        config_big = ToolSummaryConfig(max_summary_length=5000, include_meaningful_output=False)
        natural_result = generate_tool_summary(action, config_big)
        config_exact = ToolSummaryConfig(
            max_summary_length=len(natural_result), include_meaningful_output=False
        )
        result = generate_tool_summary(action, config_exact)
        assert result == natural_result
        assert not result.endswith("...")

    def test_summary_one_over_max_gets_truncated(self):
        """A summary one char over max_summary_length must be truncated."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/short.txt"},
            "output": {},
        }
        config_big = ToolSummaryConfig(max_summary_length=5000, include_meaningful_output=False)
        natural_result = generate_tool_summary(action, config_big)
        config_tight = ToolSummaryConfig(
            max_summary_length=len(natural_result) - 1, include_meaningful_output=False
        )
        result = generate_tool_summary(action, config_tight)
        assert len(result) == len(natural_result) - 1
        assert result.endswith("...")

    def test_newline_in_description_replaced_with_space(self):
        """A description containing \\n should have it replaced with a space."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/foo\n/bar.txt"},
            "output": {},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        first_line = result.split("\n")[0]
        assert "/foo /bar.txt" in first_line
        assert "\n" not in first_line

    def test_carriage_return_in_description_replaced_with_space(self):
        """A description containing \\r should have it replaced with a space."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/foo\r/bar.txt"},
            "output": {},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        first_line = result.split("\n")[0]
        assert "/foo /bar.txt" in first_line
        assert "\r" not in first_line

    def test_mixed_newlines_all_normalized(self):
        """Both \\n and \\r in description become spaces."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "echo\nhello\rworld"},
            "output": {"exitCode": 0},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        first_line = result.split("\n")[0]
        assert "\n" not in first_line
        assert "\r" not in first_line
        assert "echo hello world" in first_line

    def test_empty_description_fallback_to_no_input(self):
        """When description strips to empty, exact '(no input)' text is used."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "\n\r\n"},
            "output": {"filePath": "\r\n"},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert "(no input)" in result
        assert "(NO INPUT)" not in result


class TestDescribeActionBoundaries:
    """Precise boundary tests for _describe_action."""

    def test_long_command_exactly_100_chars(self):
        """A 150-char command truncates to exactly 97 + '...' = 100 chars."""
        long_command = "a" * 150
        action = {
            "actionType": "execute_pwsh",
            "input": {"command": long_command},
            "output": {"exitCode": 0},
        }
        description = _describe_action("execute_pwsh", action)
        command_part = description.split(" (exit")[0]
        assert len(command_part) == 100
        assert command_part == "a" * 97 + "..."

    def test_command_at_exactly_100_chars_not_truncated(self):
        """A 100-char command should NOT be truncated."""
        command = "b" * 100
        action = {"actionType": "execute_pwsh", "input": {"command": command}, "output": {"exitCode": 0}}
        description = _describe_action("execute_pwsh", action)
        assert "b" * 100 in description
        assert "..." not in description

    def test_command_at_101_chars_is_truncated(self):
        """A 101-char command should be truncated."""
        command = "c" * 101
        action = {"actionType": "execute_pwsh", "input": {"command": command}, "output": {}}
        description = _describe_action("execute_pwsh", action)
        assert description == "c" * 97 + "..."

    def test_exit_code_from_snake_case_field(self):
        """Fallback field 'exit_code' (snake_case) should be used when 'exitCode' absent."""
        action = {"actionType": "execute_pwsh", "input": {"command": "ls"}, "output": {"exit_code": 7}}
        description = _describe_action("execute_pwsh", action)
        assert "(exit 7)" in description

    def test_no_exit_code_fields_omits_exit_info(self):
        """When neither exit code field is present, no '(exit N)' in output."""
        action = {"actionType": "execute_pwsh", "input": {"command": "ls"}, "output": {}}
        description = _describe_action("execute_pwsh", action)
        assert description == "ls"

    def test_long_prompt_exactly_100_chars(self):
        """A 200-char prompt truncates to exactly 97 + '...' = 100 chars."""
        long_prompt = "p" * 200
        action = {"actionType": "invoke_sub_agent", "input": {"name": "agent", "prompt": long_prompt}, "output": {}}
        description = _describe_action("invoke_sub_agent", action)
        prompt_in_desc = description.split('"')[1]
        assert len(prompt_in_desc) == 100
        assert prompt_in_desc == "p" * 97 + "..."

    def test_prompt_at_100_chars_not_truncated(self):
        """A 100-char prompt should NOT be truncated."""
        prompt = "q" * 100
        action = {"actionType": "invoke_sub_agent", "input": {"name": "agent", "prompt": prompt}, "output": {}}
        description = _describe_action("invoke_sub_agent", action)
        prompt_in_desc = description.split('"')[1]
        assert prompt_in_desc == "q" * 100

    def test_unrecognized_action_serializes_to_exactly_100_chars(self):
        """Unrecognized action types serialize input JSON, truncated at 100."""
        long_value = "v" * 200
        action = {"actionType": "unknown", "input": {"key": long_value}, "output": {}}
        description = _describe_action("unknown", action)
        assert len(description) == 100
        assert description.endswith("...")

    def test_short_input_not_truncated(self):
        """Short input JSON should not be truncated."""
        action = {"actionType": "unknown", "input": {"x": 1}, "output": {}}
        description = _describe_action("unknown", action)
        assert description == '{"x": 1}'

    def test_missing_input_key_does_not_crash(self):
        """Should not crash when 'input' key is missing."""
        action = {"actionType": "readFile", "output": {"filePath": "/foo.txt"}}
        description = _describe_action("readFile", action)
        assert "/foo.txt" in description

    def test_missing_output_key_does_not_crash(self):
        """Should not crash when 'output' key is missing."""
        action = {"actionType": "readFile", "input": {"path": "/bar.txt"}}
        description = _describe_action("readFile", action)
        assert "/bar.txt" in description

    def test_readfile_prefers_input_path_over_output(self):
        """readFile prefers input.path over output.filePath."""
        action = {"actionType": "readFile", "input": {"path": "/input.ts"}, "output": {"filePath": "/output.ts"}}
        description = _describe_action("readFile", action)
        assert "/input.ts" in description

    def test_grep_search_shows_result_count_from_array(self):
        """grep_search shows count from results array length."""
        action = {"actionType": "grep_search", "input": {"query": "TODO"}, "output": {"results": [1, 2, 3]}}
        description = _describe_action("grep_search", action)
        assert "(3 results)" in description


class TestExtractOutcomeBoundaries:
    """Precise boundary tests for _extract_outcome."""

    def test_error_at_exactly_200_chars_not_truncated(self):
        """A 200-char error message should NOT be truncated."""
        error = "F" * 200
        action = {"actionState": "failed", "output": {"error": error}}
        outcome = _extract_outcome(action)
        error_part = outcome[len("failed: "):]
        assert error_part == "F" * 200
        assert "..." not in error_part

    def test_error_at_201_chars_is_truncated(self):
        """A 201-char error message should be truncated to 197 + '...' = 200."""
        error = "G" * 201
        action = {"actionState": "failed", "output": {"error": error}}
        outcome = _extract_outcome(action)
        error_part = outcome[len("failed: "):]
        assert len(error_part) == 200
        assert error_part == "G" * 197 + "..."

    def test_error_state_also_truncates_at_200(self):
        """The 'error' state applies the same 200-char truncation."""
        long_error = "H" * 300
        action = {"actionState": "error", "output": {"error": long_error}}
        outcome = _extract_outcome(action)
        error_part = outcome[len("error: "):]
        assert len(error_part) == 200
        assert error_part == "H" * 197 + "..."

    def test_newline_in_error_normalized_to_space(self):
        """Newlines in error messages are replaced with spaces."""
        action = {"actionState": "failed", "output": {"error": "line1\nline2\nline3"}}
        outcome = _extract_outcome(action)
        assert "\n" not in outcome
        assert "line1 line2 line3" in outcome

    def test_carriage_return_in_error_normalized(self):
        """Carriage returns in error messages are replaced with spaces."""
        action = {"actionState": "failed", "output": {"error": "foo\rbar"}}
        outcome = _extract_outcome(action)
        assert "\r" not in outcome
        assert "foo bar" in outcome

    def test_whitespace_only_error_treated_as_no_error(self):
        """Error message that's only whitespace should be treated as absent."""
        action = {"actionState": "failed", "output": {"error": "  \n\r  "}}
        outcome = _extract_outcome(action)
        assert outcome == "failed"


class TestCountSearchResults:
    """Tests for _count_search_results."""

    def test_count_from_explicit_count_field(self):
        assert _count_search_results({"count": 42}) == 42

    def test_count_from_results_array_length(self):
        assert _count_search_results({"results": [1, 2, 3, 4, 5]}) == 5

    def test_count_from_matches_array_length(self):
        assert _count_search_results({"matches": ["a", "b"]}) == 2

    def test_count_prefers_count_over_results_array(self):
        assert _count_search_results({"count": 10, "results": [1, 2, 3]}) == 10

    def test_count_returns_none_when_no_fields(self):
        assert _count_search_results({"foo": "bar"}) is None

    def test_count_returns_none_for_none_input(self):
        assert _count_search_results(None) is None

    def test_count_returns_none_for_empty_dict(self):
        assert _count_search_results({}) is None


class TestGetErrorMessageFieldPriority:
    """Tests for _get_error_message field priority: error > message > stderr."""

    def test_error_field_preferred_over_message(self):
        action = {"output": {"error": "permission denied", "message": "file access error"}}
        assert _get_error_message(action) == "permission denied"

    def test_message_field_used_when_no_error(self):
        action = {"output": {"message": "file not found"}}
        assert _get_error_message(action) == "file not found"

    def test_stderr_used_when_no_error_or_message(self):
        action = {"output": {"stderr": "command not found: foo"}}
        assert _get_error_message(action) == "command not found: foo"

    def test_empty_string_when_no_fields(self):
        action = {"output": {"exitCode": 1}}
        assert _get_error_message(action) == ""

    def test_missing_output_key_returns_empty(self):
        action = {"actionState": "failed", "input": {}}
        assert _get_error_message(action) == ""

    def test_none_output_returns_empty(self):
        action = {"output": None}
        assert _get_error_message(action) == ""


class TestLooksMeaningful:
    """Tests for _looks_meaningful keyword detection."""

    def test_lowercase_error(self):
        assert _looks_meaningful("there was an error in the build") is True

    def test_capitalized_error(self):
        assert _looks_meaningful("Error: file not found") is True

    def test_uppercase_error(self):
        assert _looks_meaningful("ERROR: connection refused") is True

    def test_lowercase_fail(self):
        assert _looks_meaningful("some tests fail here") is True

    def test_uppercase_fail(self):
        assert _looks_meaningful("FAIL tests/test_foo.py") is True

    def test_warning(self):
        assert _looks_meaningful("warning: unused variable") is True

    def test_capitalized_warning(self):
        assert _looks_meaningful("Warning: deprecated API") is True

    def test_uppercase_warn(self):
        assert _looks_meaningful("WARN: disk space low") is True

    def test_test_keyword(self):
        assert _looks_meaningful("running test suite now") is True

    def test_assert_keyword(self):
        assert _looks_meaningful("assert x == 5 failed") is True

    def test_exception_lowercase(self):
        assert _looks_meaningful("caught an exception here") is True

    def test_exception_capitalized(self):
        assert _looks_meaningful("Exception in thread main") is True

    def test_traceback_lowercase(self):
        assert _looks_meaningful("full traceback follows:") is True

    def test_traceback_capitalized(self):
        assert _looks_meaningful("Traceback (most recent call last):") is True

    def test_npm_err(self):
        assert _looks_meaningful("npm ERR! missing script") is True

    def test_passed_uppercase(self):
        assert _looks_meaningful("PASSED all integration tests") is True

    def test_failed_uppercase(self):
        assert _looks_meaningful("FAILED test_something.py") is True

    def test_passed_lowercase(self):
        assert _looks_meaningful("12 tests passed in 3s") is True

    def test_failed_lowercase(self):
        assert _looks_meaningful("3 tests failed out of 10") is True

    def test_short_text_rejected(self):
        """Text under 10 chars is not meaningful even with keywords."""
        assert _looks_meaningful("error") is False
        assert _looks_meaningful("fail") is False
        assert _looks_meaningful("123456789") is False

    def test_10_char_text_with_keyword_accepted(self):
        """Text at exactly 10 chars with a keyword passes."""
        assert _looks_meaningful("error text") is True

    def test_no_keywords_rejected(self):
        """Long text without any keywords is not meaningful."""
        assert _looks_meaningful("this is a normal file content with nothing special") is False


class TestExtractMeaningfulOutputBoundaries:
    """Boundary and priority tests for _extract_meaningful_output."""

    def test_error_field_takes_priority_over_stderr(self):
        action = {"output": {"error": "primary error with traceback", "stderr": "secondary stderr"}}
        result = _extract_meaningful_output(action)
        assert "primary error" in result
        assert "secondary" not in result

    def test_stderr_used_when_no_error_field(self):
        action = {"output": {"stderr": "compilation error on line 42"}}
        result = _extract_meaningful_output(action)
        assert "compilation error" in result

    def test_stdout_used_when_no_error_or_stderr(self):
        action = {"output": {"stdout": "FAILED test_something.py::test_case - AssertionError"}}
        result = _extract_meaningful_output(action)
        assert "FAILED test_something" in result

    def test_result_field_used_as_last_resort(self):
        action = {"output": {"result": "Error: validation failed for schema"}}
        result = _extract_meaningful_output(action)
        assert "validation failed" in result

    def test_empty_error_field_falls_through_to_stderr(self):
        action = {"output": {"error": "", "stderr": "real error with traceback"}}
        result = _extract_meaningful_output(action)
        assert "real error" in result

    def test_none_output_returns_empty(self):
        action = {"output": None}
        assert _extract_meaningful_output(action) == ""

    def test_meaningful_output_at_500_not_truncated(self):
        """Meaningful output at exactly 500 chars should NOT be truncated."""
        output = "error " + "y" * 494  # 500 chars total
        action = {"output": {"stderr": output}}
        result = _extract_meaningful_output(action)
        assert result == output
        assert not result.endswith("...")

    def test_meaningful_output_at_501_is_truncated(self):
        """Meaningful output at 501 chars should be truncated to 497 + '...' = 500."""
        output = "error " + "z" * 495  # 501 chars total
        action = {"output": {"stderr": output}}
        result = _extract_meaningful_output(action)
        assert len(result) == 500
        assert result.endswith("...")
        assert result == output[:497] + "..."
