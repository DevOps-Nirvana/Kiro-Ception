"""Targeted tests to kill surviving mutants in tool_summaries.py.

Each test pins down a specific behavior that mutmut showed was untested:
exact truncation boundaries, fallback defaults, and edge cases around
missing fields.
"""

from kiro_ception.tool_summaries import (
    ToolSummaryConfig,
    generate_tool_summary,
    _describe_action,
    _extract_outcome,
    _extract_meaningful_output,
    _count_search_results,
)


class TestMissingActionType:
    """Mutant: action.get("actionType", "") → action.get("actionType", None)"""

    def test_missing_action_type_key_produces_valid_summary(self):
        """When actionType is completely absent, summary should still work."""
        action = {
            "actionState": "completed",
            "input": {"path": "/foo.txt"},
            "output": {"filePath": "/foo.txt"},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        # Should produce a summary with empty action type in brackets
        assert result is not None
        assert result.startswith("[")
        assert "→" in result

    def test_missing_action_type_uses_empty_string_in_brackets(self):
        """The actionType in brackets should be empty string, not 'None'."""
        action = {
            "actionState": "completed",
            "input": {},
            "output": {},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert result is not None
        assert result.startswith("[] ")
        assert "None" not in result


class TestCommandTruncation:
    """Mutant: command[:97] → command[:98] or command[:96]"""

    def test_long_command_truncated_to_100_chars(self):
        """A 150-char command should be truncated to exactly 97 chars + '...' = 100."""
        long_command = "a" * 150
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": long_command},
            "output": {"exitCode": 0},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)

        # Description should contain truncated command
        # Format: [execute_pwsh] <command> (exit 0) → completed
        description = _describe_action("execute_pwsh", action)
        # The command portion before " (exit 0)" should be 100 chars
        command_part = description.split(" (exit")[0]
        assert len(command_part) == 100
        assert command_part.endswith("...")
        assert command_part == "a" * 97 + "..."

    def test_command_at_exactly_100_chars_not_truncated(self):
        """A 100-char command should NOT be truncated."""
        command = "b" * 100
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": command},
            "output": {"exitCode": 0},
        }
        description = _describe_action("execute_pwsh", action)
        assert "b" * 100 in description
        assert "..." not in description

    def test_command_at_101_chars_is_truncated(self):
        """A 101-char command should be truncated."""
        command = "c" * 101
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": command},
            "output": {},
        }
        description = _describe_action("execute_pwsh", action)
        assert description == "c" * 97 + "..."


class TestExitCodeFallback:
    """Mutant: output_data.get("exitCode", output_data.get("exit_code")) → .get("exitCode", None)"""

    def test_exit_code_from_exitCode_field(self):
        """Primary field 'exitCode' should be used."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "ls"},
            "output": {"exitCode": 42},
        }
        description = _describe_action("execute_pwsh", action)
        assert "(exit 42)" in description

    def test_exit_code_from_exit_code_field(self):
        """Fallback field 'exit_code' (snake_case) should be used when 'exitCode' absent."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "ls"},
            "output": {"exit_code": 7},
        }
        description = _describe_action("execute_pwsh", action)
        assert "(exit 7)" in description

    def test_no_exit_code_fields_omits_exit_info(self):
        """When neither exit code field is present, no '(exit N)' in output."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "ls"},
            "output": {},
        }
        description = _describe_action("execute_pwsh", action)
        assert "(exit" not in description
        assert description == "ls"


class TestMaxSummaryLengthBoundary:
    """Mutant: summary[: max - 3] + "..." → summary[: max - 4] + "..."""

    def test_summary_truncated_to_exact_max_length(self):
        """When summary exceeds max_summary_length, result length == max_summary_length."""
        # Create an action with very long output to trigger truncation
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
        """A summary exactly at max_summary_length should NOT be truncated (no '...')."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/short.txt"},
            "output": {},
        }
        # Use a large max so it won't truncate
        config = ToolSummaryConfig(max_summary_length=5000, include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        assert result is not None
        # Set max to exactly the length of the result
        config2 = ToolSummaryConfig(max_summary_length=len(result), include_meaningful_output=False)
        result2 = generate_tool_summary(action, config2)
        assert result2 == result  # No truncation applied
        assert not result2.endswith("...")

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
        natural_len = len(natural_result)

        # Set max to one less than natural length
        config_tight = ToolSummaryConfig(
            max_summary_length=natural_len - 1, include_meaningful_output=False
        )
        result = generate_tool_summary(action, config_tight)
        assert len(result) == natural_len - 1
        assert result.endswith("...")


class TestInvokeSubAgentPromptTruncation:
    """Mutant: prompt[:97] → prompt[:98] or similar."""

    def test_long_prompt_truncated_to_100_chars(self):
        """A 200-char prompt should be truncated to 97 + '...' = 100 chars."""
        long_prompt = "p" * 200
        action = {
            "actionType": "invoke_sub_agent",
            "actionState": "completed",
            "input": {"name": "agent", "prompt": long_prompt},
            "output": {},
        }
        description = _describe_action("invoke_sub_agent", action)
        # Format: 'agent: "<truncated prompt>"'
        # The prompt inside quotes should be 100 chars
        prompt_in_desc = description.split('"')[1]  # Extract between quotes
        assert len(prompt_in_desc) == 100
        assert prompt_in_desc == "p" * 97 + "..."

    def test_prompt_at_100_chars_not_truncated(self):
        """A 100-char prompt should NOT be truncated."""
        prompt = "q" * 100
        action = {
            "actionType": "invoke_sub_agent",
            "actionState": "completed",
            "input": {"name": "agent", "prompt": prompt},
            "output": {},
        }
        description = _describe_action("invoke_sub_agent", action)
        prompt_in_desc = description.split('"')[1]
        assert prompt_in_desc == "q" * 100
        assert "..." not in prompt_in_desc


class TestSearchResultCount:
    """Mutant: uses results array length vs. explicit count field."""

    def test_count_from_explicit_count_field(self):
        """The 'count' field takes priority."""
        output = {"count": 42}
        assert _count_search_results(output) == 42

    def test_count_from_results_array_length(self):
        """Falls back to len(results) when no 'count' field."""
        output = {"results": [1, 2, 3, 4, 5]}
        assert _count_search_results(output) == 5

    def test_count_from_matches_array_length(self):
        """Falls back to len(matches) when no 'count' or 'results'."""
        output = {"matches": ["a", "b"]}
        assert _count_search_results(output) == 2

    def test_count_prefers_count_over_results_array(self):
        """When both 'count' and 'results' exist, 'count' wins."""
        output = {"count": 10, "results": [1, 2, 3]}
        assert _count_search_results(output) == 10

    def test_count_returns_none_when_no_fields(self):
        """Returns None when no count-like fields exist."""
        output = {"foo": "bar"}
        assert _count_search_results(output) is None

    def test_grep_search_description_includes_result_count(self):
        """grep_search description should show result count from output."""
        action = {
            "actionType": "grep_search",
            "actionState": "completed",
            "input": {"query": "TODO"},
            "output": {"results": [1, 2, 3]},
        }
        description = _describe_action("grep_search", action)
        assert '(3 results)' in description


class TestErrorMessageTruncation:
    """Mutant: error_msg[:197] → error_msg[:198] or similar."""

    def test_error_truncated_to_200_chars_in_outcome(self):
        """A 300-char error message should be truncated to 197 + '...' = 200 chars."""
        long_error = "E" * 300
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
            "output": {"error": long_error},
        }
        outcome = _extract_outcome(action)
        # Format: "failed: <truncated error>"
        assert outcome.startswith("failed: ")
        error_part = outcome[len("failed: "):]
        assert len(error_part) == 200
        assert error_part.endswith("...")
        assert error_part == "E" * 197 + "..."

    def test_error_at_200_chars_not_truncated(self):
        """A 200-char error message should NOT be truncated."""
        error = "F" * 200
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
            "output": {"error": error},
        }
        outcome = _extract_outcome(action)
        error_part = outcome[len("failed: "):]
        assert error_part == "F" * 200
        assert "..." not in error_part

    def test_error_at_201_chars_is_truncated(self):
        """A 201-char error message should be truncated."""
        error = "G" * 201
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
            "output": {"error": error},
        }
        outcome = _extract_outcome(action)
        error_part = outcome[len("failed: "):]
        assert len(error_part) == 200
        assert error_part.endswith("...")

    def test_error_state_also_truncates(self):
        """The 'error' state applies the same 200-char truncation."""
        long_error = "H" * 300
        action = {
            "actionType": "readFile",
            "actionState": "error",
            "input": {},
            "output": {"error": long_error},
        }
        outcome = _extract_outcome(action)
        assert outcome.startswith("error: ")
        error_part = outcome[len("error: "):]
        assert len(error_part) == 200
        assert error_part == "H" * 197 + "..."


class TestMeaningfulOutputTruncation:
    """Mutant: meaningful[:497] → meaningful[:498] or similar."""

    def test_meaningful_output_truncated_to_500_chars(self):
        """Meaningful output exceeding 500 chars should be truncated."""
        long_output = "error " + "x" * 600  # starts with "error" to pass _looks_meaningful
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {"command": "test"},
            "output": {"stderr": long_output},
        }
        result = _extract_meaningful_output(action)
        assert len(result) == 500
        assert result.endswith("...")
        assert result == long_output[:497] + "..."

    def test_meaningful_output_at_500_not_truncated(self):
        """Meaningful output at exactly 500 chars should NOT be truncated."""
        output = "error " + "y" * 494  # 500 chars total
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {},
            "output": {"stderr": output},
        }
        result = _extract_meaningful_output(action)
        assert result == output
        assert not result.endswith("...")


class TestGenericInputSerialization:
    """Mutant: serialized[:97] → serialized[:98] for unrecognized action types."""

    def test_unrecognized_action_type_serializes_input_to_100_chars(self):
        """Unrecognized action types serialize input JSON, truncated at 100."""
        long_value = "v" * 200
        action = {
            "actionType": "unknown_custom_tool",
            "actionState": "completed",
            "input": {"key": long_value},
            "output": {},
        }
        description = _describe_action("unknown_custom_tool", action)
        assert len(description) == 100
        assert description.endswith("...")

    def test_short_input_not_truncated(self):
        """Short input JSON should not be truncated."""
        action = {
            "actionType": "unknown_tool",
            "actionState": "completed",
            "input": {"x": 1},
            "output": {},
        }
        description = _describe_action("unknown_tool", action)
        assert description == '{"x": 1}'
        assert "..." not in description
