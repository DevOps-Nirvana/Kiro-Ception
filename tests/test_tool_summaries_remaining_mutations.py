"""Tests targeting remaining surviving mutants in tool_summaries.py.

Covers: newline normalization in descriptions, _looks_meaningful keyword
detection, field priority in _extract_meaningful_output/_get_error_message,
empty description fallback, and missing 'output' key handling.
"""

from kiro_ception.tool_summaries import (
    ToolSummaryConfig,
    generate_tool_summary,
    _describe_action,
    _extract_meaningful_output,
    _extract_outcome,
    _get_error_message,
    _looks_meaningful,
)


class TestNewlineNormalizationInDescription:
    """Mutants: replace("\\n", " ") → replace("XX\\nXX", " ")
    and replace("\\r", " ") → replace("XX\\rXX", " ")"""

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
        # The newline in the path should be replaced, not split the line
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


class TestEmptyDescriptionFallback:
    """Mutants: description = "(no input)" → None / "(NO INPUT)" / "XX(no input)XX" """

    def test_empty_input_produces_no_input_fallback(self):
        """When _describe_action returns only whitespace, fallback kicks in."""
        # For readFile with no path in input OR output, the description is "(unknown path)"
        # which is non-empty. To trigger the empty-description fallback, we need
        # a description that becomes empty after newline replacement + strip.
        # A description that's all newlines/carriage returns will strip to empty.
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "\n\r\n"},
            "output": {"filePath": "\r\n"},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        # After replacing \n and \r with spaces, and stripping, path becomes empty
        # so the fallback should produce "(no input)"
        assert "(no input)" in result
        assert "(NO INPUT)" not in result

    def test_whitespace_only_description_falls_back_to_no_input(self):
        """If _describe_action returns only whitespace, fallback should kick in."""
        # readFile with path that's all whitespace
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "   "},
            "output": {"filePath": "   "},
        }
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)
        # After strip(), "   " becomes empty, so fallback applies
        # Actually, "   " isn't empty after .get() but the path IS whitespace
        # The description would be "   " which after strip() in generate_tool_summary = ""
        # Wait - _describe_action returns "   " for readFile, then generate_tool_summary
        # does .replace("\n", " ").replace("\r", " ").strip() → ""
        # Then if not description → description = "(no input)"
        assert "(no input)" in result


class TestLooksMeaningfulKeywords:
    """Mutants: "error" → "XXerrorXX", "FAIL" → "XXFAILXX", etc."""

    def test_lowercase_error_detected(self):
        assert _looks_meaningful("there was an error in the build") is True

    def test_capitalized_error_detected(self):
        assert _looks_meaningful("Error: file not found") is True

    def test_uppercase_error_detected(self):
        assert _looks_meaningful("ERROR: connection refused") is True

    def test_lowercase_fail_detected(self):
        assert _looks_meaningful("some tests fail here") is True

    def test_uppercase_fail_detected(self):
        assert _looks_meaningful("FAIL tests/test_foo.py") is True

    def test_warning_detected(self):
        assert _looks_meaningful("warning: unused variable") is True

    def test_capitalized_warning_detected(self):
        assert _looks_meaningful("Warning: deprecated API") is True

    def test_uppercase_warn_detected(self):
        assert _looks_meaningful("WARN: disk space low") is True

    def test_test_keyword_detected(self):
        assert _looks_meaningful("running test suite now") is True

    def test_assert_keyword_detected(self):
        assert _looks_meaningful("assert x == 5 failed") is True

    def test_exception_lowercase_detected(self):
        assert _looks_meaningful("caught an exception here") is True

    def test_exception_capitalized_detected(self):
        assert _looks_meaningful("Exception in thread main") is True

    def test_traceback_lowercase_detected(self):
        assert _looks_meaningful("full traceback follows:") is True

    def test_traceback_capitalized_detected(self):
        assert _looks_meaningful("Traceback (most recent call last):") is True

    def test_npm_err_detected(self):
        assert _looks_meaningful("npm ERR! missing script") is True

    def test_passed_uppercase_detected(self):
        assert _looks_meaningful("PASSED all integration tests") is True

    def test_failed_uppercase_detected(self):
        assert _looks_meaningful("FAILED test_something.py") is True

    def test_passed_lowercase_detected(self):
        assert _looks_meaningful("12 tests passed in 3s") is True

    def test_failed_lowercase_detected(self):
        assert _looks_meaningful("3 tests failed out of 10") is True

    def test_short_text_rejected(self):
        """Text under 10 chars should not be meaningful even with keywords."""
        assert _looks_meaningful("error") is False
        assert _looks_meaningful("fail") is False
        assert _looks_meaningful("123456789") is False

    def test_10_char_text_with_keyword_accepted(self):
        """Text at exactly 10 chars with a keyword should pass."""
        assert _looks_meaningful("error text") is True  # 10 chars

    def test_no_keywords_rejected(self):
        """Long text without any keywords is not meaningful."""
        assert _looks_meaningful("this is a normal file content with nothing special") is False


class TestExtractMeaningfulOutputFieldPriority:
    """Mutants: field priority order (error > stderr > stdout > result)."""

    def test_error_field_takes_priority_over_stderr(self):
        """When both 'error' and 'stderr' exist, 'error' is used."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "failed",
            "input": {},
            "output": {
                "error": "primary error message with traceback",
                "stderr": "secondary stderr output with error info",
            },
        }
        result = _extract_meaningful_output(action)
        assert "primary error" in result
        assert "secondary" not in result

    def test_stderr_used_when_no_error_field(self):
        """When 'error' is absent, 'stderr' is used."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "failed",
            "input": {},
            "output": {
                "stderr": "compilation error on line 42",
            },
        }
        result = _extract_meaningful_output(action)
        assert "compilation error" in result

    def test_stdout_used_when_no_error_or_stderr(self):
        """When 'error' and 'stderr' are absent, 'stdout' is used if meaningful."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {},
            "output": {
                "stdout": "FAILED test_something.py::test_case - AssertionError",
            },
        }
        result = _extract_meaningful_output(action)
        assert "FAILED test_something" in result

    def test_result_field_used_as_last_resort(self):
        """'result' field is checked last."""
        action = {
            "actionType": "custom_tool",
            "actionState": "completed",
            "input": {},
            "output": {
                "result": "Error: validation failed for schema",
            },
        }
        result = _extract_meaningful_output(action)
        assert "validation failed" in result

    def test_empty_error_field_falls_through_to_stderr(self):
        """Empty string 'error' should fall through to stderr."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "failed",
            "input": {},
            "output": {
                "error": "",
                "stderr": "real error information with traceback",
            },
        }
        result = _extract_meaningful_output(action)
        assert "real error" in result

    def test_non_meaningful_stdout_skipped(self):
        """stdout without meaningful keywords is skipped."""
        action = {
            "actionType": "execute_pwsh",
            "actionState": "completed",
            "input": {},
            "output": {
                "stdout": "just some normal output with no keywords at all here nothing special to see",
            },
        }
        result = _extract_meaningful_output(action)
        assert result == ""


class TestGetErrorMessageFieldPriority:
    """Mutants in _get_error_message: field order is error > message > stderr."""

    def test_error_field_preferred_over_message(self):
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
            "output": {
                "error": "permission denied",
                "message": "file access error",
            },
        }
        result = _get_error_message(action)
        assert result == "permission denied"

    def test_message_field_used_when_no_error(self):
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
            "output": {
                "message": "file not found",
            },
        }
        result = _get_error_message(action)
        assert result == "file not found"

    def test_stderr_used_when_no_error_or_message(self):
        action = {
            "actionType": "execute_pwsh",
            "actionState": "failed",
            "input": {},
            "output": {
                "stderr": "command not found: foo",
            },
        }
        result = _get_error_message(action)
        assert result == "command not found: foo"

    def test_empty_string_when_no_fields(self):
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
            "output": {"exitCode": 1},
        }
        result = _get_error_message(action)
        assert result == ""

    def test_missing_output_key_returns_empty(self):
        """When 'output' key is absent entirely, should return empty string."""
        action = {
            "actionType": "readFile",
            "actionState": "failed",
            "input": {},
        }
        result = _get_error_message(action)
        assert result == ""


class TestMissingOutputKeyHandling:
    """Mutants: action.get("output", {}) → action.get("output", None)"""

    def test_extract_meaningful_output_with_missing_output(self):
        """Should return empty string when 'output' key is missing."""
        action = {"actionType": "readFile", "actionState": "completed", "input": {}}
        result = _extract_meaningful_output(action)
        assert result == ""

    def test_extract_meaningful_output_with_none_output(self):
        """Should return empty string when output is explicitly None."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {},
            "output": None,
        }
        result = _extract_meaningful_output(action)
        assert result == ""

    def test_describe_action_with_missing_input(self):
        """Should not crash when 'input' key is missing."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "output": {"filePath": "/foo.txt"},
        }
        description = _describe_action("readFile", action)
        # Should fall back to output.filePath
        assert "/foo.txt" in description

    def test_describe_action_with_missing_output(self):
        """Should not crash when 'output' key is missing."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/bar.txt"},
        }
        description = _describe_action("readFile", action)
        assert "/bar.txt" in description


class TestDescribeActionPathFallbacks:
    """Mutants: path fallback from input.path to output.filePath."""

    def test_readfile_uses_input_path_first(self):
        """readFile prefers input.path over output.filePath."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {"path": "/from/input.ts"},
            "output": {"filePath": "/from/output.ts"},
        }
        description = _describe_action("readFile", action)
        assert "/from/input.ts" in description

    def test_readfile_falls_back_to_output_filepath(self):
        """readFile uses output.filePath when input.path is absent."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {},
            "output": {"filePath": "/fallback/file.ts"},
        }
        description = _describe_action("readFile", action)
        assert "/fallback/file.ts" in description

    def test_readfile_unknown_path_when_both_missing(self):
        """readFile shows '(unknown path)' when neither path field exists."""
        action = {
            "actionType": "readFile",
            "actionState": "completed",
            "input": {},
            "output": {},
        }
        description = _describe_action("readFile", action)
        assert description == "(unknown path)"

    def test_writefile_uses_input_path(self):
        """writeFile prefers input.path."""
        action = {
            "actionType": "writeFile",
            "actionState": "completed",
            "input": {"path": "/write/target.ts"},
            "output": {},
        }
        description = _describe_action("writeFile", action)
        assert "/write/target.ts" in description

    def test_fs_write_uses_input_path(self):
        """fs_write prefers input.path."""
        action = {
            "actionType": "fs_write",
            "actionState": "completed",
            "input": {"path": "/fs/write.py"},
            "output": {},
        }
        description = _describe_action("fs_write", action)
        assert "/fs/write.py" in description
