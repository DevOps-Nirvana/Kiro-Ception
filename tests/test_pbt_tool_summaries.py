"""Property-based tests for tool_summaries module.

# Feature: conversation-logger, Property 6: Summary Format Invariant

Validates: Requirements 7.1, 2.2, 7.9
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.tool_summaries import ToolSummaryConfig, generate_tool_summary

# --- Strategies ---

# Action types that are NOT "say" (say actions return None)
action_types = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "unknown_tool",
    "mcp_chrome_devtools_click",
    "semanticRename",
    "list_directory",
    "str_replace",
    "web_fetch",
])

# Action states that produce known outcomes
action_states = st.sampled_from(["completed", "failed", "error", "running", ""])

# Generate plausible input dictionaries
input_strategy = st.fixed_dictionaries({}, optional={
    "path": st.text(min_size=1, max_size=200),
    "query": st.text(min_size=1, max_size=100),
    "command": st.text(min_size=1, max_size=300),
    "name": st.text(min_size=1, max_size=50),
    "prompt": st.text(min_size=0, max_size=300),
})

# Generate plausible output dictionaries
output_strategy = st.fixed_dictionaries({}, optional={
    "filePath": st.text(min_size=1, max_size=200),
    "content": st.text(min_size=0, max_size=500),
    "exitCode": st.integers(min_value=-128, max_value=255),
    "exit_code": st.integers(min_value=-128, max_value=255),
    "error": st.text(min_size=1, max_size=500),
    "stderr": st.text(min_size=1, max_size=500),
    "message": st.text(min_size=1, max_size=200),
    "count": st.integers(min_value=0, max_value=1000),
    "results": st.lists(st.integers(), min_size=0, max_size=20),
    "matches": st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=20),
})

# Full action dict strategy (excluding "say" actionType)
action_strategy = st.fixed_dictionaries({
    "actionType": action_types,
    "actionState": action_states,
    "input": input_strategy,
    "output": output_strategy,
})

# The format pattern for tool summaries:
# [{actionType}] {description} → {outcome}
# where outcome is one of: completed, failed, failed: {msg}, error, error: {msg}
SUMMARY_FORMAT_PATTERN = re.compile(
    r"^\[([^\]]+)\] .+ → (completed|failed(?:: .+)?|error(?:: .+)?)$",
    re.DOTALL,
)


class TestProperty6SummaryFormatInvariant:
    """Property 6: Summary Format Invariant

    *For any* generated Tool_Summary string, it SHALL match the pattern
    `[{actionType}] {description} → {outcome}` where outcome is one of
    `completed`, `failed`, `failed: {msg}`, `error`, or `error: {msg}`.

    **Validates: Requirements 7.1, 2.2, 7.9**
    """

    @given(action=action_strategy)
    @settings(max_examples=100)
    def test_summary_format_matches_pattern(self, action: dict):
        """For any non-None summary, the first line must match
        [{actionType}] {description} → {outcome}."""
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)

        # If result is None, action was skipped (say or excluded) - not relevant here
        if result is None:
            return

        # The core format is on the first line (meaningful output goes on subsequent lines)
        first_line = result.split("\n")[0]

        # Verify the first line matches the expected format
        match = SUMMARY_FORMAT_PATTERN.match(first_line)
        assert match is not None, (
            f"Summary first line does not match format "
            f"'[{{actionType}}] {{description}} → {{outcome}}'.\n"
            f"Got: {first_line!r}"
        )

        # Verify the actionType in brackets matches the action's actionType
        extracted_action_type = match.group(1)
        assert extracted_action_type == action["actionType"], (
            f"Action type mismatch: expected '{action['actionType']}', "
            f"got '{extracted_action_type}' in summary"
        )

        # Verify the outcome is valid
        outcome = match.group(2)
        assert outcome.startswith(("completed", "failed", "error")), (
            f"Invalid outcome: {outcome!r}"
        )

    @given(action=action_strategy)
    @settings(max_examples=100)
    def test_summary_contains_unicode_arrow(self, action: dict):
        """For any non-None summary, it SHALL contain the → (U+2192) character."""
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)

        if result is None:
            return

        first_line = result.split("\n")[0]
        assert "→" in first_line, (
            f"Summary missing unicode arrow →.\nGot: {first_line!r}"
        )

    @given(action=action_strategy)
    @settings(max_examples=100)
    def test_summary_outcome_is_valid(self, action: dict):
        """The outcome portion after → must be one of the allowed values."""
        config = ToolSummaryConfig(include_meaningful_output=False)
        result = generate_tool_summary(action, config)

        if result is None:
            return

        first_line = result.split("\n")[0]

        # Split on " → " to get the outcome part
        parts = first_line.split(" → ")
        assert len(parts) >= 2, (
            f"Summary does not contain ' → ' separator.\nGot: {first_line!r}"
        )

        outcome = parts[-1]
        valid_outcomes = (
            outcome == "completed"
            or outcome == "failed"
            or outcome.startswith("failed: ")
            or outcome == "error"
            or outcome.startswith("error: ")
        )
        assert valid_outcomes, (
            f"Invalid outcome: {outcome!r}. "
            f"Must be 'completed', 'failed', 'failed: {{msg}}', 'error', or 'error: {{msg}}'"
        )

    @given(action=action_strategy)
    @settings(max_examples=100)
    def test_summary_with_meaningful_output_preserves_first_line_format(self, action: dict):
        """Even with meaningful output enabled, the first line must follow the format."""
        config = ToolSummaryConfig(include_meaningful_output=True)
        result = generate_tool_summary(action, config)

        if result is None:
            return

        first_line = result.split("\n")[0]

        # Even with meaningful output appended (on subsequent lines),
        # the first line must match the format pattern
        match = SUMMARY_FORMAT_PATTERN.match(first_line)
        assert match is not None, (
            f"Summary first line with meaningful output enabled does not match format.\n"
            f"Got: {first_line!r}"
        )
