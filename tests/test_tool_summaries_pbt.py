"""Property-based tests for tool_summaries module.

Uses Hypothesis to verify correctness properties across randomly generated inputs.
"""

# Feature: conversation-logger, Property 7: Summary Length Invariant

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.tool_summaries import ToolSummaryConfig, generate_tool_summary

# --- Strategies ---

# Strategy for generating long random strings (for paths, commands, prompts)
long_text_strategy = st.text(
    alphabet=string.printable,
    min_size=0,
    max_size=2000,
)

# Strategy for non-empty long strings (to ensure we get actual content)
nonempty_long_text = st.text(
    alphabet=string.printable,
    min_size=1,
    max_size=2000,
)

# Strategy for action types (both known and unknown)
action_type_strategy = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "unknown_tool",
    "custom_action",
    "some_random_tool",
])

# Strategy for action state
action_state_strategy = st.sampled_from(["completed", "failed", "error", "running"])

# Strategy for max_summary_length config values (reasonable range)
max_summary_length_strategy = st.integers(min_value=50, max_value=2000)

# Strategy for generating random output dicts with potentially long content
output_strategy = st.fixed_dictionaries(
    {},
    optional={
        "error": long_text_strategy,
        "stderr": long_text_strategy,
        "stdout": long_text_strategy,
        "result": long_text_strategy,
        "content": long_text_strategy,
        "filePath": long_text_strategy,
        "exitCode": st.integers(min_value=-1, max_value=255),
        "count": st.integers(min_value=0, max_value=10000),
        "results": st.lists(st.text(max_size=50), max_size=100),
        "matches": st.lists(st.text(max_size=50), max_size=100),
    },
)

# Strategy for generating random input dicts with potentially long content
input_strategy = st.fixed_dictionaries(
    {},
    optional={
        "path": long_text_strategy,
        "query": long_text_strategy,
        "command": long_text_strategy,
        "name": long_text_strategy,
        "prompt": long_text_strategy,
        "custom_field": long_text_strategy,
        "data": long_text_strategy,
    },
)

# Strategy for full action dicts
action_strategy = st.fixed_dictionaries({
    "actionType": action_type_strategy,
    "actionState": action_state_strategy,
    "input": input_strategy,
    "output": output_strategy,
})


# --- Property 7: Summary Length Invariant ---
# For any generated Tool_Summary, its total character length SHALL NOT exceed
# the configured max_summary_length (default 800 characters).
# Validates: Requirements 7.2, 8.3


class TestSummaryLengthInvariant:
    """Property 7: Summary Length Invariant.

    For any generated Tool_Summary, its total character length SHALL NOT exceed
    the configured max_summary_length (default 800 characters).

    **Validates: Requirements 7.2, 8.3**
    """

    @given(action=action_strategy, max_length=max_summary_length_strategy)
    @settings(max_examples=200)
    def test_summary_never_exceeds_configured_max_length(
        self, action: dict, max_length: int
    ):
        """For any action and any configured max_summary_length,
        the generated summary length SHALL NOT exceed max_summary_length."""
        config = ToolSummaryConfig(
            max_summary_length=max_length,
            include_meaningful_output=True,
        )
        result = generate_tool_summary(action, config)

        # If the result is not None (i.e., it wasn't skipped), verify length
        if result is not None:
            assert len(result) <= max_length, (
                f"Summary length {len(result)} exceeds max_summary_length {max_length}. "
                f"Summary: {result[:200]}..."
            )

    @given(action=action_strategy)
    @settings(max_examples=200)
    def test_summary_never_exceeds_default_max_length(self, action: dict):
        """With default config (max_summary_length=800), summary SHALL NOT exceed 800 chars."""
        config = ToolSummaryConfig()  # default max_summary_length=800
        result = generate_tool_summary(action, config)

        if result is not None:
            assert len(result) <= 800, (
                f"Summary length {len(result)} exceeds default max of 800 characters. "
                f"Summary: {result[:200]}..."
            )

    @given(action=action_strategy, max_length=st.integers(min_value=50, max_value=100))
    @settings(max_examples=200)
    def test_summary_respects_small_max_length(self, action: dict, max_length: int):
        """Even with very small max_summary_length values (50-100),
        the summary SHALL NOT exceed the configured limit."""
        config = ToolSummaryConfig(
            max_summary_length=max_length,
            include_meaningful_output=True,
        )
        result = generate_tool_summary(action, config)

        if result is not None:
            assert len(result) <= max_length, (
                f"Summary length {len(result)} exceeds small max_summary_length {max_length}. "
                f"Summary: {result!r}"
            )
