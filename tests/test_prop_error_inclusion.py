"""Property test for error inclusion in failed actions.

# Feature: conversation-logger, Property 10: Error Inclusion for Failed Actions

Validates: Requirements 2.9

Property: For any tool action with `actionState` of `failed` or `error`,
the Tool_Summary outcome SHALL include the error message from the output,
truncated to 200 characters.
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.tool_summaries import ToolSummaryConfig, _extract_outcome, generate_tool_summary


# --- Strategies ---

# Generate non-empty error messages of varying lengths (including > 200 chars)
error_message_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=500,
)

# Action states that indicate failure
failed_state_strategy = st.sampled_from(["failed", "error"])

# Non-say action types for generating test actions
action_type_strategy = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "custom_tool",
])

# Error location in output - could be in "error" or "message" field
error_field_strategy = st.sampled_from(["error", "message"])


def make_failed_action(action_type: str, action_state: str, error_field: str, error_msg: str):
    """Construct an action dict with a failed/error state and error message."""
    output = {error_field: error_msg}
    # Add minimal input based on action type
    if action_type == "readFile":
        input_data = {"path": "/some/file.ts"}
    elif action_type in ("writeFile", "fs_write"):
        input_data = {"path": "/some/file.ts"}
    elif action_type in ("grep_search", "file_search"):
        input_data = {"query": "search_term"}
    elif action_type == "execute_pwsh":
        input_data = {"command": "some_command"}
    elif action_type == "invoke_sub_agent":
        input_data = {"name": "test-agent", "prompt": "do something"}
    else:
        input_data = {"key": "value"}

    return {
        "actionType": action_type,
        "actionState": action_state,
        "input": input_data,
        "output": output,
    }


# --- Property Tests ---


class TestErrorInclusionProperty:
    """Property 10: Error Inclusion for Failed Actions.

    **Validates: Requirements 2.9**
    """

    @given(
        action_state=failed_state_strategy,
        error_msg=error_message_strategy,
        error_field=error_field_strategy,
    )
    @settings(max_examples=100)
    def test_extract_outcome_includes_error_message(
        self, action_state: str, error_msg: str, error_field: str
    ):
        """For any failed/error action with an error message, _extract_outcome
        SHALL include the error message (potentially truncated) in the outcome."""
        action = {
            "actionState": action_state,
            "output": {error_field: error_msg},
        }

        outcome = _extract_outcome(action)

        # The outcome should start with the state prefix
        if action_state == "failed":
            assert outcome.startswith("failed: "), (
                f"Expected outcome to start with 'failed: ' but got: {outcome!r}"
            )
            error_portion = outcome[len("failed: "):]
        else:
            assert outcome.startswith("error: "), (
                f"Expected outcome to start with 'error: ' but got: {outcome!r}"
            )
            error_portion = outcome[len("error: "):]

        # The error portion must be ≤200 chars
        assert len(error_portion) <= 200, (
            f"Error portion exceeds 200 chars: {len(error_portion)} chars"
        )

        # The error portion should be a prefix of the error message (possibly truncated)
        if len(error_msg) <= 200:
            assert error_portion == error_msg, (
                f"Expected full error message '{error_msg}' but got '{error_portion}'"
            )
        else:
            # When truncated, should end with "..." and be 200 chars
            assert error_portion.endswith("..."), (
                f"Expected truncated error to end with '...' but got: {error_portion!r}"
            )
            assert len(error_portion) == 200, (
                f"Expected truncated error to be exactly 200 chars, got {len(error_portion)}"
            )
            # The leading part (before "...") should be a prefix of the original message
            assert error_msg.startswith(error_portion[:-3]), (
                f"Truncated error is not a prefix of original message"
            )

    @given(
        action_type=action_type_strategy,
        action_state=failed_state_strategy,
        error_msg=error_message_strategy,
        error_field=error_field_strategy,
    )
    @settings(max_examples=100)
    def test_full_summary_includes_error_for_failed_actions(
        self, action_type: str, action_state: str, error_msg: str, error_field: str
    ):
        """For any failed/error action, generate_tool_summary SHALL produce a
        summary whose outcome portion includes the error message (truncated to 200 chars)."""
        action = make_failed_action(action_type, action_state, error_field, error_msg)
        config = ToolSummaryConfig(include_meaningful_output=False)

        summary = generate_tool_summary(action, config)
        assert summary is not None, "Summary should not be None for non-excluded, non-say action"

        # The summary should contain the error state prefix
        assert f"→ {action_state}: " in summary, (
            f"Expected '→ {action_state}: ' in summary but got: {summary!r}"
        )

        # Extract the outcome portion after "→ "
        arrow_idx = summary.index("→ ")
        outcome_portion = summary[arrow_idx + 2:]  # skip "→ "

        # Parse the error portion from the outcome
        prefix = f"{action_state}: "
        assert outcome_portion.startswith(prefix), (
            f"Outcome should start with '{prefix}' but got: {outcome_portion!r}"
        )
        error_in_outcome = outcome_portion[len(prefix):]

        # Error portion in outcome must be ≤200 chars
        assert len(error_in_outcome) <= 200, (
            f"Error in outcome exceeds 200 chars: {len(error_in_outcome)} chars"
        )

    @given(
        action_state=failed_state_strategy,
        error_msg=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "S", "Z"),
                blacklist_characters="\x00",
            ),
            min_size=201,
            max_size=500,
        ),
    )
    @settings(max_examples=100)
    def test_long_error_messages_are_truncated_to_200_chars(
        self, action_state: str, error_msg: str
    ):
        """For any failed/error action where the error message exceeds 200 chars,
        the error in the outcome SHALL be truncated to exactly 200 characters."""
        action = {
            "actionState": action_state,
            "output": {"error": error_msg},
        }

        outcome = _extract_outcome(action)

        # Extract error portion
        prefix = f"{action_state}: "
        assert outcome.startswith(prefix)
        error_portion = outcome[len(prefix):]

        # Must be exactly 200 chars and end with "..."
        assert len(error_portion) == 200, (
            f"Expected 200 chars, got {len(error_portion)}"
        )
        assert error_portion.endswith("..."), (
            f"Expected truncated error to end with '...' but got: {error_portion[-10:]!r}"
        )
