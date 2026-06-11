"""Property 9: Summary Payload Exclusion

# Feature: conversation-logger, Property 9: Summary Payload Exclusion

**Validates: Requirements 2.8, 2.3**

*For any* tool action with an `output.content` or `output.filePath` containing file contents
or large data, the generated Tool_Summary `searchable_text` SHALL NOT contain the full
serialized `input` or `output` payload.
"""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.tool_summaries import ToolSummaryConfig, generate_tool_summary


# --- Strategies ---

# Generate large text content (>100 chars) representing file contents
large_content_st = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "Z")),
    min_size=101,
    max_size=2000,
)

# Generate file paths
file_path_st = st.from_regex(r"/[a-z]+(/[a-z_]+){1,4}\.[a-z]{1,4}", fullmatch=True)

# Non-say action types that would have large payloads
action_type_st = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "custom_tool",
    "read_code",
    "list_directory",
])

action_state_st = st.sampled_from(["completed", "failed", "error"])


# Strategy for actions with large output.content
@st.composite
def action_with_large_output_content(draw):
    """Generate an action with large content in output.content field."""
    action_type = draw(action_type_st)
    content = draw(large_content_st)
    path = draw(file_path_st)
    state = draw(action_state_st)

    action = {
        "actionType": action_type,
        "actionState": state,
        "input": {"path": path},
        "output": {
            "content": content,
            "filePath": path,
        },
    }
    return action


# Strategy for actions with large input fields
@st.composite
def action_with_large_input(draw):
    """Generate an action with large data in input fields."""
    action_type = draw(action_type_st)
    large_data = draw(large_content_st)
    path = draw(file_path_st)
    state = draw(action_state_st)

    action = {
        "actionType": action_type,
        "actionState": state,
        "input": {
            "path": path,
            "text": large_data,
            "content": large_data,
        },
        "output": {
            "filePath": path,
        },
    }
    return action


# Strategy for readFile actions specifically (must NOT include file contents)
@st.composite
def read_file_action_with_content(draw):
    """Generate a readFile action with large file content in output."""
    content = draw(large_content_st)
    path = draw(file_path_st)

    action = {
        "actionType": "readFile",
        "actionState": "completed",
        "input": {"path": path},
        "output": {
            "content": content,
            "filePath": path,
        },
    }
    return action


# --- Property Tests ---


@given(action=action_with_large_output_content())
@settings(max_examples=100)
def test_full_output_payload_not_in_summary(action):
    """The full serialized output payload must not appear in the generated summary.

    **Validates: Requirements 2.8, 2.3**
    """
    config = ToolSummaryConfig(include_meaningful_output=False)
    summary = generate_tool_summary(action, config)

    if summary is None:
        return  # Skipped action, nothing to check

    full_output_serialized = json.dumps(action["output"])
    assert full_output_serialized not in summary, (
        f"Full serialized output payload found in summary.\n"
        f"Summary: {summary[:200]}...\n"
        f"Output payload length: {len(full_output_serialized)}"
    )


@given(action=action_with_large_input())
@settings(max_examples=100)
def test_full_input_payload_not_in_summary(action):
    """For actions with large inputs, the full serialized input must not appear in summary.

    **Validates: Requirements 2.8, 2.3**
    """
    config = ToolSummaryConfig(include_meaningful_output=False)
    summary = generate_tool_summary(action, config)

    if summary is None:
        return  # Skipped action, nothing to check

    full_input_serialized = json.dumps(action["input"])
    # Only check when input is actually large (>100 chars serialized)
    if len(full_input_serialized) > 100:
        assert full_input_serialized not in summary, (
            f"Full serialized input payload found in summary.\n"
            f"Summary: {summary[:200]}...\n"
            f"Input payload length: {len(full_input_serialized)}"
        )


@given(action=action_with_large_output_content())
@settings(max_examples=100)
def test_large_output_content_not_verbatim_in_summary(action):
    """If action output.content has large text (>100 chars), it must not appear verbatim.

    **Validates: Requirements 2.8, 2.3**
    """
    config = ToolSummaryConfig(include_meaningful_output=False)
    summary = generate_tool_summary(action, config)

    if summary is None:
        return  # Skipped action, nothing to check

    large_content = action["output"]["content"]
    # The full content (>100 chars) should not appear verbatim
    if len(large_content) > 100:
        assert large_content not in summary, (
            f"Large output content found verbatim in summary.\n"
            f"Summary length: {len(summary)}, Content length: {len(large_content)}"
        )


@given(action=read_file_action_with_content())
@settings(max_examples=100)
def test_read_file_summary_excludes_file_contents(action):
    """readFile summaries must only contain the file path, never file contents.

    **Validates: Requirements 2.8, 2.3**
    """
    config = ToolSummaryConfig(include_meaningful_output=False)
    summary = generate_tool_summary(action, config)

    assert summary is not None, "readFile action should produce a summary"

    file_content = action["output"]["content"]
    file_path = action["input"]["path"]

    # File path SHOULD be in the summary
    assert file_path in summary, f"Expected file path '{file_path}' in summary: {summary}"

    # File content must NOT be in the summary
    assert file_content not in summary, (
        f"File contents found in readFile summary.\n"
        f"Summary: {summary[:200]}...\n"
        f"Content length: {len(file_content)}"
    )


@given(action=action_with_large_output_content())
@settings(max_examples=100)
def test_payload_exclusion_with_meaningful_output_enabled(action):
    """Even with include_meaningful_output=True, full payloads must not appear in summary.

    **Validates: Requirements 2.8, 2.3**
    """
    config = ToolSummaryConfig(include_meaningful_output=True)
    summary = generate_tool_summary(action, config)

    if summary is None:
        return  # Skipped action, nothing to check

    full_output_serialized = json.dumps(action["output"])
    assert full_output_serialized not in summary, (
        f"Full serialized output payload found in summary with meaningful output enabled.\n"
        f"Summary: {summary[:200]}...\n"
        f"Output payload length: {len(full_output_serialized)}"
    )

    full_input_serialized = json.dumps(action["input"])
    if len(full_input_serialized) > 100:
        assert full_input_serialized not in summary, (
            f"Full serialized input payload found in summary with meaningful output enabled.\n"
            f"Summary: {summary[:200]}...\n"
            f"Input payload length: {len(full_input_serialized)}"
        )
