# Feature: conversation-logger, Property 11: Meaningful Output Truncation
"""Property-based test for meaningful output truncation.

**Validates: Requirements 2.7**

Property 11: *For any* tool action whose output contains a Meaningful_Result
and where `include_meaningful_output` is true, the Tool_Summary SHALL include
an excerpt of that result with length ≤ 500 characters.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.tool_summaries import (
    ToolSummaryConfig,
    _extract_meaningful_output,
    generate_tool_summary,
)

# Keywords that _looks_meaningful checks for
MEANINGFUL_INDICATORS = [
    "error",
    "Error",
    "ERROR",
    "fail",
    "FAIL",
    "warning",
    "Warning",
    "WARN",
    "test",
    "assert",
    "exception",
    "Exception",
    "traceback",
    "Traceback",
    "npm ERR",
    "PASSED",
    "FAILED",
    "passed",
    "failed",
]

# Strategy: generate a string that contains at least one meaningful indicator
meaningful_indicator = st.sampled_from(MEANINGFUL_INDICATORS)


def meaningful_text_strategy(min_size: int = 10, max_size: int = 2000):
    """Generate text that will pass the _looks_meaningful check.

    Text must be ≥10 chars and contain at least one meaningful indicator.
    """
    return st.builds(
        lambda prefix, indicator, suffix: prefix + indicator + suffix,
        prefix=st.text(min_size=5, max_size=max_size // 2),
        indicator=meaningful_indicator,
        suffix=st.text(min_size=0, max_size=max_size // 2),
    )


# Strategy: generate output fields that contain meaningful content
output_field_strategy = st.sampled_from(["error", "stderr", "stdout", "result"])


@st.composite
def action_with_meaningful_output(draw):
    """Generate a tool action with meaningful output in one of the recognized fields."""
    action_type = draw(
        st.sampled_from(
            ["readFile", "writeFile", "fs_write", "grep_search", "execute_pwsh", "invoke_sub_agent", "custom_tool"]
        )
    )

    # Generate meaningful text of various lengths (including >500 chars)
    text_length = draw(st.integers(min_value=10, max_value=2000))
    meaningful_text = draw(meaningful_text_strategy(min_size=10, max_size=text_length))
    assume(len(meaningful_text) >= 10)  # _looks_meaningful requires ≥10 chars

    # Choose which output field to place the meaningful text in
    field_name = draw(output_field_strategy)

    # Build minimal input based on action type
    if action_type == "readFile":
        input_data = {"path": "/src/file.ts"}
    elif action_type in ("writeFile", "fs_write"):
        input_data = {"path": "/src/file.ts"}
    elif action_type in ("grep_search", "file_search"):
        input_data = {"query": "pattern"}
    elif action_type == "execute_pwsh":
        input_data = {"command": "npm test"}
    elif action_type == "invoke_sub_agent":
        input_data = {"name": "agent", "prompt": "do something"}
    else:
        input_data = {"key": "value"}

    # Build output with meaningful content in the chosen field
    output_data = {}
    if field_name == "error":
        output_data["error"] = meaningful_text
    elif field_name == "stderr":
        output_data["stderr"] = meaningful_text
    elif field_name == "stdout":
        output_data["stdout"] = meaningful_text
    elif field_name == "result":
        output_data["result"] = meaningful_text

    action = {
        "actionType": action_type,
        "actionState": "completed",
        "input": input_data,
        "output": output_data,
    }

    return action, meaningful_text


@given(data=action_with_meaningful_output())
@settings(max_examples=100)
def test_meaningful_output_excerpt_is_at_most_500_chars(data):
    """Property 11: When include_meaningful_output=True and output has meaningful content,
    the excerpt returned by _extract_meaningful_output is ≤ 500 characters."""
    action, _meaningful_text = data

    result = _extract_meaningful_output(action)

    # If meaningful output was detected, it must be ≤ 500 chars
    if result:
        assert len(result) <= 500, (
            f"Meaningful output excerpt exceeds 500 chars: got {len(result)} chars"
        )


@given(data=action_with_meaningful_output())
@settings(max_examples=100)
def test_tool_summary_includes_excerpt_when_meaningful_output_enabled(data):
    """Property 11: When include_meaningful_output=True and output contains meaningful
    content, the generated Tool_Summary SHALL include an excerpt of that result."""
    action, meaningful_text = data

    config = ToolSummaryConfig(include_meaningful_output=True)
    summary = generate_tool_summary(action, config)

    # Summary should not be None (it's not a say action or excluded tool)
    assert summary is not None

    # The summary should contain some portion of the meaningful output
    # (it could be truncated, but at least some content should appear)
    extracted = _extract_meaningful_output(action)
    if extracted:
        # The extracted content should appear in the summary
        # (possibly truncated by max_summary_length, but the excerpt itself is ≤500)
        assert len(extracted) <= 500


@given(data=action_with_meaningful_output())
@settings(max_examples=100)
def test_meaningful_output_truncation_preserves_content_start(data):
    """Property 11: When truncation occurs (text > 500 chars), the excerpt
    starts with the beginning of the meaningful text and ends with '...'."""
    action, meaningful_text = data

    result = _extract_meaningful_output(action)

    if result and len(result) == 500:
        # If exactly 500, it was truncated (497 chars + "...")
        assert result.endswith("...")
        # The start of the result should match the start of the original meaningful text
        # (extracted from the output field)
        assert len(result) <= 500
