"""Property-based tests for summary generation coverage.

# Feature: conversation-logger, Property 5: Summary Generation Coverage

**Validates: Requirements 2.1, 8.2**

Property: *For any* action in an execution log that has `actionType != "say"` and
contains both `input` and `output` fields and whose `actionType` is not in the
configured `excluded_tools` list, the IDE_Loader SHALL produce exactly one
`tool_context` IndexedMessage containing a Tool_Summary.
"""

import string
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.config import Config, ToolSummariesConfig
from kiro_ception.ide_loader import _generate_tool_context_messages
from kiro_ception.models import ContentTier

# --- Strategies ---

# Non-say action types (eligible for tool context generation)
eligible_action_types = st.sampled_from([
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
    "getDiagnostics",
    "custom_action_xyz",
])

# Action states
action_states = st.sampled_from(["completed", "failed", "error", "running", ""])

# Input dict strategy
input_strategy = st.fixed_dictionaries({}, optional={
    "path": st.text(alphabet=string.printable, min_size=1, max_size=100),
    "query": st.text(alphabet=string.printable, min_size=1, max_size=100),
    "command": st.text(alphabet=string.printable, min_size=1, max_size=100),
    "name": st.text(alphabet=string.printable, min_size=1, max_size=50),
    "prompt": st.text(alphabet=string.printable, min_size=0, max_size=100),
})

# Output dict strategy
output_strategy = st.fixed_dictionaries({}, optional={
    "filePath": st.text(alphabet=string.printable, min_size=1, max_size=100),
    "content": st.text(alphabet=string.printable, min_size=0, max_size=200),
    "exitCode": st.integers(min_value=-1, max_value=255),
    "error": st.text(alphabet=string.printable, min_size=1, max_size=200),
    "stderr": st.text(alphabet=string.printable, min_size=1, max_size=200),
    "count": st.integers(min_value=0, max_value=100),
    "results": st.lists(st.text(max_size=20), max_size=10),
    "matches": st.lists(st.text(max_size=20), max_size=10),
})

# An eligible action: non-say, has both input and output
eligible_action_strategy = st.fixed_dictionaries({
    "actionType": eligible_action_types,
    "actionState": action_states,
    "input": input_strategy,
    "output": output_strategy,
}, optional={
    "actionId": st.text(alphabet=string.ascii_letters + string.digits, min_size=5, max_size=20),
    "emittedAt": st.just("2025-01-15T10:00:00Z"),
})

# A "say" action (should be excluded from tool context generation)
say_action_strategy = st.fixed_dictionaries({
    "actionType": st.just("say"),
    "actionState": action_states,
    "input": input_strategy,
    "output": output_strategy,
})

# An action missing input (should be excluded)
action_missing_input_strategy = st.fixed_dictionaries({
    "actionType": eligible_action_types,
    "actionState": action_states,
    "output": output_strategy,
})

# An action missing output (should be excluded)
action_missing_output_strategy = st.fixed_dictionaries({
    "actionType": eligible_action_types,
    "actionState": action_states,
    "input": input_strategy,
})

# Strategy for generating excluded_tools lists (subset of eligible types)
excluded_tools_strategy = st.lists(
    st.sampled_from([
        "readFile",
        "writeFile",
        "fs_write",
        "grep_search",
        "list_directory",
        "getDiagnostics",
    ]),
    min_size=0,
    max_size=3,
    unique=True,
)

# Mixed action list: combination of eligible, say, and incomplete actions
mixed_action_strategy = st.lists(
    st.one_of(
        eligible_action_strategy,
        say_action_strategy,
        action_missing_input_strategy,
        action_missing_output_strategy,
    ),
    min_size=1,
    max_size=15,
)


def _make_config(excluded_tools: list[str] | None = None) -> Config:
    """Create a Config instance with the specified excluded_tools."""
    tool_summaries = ToolSummariesConfig(
        excluded_tools=excluded_tools or [],
        max_summary_length=800,
        include_meaningful_output=True,
    )
    config = Config(tool_summaries=tool_summaries)
    return config


def _count_eligible_actions(actions: list[dict], excluded_tools: list[str]) -> int:
    """Count the number of actions that should produce tool_context messages.

    An action is eligible if:
    1. It has actionType != "say"
    2. It has both "input" and "output" fields
    3. Its actionType is not in the excluded_tools list
    """
    count = 0
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = action.get("actionType", "")
        if action_type == "say":
            continue
        if "input" not in action or "output" not in action:
            continue
        if action_type in excluded_tools:
            continue
        count += 1
    return count


# --- Tests ---


class TestProperty5SummaryGenerationCoverage:
    """Property 5: Summary Generation Coverage

    *For any* action in an execution log that has `actionType != "say"` and
    contains both `input` and `output` fields and whose `actionType` is not in
    the configured `excluded_tools` list, the IDE_Loader SHALL produce exactly
    one `tool_context` IndexedMessage containing a Tool_Summary.

    **Validates: Requirements 2.1, 8.2**
    """

    @given(actions=st.lists(eligible_action_strategy, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_eligible_actions_produce_exactly_one_tool_context_message(
        self, actions: list[dict]
    ):
        """Every eligible action (non-say, has input+output, not excluded)
        produces exactly one tool_context IndexedMessage."""
        config = _make_config(excluded_tools=[])
        timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        messages = _generate_tool_context_messages(
            actions=actions,
            config=config,
            session_id="test-session",
            workspace="/test/workspace",
            timestamp=timestamp,
        )

        expected_count = len(actions)
        actual_count = len(messages)

        assert actual_count == expected_count, (
            f"Expected {expected_count} tool_context messages for "
            f"{expected_count} eligible actions, got {actual_count}."
        )

        # Verify all produced messages are tool_context tier
        for msg in messages:
            assert msg.content_tier == ContentTier.TOOL_CONTEXT, (
                f"Expected content_tier=TOOL_CONTEXT, got {msg.content_tier}"
            )

    @given(actions=mixed_action_strategy, excluded_tools=excluded_tools_strategy)
    @settings(max_examples=100)
    def test_mixed_actions_produce_correct_count_of_tool_context_messages(
        self, actions: list[dict], excluded_tools: list[str]
    ):
        """For a mixed set of actions (eligible, say, incomplete), the number
        of tool_context messages equals the count of eligible actions."""
        config = _make_config(excluded_tools=excluded_tools)
        timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        messages = _generate_tool_context_messages(
            actions=actions,
            config=config,
            session_id="test-session",
            workspace="/test/workspace",
            timestamp=timestamp,
        )

        expected_count = _count_eligible_actions(actions, excluded_tools)
        actual_count = len(messages)

        assert actual_count == expected_count, (
            f"Expected {expected_count} tool_context messages, got {actual_count}. "
            f"Actions: {[a.get('actionType') for a in actions]}, "
            f"excluded_tools: {excluded_tools}"
        )

    @given(actions=st.lists(say_action_strategy, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_say_actions_produce_no_tool_context_messages(
        self, actions: list[dict]
    ):
        """Say actions never produce tool_context messages."""
        config = _make_config(excluded_tools=[])
        timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        messages = _generate_tool_context_messages(
            actions=actions,
            config=config,
            session_id="test-session",
            workspace="/test/workspace",
            timestamp=timestamp,
        )

        assert len(messages) == 0, (
            f"Say actions should produce 0 tool_context messages, got {len(messages)}"
        )

    @given(
        actions=st.lists(eligible_action_strategy, min_size=1, max_size=8),
        excluded_tools=st.lists(
            eligible_action_types,
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_excluded_tools_are_not_summarized(
        self, actions: list[dict], excluded_tools: list[str]
    ):
        """Actions whose actionType is in excluded_tools produce no messages."""
        config = _make_config(excluded_tools=excluded_tools)
        timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        messages = _generate_tool_context_messages(
            actions=actions,
            config=config,
            session_id="test-session",
            workspace="/test/workspace",
            timestamp=timestamp,
        )

        expected_count = _count_eligible_actions(actions, excluded_tools)
        actual_count = len(messages)

        assert actual_count == expected_count, (
            f"With excluded_tools={excluded_tools}, expected {expected_count} "
            f"messages but got {actual_count}."
        )

        # Verify none of the produced messages have an excluded tool_name
        for msg in messages:
            assert msg.tool_name not in excluded_tools, (
                f"Message with tool_name={msg.tool_name} should have been excluded. "
                f"excluded_tools={excluded_tools}"
            )

    @given(
        actions=st.lists(
            st.one_of(action_missing_input_strategy, action_missing_output_strategy),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_actions_missing_input_or_output_produce_no_messages(
        self, actions: list[dict]
    ):
        """Actions that lack either input or output fields produce no messages."""
        config = _make_config(excluded_tools=[])
        timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        messages = _generate_tool_context_messages(
            actions=actions,
            config=config,
            session_id="test-session",
            workspace="/test/workspace",
            timestamp=timestamp,
        )

        assert len(messages) == 0, (
            f"Actions missing input or output should produce 0 messages, "
            f"got {len(messages)}"
        )

    @given(actions=st.lists(eligible_action_strategy, min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_each_tool_context_message_has_non_empty_searchable_text(
        self, actions: list[dict]
    ):
        """Every tool_context message must contain a non-empty summary in
        searchable_text."""
        config = _make_config(excluded_tools=[])
        timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        messages = _generate_tool_context_messages(
            actions=actions,
            config=config,
            session_id="test-session",
            workspace="/test/workspace",
            timestamp=timestamp,
        )

        for msg in messages:
            assert msg.searchable_text.strip(), (
                f"tool_context message has empty searchable_text. "
                f"tool_name={msg.tool_name}"
            )
