# Feature: conversation-logger, Property 18: Message Ordering Across Executions
"""Property 18: Message Ordering Across Executions.

**Validates: Requirements 5.2**

Property: *For any* session containing multiple executions, the message_index
values across executions SHALL be monotonically increasing when executions are
sorted by `startTime`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.config import Config
from kiro_ception.ide_loader import _process_execution_logs_for_session


# --- Strategies ---

# Generate a list of distinct start times (as millisecond epoch timestamps)
# We use integers representing milliseconds since epoch, ensuring distinctness
# by drawing from a large range.
_start_time_st = st.integers(min_value=1_600_000_000_000, max_value=1_800_000_000_000)

# Non-say action types that produce tool context messages
_tool_action_types = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "list_directory",
])

# Strategy for a single tool action
_tool_action_st = st.builds(
    lambda action_type, path: {
        "actionType": action_type,
        "actionId": f"action-{action_type}",
        "actionState": "completed",
        "input": {"path": path, "query": "pattern", "command": "echo hi", "name": "agent", "prompt": "do something"},
        "output": {"content": "result", "exitCode": 0, "results": []},
    },
    action_type=_tool_action_types,
    path=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=30,
    ),
)

# Strategy for a say action
_say_action_st = st.builds(
    lambda msg: {
        "actionType": "say",
        "actionId": "say-1",
        "actionState": "completed",
        "input": {},
        "output": {"message": msg},
    },
    msg=st.text(min_size=1, max_size=50),
)

# Strategy for a list of actions within an execution (0 to 4 tool actions + 0 or 1 say)
_actions_st = st.builds(
    lambda tools, say: tools + ([say] if say else []),
    tools=st.lists(_tool_action_st, min_size=0, max_size=4),
    say=st.one_of(st.none(), _say_action_st),
)

# Strategy for a user prompt (sometimes present, sometimes not)
_user_prompt_st = st.one_of(
    st.text(min_size=1, max_size=50),  # Non-empty prompt
    st.just(""),  # Empty prompt (no user message generated)
)


@st.composite
def _execution_logs_st(draw):
    """Generate 2-5 execution logs with distinct startTime values and varying actions."""
    num_executions = draw(st.integers(min_value=2, max_value=5))

    # Draw distinct start times
    start_times = draw(
        st.lists(
            _start_time_st,
            min_size=num_executions,
            max_size=num_executions,
            unique=True,
        )
    )

    logs = []
    for start_time in start_times:
        user_prompt = draw(_user_prompt_st)
        actions = draw(_actions_st)

        log_data = {
            "chatSessionId": "test-session",
            "startTime": start_time,
            "actions": actions,
        }
        # Only add userPrompt if non-empty
        if user_prompt:
            log_data["input"] = {"data": {"userPrompt": user_prompt}}
        else:
            log_data["input"] = {"data": {}}

        logs.append((start_time, log_data))

    return logs


class TestProperty18MessageOrderingAcrossExecutions:
    """Property 18: Message Ordering Across Executions.

    *For any* session containing multiple executions, the message_index values
    across executions SHALL be monotonically increasing when executions are
    sorted by `startTime`.

    **Validates: Requirements 5.2**
    """

    @given(exec_logs=_execution_logs_st())
    @settings(max_examples=100)
    def test_message_indices_monotonically_increase_across_executions(
        self, exec_logs: list[tuple[int, dict]], tmp_path_factory
    ):
        """For any session with multiple executions sorted by startTime,
        all message_index values from an earlier execution must be less
        than all message_index values from a later execution."""

        # Write execution log files to disk
        tmp_path = tmp_path_factory.mktemp("exec_logs")
        file_paths = []
        for i, (start_time, log_data) in enumerate(exec_logs):
            exec_file = tmp_path / f"exec_{i}.json"
            exec_file.write_text(json.dumps(log_data), encoding="utf-8")
            file_paths.append(str(exec_file))

        # Mock _build_execution_index to return our test files
        with patch("kiro_ception.ide_loader._build_execution_index") as mock_index:
            mock_index.return_value = {"test-session": file_paths}

            config = Config()
            messages = _process_execution_logs_for_session(
                session_id="test-session",
                workspace="/test-workspace",
                fallback_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                config=config,
            )

        # If no messages were produced, the property holds vacuously
        if not messages:
            return

        # Sort executions by startTime to determine expected ordering
        sorted_start_times = sorted(set(st for st, _ in exec_logs))

        # Group messages by which execution they came from based on their timestamp.
        # Each execution produces messages with timestamp = parsed startTime.
        # We can use message_index ranges to group them:
        # execution_order * 100 is the base offset, so messages from execution i
        # have indices in range [i*100, (i+1)*100).
        execution_count = len(sorted_start_times)

        # Group messages by execution (using base_offset logic: exec i → indices in [i*100, (i+1)*100))
        executions_messages: dict[int, list[int]] = {}
        for msg in messages:
            exec_order = msg.message_index // 100
            if exec_order not in executions_messages:
                executions_messages[exec_order] = []
            executions_messages[exec_order].append(msg.message_index)

        # Verify monotonically increasing across executions
        sorted_exec_orders = sorted(executions_messages.keys())

        for i in range(len(sorted_exec_orders) - 1):
            current_order = sorted_exec_orders[i]
            next_order = sorted_exec_orders[i + 1]

            current_indices = executions_messages[current_order]
            next_indices = executions_messages[next_order]

            max_current = max(current_indices)
            min_next = min(next_indices)

            assert max_current < min_next, (
                f"Message ordering violation across executions: "
                f"execution {current_order} has max index {max_current}, "
                f"but execution {next_order} has min index {min_next}. "
                f"Expected all indices from earlier executions to be less than "
                f"all indices from later executions."
            )
