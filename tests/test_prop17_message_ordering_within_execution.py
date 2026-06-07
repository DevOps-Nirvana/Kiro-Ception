"""Property-based test for message ordering within execution.

# Feature: conversation-logger, Property 17: Message Ordering Within Execution

**Validates: Requirements 5.1**

Property: *For any* single execution log, the assigned `message_index` values
SHALL satisfy: user prompt index < all tool summary indices < all assistant say
response indices. Tool summary indices SHALL follow action-array order
(strictly increasing).

Test strategy:
- Generate single execution logs with a random user prompt, a random mix of
  tool actions (non-say with input+output), and say actions.
- Write the execution log to a temp file and mock _build_execution_index.
- Call _process_execution_logs_for_session and verify:
  1. user prompt message_index < all tool indices
  2. all tool indices < all say response indices
  3. tool indices are strictly increasing in action-array order
"""

import json
import string
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.config import Config, ToolSummariesConfig
from kiro_ception.ide_loader import _process_execution_logs_for_session
from kiro_ception.models import ContentTier


# --- Strategies ---

_safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=100,
)

# Prefixes that trigger _should_skip_message for user role
_SKIP_PREFIXES = (
    "# System Prompt",
    "Follow these instructions for user requests related to spec tasks",
    "You are operating in a workspace",
    "<system>",
    "<instructions>",
    "<identity>",
    "## Included Rules",
    "<steering-reminder>",
    "CONTEXT TRANSFER:",
)


def _non_skip_prompt() -> st.SearchStrategy[str]:
    """Strategy producing non-empty user prompts that pass _should_skip_message."""
    return st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters="\x00",
        ),
        min_size=5,
        max_size=200,
    ).filter(
        lambda s: (
            s.strip()
            and not any(s.startswith(p) for p in _SKIP_PREFIXES)
        )
    )


# Tool action types (non-say)
_TOOL_ACTION_TYPES = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "custom_tool_abc",
])

_ACTION_STATES = st.sampled_from(["completed", "failed", "error"])


@st.composite
def tool_action(draw: st.DrawFn) -> dict:
    """Generate a non-say action with both input and output fields."""
    action_type = draw(_TOOL_ACTION_TYPES)
    action_state = draw(_ACTION_STATES)

    # Build input based on action type
    if action_type == "readFile":
        input_data = {"path": f"/src/{draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20))}.ts"}
    elif action_type in ("writeFile", "fs_write"):
        input_data = {"path": f"/src/{draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20))}.ts"}
    elif action_type in ("grep_search", "file_search"):
        input_data = {"query": draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20))}
    elif action_type == "execute_pwsh":
        input_data = {"command": draw(st.text(alphabet=string.ascii_lowercase + " -_./", min_size=1, max_size=50))}
    elif action_type == "invoke_sub_agent":
        input_data = {
            "name": draw(st.text(alphabet=string.ascii_lowercase + "-", min_size=1, max_size=20)),
            "prompt": draw(st.text(alphabet=string.ascii_lowercase + " ", min_size=1, max_size=50)),
        }
    else:
        input_data = {"param": draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20))}

    # Build output
    output_data: dict = {}
    if action_type in ("grep_search", "file_search"):
        output_data["results"] = list(range(draw(st.integers(min_value=0, max_value=5))))
    elif action_type == "execute_pwsh":
        output_data["exitCode"] = draw(st.integers(min_value=0, max_value=127))
        output_data["stdout"] = "output"
    else:
        output_data["content"] = "some content"

    if action_state in ("failed", "error"):
        output_data["error"] = "Something went wrong"

    action_id = draw(st.text(alphabet=string.ascii_lowercase + string.digits, min_size=8, max_size=16))

    return {
        "actionType": action_type,
        "actionState": action_state,
        "actionId": f"tool-{action_id}",
        "input": input_data,
        "output": output_data,
    }


@st.composite
def say_action(draw: st.DrawFn) -> dict:
    """Generate a say action with a non-empty message that won't be filtered."""
    # Generate text that won't be empty after processing
    message = draw(st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters="\x00`",
        ),
        min_size=10,
        max_size=200,
    ).filter(lambda s: s.strip()))

    action_id = draw(st.text(alphabet=string.ascii_lowercase + string.digits, min_size=8, max_size=16))

    return {
        "actionType": "say",
        "actionState": "completed",
        "actionId": f"say-{action_id}",
        "input": {},
        "output": {"message": message},
    }


@st.composite
def execution_log_data(draw: st.DrawFn) -> dict:
    """Generate a single execution log with a user prompt, tool actions, and say actions.

    The actions array has tool actions first, then say actions intermixed,
    matching what a real execution log looks like (tools run first, say comes after).
    However, the property should hold regardless of the ordering in the actions array
    since tool context messages always come before say responses in message_index.
    """
    user_prompt = draw(_non_skip_prompt())
    num_tools = draw(st.integers(min_value=1, max_value=6))
    num_says = draw(st.integers(min_value=1, max_value=3))

    tool_actions = [draw(tool_action()) for _ in range(num_tools)]
    say_actions = [draw(say_action()) for _ in range(num_says)]

    # Intermix tool and say actions in various orderings
    # This tests that the function correctly separates them regardless of array order
    all_actions = tool_actions + say_actions
    # Use a permutation strategy to shuffle
    shuffled = draw(st.permutations(all_actions))

    return {
        "chatSessionId": "test-session",
        "input": {"data": {"userPrompt": user_prompt}},
        "startTime": 1717614000000,
        "actions": list(shuffled),
    }


# --- Property Tests ---


class TestProperty17MessageOrderingWithinExecution:
    """Property 17: Message Ordering Within Execution

    *For any* single execution log, the assigned `message_index` values SHALL
    satisfy: user prompt index < all tool summary indices < all assistant say
    response indices. Tool summary indices SHALL follow action-array order
    (strictly increasing).

    **Validates: Requirements 5.1**
    """

    @given(log_data=execution_log_data())
    @settings(max_examples=100)
    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_user_prompt_index_less_than_all_tool_indices(
        self, mock_index, log_data: dict
    ):
        """User prompt message_index < all tool summary indices.

        # Feature: conversation-logger, Property 17: Message Ordering Within Execution
        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = Path(tmpdir) / "exec.json"
            exec_file.write_text(json.dumps(log_data), encoding="utf-8")
            mock_index.return_value = {"test-session": [str(exec_file)]}

            config = Config()
            messages = _process_execution_logs_for_session(
                session_id="test-session",
                workspace="/workspace",
                fallback_timestamp=datetime(2026, 1, 1),
                config=config,
            )

        user_msgs = [m for m in messages if m.role == "user"]
        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]

        # Must have at least one user message and one tool message
        assume(len(user_msgs) >= 1 and len(tool_msgs) >= 1)

        user_index = user_msgs[0].message_index
        for tool_msg in tool_msgs:
            assert user_index < tool_msg.message_index, (
                f"User prompt index ({user_index}) should be less than "
                f"tool summary index ({tool_msg.message_index}), "
                f"tool_name={tool_msg.tool_name}"
            )

    @given(log_data=execution_log_data())
    @settings(max_examples=100)
    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_all_tool_indices_less_than_all_say_response_indices(
        self, mock_index, log_data: dict
    ):
        """All tool summary indices < all assistant say response indices.

        # Feature: conversation-logger, Property 17: Message Ordering Within Execution
        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = Path(tmpdir) / "exec.json"
            exec_file.write_text(json.dumps(log_data), encoding="utf-8")
            mock_index.return_value = {"test-session": [str(exec_file)]}

            config = Config()
            messages = _process_execution_logs_for_session(
                session_id="test-session",
                workspace="/workspace",
                fallback_timestamp=datetime(2026, 1, 1),
                config=config,
            )

        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]
        asst_msgs = [
            m for m in messages
            if m.role == "assistant" and m.content_tier == ContentTier.CONVERSATION
        ]

        # Must have at least one tool message and one assistant message
        assume(len(tool_msgs) >= 1 and len(asst_msgs) >= 1)

        max_tool_index = max(m.message_index for m in tool_msgs)
        min_asst_index = min(m.message_index for m in asst_msgs)

        assert max_tool_index < min_asst_index, (
            f"Maximum tool index ({max_tool_index}) should be less than "
            f"minimum assistant say index ({min_asst_index}). "
            f"Tool indices: {[m.message_index for m in tool_msgs]}, "
            f"Assistant indices: {[m.message_index for m in asst_msgs]}"
        )

    @given(log_data=execution_log_data())
    @settings(max_examples=100)
    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_tool_indices_strictly_increasing_in_action_array_order(
        self, mock_index, log_data: dict
    ):
        """Tool summary indices are strictly increasing following action-array order.

        # Feature: conversation-logger, Property 17: Message Ordering Within Execution
        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = Path(tmpdir) / "exec.json"
            exec_file.write_text(json.dumps(log_data), encoding="utf-8")
            mock_index.return_value = {"test-session": [str(exec_file)]}

            config = Config()
            messages = _process_execution_logs_for_session(
                session_id="test-session",
                workspace="/workspace",
                fallback_timestamp=datetime(2026, 1, 1),
                config=config,
            )

        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]

        assume(len(tool_msgs) >= 2)

        # Tool indices should be strictly increasing
        indices = [m.message_index for m in tool_msgs]
        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1], (
                f"Tool indices must be strictly increasing. "
                f"Index at position {i} ({indices[i]}) is not greater than "
                f"index at position {i-1} ({indices[i-1]}). "
                f"All tool indices: {indices}"
            )

    @given(log_data=execution_log_data())
    @settings(max_examples=100)
    @patch("kiro_ception.ide_loader._build_execution_index")
    def test_full_ordering_property(
        self, mock_index, log_data: dict
    ):
        """Combined property: user_index < all tool_indices < all say_indices,
        and tool indices are strictly increasing.

        # Feature: conversation-logger, Property 17: Message Ordering Within Execution
        **Validates: Requirements 5.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = Path(tmpdir) / "exec.json"
            exec_file.write_text(json.dumps(log_data), encoding="utf-8")
            mock_index.return_value = {"test-session": [str(exec_file)]}

            config = Config()
            messages = _process_execution_logs_for_session(
                session_id="test-session",
                workspace="/workspace",
                fallback_timestamp=datetime(2026, 1, 1),
                config=config,
            )

        user_msgs = [m for m in messages if m.role == "user"]
        tool_msgs = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]
        asst_msgs = [
            m for m in messages
            if m.role == "assistant" and m.content_tier == ContentTier.CONVERSATION
        ]

        # Need all three message types for the full ordering check
        assume(len(user_msgs) >= 1 and len(tool_msgs) >= 1 and len(asst_msgs) >= 1)

        user_index = user_msgs[0].message_index
        tool_indices = [m.message_index for m in tool_msgs]
        asst_indices = [m.message_index for m in asst_msgs]

        # 1. User prompt < all tools
        assert all(user_index < ti for ti in tool_indices), (
            f"User index ({user_index}) must be less than all tool indices ({tool_indices})"
        )

        # 2. All tools < all assistant responses
        max_tool = max(tool_indices)
        assert all(max_tool < ai for ai in asst_indices), (
            f"Max tool index ({max_tool}) must be less than all "
            f"assistant indices ({asst_indices})"
        )

        # 3. Tool indices are strictly increasing
        for i in range(1, len(tool_indices)):
            assert tool_indices[i] > tool_indices[i - 1], (
                f"Tool indices must be strictly increasing: {tool_indices}"
            )

        # 4. Assistant indices are strictly increasing
        for i in range(1, len(asst_indices)):
            assert asst_indices[i] > asst_indices[i - 1], (
                f"Assistant indices must be strictly increasing: {asst_indices}"
            )
