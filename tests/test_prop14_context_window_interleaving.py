"""Property-based tests for context window interleaving both content tiers.

# Feature: conversation-logger, Property 14: Context Window Interleaves Both Tiers

**Validates: Requirements 4.1, 4.2**

Property: *For any* context window built around a matched message, the window SHALL
include both `conversation` and `tool_context` messages from the same session,
ordered by `message_index`, and each context message SHALL include a `content_tier`
field.
"""

import time

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.search_utils import build_context_window


# --- Strategies ---

# Content tiers
CONVERSATION = "conversation"
TOOL_CONTEXT = "tool_context"

# Generate a conversation message dict (user or assistant say)
conversation_message_strategy = st.builds(
    lambda idx, role, text: {
        "uuid": f"conv-msg-{idx}",
        "role": role,
        "searchable_text": text,
        "content_tier": CONVERSATION,
        "message_index": idx,
        "timestamp": 1717614000.0 + idx * 60,
    },
    idx=st.integers(min_value=0, max_value=999),
    role=st.sampled_from(["user", "assistant"]),
    text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
        min_size=3,
        max_size=100,
    ).filter(lambda t: t.strip()),
)

# Generate a tool_context message dict
tool_context_message_strategy = st.builds(
    lambda idx, text, tool_name: {
        "uuid": f"tool-msg-{idx}",
        "role": "assistant",
        "searchable_text": text,
        "content_tier": TOOL_CONTEXT,
        "message_index": idx,
        "timestamp": 1717614000.0 + idx * 60,
        "tool_name": tool_name,
    },
    idx=st.integers(min_value=0, max_value=999),
    text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
        min_size=10,
        max_size=100,
    ).filter(lambda t: t.strip()),
    tool_name=st.sampled_from([
        "readFile", "writeFile", "grep_search", "execute_pwsh", "invoke_sub_agent",
    ]),
)


@st.composite
def session_with_both_tiers(draw):
    """Generate a session message list containing both conversation and tool_context
    messages, with unique monotonically increasing message_index values and one
    designated match UUID.

    Returns (session_messages, match_uuid, context_size).
    """
    # Generate at least 2 conversation and 2 tool_context messages to ensure
    # both tiers can appear in the context window
    num_conv = draw(st.integers(min_value=2, max_value=8))
    num_tool = draw(st.integers(min_value=2, max_value=8))
    total = num_conv + num_tool

    # Generate unique message indices
    indices = sorted(draw(
        st.lists(
            st.integers(min_value=0, max_value=500),
            min_size=total,
            max_size=total,
            unique=True,
        )
    ))

    # Decide which indices are conversation vs tool_context
    # Ensure we have exactly num_conv conversation and num_tool tool_context
    conv_positions = sorted(draw(
        st.lists(
            st.sampled_from(list(range(total))),
            min_size=num_conv,
            max_size=num_conv,
            unique=True,
        )
    ))
    tool_positions = [i for i in range(total) if i not in conv_positions]

    messages = []
    base_ts = 1717614000.0

    for pos in range(total):
        idx = indices[pos]
        if pos in conv_positions:
            role = draw(st.sampled_from(["user", "assistant"]))
            messages.append({
                "uuid": f"msg-{idx}",
                "role": role,
                "searchable_text": f"Conversation message at index {idx}",
                "content_tier": CONVERSATION,
                "message_index": idx,
                "timestamp": base_ts + idx * 60,
            })
        else:
            messages.append({
                "uuid": f"msg-{idx}",
                "role": "assistant",
                "searchable_text": f"[readFile] /src/file_{idx}.ts → completed",
                "content_tier": TOOL_CONTEXT,
                "message_index": idx,
                "timestamp": base_ts + idx * 60,
            })

    # Pick a match message - choose one that has neighbours of both tiers
    # Use a match position that allows context to include both tiers
    match_pos = draw(st.integers(min_value=0, max_value=total - 1))
    match_uuid = messages[match_pos]["uuid"]

    # Context size large enough to include neighbours but not too large
    context_size = draw(st.integers(min_value=2, max_value=max(2, total - 1)))

    return messages, match_uuid, context_size


class TestProperty14ContextWindowInterleavesBothTiers:
    """Property 14: Context Window Interleaves Both Tiers

    *For any* context window built around a matched message, the window SHALL
    include both `conversation` and `tool_context` messages from the same session,
    ordered by `message_index`, and each context message SHALL include a
    `content_tier` field.

    **Validates: Requirements 4.1, 4.2**
    """

    @given(data=session_with_both_tiers())
    @settings(max_examples=100)
    def test_context_window_includes_content_tier_field(self, data):
        """Every message in a context window SHALL include a content_tier field."""
        session_messages, match_uuid, context_size = data

        context = build_context_window(session_messages, match_uuid, context_size)

        # The window should not be empty since we have a valid match
        assert len(context) > 0, "Context window should not be empty for valid match"

        # Every context message must have a content_tier field
        for i, msg in enumerate(context):
            assert "content_tier" in msg, (
                f"Context message at position {i} is missing 'content_tier' field. "
                f"Keys present: {list(msg.keys())}"
            )
            assert msg["content_tier"] in (CONVERSATION, TOOL_CONTEXT), (
                f"Context message at position {i} has invalid content_tier: "
                f"{msg['content_tier']!r}. Expected 'conversation' or 'tool_context'."
            )

    @given(data=session_with_both_tiers())
    @settings(max_examples=100)
    def test_context_window_interleaves_both_tiers(self, data):
        """When the session contains both tiers within the context window range,
        the window SHALL include messages from both tiers."""
        session_messages, match_uuid, context_size = data

        context = build_context_window(session_messages, match_uuid, context_size)

        assert len(context) > 0, "Context window should not be empty"

        # Determine which messages from the input are within the window range
        # Find the match position
        match_pos = None
        for i, msg in enumerate(session_messages):
            if msg["uuid"] == match_uuid:
                match_pos = i
                break

        assert match_pos is not None

        start = max(0, match_pos - context_size)
        end = min(len(session_messages), match_pos + context_size + 1)
        window_input = session_messages[start:end]

        # Check if the input window range contains both tiers
        input_tiers = {msg["content_tier"] for msg in window_input}

        # If the input range contains both tiers, the output should too
        # (unless overflow logic dropped tool_context messages, but that only
        # happens when window > 20 messages - which our generator stays under)
        if CONVERSATION in input_tiers and TOOL_CONTEXT in input_tiers:
            output_tiers = {msg["content_tier"] for msg in context}
            assert CONVERSATION in output_tiers, (
                "Context window should include conversation tier messages "
                "when input range contains them"
            )
            assert TOOL_CONTEXT in output_tiers, (
                "Context window should include tool_context tier messages "
                "when input range contains them and window <= 20 messages"
            )

    @given(data=session_with_both_tiers())
    @settings(max_examples=100)
    def test_context_window_ordered_by_message_index(self, data):
        """Messages in the context window SHALL be ordered by message_index
        (chronological order from the session)."""
        session_messages, match_uuid, context_size = data

        context = build_context_window(session_messages, match_uuid, context_size)

        assert len(context) > 0, "Context window should not be empty"

        # The context window should preserve the input ordering which is by
        # message_index. Since session_messages are already ordered by
        # message_index, the window slice should maintain that order.
        # We verify by checking timestamps are non-decreasing (timestamps
        # correlate with message_index in our generated data).
        from datetime import datetime as dt

        timestamps = []
        for msg in context:
            ts = dt.fromisoformat(msg["timestamp"])
            timestamps.append(ts)

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Context messages not ordered by message_index. "
                f"Message at position {i} has timestamp {timestamps[i]} "
                f"which is before position {i-1} timestamp {timestamps[i-1]}"
            )

    @given(data=session_with_both_tiers())
    @settings(max_examples=100)
    def test_context_window_preserves_interleaved_tier_order(self, data):
        """When both tiers are present, they SHALL be interleaved by their
        message_index position, not grouped by tier."""
        session_messages, match_uuid, context_size = data

        context = build_context_window(session_messages, match_uuid, context_size)

        if len(context) < 3:
            # Need at least 3 messages to verify interleaving
            return

        # Get the tiers and check they follow the same order as the input
        match_pos = None
        for i, msg in enumerate(session_messages):
            if msg["uuid"] == match_uuid:
                match_pos = i
                break

        start = max(0, match_pos - context_size)
        end = min(len(session_messages), match_pos + context_size + 1)
        window_input = session_messages[start:end]

        # The content_tier sequence in the output should match the input order
        # (messages are interleaved by message_index, not grouped by tier)
        # Note: overflow logic only kicks in >20 messages, which we don't hit
        if len(window_input) <= 20:
            expected_tiers = [msg["content_tier"] for msg in window_input]
            actual_tiers = [msg["content_tier"] for msg in context]
            assert actual_tiers == expected_tiers, (
                f"Context window tiers don't match input order. "
                f"Expected: {expected_tiers}, Got: {actual_tiers}. "
                f"Messages should be interleaved by message_index, not grouped by tier."
            )
