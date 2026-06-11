"""Property-based test for context window overflow prioritization.

# Feature: conversation-logger, Property 16: Context Window Overflow Prioritization

**Validates: Requirements 4.4**

Property: *For any* context window that would exceed 20 messages total, the final
window SHALL contain ≤ 20 messages, all conversation-tier messages from the
original window SHALL be retained, and only tool_context messages SHALL be
dropped (oldest first).

Test strategy:
- Generate large sessions with many messages (enough to trigger overflow >20).
- Use a mix of conversation and tool_context messages.
- Use context_size large enough to produce a window exceeding 20 messages.
- Call build_context_window and verify:
  1. Final window ≤ 20 messages.
  2. All conversation messages from the pre-overflow window are retained.
  3. Only tool_context messages are dropped.
  4. Dropped tool_context messages are the oldest ones.
"""

import time

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.search_utils import build_context_window


# --- Strategies ---


@st.composite
def overflow_session(draw: st.DrawFn) -> tuple[list[dict], str, int]:
    """Generate a session large enough to trigger overflow (>20 messages in window).

    Returns (session_messages, match_uuid, context_size).
    We ensure:
    - The session has enough messages (25-60).
    - context_size is large enough that the window will exceed 20 messages.
    - There is a mix of conversation and tool_context tiers.
    - The number of conversation messages in the window is ≤ 20
      (otherwise overflow can't satisfy both ≤20 and retain-all-conversation).
    """
    # Generate a session with 25 to 60 messages
    num_messages = draw(st.integers(min_value=25, max_value=60))

    base_ts = time.time()
    messages = []
    for i in range(num_messages):
        role = draw(st.sampled_from(["user", "assistant"]))
        # Mix of tiers — ensure some of each
        content_tier = draw(st.sampled_from(["conversation", "tool_context"]))
        messages.append({
            "uuid": f"msg-{i}",
            "role": role,
            "searchable_text": f"Message content {i}",
            "timestamp": base_ts + i * 60,
            "content_tier": content_tier,
            "message_index": i,
        })

    # Pick a match roughly in the middle to maximize window size
    middle = num_messages // 2
    match_range_start = max(0, middle - 5)
    match_range_end = min(num_messages - 1, middle + 5)
    match_idx = draw(st.integers(min_value=match_range_start, max_value=match_range_end))
    match_uuid = messages[match_idx]["uuid"]

    # Use a context_size large enough to produce >20 messages in the window
    # context_size of 11+ means window of 23+ messages if there's enough data
    context_size = draw(st.integers(min_value=11, max_value=25))

    return (messages, match_uuid, context_size)


@st.composite
def overflow_session_many_tool_context(draw: st.DrawFn) -> tuple[list[dict], str, int]:
    """Generate a session with many tool_context messages to ensure overflow drops them.

    Guarantees the window has more tool_context than conversation messages,
    ensuring the overflow logic can trim to ≤ 20 while retaining all conversation.
    """
    # 30-50 messages total
    num_messages = draw(st.integers(min_value=30, max_value=50))

    base_ts = time.time()
    messages = []
    for i in range(num_messages):
        role = draw(st.sampled_from(["user", "assistant"]))
        # Bias heavily toward tool_context (75% tool_context, 25% conversation)
        content_tier = draw(
            st.sampled_from(["tool_context", "tool_context", "tool_context", "conversation"])
        )
        messages.append({
            "uuid": f"msg-{i}",
            "role": role,
            "searchable_text": f"Message content {i}",
            "timestamp": base_ts + i * 60,
            "content_tier": content_tier,
            "message_index": i,
        })

    # Match in middle for maximum window
    match_idx = num_messages // 2
    match_uuid = messages[match_idx]["uuid"]

    # Large context_size to ensure >20 messages
    context_size = draw(st.integers(min_value=12, max_value=25))

    return (messages, match_uuid, context_size)


# --- Helper ---


def _get_pre_overflow_window(
    session_messages: list[dict], match_uuid: str, context_size: int
) -> list[dict]:
    """Compute what the window would be before overflow trimming is applied.

    This replicates the slicing logic in build_context_window before
    _apply_overflow_logic is called.
    """
    match_pos = None
    for i, msg in enumerate(session_messages):
        if msg["uuid"] == match_uuid:
            match_pos = i
            break
    if match_pos is None:
        return []

    start = max(0, match_pos - context_size)
    end = min(len(session_messages), match_pos + context_size + 1)
    return session_messages[start:end]


# --- Property Tests ---


class TestProperty16ContextWindowOverflowPrioritization:
    """Property 16: Context Window Overflow Prioritization

    *For any* context window that would exceed 20 messages total, the final
    window SHALL contain ≤ 20 messages, all conversation-tier messages from the
    original window SHALL be retained, and only tool_context messages SHALL be
    dropped (oldest first).

    **Validates: Requirements 4.4**
    """

    @given(data=overflow_session())
    @settings(max_examples=100)
    def test_final_window_at_most_20_messages(
        self, data: tuple[list[dict], str, int]
    ):
        """The final window contains at most 20 messages after overflow trimming.

        # Feature: conversation-logger, Property 16: Context Window Overflow Prioritization
        **Validates: Requirements 4.4**
        """
        session_messages, match_uuid, context_size = data

        # Ensure the pre-overflow window actually exceeds 20
        pre_overflow = _get_pre_overflow_window(session_messages, match_uuid, context_size)
        assume(len(pre_overflow) > 20)

        # The overflow logic can only drop tool_context messages.
        # If there are more than 20 conversation messages (including the match),
        # the window cannot be trimmed to ≤20. We only test the case where
        # there are enough tool_context messages to trim.
        conversation_count = sum(
            1 for msg in pre_overflow
            if msg.get("content_tier", "conversation") == "conversation"
            or msg["uuid"] == match_uuid
        )
        assume(conversation_count <= 20)

        window = build_context_window(session_messages, match_uuid, context_size)

        assert len(window) <= 20, (
            f"Final window has {len(window)} messages but should have at most 20. "
            f"Pre-overflow window had {len(pre_overflow)} messages, "
            f"with {conversation_count} conversation messages."
        )

    @given(data=overflow_session())
    @settings(max_examples=100)
    def test_all_conversation_messages_retained(
        self, data: tuple[list[dict], str, int]
    ):
        """All conversation-tier messages from the pre-overflow window are retained.

        # Feature: conversation-logger, Property 16: Context Window Overflow Prioritization
        **Validates: Requirements 4.4**
        """
        session_messages, match_uuid, context_size = data

        # Get the pre-overflow window
        pre_overflow = _get_pre_overflow_window(session_messages, match_uuid, context_size)
        assume(len(pre_overflow) > 20)

        # Count conversation messages in the pre-overflow window
        pre_conversation_uuids = {
            msg["uuid"]
            for msg in pre_overflow
            if msg.get("content_tier", "conversation") == "conversation"
        }

        # Ensure the conversation count allows the overflow to work
        # (if there are more than 20 conversation messages, the spec says retain all,
        # so the window may still be >20 which contradicts ≤20 — but that's an edge
        # case the spec doesn't fully address. We test the normal case.)
        assume(len(pre_conversation_uuids) <= 20)

        window = build_context_window(session_messages, match_uuid, context_size)

        # All conversation messages should still be present
        window_uuids = {
            msg["content"]  # we need to check by content or role
            for msg in window
        }

        # Check by matching content_tier in the output
        window_conversation_content = [
            msg for msg in window if msg.get("content_tier", "conversation") == "conversation"
        ]

        assert len(window_conversation_content) == len(pre_conversation_uuids), (
            f"Expected {len(pre_conversation_uuids)} conversation messages retained, "
            f"but found {len(window_conversation_content)} in final window. "
            f"Pre-overflow had {len(pre_overflow)} total messages."
        )

    @given(data=overflow_session_many_tool_context())
    @settings(max_examples=100)
    def test_only_tool_context_messages_dropped(
        self, data: tuple[list[dict], str, int]
    ):
        """Only tool_context messages are dropped — never conversation messages.

        # Feature: conversation-logger, Property 16: Context Window Overflow Prioritization
        **Validates: Requirements 4.4**
        """
        session_messages, match_uuid, context_size = data

        pre_overflow = _get_pre_overflow_window(session_messages, match_uuid, context_size)
        assume(len(pre_overflow) > 20)

        # Conversation messages in pre-overflow window
        pre_conversation_uuids = {
            msg["uuid"]
            for msg in pre_overflow
            if msg.get("content_tier", "conversation") == "conversation"
        }
        assume(len(pre_conversation_uuids) <= 20)

        window = build_context_window(session_messages, match_uuid, context_size)

        # Every conversation message from the pre-overflow window must appear in final
        # We verify by checking that the count of conversation messages is preserved
        window_conversation_count = sum(
            1 for msg in window
            if msg.get("content_tier", "conversation") == "conversation"
        )

        assert window_conversation_count == len(pre_conversation_uuids), (
            f"Some conversation messages were dropped! "
            f"Pre-overflow had {len(pre_conversation_uuids)} conversation messages, "
            f"but final window has {window_conversation_count}."
        )

        # The total dropped count should equal pre_overflow - final window
        dropped_count = len(pre_overflow) - len(window)
        pre_tool_context_count = sum(
            1 for msg in pre_overflow
            if msg.get("content_tier", "conversation") == "tool_context"
        )

        # All drops come from tool_context
        assert dropped_count <= pre_tool_context_count, (
            f"Dropped {dropped_count} messages but only {pre_tool_context_count} "
            f"tool_context messages existed in pre-overflow window."
        )

    @given(data=overflow_session_many_tool_context())
    @settings(max_examples=100)
    def test_oldest_tool_context_dropped_first(
        self, data: tuple[list[dict], str, int]
    ):
        """When tool_context messages are dropped, the oldest ones are dropped first.

        # Feature: conversation-logger, Property 16: Context Window Overflow Prioritization
        **Validates: Requirements 4.4**
        """
        session_messages, match_uuid, context_size = data

        pre_overflow = _get_pre_overflow_window(session_messages, match_uuid, context_size)
        assume(len(pre_overflow) > 20)

        # Get tool_context messages from pre-overflow (excluding the match itself,
        # since the match is never dropped regardless of its tier)
        pre_tool_context = [
            msg for msg in pre_overflow
            if msg.get("content_tier", "conversation") == "tool_context"
            and msg["uuid"] != match_uuid
        ]

        # Count messages that are protected (conversation + the match)
        protected_count = sum(
            1 for msg in pre_overflow
            if msg.get("content_tier", "conversation") == "conversation"
            or msg["uuid"] == match_uuid
        )

        # Ensure we can actually satisfy the constraint
        assume(protected_count <= 20)
        assume(len(pre_tool_context) > 0)

        # Need enough tool_context to actually trigger dropping
        num_to_drop = len(pre_overflow) - 20
        assume(num_to_drop > 0)
        assume(num_to_drop <= len(pre_tool_context))

        window = build_context_window(session_messages, match_uuid, context_size)

        # Identify tool_context messages in the final window (excluding the match
        # if the match happens to be tool_context — it appears as its own tier)
        window_tool_content = [
            msg["content"] for msg in window
            if msg.get("content_tier", "conversation") == "tool_context"
            and not msg["is_match"]
        ]

        # The pre-overflow tool_context messages are ordered by position (oldest first).
        # After overflow, the OLDEST should be dropped, so the NEWEST should remain.
        num_actually_dropped = len(pre_tool_context) - len(window_tool_content)

        if num_actually_dropped > 0:
            # The surviving tool_context messages should be from the end of the list
            # (i.e., the newest ones)
            expected_survivors = pre_tool_context[num_actually_dropped:]
            expected_survivor_content = [msg["searchable_text"] for msg in expected_survivors]

            assert window_tool_content == expected_survivor_content, (
                f"Oldest tool_context should be dropped first. "
                f"Expected surviving content: {expected_survivor_content[:3]}..., "
                f"but got: {window_tool_content[:3]}..."
            )
