"""Property-based test for context window size constraint.

# Feature: conversation-logger, Property 15: Context Window Size Constraint

**Validates: Requirements 4.3**

Property: *For any* context window with parameter `context_size = N`, the window
SHALL contain at most `2N + 1` messages (N before the match, the match itself,
N after).

Test strategy:
- Generate sessions of varying length (1 to 50 messages).
- Generate various context_size values (0 to 25).
- Pick a random message as the match.
- Call build_context_window and verify len(window) <= 2*N + 1.
"""

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.search_utils import build_context_window


# --- Strategies ---


@st.composite
def session_with_match_and_context_size(draw: st.DrawFn) -> tuple[list[dict], str, int]:
    """Generate a session of messages, a match UUID, and a context_size value.

    Returns (session_messages, match_uuid, context_size).
    """
    num_messages = draw(st.integers(min_value=1, max_value=50))
    context_size = draw(st.integers(min_value=0, max_value=25))

    base_ts = time.time()
    messages = []
    for i in range(num_messages):
        role = draw(st.sampled_from(["user", "assistant"]))
        content_tier = draw(st.sampled_from(["conversation", "tool_context"]))
        messages.append({
            "uuid": f"msg-{i}",
            "role": role,
            "searchable_text": f"Message content {i} " + draw(
                st.text(min_size=0, max_size=50, alphabet=st.characters(
                    blacklist_categories=("Cs",), blacklist_characters="\x00"
                ))
            ),
            "timestamp": base_ts + i * 60,
            "content_tier": content_tier,
        })

    # Pick a random message as the match
    match_idx = draw(st.integers(min_value=0, max_value=num_messages - 1))
    match_uuid = messages[match_idx]["uuid"]

    return (messages, match_uuid, context_size)


# --- Property Tests ---


class TestProperty15ContextWindowSizeConstraint:
    """Property 15: Context Window Size Constraint

    *For any* context window with parameter `context_size = N`, the window SHALL
    contain at most `2N + 1` messages (N before the match, the match itself,
    N after).

    **Validates: Requirements 4.3**
    """

    @given(data=session_with_match_and_context_size())
    @settings(max_examples=200)
    def test_context_window_size_at_most_2n_plus_1(
        self, data: tuple[list[dict], str, int]
    ):
        """The context window contains at most 2*context_size + 1 messages.

        # Feature: conversation-logger, Property 15: Context Window Size Constraint
        **Validates: Requirements 4.3**
        """
        session_messages, match_uuid, context_size = data

        window = build_context_window(session_messages, match_uuid, context_size)

        max_allowed = 2 * context_size + 1

        assert len(window) <= max_allowed, (
            f"Context window has {len(window)} messages but should have at most "
            f"2*{context_size} + 1 = {max_allowed}. "
            f"Session has {len(session_messages)} messages."
        )

    @given(data=session_with_match_and_context_size())
    @settings(max_examples=200)
    def test_context_window_contains_the_match(
        self, data: tuple[list[dict], str, int]
    ):
        """The context window always contains the matched message.

        # Feature: conversation-logger, Property 15: Context Window Size Constraint
        **Validates: Requirements 4.3**
        """
        session_messages, match_uuid, context_size = data

        window = build_context_window(session_messages, match_uuid, context_size)

        # Window must be non-empty and contain exactly one match
        assert len(window) >= 1, "Window must contain at least the matched message"
        matched = [msg for msg in window if msg["is_match"]]
        assert len(matched) == 1, (
            f"Expected exactly 1 matched message in window, got {len(matched)}"
        )

    @given(
        num_messages=st.integers(min_value=1, max_value=50),
        context_size=st.integers(min_value=0, max_value=25),
    )
    @settings(max_examples=200)
    def test_context_window_size_for_match_at_boundaries(
        self, num_messages: int, context_size: int
    ):
        """Window size respects 2N+1 even when match is at start or end.

        # Feature: conversation-logger, Property 15: Context Window Size Constraint
        **Validates: Requirements 4.3**
        """
        base_ts = time.time()
        messages = [
            {
                "uuid": f"msg-{i}",
                "role": "user" if i % 2 == 0 else "assistant",
                "searchable_text": f"Content {i}",
                "timestamp": base_ts + i * 60,
            }
            for i in range(num_messages)
        ]

        max_allowed = 2 * context_size + 1

        # Test match at start
        window_start = build_context_window(messages, "msg-0", context_size)
        assert len(window_start) <= max_allowed, (
            f"Match at start: window has {len(window_start)} messages, "
            f"max allowed is {max_allowed}"
        )

        # Test match at end
        last_uuid = f"msg-{num_messages - 1}"
        window_end = build_context_window(messages, last_uuid, context_size)
        assert len(window_end) <= max_allowed, (
            f"Match at end: window has {len(window_end)} messages, "
            f"max allowed is {max_allowed}"
        )
