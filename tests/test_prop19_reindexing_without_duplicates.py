"""Property-based test for Property 19: Re-indexing Augments Without Duplicates.

# Feature: conversation-logger, Property 19: Re-indexing Augments Without Duplicates

**Validates: Requirements 9.6**

Property: *For any* session previously indexed by unmodified Kiro Ception, re-indexing
with the fork SHALL not create duplicate conversation-tier messages. The count of
conversation messages SHALL remain the same (updated in place), and new tool_context
messages SHALL be added.

Test strategy:
- Generate existing session messages simulating unmodified Kiro Ception output:
  conversation-tier only, some stubs (short user messages like "continue", assistant
  messages like "On it.")
- Generate new fork-produced messages: conversation messages (full prompts/responses)
  at the same positions as existing stubs, plus new tool_context messages
- Verify: conversation message count doesn't increase (same or replaced, never
  duplicated), and tool_context messages are added
"""

import uuid as uuid_mod
from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.ide_loader import _replace_stub_messages
from kiro_ception.models import ContentTier, IndexedMessage, Source


# --- Strategies ---

# Stub texts that unmodified Kiro Ception would have indexed from session files
_USER_STUBS = st.sampled_from([
    "continue",
    "go ahead",
    "yes",
    "ok",
    "proceed",
    "do it",
])

_ASSISTANT_STUBS = st.sampled_from([
    "On it.",
    "On it",
])

# Non-stub conversation text (full prompts/responses from execution logs)
_FULL_TEXT = st.text(
    min_size=15, max_size=300,
    alphabet=st.characters(categories=("L", "N", "P", "Z"), exclude_characters="\x00"),
).filter(
    lambda t: (
        len(t.strip()) > 10
        and t.strip().lower() not in {
            "continue", "go ahead", "yes", "ok", "proceed", "do it"
        }
        and t.strip() not in ("On it.", "On it")
    )
)

# Tool names for tool_context messages
_TOOL_NAMES = st.sampled_from([
    "readFile", "fs_write", "grep_search", "execute_pwsh",
    "invoke_sub_agent", "file_search", "list_directory",
])


def _make_msg(
    msg_uuid: str,
    role: str,
    text: str,
    message_index: int = 0,
    content_tier: ContentTier = ContentTier.CONVERSATION,
    tool_name: str | None = None,
) -> IndexedMessage:
    """Helper to create an IndexedMessage with defaults."""
    return IndexedMessage(
        uuid=msg_uuid,
        session_id="session-reindex-test",
        workspace="/workspace",
        timestamp=datetime(2026, 1, 1),
        role=role,
        searchable_text=text,
        message_index=message_index,
        source=Source.IDE,
        content_tier=content_tier,
        tool_name=tool_name,
    )


@st.composite
def existing_session_messages(draw: st.DrawFn) -> list[IndexedMessage]:
    """Generate messages as unmodified Kiro Ception would produce.

    These are conversation-tier only (no tool_context), with a mix of:
    - Stub user messages (short session-file placeholders)
    - Stub assistant messages ("On it.")
    - Some non-stub messages (existing full conversation content)
    """
    num_turns = draw(st.integers(min_value=1, max_value=6))
    messages: list[IndexedMessage] = []

    for i in range(num_turns):
        # User message: mostly stubs, occasionally full text
        is_stub_user = draw(st.booleans())
        if is_stub_user:
            user_text = draw(_USER_STUBS)
        else:
            user_text = draw(_FULL_TEXT)

        user_idx = i * 100  # Simulating base offset pattern
        messages.append(_make_msg(
            msg_uuid=f"existing-user-{i}",
            role="user",
            text=user_text,
            message_index=user_idx,
            content_tier=ContentTier.CONVERSATION,
        ))

        # Assistant message: mostly stubs, occasionally full text
        is_stub_assistant = draw(st.booleans())
        if is_stub_assistant:
            asst_text = draw(_ASSISTANT_STUBS)
        else:
            asst_text = draw(_FULL_TEXT)

        asst_idx = user_idx + 50  # Assistant comes after user in same execution
        messages.append(_make_msg(
            msg_uuid=f"existing-assistant-{i}",
            role="assistant",
            text=asst_text,
            message_index=asst_idx,
            content_tier=ContentTier.CONVERSATION,
        ))

    return messages


@st.composite
def fork_produced_messages(
    draw: st.DrawFn, existing_msgs: list[IndexedMessage]
) -> list[IndexedMessage]:
    """Generate messages as the fork would produce from execution logs.

    For each existing conversation message position:
    - Produce a new conversation-tier message with full text (replacing stubs)
    Also produce new tool_context messages between user and assistant messages.
    """
    new_messages: list[IndexedMessage] = []

    # For each existing conversation message, produce a full replacement
    for msg in existing_msgs:
        full_text = draw(_FULL_TEXT)
        new_messages.append(_make_msg(
            msg_uuid=f"new-{msg.role}-{msg.message_index}",
            role=msg.role,
            text=full_text,
            message_index=msg.message_index,
            content_tier=ContentTier.CONVERSATION,
        ))

    # Add tool_context messages (these are new additions from the fork)
    num_tool_msgs = draw(st.integers(min_value=1, max_value=8))
    used_indices: set[int] = {m.message_index for m in new_messages}

    for t in range(num_tool_msgs):
        # Pick a message_index that doesn't collide with conversation positions
        tool_idx = draw(st.integers(min_value=1, max_value=500).filter(
            lambda x: x not in used_indices
        ))
        used_indices.add(tool_idx)

        tool_name = draw(_TOOL_NAMES)
        summary_text = f"[{tool_name}] /some/path/file.ts → completed"

        new_messages.append(_make_msg(
            msg_uuid=f"new-tool-{t}-{tool_idx}",
            role="assistant",
            text=summary_text,
            message_index=tool_idx,
            content_tier=ContentTier.TOOL_CONTEXT,
            tool_name=tool_name,
        ))

    return new_messages


@st.composite
def reindex_scenario(
    draw: st.DrawFn,
) -> tuple[list[IndexedMessage], list[IndexedMessage]]:
    """Generate a complete re-indexing scenario.

    Returns (existing_messages, new_fork_messages) where:
    - existing_messages: what unmodified Kiro Ception previously indexed
    - new_fork_messages: what the fork now produces (conversation + tool_context)
    """
    existing = draw(existing_session_messages())
    new_msgs = draw(fork_produced_messages(existing))
    return existing, new_msgs


# --- Property Tests ---


@settings(max_examples=100)
@given(scenario=reindex_scenario())
def test_prop19_conversation_count_does_not_increase(
    scenario: tuple[list[IndexedMessage], list[IndexedMessage]],
):
    """Property 19: Re-indexing shall not create duplicate conversation-tier messages.
    The count of conversation messages shall remain the same (updated in place).

    # Feature: conversation-logger, Property 19: Re-indexing Augments Without Duplicates
    **Validates: Requirements 9.6**
    """
    existing_msgs, new_msgs = scenario

    # Count existing conversation messages
    existing_conv_count = sum(
        1 for m in existing_msgs if m.content_tier == ContentTier.CONVERSATION
    )

    result = _replace_stub_messages(
        new_messages=new_msgs,
        existing_messages=existing_msgs,
    )

    # Count conversation messages in result
    result_conv_count = sum(
        1 for m in result if m.content_tier == ContentTier.CONVERSATION
    )

    # Conversation count must not increase — stubs are replaced in place
    assert result_conv_count <= existing_conv_count, (
        f"Conversation message count increased from {existing_conv_count} to "
        f"{result_conv_count} — duplicates were created.\n"
        f"Existing conv UUIDs: {[m.uuid for m in existing_msgs if m.content_tier == ContentTier.CONVERSATION]}\n"
        f"Result conv UUIDs: {[m.uuid for m in result if m.content_tier == ContentTier.CONVERSATION]}"
    )


@settings(max_examples=100)
@given(scenario=reindex_scenario())
def test_prop19_no_duplicate_conversation_at_same_position(
    scenario: tuple[list[IndexedMessage], list[IndexedMessage]],
):
    """Property 19: After re-indexing, no two conversation-tier messages shall
    occupy the same (role, message_index) position.

    # Feature: conversation-logger, Property 19: Re-indexing Augments Without Duplicates
    **Validates: Requirements 9.6**
    """
    existing_msgs, new_msgs = scenario

    result = _replace_stub_messages(
        new_messages=new_msgs,
        existing_messages=existing_msgs,
    )

    # Check for duplicate positions in conversation tier
    conv_positions: dict[tuple[str, int], list[IndexedMessage]] = {}
    for m in result:
        if m.content_tier == ContentTier.CONVERSATION:
            key = (m.role, m.message_index)
            conv_positions.setdefault(key, []).append(m)

    for position, msgs in conv_positions.items():
        assert len(msgs) == 1, (
            f"Duplicate conversation messages at position {position}:\n"
            f"  {[(m.uuid, m.searchable_text[:50]) for m in msgs]}"
        )


@settings(max_examples=100)
@given(scenario=reindex_scenario())
def test_prop19_tool_context_messages_are_added(
    scenario: tuple[list[IndexedMessage], list[IndexedMessage]],
):
    """Property 19: Re-indexing shall add new tool_context messages from the fork.

    # Feature: conversation-logger, Property 19: Re-indexing Augments Without Duplicates
    **Validates: Requirements 9.6**
    """
    existing_msgs, new_msgs = scenario

    # Count tool_context messages in new messages
    new_tool_msgs = [
        m for m in new_msgs if m.content_tier == ContentTier.TOOL_CONTEXT
    ]

    result = _replace_stub_messages(
        new_messages=new_msgs,
        existing_messages=existing_msgs,
    )

    # Count tool_context messages in result
    result_tool_msgs = [
        m for m in result if m.content_tier == ContentTier.TOOL_CONTEXT
    ]

    # All new tool_context messages should be present in the result
    # (existing had none since it came from unmodified Kiro Ception)
    assert len(result_tool_msgs) >= len(new_tool_msgs), (
        f"Expected at least {len(new_tool_msgs)} tool_context messages in result, "
        f"got {len(result_tool_msgs)}.\n"
        f"New tool UUIDs: {[m.uuid for m in new_tool_msgs]}\n"
        f"Result tool UUIDs: {[m.uuid for m in result_tool_msgs]}"
    )

    # Verify each new tool_context message UUID appears in result
    result_uuids = {m.uuid for m in result}
    for tool_msg in new_tool_msgs:
        assert tool_msg.uuid in result_uuids, (
            f"Tool context message {tool_msg.uuid} ({tool_msg.tool_name}) "
            f"was not added to the result."
        )


@settings(max_examples=100)
@given(scenario=reindex_scenario())
def test_prop19_existing_non_stub_conversation_preserved(
    scenario: tuple[list[IndexedMessage], list[IndexedMessage]],
):
    """Property 19: Re-indexing preserves existing non-stub conversation messages
    (they are updated in place, not duplicated or lost).

    # Feature: conversation-logger, Property 19: Re-indexing Augments Without Duplicates
    **Validates: Requirements 9.6**
    """
    existing_msgs, new_msgs = scenario

    result = _replace_stub_messages(
        new_messages=new_msgs,
        existing_messages=existing_msgs,
    )

    # For every (role, message_index) position that existed before,
    # exactly one conversation message should exist in the result
    existing_positions = set()
    for m in existing_msgs:
        if m.content_tier == ContentTier.CONVERSATION:
            existing_positions.add((m.role, m.message_index))

    for position in existing_positions:
        matching = [
            m for m in result
            if m.content_tier == ContentTier.CONVERSATION
            and (m.role, m.message_index) == position
        ]
        assert len(matching) == 1, (
            f"Position {position} should have exactly 1 conversation message, "
            f"got {len(matching)}.\n"
            f"Messages: {[(m.uuid, m.searchable_text[:40]) for m in matching]}"
        )
