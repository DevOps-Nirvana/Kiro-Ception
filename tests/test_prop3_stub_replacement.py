"""Property-based test for Property 3: Stub Replacement.

# Feature: conversation-logger, Property 3: Stub Replacement

**Validates: Requirements 1.3**

Property: *For any* session where both a session-file stub message and a corresponding
execution log with a non-empty `input.data.userPrompt` exist, the final indexed messages
for that turn SHALL contain the full prompt text from the execution log, not the stub text.

Test strategy:
- Generate pairs of (stub existing messages, new full-content messages) at the same position
- Generate stub user messages ("continue", "yes", "ok") and stub assistant messages ("On it.")
  paired with corresponding full replacement messages at the same position
- Verify that _replace_stub_messages returns the full content, not the stub
"""

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.ide_loader import _replace_stub_messages
from kiro_ception.models import ContentTier, IndexedMessage, Source


# --- Strategies ---

# Known stub texts for user role
_USER_STUBS = st.sampled_from([
    "continue",
    "go ahead",
    "yes",
    "ok",
    "proceed",
    "do it",
    "",
    "   ",
    "Continue",
    "YES",
    "Ok",
])

# Known stub texts for assistant role
_ASSISTANT_STUBS = st.sampled_from([
    "On it.",
    "On it",
    "",
    "   ",
])

# Full replacement text: non-stub content that should replace the stub
_FULL_USER_TEXT = st.text(
    min_size=10, max_size=200,
    alphabet=st.characters(categories=("L", "N", "P", "Z"), exclude_characters="\x00")
).filter(
    # Ensure the generated text is NOT itself a stub
    lambda t: t.strip().lower() not in {
        "continue", "go ahead", "yes", "ok", "proceed", "do it"
    } and len(t.strip()) > 0
)

_FULL_ASSISTANT_TEXT = st.text(
    min_size=10, max_size=200,
    alphabet=st.characters(categories=("L", "N", "P", "Z"), exclude_characters="\x00")
).filter(
    # Ensure the generated text is NOT itself a stub
    lambda t: t.strip() not in ("On it.", "On it") and len(t.strip()) > 0
)


def _make_msg(
    uuid: str,
    role: str,
    text: str,
    message_index: int = 0,
    content_tier: ContentTier = ContentTier.CONVERSATION,
) -> IndexedMessage:
    """Helper to create an IndexedMessage with defaults."""
    return IndexedMessage(
        uuid=uuid,
        session_id="session-1",
        workspace="/workspace",
        timestamp=datetime(2026, 1, 1),
        role=role,
        searchable_text=text,
        message_index=message_index,
        source=Source.IDE,
        content_tier=content_tier,
    )


@st.composite
def stub_and_replacement_pair(draw: st.DrawFn) -> tuple[IndexedMessage, IndexedMessage]:
    """Generate a (stub existing message, full new message) pair at the same position.

    Randomly chooses between user stubs and assistant stubs.
    """
    role = draw(st.sampled_from(["user", "assistant"]))
    message_index = draw(st.integers(min_value=0, max_value=50))

    if role == "user":
        stub_text = draw(_USER_STUBS)
        full_text = draw(_FULL_USER_TEXT)
    else:
        stub_text = draw(_ASSISTANT_STUBS)
        full_text = draw(_FULL_ASSISTANT_TEXT)

    existing_stub = _make_msg(
        uuid=f"existing-{role}-{message_index}",
        role=role,
        text=stub_text,
        message_index=message_index,
    )

    new_full = _make_msg(
        uuid=f"new-{role}-{message_index}",
        role=role,
        text=full_text,
        message_index=message_index,
    )

    return existing_stub, new_full


@st.composite
def multiple_stub_replacement_pairs(
    draw: st.DrawFn,
) -> tuple[list[IndexedMessage], list[IndexedMessage]]:
    """Generate multiple stub/replacement pairs ensuring unique positions."""
    count = draw(st.integers(min_value=1, max_value=5))
    existing_msgs: list[IndexedMessage] = []
    new_msgs: list[IndexedMessage] = []
    used_positions: set[tuple[str, int]] = set()

    for i in range(count):
        role = draw(st.sampled_from(["user", "assistant"]))
        # Use unique index per role to avoid position collisions within same role
        message_index = i * 10 + draw(st.integers(min_value=0, max_value=5))

        position_key = (role, message_index)
        if position_key in used_positions:
            continue
        used_positions.add(position_key)

        if role == "user":
            stub_text = draw(_USER_STUBS)
            full_text = draw(_FULL_USER_TEXT)
        else:
            stub_text = draw(_ASSISTANT_STUBS)
            full_text = draw(_FULL_ASSISTANT_TEXT)

        existing_msgs.append(_make_msg(
            uuid=f"existing-{role}-{message_index}",
            role=role,
            text=stub_text,
            message_index=message_index,
        ))

        new_msgs.append(_make_msg(
            uuid=f"new-{role}-{message_index}",
            role=role,
            text=full_text,
            message_index=message_index,
        ))

    return existing_msgs, new_msgs


# --- Property Tests ---


@settings(max_examples=100)
@given(pair=stub_and_replacement_pair())
def test_prop3_stub_replaced_by_full_content(
    pair: tuple[IndexedMessage, IndexedMessage],
):
    """Property 3: When a stub message exists at a position and a new full-content message
    arrives at the same position, the result contains the full content, not the stub.

    # Feature: conversation-logger, Property 3: Stub Replacement
    **Validates: Requirements 1.3**
    """
    existing_stub, new_full = pair

    result = _replace_stub_messages(
        new_messages=[new_full],
        existing_messages=[existing_stub],
    )

    # The result should contain exactly one conversation message at this position
    conversation_msgs = [
        m for m in result
        if m.content_tier == ContentTier.CONVERSATION
        and m.role == new_full.role
        and m.message_index == new_full.message_index
    ]
    assert len(conversation_msgs) == 1, (
        f"Expected exactly 1 conversation message at position "
        f"({new_full.role}, {new_full.message_index}), got {len(conversation_msgs)}.\n"
        f"Result messages: {[(m.uuid, m.searchable_text[:50]) for m in result]}"
    )

    # The surviving message should have the full content, NOT the stub text
    surviving_msg = conversation_msgs[0]
    assert surviving_msg.searchable_text == new_full.searchable_text, (
        f"Expected full content text, got stub text.\n"
        f"Expected: {new_full.searchable_text!r}\n"
        f"Got: {surviving_msg.searchable_text!r}"
    )

    # Verify the stub text is NOT present in any result message at this position
    assert surviving_msg.searchable_text != existing_stub.searchable_text, (
        f"Stub text should have been replaced.\n"
        f"Stub: {existing_stub.searchable_text!r}\n"
        f"Result: {surviving_msg.searchable_text!r}"
    )


@settings(max_examples=100)
@given(data=multiple_stub_replacement_pairs())
def test_prop3_multiple_stubs_all_replaced(
    data: tuple[list[IndexedMessage], list[IndexedMessage]],
):
    """Property 3: When multiple stub messages exist at different positions and
    corresponding full-content messages arrive, ALL stubs are replaced with full content.

    # Feature: conversation-logger, Property 3: Stub Replacement
    **Validates: Requirements 1.3**
    """
    existing_msgs, new_msgs = data

    # Skip if no pairs were generated (due to position collision filtering)
    if not existing_msgs or not new_msgs:
        return

    result = _replace_stub_messages(
        new_messages=new_msgs,
        existing_messages=existing_msgs,
    )

    # For each new full-content message, verify it appears in the result
    for new_msg in new_msgs:
        matching = [
            m for m in result
            if m.content_tier == ContentTier.CONVERSATION
            and m.role == new_msg.role
            and m.message_index == new_msg.message_index
        ]

        assert len(matching) == 1, (
            f"Expected exactly 1 message at position ({new_msg.role}, {new_msg.message_index}), "
            f"got {len(matching)}.\n"
            f"Result: {[(m.uuid, m.role, m.message_index, m.searchable_text[:30]) for m in result]}"
        )

        assert matching[0].searchable_text == new_msg.searchable_text, (
            f"At position ({new_msg.role}, {new_msg.message_index}): "
            f"expected full content, got different text.\n"
            f"Expected: {new_msg.searchable_text!r}\n"
            f"Got: {matching[0].searchable_text!r}"
        )

    # Verify no stub text survives for any position that had a replacement
    for existing_stub in existing_msgs:
        position_key = (existing_stub.role, existing_stub.message_index)
        # Check if there was a new message for this position
        has_replacement = any(
            m.role == existing_stub.role and m.message_index == existing_stub.message_index
            for m in new_msgs
        )
        if has_replacement:
            surviving = [
                m for m in result
                if m.role == existing_stub.role
                and m.message_index == existing_stub.message_index
                and m.content_tier == ContentTier.CONVERSATION
            ]
            for msg in surviving:
                assert msg.searchable_text != existing_stub.searchable_text, (
                    f"Stub text at position {position_key} should have been replaced.\n"
                    f"Stub: {existing_stub.searchable_text!r}\n"
                    f"Surviving: {msg.searchable_text!r}"
                )


@settings(max_examples=100)
@given(pair=stub_and_replacement_pair())
def test_prop3_no_duplicate_messages_after_replacement(
    pair: tuple[IndexedMessage, IndexedMessage],
):
    """Property 3: After stub replacement, no duplicate conversation-tier messages
    exist at the same (role, message_index) position.

    # Feature: conversation-logger, Property 3: Stub Replacement
    **Validates: Requirements 1.3**
    """
    existing_stub, new_full = pair

    result = _replace_stub_messages(
        new_messages=[new_full],
        existing_messages=[existing_stub],
    )

    # Count conversation messages at the same position
    position_key = (new_full.role, new_full.message_index)
    msgs_at_position = [
        m for m in result
        if m.content_tier == ContentTier.CONVERSATION
        and (m.role, m.message_index) == position_key
    ]

    assert len(msgs_at_position) == 1, (
        f"Expected exactly 1 conversation message at position {position_key}, "
        f"got {len(msgs_at_position)} (duplicates detected).\n"
        f"Messages: {[(m.uuid, m.searchable_text[:50]) for m in msgs_at_position]}"
    )
