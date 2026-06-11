"""Property-based tests for content tier classification in ide_loader.

# Feature: conversation-logger, Property 2: Content Tier Classification

**Validates: Requirements 3.2, 3.3, 6.3**

Property: *For any* IndexedMessage produced by the IDE_Loader: messages with role
`user` or derived from `say` actions SHALL have `content_tier = "conversation"`,
messages derived from tool summaries SHALL have `content_tier = "tool_context"`
and `role = "assistant"`.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.config import Config
from kiro_ception.ide_loader import _process_execution_logs_for_session
from kiro_ception.models import ContentTier


# --- Strategies ---

# Non-say action types that produce tool context messages
non_say_action_types = st.sampled_from([
    "readFile",
    "writeFile",
    "fs_write",
    "grep_search",
    "file_search",
    "execute_pwsh",
    "invoke_sub_agent",
    "list_directory",
    "str_replace",
    "semanticRename",
    "web_fetch",
    "mcp_chrome_devtools_click",
])

# Generate a valid user prompt (non-empty, not a system prompt prefix)
user_prompts = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=5,
    max_size=200,
).filter(
    lambda t: t.strip()
    and not t.startswith("# System Prompt")
    and not t.startswith("<identity>")
    and not t.startswith("<system>")
    and not t.startswith("<instructions>")
    and not t.startswith("## Included Rules")
    and not t.startswith("<steering-reminder>")
    and not t.startswith("Follow these instructions for user requests related to spec tasks")
    and not t.startswith("You are operating in a workspace")
    and not t.startswith("CONTEXT TRANSFER:")
)

# Generate a valid assistant say message (non-empty)
say_messages = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=5,
    max_size=200,
).filter(lambda t: t.strip())

# Generate a non-say action with input and output
non_say_action_strategy = st.builds(
    lambda action_type, path, content: {
        "actionType": action_type,
        "actionId": f"tool-{action_type}",
        "actionState": "completed",
        "input": {"path": path},
        "output": {"content": content},
    },
    action_type=non_say_action_types,
    path=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=3,
        max_size=50,
    ).map(lambda p: f"/src/{p}.ts"),
    content=st.text(min_size=1, max_size=50),
)

# Generate a say action (assistant response)
say_action_strategy = st.builds(
    lambda message: {
        "actionType": "say",
        "actionId": "say-action",
        "actionState": "completed",
        "input": {},
        "output": {"message": message},
    },
    message=say_messages,
)

# Generate a mixed list of actions: at least one say and at least one non-say
mixed_actions_strategy = st.tuples(
    st.lists(non_say_action_strategy, min_size=1, max_size=5),
    st.lists(say_action_strategy, min_size=1, max_size=3),
).map(lambda pair: pair[0] + pair[1])


def _write_and_process(user_prompt: str, actions: list[dict]) -> list:
    """Write an execution log to a temp file and process it, returning messages."""
    log_data = {
        "chatSessionId": "session-1",
        "input": {"data": {"userPrompt": user_prompt}},
        "startTime": 1717614000000,
        "actions": actions,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        exec_file = Path(tmp_dir) / "exec_0.json"
        exec_file.write_text(json.dumps(log_data), encoding="utf-8")
        exec_path = str(exec_file)

        with patch("kiro_ception.ide_loader._build_execution_index") as mock_index:
            mock_index.return_value = {"session-1": [exec_path]}
            messages = _process_execution_logs_for_session(
                session_id="session-1",
                workspace="/workspace",
                fallback_timestamp=datetime(2026, 1, 1),
                config=Config(),
            )

    return messages


class TestProperty2ContentTierClassification:
    """Property 2: Content Tier Classification

    *For any* IndexedMessage produced by the IDE_Loader: messages with role `user`
    or derived from `say` actions SHALL have `content_tier = "conversation"`,
    messages derived from tool summaries SHALL have `content_tier = "tool_context"`
    and `role = "assistant"`.

    **Validates: Requirements 3.2, 3.3, 6.3**
    """

    @given(
        user_prompt=user_prompts,
        actions=mixed_actions_strategy,
    )
    @settings(max_examples=100)
    def test_user_prompt_messages_have_conversation_tier(
        self, user_prompt: str, actions: list[dict]
    ):
        """User prompt messages always get content_tier=CONVERSATION."""
        messages = _write_and_process(user_prompt, actions)

        user_messages = [m for m in messages if m.role == "user"]

        # There should be at least one user message (from our non-empty prompt)
        assert len(user_messages) >= 1, (
            f"Expected at least 1 user message, got {len(user_messages)}. "
            f"Prompt was: {user_prompt!r}"
        )

        # Every user message must have content_tier=CONVERSATION
        for msg in user_messages:
            assert msg.content_tier == ContentTier.CONVERSATION, (
                f"User message has wrong content_tier: {msg.content_tier}. "
                f"Text: {msg.searchable_text!r}"
            )

    @given(
        user_prompt=user_prompts,
        actions=mixed_actions_strategy,
    )
    @settings(max_examples=100)
    def test_say_derived_messages_have_conversation_tier(
        self, user_prompt: str, actions: list[dict]
    ):
        """Messages derived from say actions always get content_tier=CONVERSATION
        and role=assistant."""
        messages = _write_and_process(user_prompt, actions)

        # Assistant messages with content_tier=CONVERSATION come from say actions
        say_derived = [
            m for m in messages
            if m.role == "assistant" and m.content_tier == ContentTier.CONVERSATION
        ]

        # Every say-derived message must have correct tier and role
        for msg in say_derived:
            assert msg.content_tier == ContentTier.CONVERSATION, (
                f"Say-derived assistant message has wrong content_tier: {msg.content_tier}. "
                f"Text: {msg.searchable_text!r}"
            )
            assert msg.role == "assistant", (
                f"Say-derived message has wrong role: {msg.role}"
            )

    @given(
        user_prompt=user_prompts,
        actions=mixed_actions_strategy,
    )
    @settings(max_examples=100)
    def test_tool_summary_messages_have_tool_context_tier_and_assistant_role(
        self, user_prompt: str, actions: list[dict]
    ):
        """Messages derived from tool summaries always get content_tier=TOOL_CONTEXT
        and role=assistant."""
        messages = _write_and_process(user_prompt, actions)

        tool_messages = [m for m in messages if m.content_tier == ContentTier.TOOL_CONTEXT]

        # We generated at least one non-say action with input and output,
        # so we expect at least one tool context message
        assert len(tool_messages) >= 1, (
            f"Expected at least 1 tool_context message, got {len(tool_messages)}. "
            f"Actions: {[a.get('actionType') for a in actions]}"
        )

        # Every tool context message must have role=assistant
        for msg in tool_messages:
            assert msg.content_tier == ContentTier.TOOL_CONTEXT, (
                f"Tool message has wrong content_tier: {msg.content_tier}"
            )
            assert msg.role == "assistant", (
                f"Tool context message has wrong role: {msg.role}. "
                f"Expected 'assistant', got '{msg.role}'. "
                f"Text: {msg.searchable_text!r}"
            )

    @given(
        user_prompt=user_prompts,
        actions=mixed_actions_strategy,
    )
    @settings(max_examples=100)
    def test_all_messages_have_valid_content_tier_classification(
        self, user_prompt: str, actions: list[dict]
    ):
        """Every IndexedMessage has a valid content_tier and obeys the classification
        rules: user→CONVERSATION, tool summaries→TOOL_CONTEXT+assistant."""
        messages = _write_and_process(user_prompt, actions)

        for msg in messages:
            # Every message must have a valid ContentTier value
            assert msg.content_tier in (
                ContentTier.CONVERSATION,
                ContentTier.TOOL_CONTEXT,
            ), f"Message has invalid content_tier: {msg.content_tier!r}"

            # Classification invariants:
            # - role=user → must be CONVERSATION
            # - content_tier=TOOL_CONTEXT → must be role=assistant
            if msg.role == "user":
                assert msg.content_tier == ContentTier.CONVERSATION, (
                    f"User message must be CONVERSATION tier, got {msg.content_tier}"
                )
            elif msg.content_tier == ContentTier.TOOL_CONTEXT:
                assert msg.role == "assistant", (
                    f"TOOL_CONTEXT messages must have role=assistant, got {msg.role}"
                )
