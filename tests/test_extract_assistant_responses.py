"""Unit tests for _extract_assistant_responses_from_actions in ide_loader.py.

Tests the extraction of full assistant responses from `say` actions in
execution logs. Validates content_tier assignment, role assignment, filtering,
and code-block replacement.
"""

from datetime import datetime

import pytest

from kiro_ception.ide_loader import _extract_assistant_responses_from_actions
from kiro_ception.models import ContentTier


FALLBACK_TS = datetime(2026, 6, 1, 12, 0, 0)
SESSION_ID = "test-session-123"
WORKSPACE = "/home/user/project"


class TestExtractAssistantResponsesBasic:
    """Test basic extraction of say action responses."""

    def test_extracts_single_say_action(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "Here is the full assistant response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 1
        assert result[0].searchable_text == "Here is the full assistant response."

    def test_extracts_multiple_say_actions(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "First response."},
            },
            {
                "actionType": "readFile",
                "actionState": "completed",
                "input": {"path": "/src/foo.py"},
                "output": {"content": "file content"},
            },
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "Second response."},
            },
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 2
        assert result[0].searchable_text == "First response."
        assert result[1].searchable_text == "Second response."

    def test_skips_non_say_actions(self):
        actions = [
            {
                "actionType": "readFile",
                "actionState": "completed",
                "input": {"path": "/src/foo.py"},
                "output": {"content": "file content"},
            },
            {
                "actionType": "execute_pwsh",
                "actionState": "completed",
                "input": {"command": "npm test"},
                "output": {"exitCode": 0},
            },
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_empty_actions_list(self):
        result = _extract_assistant_responses_from_actions(
            [], SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result == []


class TestContentTierAndRole:
    """Test that content_tier and role are correctly assigned."""

    def test_content_tier_is_conversation(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "A response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].content_tier == ContentTier.CONVERSATION

    def test_role_is_assistant(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "A response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].role == "assistant"


class TestFiltering:
    """Test filtering of say actions based on various conditions."""

    def test_skips_empty_message(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": ""},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_whitespace_only_message(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "   \n  \t  "},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_non_completed_say_actions(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "pending",
                "input": {},
                "output": {"message": "Pending response."},
            },
            {
                "actionType": "say",
                "actionState": "failed",
                "input": {},
                "output": {"message": "Failed response."},
            },
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_missing_output(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_non_dict_output(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": "not a dict",
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_null_message(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": None},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_non_string_message(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": 12345},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0

    def test_skips_non_dict_action(self):
        actions = [None, "not an action", 42]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 0


class TestCodeBlockReplacement:
    """Test that code blocks are replaced with compact placeholders."""

    def test_code_blocks_replaced(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {
                    "message": "Here is the fix:\n```python\ndef foo():\n    return 42\n```\nThat should work."
                },
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 1
        assert "[code:python]" in result[0].searchable_text
        assert "def foo():" not in result[0].searchable_text
        assert "Here is the fix:" in result[0].searchable_text
        assert "That should work." in result[0].searchable_text

    def test_message_only_code_block_produces_non_empty(self):
        """A message that is only a code block should still produce a result with the placeholder."""
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {
                    "message": "```typescript\nconst x = 1;\n```"
                },
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 1
        assert result[0].searchable_text == "[code:typescript]"


class TestTimestampHandling:
    """Test timestamp extraction from emittedAt."""

    def test_uses_emitted_at_when_available(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "emittedAt": "2026-07-15T10:30:00Z",
                "input": {},
                "output": {"message": "Response with timestamp."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].timestamp.year == 2026
        assert result[0].timestamp.month == 7
        assert result[0].timestamp.day == 15

    def test_uses_fallback_when_no_emitted_at(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "No timestamp action."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].timestamp == FALLBACK_TS

    def test_uses_fallback_for_invalid_emitted_at(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "emittedAt": "not-a-date",
                "input": {},
                "output": {"message": "Invalid timestamp action."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].timestamp == FALLBACK_TS


class TestMessageMetadata:
    """Test uuid, session_id, workspace, and message_index assignment."""

    def test_uses_action_id_as_uuid(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "actionId": "action-uuid-123",
                "input": {},
                "output": {"message": "Response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].uuid == "action-uuid-123"

    def test_generates_fallback_uuid_when_no_action_id(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "Response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert SESSION_ID in result[0].uuid
        assert "say" in result[0].uuid

    def test_session_id_and_workspace_propagated(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "Response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].session_id == SESSION_ID
        assert result[0].workspace == WORKSPACE

    def test_message_index_increments(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "First."},
            },
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "Second."},
            },
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert result[0].message_index == 30000
        assert result[1].message_index == 30001

    def test_custom_msg_index_start(self):
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": "Response."},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS, msg_index_start=500
        )
        assert result[0].message_index == 500


class TestStubReplacement:
    """Test that the function extracts full content (replacing stubs like 'On it.')."""

    def test_full_response_extracted_not_stub(self):
        """The function extracts the full message, not a stub."""
        full_response = (
            "I've analyzed the code and found the issue. The problem is in the "
            "authentication module where the token validation skips expiry checks. "
            "Here's how to fix it: you need to add a timestamp comparison in the "
            "validate_token function."
        )
        actions = [
            {
                "actionType": "say",
                "actionState": "completed",
                "input": {},
                "output": {"message": full_response},
            }
        ]
        result = _extract_assistant_responses_from_actions(
            actions, SESSION_ID, WORKSPACE, FALLBACK_TS
        )
        assert len(result) == 1
        assert result[0].searchable_text == full_response
        # Verify it's NOT the stub
        assert result[0].searchable_text != "On it."
