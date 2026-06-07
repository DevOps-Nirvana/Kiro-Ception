"""Property-based test for prompt extraction fidelity.

# Feature: conversation-logger, Property 1: Prompt Extraction Fidelity

**Validates: Requirements 1.1, 1.5**

Property: *For any* execution log containing a non-empty `input.data.userPrompt`
field, the IDE_Loader SHALL produce an IndexedMessage whose `searchable_text`
equals the userPrompt value (after code-block replacement and skip filtering
are applied).
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.ide_loader import (
    _extract_user_prompt_from_execution_log,
    _replace_code_blocks,
    _should_skip_message,
)
from kiro_ception.models import ContentTier

# Prefixes that cause _should_skip_message to filter user messages
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

# --- Strategies ---

# Generate non-empty strings that won't be filtered by _should_skip_message.
_valid_prompt_chars = st.characters(
    blacklist_categories=("Cs",),  # Exclude surrogate characters
)


def _non_skip_prompt() -> st.SearchStrategy[str]:
    """Strategy producing non-empty prompts that pass _should_skip_message."""
    return st.text(
        alphabet=_valid_prompt_chars,
        min_size=1,
        max_size=500,
    ).filter(
        lambda s: (
            s.strip()  # Non-empty after stripping
            and not _should_skip_message("user", s)
        )
    )


# Strategy for prompts that contain fenced code blocks
@st.composite
def _prompt_with_code_blocks(draw):
    """Strategy producing prompts that contain fenced code blocks."""
    lang = draw(st.sampled_from(["python", "javascript", "typescript", "rust", "go", ""]))
    code_body = draw(st.text(
        alphabet=st.characters(
            blacklist_characters="`",
            blacklist_categories=("Cs",),
        ),
        min_size=1,
        max_size=100,
    ))
    prefix = draw(st.text(
        alphabet=_valid_prompt_chars,
        min_size=1,
        max_size=100,
    ).filter(
        lambda s: s.strip() and not any(s.startswith(p) for p in _SKIP_PREFIXES)
    ))
    # Build a fenced code block that matches the regex ```(\w*)\n...\n```
    block = f"```{lang}\n{code_body}\n```"
    return f"{prefix}\n{block}"


def _write_exec_log(tmpdir: Path, user_prompt: str) -> Path:
    """Write a minimal execution log with the given userPrompt."""
    data = {
        "input": {"data": {"userPrompt": user_prompt}},
        "startTime": 1700000000000,
        "actions": [],
    }
    exec_file = tmpdir / "exec.json"
    exec_file.write_text(json.dumps(data), encoding="utf-8")
    return exec_file


class TestProperty1PromptExtractionFidelity:
    """Property 1: Prompt Extraction Fidelity

    *For any* execution log containing a non-empty `input.data.userPrompt`
    field, the IDE_Loader SHALL produce an IndexedMessage whose
    `searchable_text` equals the userPrompt value (after code-block
    replacement and skip filtering are applied).

    **Validates: Requirements 1.1, 1.5**
    """

    @given(prompt=_non_skip_prompt())
    @settings(max_examples=100)
    def test_extraction_equals_code_block_replaced_prompt(self, prompt: str):
        """For any valid non-skip prompt, extraction produces searchable_text
        equal to _replace_code_blocks(prompt)."""
        # Compute expected text after code-block replacement
        expected = _replace_code_blocks(prompt)

        # Skip if the result would be empty after processing
        assume(expected.strip())

        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = _write_exec_log(Path(tmpdir), prompt)
            result = _extract_user_prompt_from_execution_log(
                exec_file, "session-test", "/workspace", datetime(2026, 1, 1)
            )

        assert result is not None, (
            f"Expected extraction to succeed for prompt: {prompt!r}"
        )
        assert result.searchable_text == expected, (
            f"Extracted text mismatch.\n"
            f"Input prompt: {prompt!r}\n"
            f"Expected (after code-block replace): {expected!r}\n"
            f"Got: {result.searchable_text!r}"
        )

    @given(prompt=_non_skip_prompt())
    @settings(max_examples=100)
    def test_extraction_produces_conversation_tier(self, prompt: str):
        """For any valid prompt, the extracted message has content_tier=CONVERSATION."""
        expected = _replace_code_blocks(prompt)
        assume(expected.strip())

        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = _write_exec_log(Path(tmpdir), prompt)
            result = _extract_user_prompt_from_execution_log(
                exec_file, "session-test", "/workspace", datetime(2026, 1, 1)
            )

        assert result is not None
        assert result.content_tier == ContentTier.CONVERSATION

    @given(prompt=_non_skip_prompt())
    @settings(max_examples=100)
    def test_extraction_produces_user_role(self, prompt: str):
        """For any valid prompt, the extracted message has role='user'."""
        expected = _replace_code_blocks(prompt)
        assume(expected.strip())

        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = _write_exec_log(Path(tmpdir), prompt)
            result = _extract_user_prompt_from_execution_log(
                exec_file, "session-test", "/workspace", datetime(2026, 1, 1)
            )

        assert result is not None
        assert result.role == "user"

    @given(prompt=_prompt_with_code_blocks())
    @settings(max_examples=100)
    def test_code_blocks_are_replaced_in_extraction(self, prompt: str):
        """For any prompt containing code blocks, extraction replaces them with
        [code:lang] placeholders."""
        # Ensure the prompt won't be skipped
        assume(not _should_skip_message("user", prompt))
        expected = _replace_code_blocks(prompt)
        assume(expected.strip())

        with tempfile.TemporaryDirectory() as tmpdir:
            exec_file = _write_exec_log(Path(tmpdir), prompt)
            result = _extract_user_prompt_from_execution_log(
                exec_file, "session-test", "/workspace", datetime(2026, 1, 1)
            )

        assert result is not None
        assert result.searchable_text == expected
        # Verify that code blocks are actually replaced (no raw ``` in output
        # unless they're unmatched single backticks)
        assert "```" not in result.searchable_text or "```" in expected
