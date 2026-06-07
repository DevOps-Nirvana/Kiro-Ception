"""Property-based tests for fallback to session file.

# Feature: conversation-logger, Property 4: Fallback to Session File

Validates: Requirements 1.4

Property: *For any* execution log where `input.data.userPrompt` is absent, null,
or an empty string, the IDE_Loader SHALL produce a user message using the session
file's content for that turn (if available).

We test the underlying function `_extract_user_prompt_from_execution_log` which returns
None when userPrompt is missing/null/empty/whitespace — indicating that the caller
should fall back to the session file's content.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.ide_loader import _extract_user_prompt_from_execution_log


# --- Strategies ---

# Case 1: userPrompt key is absent entirely from data dict
absent_user_prompt_data = st.fixed_dictionaries({
    "input": st.fixed_dictionaries({
        "data": st.fixed_dictionaries({}, optional={
            "otherField": st.text(min_size=0, max_size=50),
            "context": st.text(min_size=0, max_size=50),
        }),
    }),
})

# Case 2: userPrompt is explicitly null
null_user_prompt_data = st.fixed_dictionaries({
    "input": st.fixed_dictionaries({
        "data": st.fixed_dictionaries({
            "userPrompt": st.none(),
        }),
    }),
})

# Case 3: userPrompt is an empty string
empty_user_prompt_data = st.fixed_dictionaries({
    "input": st.fixed_dictionaries({
        "data": st.fixed_dictionaries({
            "userPrompt": st.just(""),
        }),
    }),
})

# Case 4: userPrompt is whitespace-only (spaces, tabs, newlines)
whitespace_chars = st.sampled_from([" ", "\t", "\n", "\r", "\r\n", "  ", "\t\t"])
whitespace_user_prompt_data = st.fixed_dictionaries({
    "input": st.fixed_dictionaries({
        "data": st.fixed_dictionaries({
            "userPrompt": st.lists(
                whitespace_chars, min_size=1, max_size=10
            ).map("".join),
        }),
    }),
})

# Case 5: input field is missing entirely
missing_input_data = st.fixed_dictionaries({
    "actions": st.just([]),
})

# Case 6: input is not a dict
non_dict_input_data = st.fixed_dictionaries({
    "input": st.one_of(
        st.just("not a dict"),
        st.just(42),
        st.just([]),
        st.just(True),
    ),
})

# Case 7: input.data is not a dict
non_dict_data_field = st.fixed_dictionaries({
    "input": st.fixed_dictionaries({
        "data": st.one_of(
            st.just("string value"),
            st.just(123),
            st.just([]),
            st.just(True),
        ),
    }),
})

# Case 8: userPrompt is not a string (non-string truthy values)
non_string_user_prompt_data = st.fixed_dictionaries({
    "input": st.fixed_dictionaries({
        "data": st.fixed_dictionaries({
            "userPrompt": st.one_of(
                st.integers(min_value=1, max_value=9999),
                st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
                st.dictionaries(
                    st.text(min_size=1, max_size=5),
                    st.text(min_size=1, max_size=5),
                    min_size=1, max_size=3,
                ),
                st.just(True),
            ),
        }),
    }),
})

# Combined strategy: any of the missing/empty/null states
fallback_exec_log_strategy = st.one_of(
    absent_user_prompt_data,
    null_user_prompt_data,
    empty_user_prompt_data,
    whitespace_user_prompt_data,
    missing_input_data,
    non_dict_input_data,
    non_dict_data_field,
    non_string_user_prompt_data,
)

# Optional extra fields that might appear in the execution log
optional_extra_fields = st.fixed_dictionaries({}, optional={
    "actions": st.just([]),
    "startTime": st.one_of(
        st.integers(min_value=1000000000000, max_value=2000000000000),
        st.just("2026-01-01T00:00:00Z"),
        st.none(),
    ),
    "executionId": st.text(min_size=5, max_size=20),
})


def _write_exec_log_to_tempfile(data: dict) -> Path:
    """Write execution log data to a named temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


class TestProperty4FallbackToSessionFile:
    """Property 4: Fallback to Session File

    *For any* execution log where `input.data.userPrompt` is absent, null,
    or an empty string, the IDE_Loader SHALL produce a user message using the
    session file's content for that turn (if available).

    We verify the precondition: _extract_user_prompt_from_execution_log returns
    None for all such cases, signaling the caller to fall back to session file.

    **Validates: Requirements 1.4**
    """

    @given(exec_data=fallback_exec_log_strategy, extras=optional_extra_fields)
    @settings(max_examples=200)
    def test_returns_none_for_missing_or_empty_user_prompt(
        self, exec_data: dict, extras: dict
    ):
        """For any execution log with absent/null/empty/whitespace userPrompt,
        _extract_user_prompt_from_execution_log returns None, indicating
        fallback to session file content should be used."""
        # Merge extra fields into the exec data
        full_data = {**exec_data, **extras}

        # Write to a temporary JSON file
        exec_file = _write_exec_log_to_tempfile(full_data)
        try:
            result = _extract_user_prompt_from_execution_log(
                exec_file=exec_file,
                session_id="test-session",
                workspace="/test/workspace",
                fallback_timestamp=datetime(2026, 1, 1, 12, 0, 0),
            )

            # The function MUST return None for all these cases,
            # indicating the caller should fall back to session file content
            assert result is None, (
                f"Expected None (fallback to session file) but got a result.\n"
                f"Execution log data: {json.dumps(full_data, default=str)[:500]}\n"
                f"Result text: {result.searchable_text if result else 'N/A'}"
            )
        finally:
            exec_file.unlink(missing_ok=True)

    @given(
        whitespace=st.lists(
            st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
            min_size=1,
            max_size=20,
        ).map("".join)
    )
    @settings(max_examples=100)
    def test_whitespace_only_prompts_trigger_fallback(self, whitespace: str):
        """For any whitespace-only userPrompt string, the function returns None."""
        exec_data = {
            "input": {"data": {"userPrompt": whitespace}},
            "actions": [],
        }

        exec_file = _write_exec_log_to_tempfile(exec_data)
        try:
            result = _extract_user_prompt_from_execution_log(
                exec_file=exec_file,
                session_id="test-session",
                workspace="/test/workspace",
                fallback_timestamp=datetime(2026, 1, 1, 12, 0, 0),
            )

            assert result is None, (
                f"Expected None for whitespace-only prompt but got result.\n"
                f"Whitespace repr: {whitespace!r}\n"
                f"Result: {result.searchable_text if result else 'N/A'}"
            )
        finally:
            exec_file.unlink(missing_ok=True)

    @given(exec_data=absent_user_prompt_data)
    @settings(max_examples=100)
    def test_absent_key_triggers_fallback(self, exec_data: dict):
        """For any execution log where userPrompt key is completely absent,
        the function returns None."""
        exec_file = _write_exec_log_to_tempfile(exec_data)
        try:
            result = _extract_user_prompt_from_execution_log(
                exec_file=exec_file,
                session_id="test-session",
                workspace="/test/workspace",
                fallback_timestamp=datetime(2026, 1, 1, 12, 0, 0),
            )

            assert result is None, (
                f"Expected None when userPrompt key is absent but got result.\n"
                f"Data: {exec_data}"
            )
        finally:
            exec_file.unlink(missing_ok=True)

    @given(exec_data=null_user_prompt_data)
    @settings(max_examples=100)
    def test_null_value_triggers_fallback(self, exec_data: dict):
        """For any execution log where userPrompt is explicitly null,
        the function returns None."""
        exec_file = _write_exec_log_to_tempfile(exec_data)
        try:
            result = _extract_user_prompt_from_execution_log(
                exec_file=exec_file,
                session_id="test-session",
                workspace="/test/workspace",
                fallback_timestamp=datetime(2026, 1, 1, 12, 0, 0),
            )

            assert result is None, (
                "Expected None when userPrompt is null but got result."
            )
        finally:
            exec_file.unlink(missing_ok=True)
