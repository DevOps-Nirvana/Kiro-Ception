"""Property 8: Summary Description Correctness by Action Type.

# Feature: conversation-logger, Property 8: Summary Description Correctness by Action Type

Validates: Requirements 7.3, 7.4, 7.5, 7.6, 7.7, 7.8

*For any* tool action, the description portion of its Tool_Summary SHALL:
- For `readFile`: contain the file path and NOT contain file content
- For `writeFile`/`fs_write`: contain the target file path
- For `grep_search`/`file_search`: contain the query pattern and result count
- For `execute_pwsh`: contain the command (<=100 chars) and exit status
- For `invoke_sub_agent`: contain the agent name and prompt (<=100 chars)
- For unrecognized types: contain the first 100 chars of JSON-serialized input
"""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro_ception.tool_summaries import _describe_action


# --- Strategies ---

# File paths: non-empty strings resembling paths
file_paths = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-."
    ),
    min_size=1,
    max_size=300,
).map(lambda s: "/" + s.lstrip("/"))

# File content that should NOT appear in descriptions
file_contents = st.text(min_size=10, max_size=500)

# Search queries
search_queries = st.text(min_size=1, max_size=100)

# Result count lists
result_lists = st.lists(st.integers(), min_size=0, max_size=50)

# Shell commands (variable length, including very long ones)
shell_commands = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyz0123456789 -_./"
    ),
    min_size=1,
    max_size=300,
)

# Exit codes
exit_codes = st.integers(min_value=-1, max_value=255)

# Agent names
agent_names = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz-_"),
    min_size=1,
    max_size=50,
)

# Agent prompts (variable length)
agent_prompts = st.text(min_size=1, max_size=300)

# Unrecognized action type names (avoid known ones)
KNOWN_TYPES = {"readFile", "writeFile", "fs_write", "grep_search", "file_search",
               "execute_pwsh", "invoke_sub_agent", "say"}
unrecognized_types = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
    min_size=3,
    max_size=30,
).filter(lambda t: t not in KNOWN_TYPES)

# Arbitrary JSON-serializable input dicts for unrecognized types
arbitrary_inputs = st.dictionaries(
    keys=st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
        min_size=1,
        max_size=20,
    ),
    values=st.one_of(
        st.text(min_size=0, max_size=200),
        st.integers(),
        st.booleans(),
    ),
    min_size=1,
    max_size=5,
)


# --- Property Tests ---


class TestProperty8ReadFile:
    """Validates: Requirement 7.3 — readFile description contains path, not content."""

    @settings(max_examples=100)
    @given(path=file_paths, content=file_contents)
    def test_readfile_contains_path_not_content(self, path: str, content: str):
        """**Validates: Requirements 7.3**"""
        action = {
            "input": {"path": path},
            "output": {"filePath": path, "content": content},
        }
        description = _describe_action("readFile", action)

        # Description SHALL contain the file path
        assert path in description or description == "(unknown path)"

        # Description SHALL NOT contain file content
        # Only check if content is long enough to be distinguishable from path
        if len(content) > 10 and content not in path:
            assert content not in description


class TestProperty8WriteFile:
    """Validates: Requirement 7.4 — writeFile/fs_write description contains target path."""

    @settings(max_examples=100)
    @given(path=file_paths)
    def test_writefile_contains_target_path(self, path: str):
        """**Validates: Requirements 7.4**"""
        action = {
            "input": {"path": path},
            "output": {},
        }
        description = _describe_action("writeFile", action)
        assert path in description or description == "(unknown path)"

    @settings(max_examples=100)
    @given(path=file_paths)
    def test_fs_write_contains_target_path(self, path: str):
        """**Validates: Requirements 7.4**"""
        action = {
            "input": {"path": path},
            "output": {},
        }
        description = _describe_action("fs_write", action)
        assert path in description or description == "(unknown path)"


class TestProperty8SearchActions:
    """Validates: Requirement 7.5 — grep_search/file_search contains query and result count."""

    @settings(max_examples=100)
    @given(query=search_queries, results=result_lists)
    def test_grep_search_contains_query_and_count(self, query: str, results: list):
        """**Validates: Requirements 7.5**"""
        action = {
            "input": {"query": query},
            "output": {"results": results},
        }
        description = _describe_action("grep_search", action)

        # Description SHALL contain the query pattern
        assert query in description

        # Description SHALL contain result count
        expected_count = str(len(results))
        assert expected_count in description

    @settings(max_examples=100)
    @given(query=search_queries, matches=result_lists)
    def test_file_search_contains_query_and_count(self, query: str, matches: list):
        """**Validates: Requirements 7.5**"""
        action = {
            "input": {"query": query},
            "output": {"matches": matches},
        }
        description = _describe_action("file_search", action)

        # Description SHALL contain the query pattern
        assert query in description

        # Description SHALL contain result count
        expected_count = str(len(matches))
        assert expected_count in description


class TestProperty8ExecutePwsh:
    """Validates: Requirement 7.6 — execute_pwsh contains command (<=100 chars) and exit status."""

    @settings(max_examples=100)
    @given(command=shell_commands, exit_code=exit_codes)
    def test_execute_pwsh_contains_command_and_exit(self, command: str, exit_code: int):
        """**Validates: Requirements 7.6**"""
        action = {
            "input": {"command": command},
            "output": {"exitCode": exit_code},
        }
        description = _describe_action("execute_pwsh", action)

        # Description SHALL contain the command (possibly truncated)
        if len(command) <= 100:
            assert command in description
        else:
            # Truncated: first 97 chars + "..."
            assert command[:97] in description

        # The command portion in description SHALL be <=100 chars
        # Command portion is before " (exit "
        cmd_part = description.split(" (exit ")[0]
        assert len(cmd_part) <= 100

        # Description SHALL contain exit status
        assert str(exit_code) in description


class TestProperty8InvokeSubAgent:
    """Validates: Requirement 7.7 — invoke_sub_agent contains agent name and prompt (<=100)."""

    @settings(max_examples=100)
    @given(name=agent_names, prompt=agent_prompts)
    def test_invoke_sub_agent_contains_name_and_prompt(self, name: str, prompt: str):
        """**Validates: Requirements 7.7**"""
        action = {
            "input": {"name": name, "prompt": prompt},
            "output": {},
        }
        description = _describe_action("invoke_sub_agent", action)

        # Description SHALL contain the agent name
        assert name in description

        # Description SHALL contain the prompt (possibly truncated)
        if len(prompt) <= 100:
            assert prompt in description
        else:
            # Truncated: first 97 chars + "..."
            assert prompt[:97] in description

        # The prompt portion in the description SHALL be <=100 chars
        # Extract the quoted part after ': "'
        prefix = f'{name}: "'
        assert description.startswith(prefix)
        # Get prompt portion (between quotes)
        prompt_in_desc = description[len(prefix):]
        if prompt_in_desc.endswith('"'):
            prompt_in_desc = prompt_in_desc[:-1]
        assert len(prompt_in_desc) <= 100


class TestProperty8UnrecognizedTypes:
    """Validates: Requirement 7.8 — unrecognized types contain first 100 chars of JSON input."""

    @settings(max_examples=100)
    @given(action_type=unrecognized_types, input_data=arbitrary_inputs)
    def test_unrecognized_type_contains_serialized_input(
        self, action_type: str, input_data: dict
    ):
        """**Validates: Requirements 7.8**"""
        action = {
            "input": input_data,
            "output": {},
        }
        description = _describe_action(action_type, action)

        # Description SHALL contain the first 100 chars of JSON-serialized input
        serialized = json.dumps(input_data, ensure_ascii=False)

        if len(serialized) <= 100:
            # Full serialization should appear
            assert description == serialized
        else:
            # Truncated: first 97 chars + "..."
            assert description == serialized[:97] + "..."

        # Description length SHALL NOT exceed 100 chars
        assert len(description) <= 100
