"""Tool summary generation for execution log actions.

Pure-function module responsible for generating compact, readable summaries
of tool calls from execution log action dictionaries.
"""

import json
from dataclasses import dataclass, field


@dataclass
class ToolSummaryConfig:
    """Configuration for tool summary generation."""

    excluded_tools: list[str] = field(default_factory=list)
    max_summary_length: int = 800
    include_meaningful_output: bool = True


def generate_tool_summary(action: dict, config: ToolSummaryConfig) -> str | None:
    """Generate a compact summary string for a tool action.

    Returns None if the action should be skipped (excluded tool, or say action).
    Format: [{actionType}] {description} → {outcome}
    """
    action_type = action.get("actionType", "")

    # Skip say actions
    if action_type == "say":
        return None

    # Skip excluded tools
    if action_type in config.excluded_tools:
        return None

    description = _describe_action(action_type, action)
    outcome = _extract_outcome(action)

    # Ensure description is non-empty and does not contain newlines
    # (newlines would break the single-line format invariant)
    description = description.replace("\n", " ").replace("\r", " ").strip()
    if not description:
        description = "(no input)"

    summary = f"[{action_type}] {description} → {outcome}"

    # Append meaningful output if configured
    if config.include_meaningful_output:
        meaningful = _extract_meaningful_output(action)
        if meaningful:
            summary = f"{summary}\n{meaningful}"
        else:
            summary = f"{summary}\n(no meaningful output)"

    # Enforce max_summary_length truncation
    if len(summary) > config.max_summary_length:
        summary = summary[: config.max_summary_length - 3] + "..."

    return summary


def _describe_action(action_type: str, action: dict) -> str:
    """Generate the description portion based on action type."""
    input_data = action.get("input", {})
    output_data = action.get("output", {})

    try:
        if action_type == "readFile":
            # File path only, no contents
            path = input_data.get("path", "") or output_data.get("filePath", "")
            return path or "(unknown path)"

        if action_type in ("writeFile", "fs_write"):
            # Target file path only
            path = input_data.get("path", "") or output_data.get("filePath", "")
            return path or "(unknown path)"

        if action_type in ("grep_search", "file_search"):
            # Query pattern + result count
            query = input_data.get("query", "")
            # Try to extract result count from output
            result_count = _count_search_results(output_data)
            if result_count is not None:
                return f'query="{query}" ({result_count} results)'
            return f'query="{query}"'

        if action_type == "execute_pwsh":
            # Command (≤100 chars) + exit status
            command = input_data.get("command", "")
            if len(command) > 100:
                command = command[:97] + "..."
            exit_code = output_data.get("exitCode", output_data.get("exit_code"))
            if exit_code is not None:
                return f"{command} (exit {exit_code})"
            return command

        if action_type == "invoke_sub_agent":
            # Agent name + prompt (≤100 chars)
            agent_name = input_data.get("name", "unknown")
            prompt = input_data.get("prompt", "")
            if len(prompt) > 100:
                prompt = prompt[:97] + "..."
            return f'{agent_name}: "{prompt}"'

        # Unrecognized: first 100 chars of JSON-serialized input
        serialized = json.dumps(input_data, ensure_ascii=False)
        if len(serialized) > 100:
            serialized = serialized[:97] + "..."
        return serialized

    except (TypeError, ValueError):
        # Handle serialization errors gracefully
        return ""


def _extract_outcome(action: dict) -> str:
    """Extract outcome status from action state and output.

    Returns "completed", "failed", or "error: {brief message}".
    """
    state = action.get("actionState", "")

    if state == "completed":
        return "completed"

    if state == "failed":
        error_msg = _get_error_message(action)
        if error_msg:
            # Truncate error message to 200 chars
            if len(error_msg) > 200:
                error_msg = error_msg[:197] + "..."
            return f"failed: {error_msg}"
        return "failed"

    if state == "error":
        error_msg = _get_error_message(action)
        if error_msg:
            if len(error_msg) > 200:
                error_msg = error_msg[:197] + "..."
            return f"error: {error_msg}"
        return "error"

    # Unknown state - treat as completed
    return "completed"


def _extract_meaningful_output(action: dict) -> str:
    """Extract truncated meaningful result from output if present.

    Returns a truncated excerpt (≤500 chars) from output for build errors,
    test failures, diagnostics, etc. Returns empty string if no meaningful
    output is found.
    """
    output_data = action.get("output", {})
    if not output_data or not isinstance(output_data, dict):
        return ""

    # Look for meaningful output in common fields
    # Priority: error messages, stderr, stdout, message
    meaningful = ""

    # Check for error output
    error = output_data.get("error", "")
    if error and isinstance(error, str):
        meaningful = error

    # Check stderr (common in execute_pwsh)
    if not meaningful:
        stderr = output_data.get("stderr", "")
        if stderr and isinstance(stderr, str):
            meaningful = stderr

    # Check stdout for build/test output
    if not meaningful:
        stdout = output_data.get("stdout", "")
        if stdout and isinstance(stdout, str) and _looks_meaningful(stdout):
            meaningful = stdout

    # Check generic output/result fields
    if not meaningful:
        result = output_data.get("result", "")
        if result and isinstance(result, str) and _looks_meaningful(result):
            meaningful = result

    if not meaningful:
        return ""

    # Truncate to 500 chars
    if len(meaningful) > 500:
        meaningful = meaningful[:497] + "..."

    return meaningful


def _get_error_message(action: dict) -> str:
    """Extract error message from action output."""
    output_data = action.get("output", {})
    if not output_data or not isinstance(output_data, dict):
        return ""

    # Try common error fields
    error = output_data.get("error", "")
    if error and isinstance(error, str):
        return error

    message = output_data.get("message", "")
    if message and isinstance(message, str):
        return message

    stderr = output_data.get("stderr", "")
    if stderr and isinstance(stderr, str):
        return stderr

    return ""


def _count_search_results(output_data: dict) -> int | None:
    """Try to extract result count from search output."""
    if not output_data or not isinstance(output_data, dict):
        return None

    # Check for explicit count field
    count = output_data.get("count")
    if count is not None and isinstance(count, int):
        return count

    # Check for results array length
    results = output_data.get("results")
    if isinstance(results, list):
        return len(results)

    # Check for matches array
    matches = output_data.get("matches")
    if isinstance(matches, list):
        return len(matches)

    return None


def _looks_meaningful(text: str) -> bool:
    """Heuristic check if text contains meaningful output worth preserving.

    Meaningful output includes build errors, test failures, diagnostic info.
    We skip pure file contents and very short/trivial output.
    """
    if len(text) < 10:
        return False

    # Keywords that suggest meaningful output
    meaningful_indicators = [
        "error",
        "Error",
        "ERROR",
        "fail",
        "FAIL",
        "warning",
        "Warning",
        "WARN",
        "test",
        "assert",
        "exception",
        "Exception",
        "traceback",
        "Traceback",
        "npm ERR",
        "PASSED",
        "FAILED",
        "passed",
        "failed",
    ]

    return any(indicator in text for indicator in meaningful_indicators)
