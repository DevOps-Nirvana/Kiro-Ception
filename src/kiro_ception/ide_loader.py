"""Load conversations from Kiro IDE storage.

Supports two formats:
- Legacy: .chat files in globalStorage/kiro.kiroagent/*/*.chat
- Current: workspace-sessions/<base64_workspace>/<uuid>.json
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from .config import get_config
from .models import IndexedMessage, SessionInfo, Source

logger = logging.getLogger(__name__)

# Prefixes for messages that are system/boilerplate content (no search value)
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

# Regex to match fenced code blocks: ```lang\n...\n```
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\n.*?```",
    re.DOTALL,
)


def _replace_code_blocks(text: str) -> str:
    """Replace fenced code blocks with a compact placeholder.

    ```python
    def foo():
        pass
    ```

    becomes: [code:python]

    Preserves the language tag for searchability.
    The placeholder format is simple and parseable for future use.
    """
    def _replacer(match: re.Match) -> str:
        lang = match.group(1).strip().lower()
        if lang:
            return f"[code:{lang}]"
        return "[code]"

    return _CODE_BLOCK_RE.sub(_replacer, text)


def _parse_timestamp(ts: int | str | None) -> datetime | None:
    """Parse timestamp from various formats."""
    if ts is None:
        return None
    if isinstance(ts, int):
        return datetime.fromtimestamp(ts / 1000)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _extract_text_from_content(content: str | list | dict | None) -> str:
    """Extract searchable text from message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _should_skip_message(role: str, text: str) -> bool:
    """Check if a message should be filtered out (system prompts, boilerplate)."""
    if role in ("user", "human") and any(text.startswith(p) for p in _SKIP_PREFIXES):
        return True
    return False


# --- Legacy .chat format ---


def get_chat_files() -> list[Path]:
    """Get all IDE .chat files (legacy format)."""
    return get_config().ide.get_chat_files()


def _list_legacy_sessions() -> list[SessionInfo]:
    """List all IDE sessions from legacy .chat files."""
    sessions = []
    for chat_file in get_chat_files():
        try:
            stat = chat_file.stat()
            session_id = chat_file.stem
            workspace = chat_file.parent.name

            sessions.append(
                SessionInfo(
                    session_id=session_id,
                    workspace=workspace,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    modified=datetime.fromtimestamp(stat.st_mtime),
                    source=Source.IDE,
                )
            )
        except OSError as e:
            logger.debug(f"Could not stat {chat_file}: {e}")

    return sessions


def _load_legacy_session_messages(session: SessionInfo) -> list[IndexedMessage]:
    """Load messages from a legacy .chat file."""
    chat_files = get_chat_files()
    chat_file = next((f for f in chat_files if f.stem == session.session_id), None)

    if not chat_file or not chat_file.exists():
        return []

    messages = []
    try:
        with open(chat_file, encoding="utf-8") as f:
            data = json.load(f)

        msg_list = data.get("chat", [])

        # Fallback patterns
        if not msg_list:
            msg_list = data.get("messages", data.get("history", []))
        if not msg_list and "conversation" in data:
            msg_list = data["conversation"].get("messages", [])
        if not msg_list and isinstance(data, list):
            msg_list = data

        for idx, msg in enumerate(msg_list):
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", msg.get("type", ""))
            if role not in ("user", "assistant", "human", "ai"):
                continue

            role = "user" if role in ("user", "human") else "assistant"
            content = msg.get("content", msg.get("text", msg.get("message", "")))
            text = _extract_text_from_content(content)

            if _should_skip_message(role, text):
                continue

            if not text.strip():
                continue

            # Replace fenced code blocks with compact placeholders
            text = _replace_code_blocks(text)

            timestamp = _parse_timestamp(msg.get("timestamp", msg.get("created_at")))
            if not timestamp:
                timestamp = session.modified or datetime.now()

            messages.append(
                IndexedMessage(
                    uuid=msg.get("id", msg.get("uuid", f"{session.session_id}-{idx}")),
                    session_id=session.session_id,
                    workspace=session.workspace,
                    timestamp=timestamp,
                    role=role,
                    searchable_text=text,
                    message_index=idx,
                    source=Source.IDE,
                )
            )

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Error loading legacy IDE session {session.session_id}: {e}")

    return messages


# --- Current workspace-sessions format ---


def _get_workspace_sessions_dirs() -> list[Path]:
    """Get workspace-sessions base directories across platforms."""
    config = get_config()
    # Derive workspace-sessions path from existing IDE patterns
    # The patterns point to globalStorage/kiro.kiroagent/*/*.chat
    # workspace-sessions is at globalStorage/kiro.kiroagent/workspace-sessions/
    candidates = [
        "~/Library/Application Support/Kiro/User/globalStorage/kiro.kiroagent/workspace-sessions",
        "~/.config/Kiro/User/globalStorage/kiro.kiroagent/workspace-sessions",
        "~/AppData/Roaming/Kiro/User/globalStorage/kiro.kiroagent/workspace-sessions",
    ]
    dirs = []
    for candidate in candidates:
        p = Path(candidate).expanduser()
        if p.is_dir():
            dirs.append(p)
    return dirs


def _list_workspace_sessions() -> list[SessionInfo]:
    """List all sessions from workspace-sessions directories.

    Uses stat() only for timestamps — skips expensive sessions.json parsing
    to minimize memory allocations during periodic rescans.
    """
    sessions = []

    for ws_dir in _get_workspace_sessions_dirs():
        # Each subdirectory is a base64-encoded workspace path
        for workspace_dir in ws_dir.iterdir():
            if not workspace_dir.is_dir():
                continue

            # Decode workspace path from directory name
            import base64
            try:
                encoded = workspace_dir.name
                # URL-safe base64: _ -> /, - -> +, ? -> =
                fixed = encoded.replace('-', '+').replace('_', '/').replace('?', '=')
                # Add padding if needed
                padding = 4 - len(fixed) % 4
                if padding != 4:
                    fixed += "=" * padding
                workspace_path = base64.b64decode(fixed).decode("utf-8").rstrip("?")
            except Exception:
                workspace_path = workspace_dir.name

            # Find session JSON files (UUIDs, not sessions.json)
            for session_file in workspace_dir.glob("*.json"):
                if session_file.name == "sessions.json":
                    continue

                try:
                    stat = session_file.stat()
                    session_id = session_file.stem

                    sessions.append(
                        SessionInfo(
                            session_id=session_id,
                            workspace=workspace_path,
                            created=datetime.fromtimestamp(stat.st_ctime),
                            modified=datetime.fromtimestamp(stat.st_mtime),
                            source=Source.IDE,
                        )
                    )
                except OSError as e:
                    logger.debug(f"Could not stat {session_file}: {e}")

    return sessions


def _get_execution_log_dirs() -> list[Path]:
    """Get all directories containing execution logs across workspaces.

    Execution logs are stored at:
    globalStorage/kiro.kiroagent/<workspace_hash>/414d1636299d2b9e4ce7e17fb11f63e9/
    """
    base_dirs = [
        Path("~/Library/Application Support/Kiro/User/globalStorage/kiro.kiroagent").expanduser(),
        Path("~/.config/Kiro/User/globalStorage/kiro.kiroagent").expanduser(),
        Path("~/AppData/Roaming/Kiro/User/globalStorage/kiro.kiroagent").expanduser(),
    ]
    exec_subdir = "414d1636299d2b9e4ce7e17fb11f63e9"
    dirs = []

    for base in base_dirs:
        if not base.is_dir():
            continue
        for workspace_hash_dir in base.iterdir():
            if not workspace_hash_dir.is_dir():
                continue
            if workspace_hash_dir.name in ("workspace-sessions", "dev_data"):
                continue
            exec_dir = workspace_hash_dir / exec_subdir
            if exec_dir.is_dir():
                dirs.append(exec_dir)

    return dirs


def _get_exec_cache():
    """Get the EmbeddingCache instance for execution index lookups.

    Tries to use the background indexer's cache, or creates a standalone one.
    """
    try:
        from .background_indexer import get_background_indexer
        indexer = get_background_indexer()
        if indexer.cache is not None:
            return indexer.cache
    except Exception:
        pass

    # Fallback: create cache directly
    from .embeddings import get_embedding_backend
    backend = get_embedding_backend()
    from .cache import EmbeddingCache
    return EmbeddingCache(backend.fingerprint())


# In-memory cache for the execution index (session_id -> [file_paths])
_execution_index: dict[str, list[str]] | None = None
_execution_index_built = False


def _build_execution_index() -> dict[str, list[str]]:
    """Build/update the execution index using SQLite cache for persistence.

    On first call: loads cached mappings from sqlite, then only parses
    new/changed files (by mtime). Subsequent calls use the in-memory cache.
    """
    global _execution_index, _execution_index_built

    if _execution_index_built and _execution_index is not None:
        return _execution_index

    cache = _get_exec_cache()

    # Load existing cached mappings
    stored_mtimes = cache.get_all_exec_file_mtimes()

    # Discover all execution log files on disk
    all_files: list[tuple[Path, float]] = []
    for exec_dir in _get_execution_log_dirs():
        try:
            for exec_file in exec_dir.iterdir():
                if not exec_file.is_file():
                    continue
                try:
                    mtime = exec_file.stat().st_mtime
                    all_files.append((exec_file, mtime))
                except OSError:
                    continue
        except OSError:
            continue

    # Determine which files need (re-)parsing
    new_entries: list[tuple[str, str, float]] = []  # (path, session_id, mtime)
    files_to_parse: list[tuple[Path, float]] = []

    for exec_file, mtime in all_files:
        file_str = str(exec_file)
        stored_mtime = stored_mtimes.get(file_str)
        if stored_mtime is not None and mtime == stored_mtime:
            continue  # Unchanged, already in cache
        files_to_parse.append((exec_file, mtime))

    # Parse only new/changed files
    if files_to_parse:
        logger.info(f"Execution index: parsing {len(files_to_parse)} new/changed files (of {len(all_files)} total)")
        for exec_file, mtime in files_to_parse:
            try:
                with open(exec_file, encoding="utf-8") as f:
                    data = json.load(f)

                session_id = data.get("chatSessionId", "")
                if not session_id:
                    for action in data.get("actions", []):
                        session_id = action.get("chatSessionId", "")
                        if session_id:
                            break

                if session_id:
                    new_entries.append((str(exec_file), session_id, mtime))
            except (json.JSONDecodeError, OSError):
                continue

        # Store new entries in sqlite
        if new_entries:
            cache.put_exec_files_batch([(path, sid, mt) for path, sid, mt in new_entries])

    # Build in-memory index from sqlite
    cursor = cache.conn.execute("SELECT exec_file_path, session_id FROM execution_index")
    index: dict[str, list[str]] = {}
    for file_path, session_id in cursor:
        if session_id not in index:
            index[session_id] = []
        index[session_id].append(file_path)

    _execution_index = index
    _execution_index_built = True

    total_cached = cache.get_exec_index_count()
    if files_to_parse:
        logger.info(f"Execution index ready: {total_cached} files, {len(index)} sessions")

    return index


def _load_execution_log_messages(session_id: str, workspace: str, timestamp: datetime) -> list[IndexedMessage]:
    """Load assistant responses from execution logs for a given session.

    Uses the cached execution index for fast lookup.
    Only reads the full file for executions belonging to this session.
    """
    index = _build_execution_index()
    exec_file_paths = index.get(session_id, [])

    if not exec_file_paths:
        return []

    messages = []
    msg_index_offset = 10000  # Offset to avoid collision with session message indices

    for file_path in exec_file_paths:
        exec_file = Path(file_path)
        if not exec_file.exists():
            continue

        try:
            with open(exec_file, encoding="utf-8") as f:
                data = json.load(f)

            actions = data.get("actions", [])
            for action in actions:
                if action.get("actionType") != "say":
                    continue

                output = action.get("output", {})
                if not isinstance(output, dict):
                    continue

                text = output.get("message", "")
                if not text or not text.strip():
                    continue

                if _should_skip_message("assistant", text):
                    continue

                # Replace fenced code blocks with compact placeholders
                text = _replace_code_blocks(text)

                if not text.strip():
                    continue

                # Use emittedAt for timestamp if available
                emitted_at = action.get("emittedAt")
                msg_ts = _parse_timestamp(emitted_at) or timestamp

                action_id = action.get("actionId", f"{session_id}-exec-{msg_index_offset}")

                messages.append(
                    IndexedMessage(
                        uuid=action_id,
                        session_id=session_id,
                        workspace=workspace,
                        timestamp=msg_ts,
                        role="assistant",
                        searchable_text=text,
                        message_index=msg_index_offset,
                        source=Source.IDE,
                    )
                )
                msg_index_offset += 1

        except (json.JSONDecodeError, OSError):
            continue

    return messages


def _load_workspace_session_messages(session: SessionInfo) -> list[IndexedMessage]:
    """Load messages from a workspace-sessions JSON file."""
    # Find the session file
    session_file = None
    for ws_dir in _get_workspace_sessions_dirs():
        for workspace_dir in ws_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            candidate = workspace_dir / f"{session.session_id}.json"
            if candidate.exists():
                session_file = candidate
                break
        if session_file:
            break

    if not session_file:
        return []

    messages = []
    try:
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)

        # Get workspace from file metadata (more accurate than directory name)
        workspace = data.get("workspaceDirectory", data.get("workspacePath", session.workspace))
        history = data.get("history", [])

        for idx, item in enumerate(history):
            if not isinstance(item, dict):
                continue

            # Messages are nested inside a 'message' key
            msg = item.get("message", {})
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            text = _extract_text_from_content(content)

            if _should_skip_message(role, text):
                continue

            if not text.strip():
                continue

            # Replace fenced code blocks with compact placeholders
            text = _replace_code_blocks(text)

            # No per-message timestamps in this format; use session dates
            timestamp = session.modified or session.created or datetime.now()

            messages.append(
                IndexedMessage(
                    uuid=msg.get("id", f"{session.session_id}-{idx}"),
                    session_id=session.session_id,
                    workspace=workspace,
                    timestamp=timestamp,
                    role=role,
                    searchable_text=text,
                    message_index=idx,
                    source=Source.IDE,
                )
            )

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Error loading workspace session {session.session_id}: {e}")

    # Also load assistant responses from execution logs
    try:
        exec_messages = _load_execution_log_messages(
            session_id=session.session_id,
            workspace=session.workspace,
            timestamp=session.modified or session.created or datetime.now(),
        )
        messages.extend(exec_messages)
    except Exception as e:
        logger.warning(f"Error loading execution logs for session {session.session_id}: {e}")

    # Sort by message_index to maintain order (user messages have low indices, exec messages have high)
    messages.sort(key=lambda m: m.message_index)

    return messages


# --- Public API ---


def list_ide_sessions() -> list[SessionInfo]:
    """List all IDE sessions from both legacy and current formats."""
    sessions = []
    sessions.extend(_list_legacy_sessions())
    sessions.extend(_list_workspace_sessions())
    return sessions


def load_ide_session_messages(session: SessionInfo) -> list[IndexedMessage]:
    """Load messages for a session, trying both formats."""
    # Try workspace-sessions format first (check if file exists there)
    for ws_dir in _get_workspace_sessions_dirs():
        for workspace_dir in ws_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            candidate = workspace_dir / f"{session.session_id}.json"
            if candidate.exists():
                return _load_workspace_session_messages(session)

    # Fall back to legacy .chat format
    return _load_legacy_session_messages(session)
