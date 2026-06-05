"""Unified loader combining CLI and IDE sources."""

from .cli_loader import list_cli_sessions, load_cli_session_messages
from .config import get_config
from .ide_loader import list_ide_sessions, load_ide_session_messages
from .models import IndexedMessage, SessionInfo, Source


def list_all_sessions() -> list[SessionInfo]:
    """List all sessions from enabled sources, sorted by modified time."""
    config = get_config()
    sessions = []

    if config.cli.enabled:
        sessions.extend(list_cli_sessions())

    if config.ide.enabled:
        sessions.extend(list_ide_sessions())

    return sorted(sessions, key=lambda s: s.timestamp_fallback, reverse=True)


def load_session_messages(session: SessionInfo) -> list[IndexedMessage]:
    """Load messages for a session based on its source."""
    if session.source == Source.CLI:
        return load_cli_session_messages(session)
    return load_ide_session_messages(session)


def load_messages_for_sessions(
    sessions: list[SessionInfo],
) -> list[IndexedMessage]:
    """Load messages for multiple sessions."""
    messages = []
    for session in sessions:
        messages.extend(load_session_messages(session))
    return messages
