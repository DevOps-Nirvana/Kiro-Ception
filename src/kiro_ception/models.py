"""Pydantic data models for Kiro Ception."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Source(str, Enum):
    """Conversation source."""

    CLI = "cli"
    IDE = "ide"


class IndexedMessage(BaseModel):
    """A message indexed for search."""

    uuid: str
    session_id: str
    workspace: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    searchable_text: str
    message_index: int = 0  # Position in session for context retrieval
    source: Source = Source.CLI


class SessionInfo(BaseModel):
    """Metadata about a conversation session."""

    session_id: str
    workspace: str
    message_count: int = 0
    created: datetime | None = None
    modified: datetime | None = None
    source: Source = Source.CLI

    @property
    def timestamp_fallback(self) -> datetime:
        """Get a timestamp for sorting, with fallback to epoch."""
        return self.modified or self.created or datetime.min
