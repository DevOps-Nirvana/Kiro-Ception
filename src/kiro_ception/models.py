"""Pydantic data models for Kiro Ception."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


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


class MatchedMessage(BaseModel):
    """A message that matched a search query."""

    role: str
    content: str
    timestamp: datetime
    workspace: str
    session_id: str
    uuid: str
    source: Source


class ContextMessage(BaseModel):
    """A message in the context window around a match."""

    role: str
    content: str
    timestamp: datetime
    is_match: bool = False


class SearchResult(BaseModel):
    """A search result with context."""

    matched_message: MatchedMessage
    score: float
    context: list[ContextMessage] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from search tools."""

    results: list[SearchResult]
    query: str
    total_matches: int
    offset: int = 0
    has_more: bool = False
    excluded_sessions: int = 0
    hint: str | None = None
