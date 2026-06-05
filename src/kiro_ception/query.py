"""Query engine for conversation search."""

from collections import defaultdict
from datetime import datetime

from .config import get_config
from .indexer import get_index
from .models import (
    ContextMessage,
    IndexedMessage,
    MatchedMessage,
    SearchResponse,
    SearchResult,
    Source,
)


def parse_date_filter(value: str | None) -> datetime | None:
    """Parse ISO 8601 date string to datetime."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError as e:
        raise ValueError(
            f"Invalid date format: {value}. Use ISO 8601 (e.g., 2025-01-15)"
        ) from e


def _truncate(content: str, max_length: int = 2000) -> str:
    """Truncate content to max length."""
    return content if len(content) <= max_length else content[: max_length - 3] + "..."


def _deduplicate_results(
    results: list[tuple[IndexedMessage, float]], context_size: int
) -> list[tuple[IndexedMessage, float]]:
    """Deduplicate by merging overlapping context windows."""
    if not results:
        return []

    by_session: dict[str, list[tuple[IndexedMessage, float]]] = defaultdict(list)
    for msg, score in results:
        by_session[msg.session_id].append((msg, score))

    dedup_distance = 2 * context_size
    deduplicated = []

    for session_results in by_session.values():
        session_results.sort(key=lambda x: x[0].message_index)
        kept: list[tuple[IndexedMessage, float]] = []

        for msg, score in session_results:
            if not kept:
                kept.append((msg, score))
                continue

            last_msg, last_score = kept[-1]
            if msg.message_index - last_msg.message_index <= dedup_distance:
                if score > last_score:
                    kept[-1] = (msg, score)
            else:
                kept.append((msg, score))

        deduplicated.extend(kept)

    deduplicated.sort(key=lambda x: x[1], reverse=True)
    return deduplicated


def _generate_hint(
    total: int, offset: int, count: int, max_results: int, has_more: bool
) -> str:
    """Generate pagination hint."""
    if total == 0:
        return "No matches found. Try different search terms."

    start, end = offset + 1, offset + count
    if has_more:
        return f"Showing {start}-{end} of {total}. Use offset: {offset + max_results} for more."
    if start == 1:
        return f"Showing all {total} matches."
    return f"Showing {start}-{end} of {total} (final page)."


def search_conversations(
    query: str,
    workspace: str | None = None,
    source: Source | None = None,
    after: str | None = None,
    before: str | None = None,
    context_size: int | None = None,
    threshold: float | None = None,
    max_results: int | None = None,
    offset: int = 0,
) -> SearchResponse:
    """Search conversations with filters and pagination."""
    config = get_config()
    context_size = context_size or config.search.default_context_window
    threshold = threshold or config.search.default_threshold
    max_results = max_results or config.search.default_max_results

    index = get_index()
    after_dt = parse_date_filter(after)
    before_dt = parse_date_filter(before)

    raw_results = index.search(
        query=query,
        workspace=workspace,
        source=source,
        threshold=threshold,
        max_results=(offset + max_results) * 3,
        after=after_dt,
        before=before_dt,
    )

    if not raw_results:
        return SearchResponse(
            results=[],
            query=query,
            total_matches=0,
            excluded_sessions=index.excluded_session_count,
            hint=_generate_hint(0, 0, 0, max_results, False),
        )

    deduplicated = _deduplicate_results(raw_results, context_size)
    total = len(deduplicated)
    paginated = deduplicated[offset : offset + max_results]

    results = []
    for msg, score in paginated:
        context_window = index.get_context_window(msg, context_size)
        context = [
            ContextMessage(
                role=m.role,
                content=_truncate(m.searchable_text),
                timestamp=m.timestamp,
                is_match=(m.uuid == msg.uuid),
            )
            for m in context_window
        ]

        results.append(
            SearchResult(
                matched_message=MatchedMessage(
                    role=msg.role,
                    content=_truncate(msg.searchable_text),
                    timestamp=msg.timestamp,
                    workspace=msg.workspace,
                    session_id=msg.session_id,
                    uuid=msg.uuid,
                    source=msg.source,
                ),
                score=round(score, 4),
                context=context,
            )
        )

    has_more = offset + len(results) < total
    return SearchResponse(
        results=results,
        query=query,
        total_matches=total,
        offset=offset,
        has_more=has_more,
        excluded_sessions=index.excluded_session_count,
        hint=_generate_hint(total, offset, len(results), max_results, has_more),
    )
