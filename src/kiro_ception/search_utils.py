"""Pure utility functions for search post-processing.

These functions handle deduplication, pagination, context windowing,
date parsing, and response formatting — all without external I/O dependencies.
"""

from collections import defaultdict
from datetime import datetime


def parse_date(value: str | None) -> datetime | None:
    """Parse ISO 8601 date string to datetime.

    Returns None for None input or unparseable strings.
    """
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError:
        return None


def deduplicate_results(results: list[dict], context_size: int) -> list[dict]:
    """Deduplicate results with overlapping context windows.

    Within each session, if two matches are within 2*context_size message
    indices of each other, keep only the higher-scoring one. Results are
    returned sorted by score descending.
    """
    if not results:
        return []

    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_session[r["session_id"]].append(r)

    dedup_distance = 2 * context_size
    deduplicated = []

    for session_results in by_session.values():
        session_results.sort(key=lambda x: x["message_index"])
        kept = []
        for r in session_results:
            if not kept:
                kept.append(r)
                continue
            if r["message_index"] - kept[-1]["message_index"] <= dedup_distance:
                if r["score"] > kept[-1]["score"]:
                    kept[-1] = r
            else:
                kept.append(r)
        deduplicated.extend(kept)

    deduplicated.sort(key=lambda x: x["score"], reverse=True)
    return deduplicated


def generate_hint(total: int, offset: int, count: int, max_results: int, has_more: bool) -> str:
    """Generate a human-readable pagination hint string."""
    if total == 0:
        return "No matches found. Try different search terms or lower the threshold."
    start, end = offset + 1, offset + count
    if has_more:
        return f"Showing {start}-{end} of {total}. Use offset: {offset + max_results} for more."
    if start == 1:
        return f"Showing all {total} matches."
    return f"Showing {start}-{end} of {total} (final page)."


def truncate_content(text: str, max_length: int = 2000) -> str:
    """Truncate text to max_length, appending '...' if truncated."""
    return text if len(text) <= max_length else text[:max_length] + "..."


def build_context_window(
    session_messages: list[dict],
    match_uuid: str,
    context_size: int,
) -> list[dict]:
    """Build the context window around a matched message.

    Args:
        session_messages: All messages in the session, ordered by message_index.
            Each dict must have keys: uuid, role, searchable_text, timestamp.
        match_uuid: The UUID of the matched message.
        context_size: Number of messages before and after the match to include.

    Returns:
        List of context message dicts with role, content, timestamp, is_match.
    """
    if not session_messages:
        return []

    match_pos = None
    for i, msg in enumerate(session_messages):
        if msg["uuid"] == match_uuid:
            match_pos = i
            break

    if match_pos is None:
        return []

    start = max(0, match_pos - context_size)
    end = min(len(session_messages), match_pos + context_size + 1)

    context = []
    for msg in session_messages[start:end]:
        text = msg["searchable_text"]
        context.append({
            "role": msg["role"],
            "content": truncate_content(text),
            "timestamp": datetime.fromtimestamp(msg["timestamp"]).isoformat(),
            "is_match": msg["uuid"] == match_uuid,
        })
    return context


def format_search_response(
    scored_results: list[dict],
    query: str,
    offset: int,
    max_results: int,
    context_size: int,
    get_session_messages: callable,
) -> dict:
    """Format raw scored results into the final search response dict.

    Args:
        scored_results: Already-deduplicated results sorted by score.
        query: The original search query string.
        offset: Pagination offset.
        max_results: Maximum results to return.
        context_size: Messages before/after each match for context.
        get_session_messages: Callable(session_id) -> list[dict] that
            returns messages for a session (for context window assembly).

    Returns:
        Complete response dict with results, pagination info, and hint.
    """
    total = len(scored_results)
    paginated = scored_results[offset:offset + max_results]

    results = []
    for match in paginated:
        session_msgs = get_session_messages(match["session_id"])
        context = build_context_window(session_msgs, match["uuid"], context_size)

        results.append({
            "matched_message": {
                "role": match["role"],
                "content": match["content"],
                "timestamp": datetime.fromtimestamp(match["timestamp"]).isoformat(),
                "workspace": match["workspace"],
                "session_id": match["session_id"],
                "uuid": match["uuid"],
                "source": match["source"],
            },
            "score": round(match["score"], 4),
            "context": context,
        })

    has_more = offset + len(results) < total
    hint = generate_hint(total, offset, len(results), max_results, has_more)

    return {
        "results": results,
        "query": query,
        "total_matches": total,
        "offset": offset,
        "has_more": has_more,
        "hint": hint,
    }
