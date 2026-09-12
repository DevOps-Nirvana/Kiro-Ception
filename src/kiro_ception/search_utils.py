"""Pure utility functions for search post-processing.

These functions handle deduplication, pagination, context windowing,
date parsing, and response formatting — all without external I/O dependencies.
"""

import re
from collections import defaultdict
from datetime import datetime

# Word-token pattern for whole-token exclusion matching. Splits on any
# non-word character, so "[code:python]" yields tokens "code" and "python".
_TOKEN_PATTERN = re.compile(r"\w+")


def token_excluded(text: str, negative_terms: list[str] | None) -> bool:
    """Return True if any negative term appears as a whole token in text.

    Matching is case-insensitive and whole-token: excluding "cat" matches
    "the cat sat" but NOT "category". Text is tokenized with \\w+, so
    punctuation is a delimiter and placeholders like "[code:python]" become
    the tokens "code" and "python".

    Empty or None negative_terms is a no-op (returns False). Empty/None text
    returns False. Negative terms that contain no word characters (e.g. "!!")
    can never match and are ignored.
    """
    if not negative_terms or not text:
        return False

    tokens = set(_TOKEN_PATTERN.findall(text.lower()))
    if not tokens:
        return False

    for term in negative_terms:
        if not term:
            continue
        # A negative term may itself be multi-token (e.g. "unit test"); require
        # every word-token of the term to be present for it to exclude.
        term_tokens = _TOKEN_PATTERN.findall(term.lower())
        if term_tokens and all(t in tokens for t in term_tokens):
            return True
    return False


def apply_set_operators(
    scored_results: list[dict],
    require_uuids: set[str] | None = None,
    exclude_uuids: set[str] | None = None,
    promote_uuids: set[str] | None = None,
    demote_uuids: set[str] | None = None,
) -> list[dict]:
    """Apply the require/exclude/promote/demote operator model to results.

    All four operators act on set membership by uuid. The sets are computed
    upstream by retrieving each operator's term(s) as their own search (set C)
    and collecting the uuids of the messages that match. This function is the
    pure set-logic core, independent of any retrieval or I/O.

    Operators form a 2x2:
      - Hard membership (change WHICH results exist):
          require  -> keep only results whose uuid is in require_uuids  (∩)
          exclude  -> drop results whose uuid is in exclude_uuids       (−)
      - Soft ranking (change ORDER only, never drop):
          promote  -> results in promote_uuids sort above the rest
          demote   -> results in demote_uuids sort below the rest

    Precedence (per design): require, then exclude, then promote/demote.
    Hard membership changes run before soft reordering — there is no point
    ranking results that are about to be removed.

    Input is assumed already sorted by score descending. Relative order within
    each promote/neutral/demote partition is preserved (stable partition), so
    relevance ordering is retained inside each band.

    Args:
        scored_results: results (dicts with at least "uuid"), score-desc order.
        require_uuids: if not None/empty, keep only results in this set.
        exclude_uuids: drop results in this set.
        promote_uuids: results in this set rank above non-members.
        demote_uuids: results in this set rank below non-members.

    Returns:
        Filtered and reordered results.
    """
    results = scored_results

    # --- Hard membership: require (intersection) ---
    if require_uuids:
        results = [r for r in results if r["uuid"] in require_uuids]

    # --- Hard membership: exclude (difference) ---
    if exclude_uuids:
        results = [r for r in results if r["uuid"] not in exclude_uuids]

    # --- Soft ranking: promote/demote (stable three-way partition) ---
    if promote_uuids or demote_uuids:
        promote_uuids = promote_uuids or set()
        demote_uuids = demote_uuids or set()
        promoted, neutral, demoted = [], [], []
        for r in results:
            uid = r["uuid"]
            # promote wins ties with demote if a uuid is somehow in both:
            # promoting is the more specific positive intent.
            if uid in promote_uuids:
                promoted.append(r)
            elif uid in demote_uuids:
                demoted.append(r)
            else:
                neutral.append(r)
        results = promoted + neutral + demoted

    return results


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


def deduplicate_results(
    results: list[dict], context_size: int, preserve_order: bool = False
) -> list[dict]:
    """Deduplicate results with overlapping context windows.

    Within each session, if two matches are within 2*context_size message
    indices of each other, keep only the higher-scoring one.

    By default the deduplicated results are returned sorted by score descending.
    When preserve_order=True, the input order is preserved instead — used when a
    promote/demote operator has already imposed a deliberate ordering that must
    not be undone by a score re-sort.
    """
    if not results:
        return []

    # Remember the incoming order so we can restore it when preserving order.
    order_index = {id(r): i for i, r in enumerate(results)}

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

    if preserve_order:
        # Restore the order the results arrived in (carries promote/demote bands).
        deduplicated.sort(key=lambda x: order_index.get(id(x), 0))
    else:
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
            May include content_tier field (defaults to "conversation").
        match_uuid: The UUID of the matched message.
        context_size: Number of messages before and after the match to include.
            Both conversation and tool_context messages count toward this limit.

    Returns:
        List of context message dicts with role, content, content_tier,
        timestamp, is_match. Messages are interleaved by message_index order.

    Overflow logic:
        When the window exceeds 20 messages, drop oldest tool_context messages
        first while retaining all conversation messages.
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

    window_messages = session_messages[start:end]

    # Apply overflow logic: when window exceeds 20 messages,
    # drop oldest tool_context first, retain all conversation messages
    if len(window_messages) > 20:
        window_messages = _apply_overflow_logic(window_messages, match_uuid)

    context = []
    for msg in window_messages:
        text = msg["searchable_text"]
        content_tier = msg.get("content_tier", "conversation")
        context.append({
            "role": msg["role"],
            "content": truncate_content(text),
            "content_tier": content_tier,
            "timestamp": datetime.fromtimestamp(msg["timestamp"]).isoformat(),
            "is_match": msg["uuid"] == match_uuid,
        })
    return context


def _apply_overflow_logic(window_messages: list[dict], match_uuid: str) -> list[dict]:
    """Apply overflow logic to trim window to 20 messages.

    Drops oldest tool_context messages first while retaining all
    conversation messages. The matched message is never dropped.

    Args:
        window_messages: Messages in the window, ordered by message_index.
        match_uuid: UUID of the matched message (never dropped).

    Returns:
        Trimmed list of messages, still ordered by message_index.
    """
    max_window = 20

    if len(window_messages) <= max_window:
        return window_messages

    # Separate messages into:
    # - protected: conversation messages + the matched message (never dropped)
    # - droppable: tool_context messages that are NOT the match
    protected_msgs = []
    droppable_tool_msgs = []

    for msg in window_messages:
        tier = msg.get("content_tier", "conversation")
        if tier == "tool_context" and msg["uuid"] != match_uuid:
            droppable_tool_msgs.append(msg)
        else:
            protected_msgs.append(msg)

    # Calculate how many tool_context messages we need to drop
    num_to_drop = len(window_messages) - max_window
    # Drop oldest tool_context first (they're already in message_index order)
    if num_to_drop < len(droppable_tool_msgs):
        kept_tool_msgs = droppable_tool_msgs[num_to_drop:]
    else:
        # Not enough droppable tool_context — keep all protected messages
        # (window may still exceed 20 if there are >20 protected messages)
        kept_tool_msgs = []

    result = protected_msgs + kept_tool_msgs
    # Re-sort by original message_index order
    result.sort(key=lambda m: m.get("message_index", 0))

    return result


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
