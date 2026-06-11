# Feature: conversation-logger, Property 13: Default Search Returns Conversation Tier Only
"""Property 13: Default Search Returns Conversation Tier Only

**Validates: Requirements 3.4, 9.2**

*For any* search query executed without `include_tool_context=true`, all matched results
SHALL have `content_tier = "conversation"`. No `tool_context` messages SHALL appear as
direct matches.

Tests the fts_search() method directly against a SQLite database populated with
mixed-tier messages. Generates random message sets with both conversation and
tool_context tiers, runs searches without include_tool_context, and verifies all
results have content_tier="conversation".
"""

import time

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kiro_ception.cache import EmbeddingCache


# --- Fixtures ---


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Create an EmbeddingCache with a temp database for testing."""
    monkeypatch.setattr(
        "kiro_ception.cache._get_cache_db_path",
        lambda fp: tmp_path / f"cache_{fp}.db",
    )
    c = EmbeddingCache("test-fingerprint")
    _ = c.conn  # Force table creation and migrations
    yield c
    c.close()


# --- Strategies ---

# Words that can appear in searchable text — using simple alpha words
# that FTS5 can tokenize and match reliably
search_word_st = st.sampled_from([
    "deploy", "server", "database", "config", "module", "function",
    "endpoint", "query", "refactor", "implement", "migration", "handler",
    "parser", "logger", "cache", "index", "update", "create",
    "delete", "fetch", "render", "compile", "build", "test",
])

# Generate a sentence from search words (ensures FTS can match)
sentence_st = st.lists(search_word_st, min_size=2, max_size=8).map(lambda ws: " ".join(ws))

# Content tiers
content_tier_st = st.sampled_from(["conversation", "tool_context"])

# Tool names for tool_context messages
tool_name_st = st.sampled_from([
    "readFile", "writeFile", "fs_write", "grep_search",
    "execute_pwsh", "invoke_sub_agent", "file_search",
])


@st.composite
def mixed_tier_message_set(draw):
    """Generate a set of messages with a mix of conversation and tool_context tiers.

    Ensures at least one message of each tier exists, and that at least one
    conversation message and one tool_context message share a common search word.
    """
    # Pick a shared word that will appear in both tiers
    shared_word = draw(search_word_st)

    # Generate conversation messages (at least 1)
    n_conv = draw(st.integers(min_value=1, max_value=5))
    conv_messages = []
    for i in range(n_conv):
        extra_words = draw(st.lists(search_word_st, min_size=1, max_size=4))
        text = shared_word + " " + " ".join(extra_words)
        role = draw(st.sampled_from(["user", "assistant"]))
        conv_messages.append({
            "uuid": f"conv-{i}",
            "text": text,
            "role": role,
            "content_tier": "conversation",
            "tool_name": None,
        })

    # Generate tool_context messages (at least 1)
    n_tool = draw(st.integers(min_value=1, max_value=5))
    tool_messages = []
    for i in range(n_tool):
        tool_name = draw(tool_name_st)
        extra_words = draw(st.lists(search_word_st, min_size=1, max_size=4))
        text = f"[{tool_name}] {shared_word} " + " ".join(extra_words) + " → completed"
        tool_messages.append({
            "uuid": f"tool-{i}",
            "text": text,
            "role": "assistant",
            "content_tier": "tool_context",
            "tool_name": tool_name,
        })

    return {
        "shared_word": shared_word,
        "conversation_messages": conv_messages,
        "tool_messages": tool_messages,
    }


# --- Helper ---


def populate_cache(cache, message_set):
    """Insert a mixed-tier message set into the cache."""
    now = time.time()
    all_msgs = []
    idx = 0

    for msg in message_set["conversation_messages"]:
        all_msgs.append((
            msg["uuid"], "sess-1", "/test/workspace", now,
            msg["role"], msg["text"], idx, "ide",
            f"hash-{msg['uuid']}", msg["content_tier"], msg["tool_name"],
        ))
        idx += 1

    for msg in message_set["tool_messages"]:
        all_msgs.append((
            msg["uuid"], "sess-1", "/test/workspace", now,
            msg["role"], msg["text"], idx, "ide",
            f"hash-{msg['uuid']}", msg["content_tier"], msg["tool_name"],
        ))
        idx += 1

    cache.put_messages_batch(all_msgs)


# --- Property Tests ---


@given(message_set=mixed_tier_message_set())
@settings(max_examples=100)
def test_default_search_returns_only_conversation_tier(message_set, tmp_path_factory):
    """Without include_tool_context, fts_search returns only conversation-tier results.

    **Validates: Requirements 3.4, 9.2**

    For any search query executed without include_tool_context=true, all matched
    results SHALL have content_tier="conversation". No tool_context messages SHALL
    appear as direct matches.
    """
    # Set up a fresh cache for each test iteration
    tmp_path = tmp_path_factory.mktemp("cache")

    import kiro_ception.cache as cache_mod
    original_fn = cache_mod._get_cache_db_path

    cache_mod._get_cache_db_path = lambda fp: tmp_path / f"cache_{fp}.db"
    try:
        cache = EmbeddingCache("test-fp")
        _ = cache.conn  # Initialize tables

        populate_cache(cache, message_set)

        # Search using the shared word (which exists in both tiers)
        query = message_set["shared_word"]
        results = cache.fts_search(query, include_tool_context=False)

        # ALL results must be conversation tier
        for result in results:
            assert result["content_tier"] == "conversation", (
                f"Default search returned a tool_context result!\n"
                f"Query: '{query}'\n"
                f"Result content_tier: {result['content_tier']}\n"
                f"Result content: {result.get('content', '')[:100]}"
            )

        cache.close()
    finally:
        cache_mod._get_cache_db_path = original_fn


@given(message_set=mixed_tier_message_set())
@settings(max_examples=100)
def test_default_search_excludes_tool_context_even_when_matching(
    message_set, tmp_path_factory
):
    """Tool_context messages matching the query must NOT appear in default search results.

    **Validates: Requirements 3.4, 9.2**

    Verifies the negative constraint: even when tool_context messages contain text
    that matches the search query, they are excluded from default (non-tool-context)
    search results.
    """
    tmp_path = tmp_path_factory.mktemp("cache")

    import kiro_ception.cache as cache_mod
    original_fn = cache_mod._get_cache_db_path

    cache_mod._get_cache_db_path = lambda fp: tmp_path / f"cache_{fp}.db"
    try:
        cache = EmbeddingCache("test-fp")
        _ = cache.conn

        populate_cache(cache, message_set)

        query = message_set["shared_word"]

        # First confirm tool_context messages DO match when explicitly included
        results_with_tool = cache.fts_search(query, include_tool_context=True)
        tool_results = [r for r in results_with_tool if r["content_tier"] == "tool_context"]

        # Now verify default search excludes them
        results_default = cache.fts_search(query, include_tool_context=False)
        default_uuids = {r["uuid"] for r in results_default}

        # None of the tool_context UUIDs should appear in default results
        for tool_result in tool_results:
            assert tool_result["uuid"] not in default_uuids, (
                f"Tool_context message appeared in default search results!\n"
                f"UUID: {tool_result['uuid']}\n"
                f"Content: {tool_result.get('content', '')[:100]}\n"
                f"Query: '{query}'"
            )

        cache.close()
    finally:
        cache_mod._get_cache_db_path = original_fn
