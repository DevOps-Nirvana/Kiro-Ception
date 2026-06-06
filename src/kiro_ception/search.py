"""Search engine: in-memory numpy matrix and search routing.

Contains the SearchIndex class (vectorized cosine similarity search),
leader search logic, and peer federation integration.
"""

import time

import numpy as np

from .background_indexer import get_background_indexer
from .coordination import get_instance_manager
from .embeddings import get_embedding_backend
from .models import Source
from .peers import fan_out_search, merge_peer_results
from .search_utils import (
    deduplicate_results,
    format_search_response,
    parse_date,
)

# --- In-Memory Search Index (leader only) ---

_MIN_REFRESH_INTERVAL = 60  # seconds


class SearchIndex:
    """In-memory numpy matrix for fast vectorized cosine similarity search.

    Loaded from the SQLite cache. Refreshes at most once per minute to pick
    up new embeddings from the background indexer.
    """

    def __init__(self):
        self._embeddings: np.ndarray | None = None  # shape: (N, dims)
        self._metadata: list[dict] = []  # parallel to embeddings rows
        self._last_refresh: float = 0
        self._count: int = 0
        self._loading: bool = False  # True when embeddings exist but metadata not yet populated
        self._embedding_count_hint: int = 0  # For informative "loading" messages
        self._ever_loaded: bool = False  # Has the index ever successfully loaded data?

    @property
    def message_count(self) -> int:
        return self._count

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def embedding_count_hint(self) -> int:
        return self._embedding_count_hint

    def refresh_if_needed(self):
        """Reload from sqlite if needed.

        The 60-second throttle only applies after a successful first load.
        If the index has never loaded data, allow refresh on every call
        (so we pick up data as soon as the indexer populates the tables).
        """
        now = time.time()
        if self._ever_loaded and now - self._last_refresh < _MIN_REFRESH_INTERVAL:
            return
        self._refresh()

    def _refresh(self):
        """Load all messages + embeddings from sqlite into numpy arrays."""
        indexer = get_background_indexer()
        cache = indexer.cache
        if cache is None:
            self._loading = True
            return

        # Load all messages with their text_hash
        cursor = cache.conn.execute(
            """SELECT uuid, session_id, workspace, timestamp, role, searchable_text,
                      message_index, source, text_hash
               FROM messages"""
        )
        rows = cursor.fetchall()

        if not rows:
            # Check if embeddings exist — if so, the indexer just hasn't
            # stored message metadata yet (still loading)
            emb_count = cache.embedding_count
            if emb_count > 0:
                self._loading = True
                self._embedding_count_hint = emb_count
            else:
                self._loading = False
                self._embedding_count_hint = 0

            self._embeddings = None
            self._metadata = []
            self._count = 0
            # Don't set _last_refresh or _ever_loaded — allow retry on next call
            return

        # Collect text hashes and load embeddings in batch
        text_hashes = [row[8] for row in rows]
        embeddings_dict = cache.get_embeddings_batch(text_hashes)

        # Build parallel arrays (only for messages that have embeddings)
        metadata = []
        embedding_list = []

        for row in rows:
            uuid, session_id, ws, ts, role, text, msg_idx, src, text_hash = row
            emb = embeddings_dict.get(text_hash)
            if emb is None:
                continue
            # Skip embeddings with mismatched dimensions (corrupt or mixed-model data)
            if embedding_list and emb.shape[0] != embedding_list[0].shape[0]:
                continue
            metadata.append({
                "uuid": uuid,
                "session_id": session_id,
                "workspace": ws,
                "timestamp": ts,
                "role": role,
                "content": text[:2000] + "..." if len(text) > 2000 else text,
                "message_index": msg_idx,
                "source": src,
            })
            embedding_list.append(emb)

        if embedding_list:
            try:
                self._embeddings = np.vstack(embedding_list)
            except ValueError:
                # Fallback: dimension mismatch despite filtering — clear and retry next cycle
                self._embeddings = None
                self._metadata = []
                self._count = 0
                return
            self._metadata = metadata
            self._count = len(metadata)
            self._loading = False
            self._ever_loaded = True
        else:
            self._embeddings = None
            self._metadata = []
            self._count = 0

        self._last_refresh = time.time()

    def search(
        self,
        query_embedding: np.ndarray,
        workspace: str | None = None,
        source: str | None = None,
        after_ts: float | None = None,
        before_ts: float | None = None,
        threshold: float = 0.2,
        max_results: int = 100,
    ) -> list[dict]:
        """Fast vectorized search against the in-memory matrix."""
        self.refresh_if_needed()

        if self._embeddings is None or self._count == 0:
            return []

        # Compute all similarities at once
        similarities = np.dot(self._embeddings, query_embedding)

        # Build candidate mask for filtering
        mask = similarities >= threshold

        # Apply filters
        if workspace or source or after_ts or before_ts:
            for i in range(self._count):
                if not mask[i]:
                    continue
                meta = self._metadata[i]
                if workspace and not meta["workspace"].startswith(workspace):
                    mask[i] = False
                elif source and meta["source"] != source:
                    mask[i] = False
                elif after_ts and meta["timestamp"] < after_ts:
                    mask[i] = False
                elif before_ts and meta["timestamp"] >= before_ts:
                    mask[i] = False

        # Get top results
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = similarities[valid_indices]
        top_order = np.argsort(valid_scores)[::-1][:max_results]

        results = []
        for idx in top_order:
            global_idx = valid_indices[idx]
            score = float(valid_scores[idx])
            meta = self._metadata[global_idx]
            results.append({**meta, "score": score})

        return results


# Global search index singleton (leader only)
_search_index: SearchIndex | None = None


def get_search_index() -> SearchIndex:
    """Get or create the in-memory search index."""
    global _search_index
    if _search_index is None:
        _search_index = SearchIndex()
    _search_index.refresh_if_needed()
    return _search_index


def invalidate_search_index():
    """Force the search index to reload on next access.

    Called when the backend/cache switches (e.g., after a config change
    that changes the embedding model/dimensions).
    """
    global _search_index
    if _search_index is not None:
        _search_index._ever_loaded = False
        _search_index._last_refresh = 0
        _search_index._embeddings = None
        _search_index._metadata = []
        _search_index._count = 0


def handle_search_request(request: dict) -> dict:
    """Handle a search request (used by both leader directly and HTTP API)."""
    return search(
        query=request.get("query", ""),
        workspace=request.get("workspace"),
        source=Source(request["source"]) if request.get("source") else None,
        after=request.get("after"),
        before=request.get("before"),
        context_size=request.get("context_size", 3),
        threshold=request.get("threshold", 0.2),
        max_results=request.get("max_results", 10),
        offset=request.get("offset", 0),
    )


def search(
    query: str,
    workspace: str | None,
    source: Source | None,
    after: str | None,
    before: str | None,
    context_size: int,
    threshold: float,
    max_results: int,
    offset: int,
) -> dict:
    """Search implementation with leader/follower routing.

    Leader: searches directly against SQLite cache.
    Follower: forwards to leader via HTTP. On failure, attempts promotion.
    """
    manager = get_instance_manager()

    if not manager.is_leader:
        # Follower: forward to leader
        follower = manager.follower_instance
        if follower:
            try:
                return follower.search({
                    "query": query,
                    "workspace": workspace,
                    "source": source.value if source else None,
                    "after": after,
                    "before": before,
                    "context_size": context_size,
                    "threshold": threshold,
                    "max_results": max_results,
                    "offset": offset,
                })
            except Exception:
                # Leader unreachable — attempt promotion
                if manager.promote_to_leader(handle_search_request):
                    # We're now leader, start indexer
                    indexer = get_background_indexer()
                    indexer.start()
                    # Fall through to leader search below
                else:
                    return {
                        "results": [],
                        "query": query,
                        "total_matches": 0,
                        "error": "Leader unavailable and could not promote",
                        "hint": "Try again in a moment.",
                    }

    # Leader path: search directly
    local_response = leader_search(query, workspace, source, after, before,
                                   context_size, threshold, max_results, offset)

    # Fan out to peers if configured
    return _search_with_peers(
        local_response, query, workspace, source, after, before,
        context_size, threshold, max_results, offset,
    )


def leader_search(
    query: str,
    workspace: str | None,
    source: Source | None,
    after: str | None,
    before: str | None,
    context_size: int,
    threshold: float,
    max_results: int,
    offset: int,
) -> dict:
    """Direct search using in-memory numpy matrix (leader only)."""
    indexer = get_background_indexer()
    backend = indexer.backend

    if backend is None:
        try:
            backend = get_embedding_backend()
        except Exception as e:
            return {
                "results": [],
                "query": query,
                "total_matches": 0,
                "error": f"Backend not ready: {e}",
                "hint": "The embedding backend is still initializing. Try again in a moment.",
            }

    # Ensure in-memory index is loaded/refreshed
    search_index = get_search_index()
    if search_index.message_count == 0:
        if search_index.is_loading:
            emb_hint = search_index.embedding_count_hint
            return {
                "results": [],
                "query": query,
                "total_matches": 0,
                "hint": (
                    f"The search index is still loading ({emb_hint} embeddings available "
                    f"but metadata not yet populated). Try again in a few seconds."
                ),
            }
        return {
            "results": [],
            "query": query,
            "total_matches": 0,
            "hint": "Index is empty. Indexing may still be in progress — try again shortly.",
        }

    # Parse date filters
    after_dt = parse_date(after)
    before_dt = parse_date(before)

    # Embed the query
    try:
        query_embedding = backend.encode_query(query)
    except Exception as e:
        return {
            "results": [],
            "query": query,
            "total_matches": 0,
            "error": f"Failed to embed query: {e}",
            "hint": "The embedding backend may be unavailable.",
        }

    # Fast vectorized search
    scored_results = search_index.search(
        query_embedding=query_embedding,
        workspace=workspace,
        source=source.value if source else None,
        after_ts=after_dt.timestamp() if after_dt else None,
        before_ts=before_dt.timestamp() if before_dt else None,
        threshold=threshold,
        max_results=(offset + max_results) * 3,  # Over-fetch for dedup
    )

    # Deduplicate overlapping context windows
    scored_results = deduplicate_results(scored_results, context_size)

    # Build response with context windows
    cache = indexer.cache

    def get_session_messages(session_id: str) -> list[dict]:
        if cache is None:
            return []
        return cache.get_session_messages(session_id)

    return format_search_response(
        scored_results=scored_results,
        query=query,
        offset=offset,
        max_results=max_results,
        context_size=context_size,
        get_session_messages=get_session_messages,
    )


def _search_with_peers(
    local_response: dict,
    query: str,
    workspace: str | None,
    source: Source | None,
    after: str | None,
    before: str | None,
    context_size: int,
    threshold: float,
    max_results: int,
    offset: int,
) -> dict:
    """Merge local results with peer results if peering is enabled."""
    peer_request = {
        "query": query,
        "workspace": workspace,
        "source": source.value if source else None,
        "after": after,
        "before": before,
        "context_size": context_size,
        "threshold": threshold,
        "max_results": max_results,
        "offset": offset,
    }

    peer_responses = fan_out_search(peer_request)
    if peer_responses:
        return merge_peer_results(local_response, peer_responses)
    return local_response
