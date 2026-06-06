"""Search engine: in-memory numpy matrix and search routing.

Contains the SearchIndex class (vectorized cosine similarity search),
leader search logic, and peer federation integration.
"""

import math
import time

import numpy as np

from .background_indexer import get_background_indexer
from .config import get_config
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
        self._oldest_timestamp: float = 0  # Oldest message timestamp (for recency boost calc)

    @property
    def message_count(self) -> int:
        return self._count

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def embedding_count_hint(self) -> int:
        return self._embedding_count_hint

    @property
    def oldest_timestamp(self) -> float:
        return self._oldest_timestamp

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
            # Track oldest timestamp for recency boost calculation
            self._oldest_timestamp = min(m["timestamp"] for m in metadata) if metadata else 0
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

    # Hybrid search: also run FTS5 full-text search and merge results
    cache = indexer.cache
    if cache is not None:
        fts_results = cache.fts_search(
            query=query,
            workspace=workspace,
            source=source.value if source else None,
            after_ts=after_dt.timestamp() if after_dt else None,
            before_ts=before_dt.timestamp() if before_dt else None,
            limit=(offset + max_results) * 3,
        )
        scored_results = _merge_hybrid_results(scored_results, fts_results)

    # Apply recency boost: recent messages get a slight score advantage
    scored_results = _apply_recency_boost(scored_results, search_index.oldest_timestamp)

    # Deduplicate overlapping context windows
    scored_results = deduplicate_results(scored_results, context_size)

    # Build response with context windows

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


# --- Hybrid search merging ---

# Weight allocation for hybrid scoring
_VECTOR_WEIGHT = 0.7  # Semantic similarity contributes 70%
_FTS_WEIGHT = 0.3     # Full-text BM25 contributes 30%


def _merge_hybrid_results(
    vector_results: list[dict],
    fts_results: list[dict],
) -> list[dict]:
    """Merge vector similarity results with FTS5 full-text results.

    Strategy:
    - Results found by both methods get a combined score (weighted sum)
    - Results found only by vector search keep their semantic score × vector_weight
    - Results found only by FTS get their FTS score × fts_weight
    - Final list is sorted by combined score descending

    This ensures exact keyword matches boost results that are also semantically
    relevant, while pure semantic matches still surface when keywords don't match.
    """
    if not fts_results:
        return vector_results

    # Build lookup by UUID for FTS results
    fts_by_uuid: dict[str, dict] = {}
    for r in fts_results:
        fts_by_uuid[r["uuid"]] = r

    # Build lookup by UUID for vector results
    vector_by_uuid: dict[str, dict] = {}
    for r in vector_results:
        vector_by_uuid[r["uuid"]] = r

    # Merge: start with all vector results, boost those also found by FTS
    merged: dict[str, dict] = {}

    for uuid, vr in vector_by_uuid.items():
        fts_match = fts_by_uuid.get(uuid)
        if fts_match:
            # Found by both — combine scores
            combined_score = (vr["score"] * _VECTOR_WEIGHT) + (fts_match["score"] * _FTS_WEIGHT)
            merged[uuid] = {**vr, "score": combined_score}
        else:
            # Vector-only result
            merged[uuid] = {**vr, "score": vr["score"] * _VECTOR_WEIGHT}

    # Add FTS-only results (not found by vector search)
    for uuid, fr in fts_by_uuid.items():
        if uuid not in merged:
            merged[uuid] = {**fr, "score": fr["score"] * _FTS_WEIGHT}

    # Sort by combined score descending
    results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
    return results


# --- Recency boost ---


def _compute_halflife(oldest_timestamp: float, floor: float) -> float:
    """Compute the halflife (in seconds) so that a message at oldest_timestamp
    gets exactly the floor multiplier.

    Derived from: floor = exp(-0.693 * max_age / halflife)
    Solving:      halflife = -0.693 * max_age / ln(floor)

    Returns 0 if recency boost is not applicable (no valid history span).
    """
    if oldest_timestamp <= 0 or floor >= 1.0 or floor <= 0:
        return 0  # No recency boost possible
    now = time.time()
    max_age = now - oldest_timestamp
    if max_age <= 86400:  # Less than 1 day of history — not meaningful
        return 0
    ln_floor = math.log(floor)
    if ln_floor == 0:
        return 0
    return -0.693 * max_age / ln_floor


def _apply_recency_boost(results: list[dict], oldest_timestamp: float) -> list[dict]:
    """Apply a recency multiplier to search results.

    The halflife is auto-calculated so that the oldest message in the index
    gets exactly `recency_floor` as its multiplier. Recent messages get ~1.0.
    The decay is smooth exponential between these two points.

    If recency_floor is 1.0 or oldest_timestamp is 0, no boost is applied.
    """
    if not results:
        return results

    config = get_config()
    floor = config.search.recency_floor

    # Disabled if floor is 1.0 (no decay) or no history span
    if floor >= 1.0 or oldest_timestamp <= 0:
        return results

    halflife = _compute_halflife(oldest_timestamp, floor)
    if halflife <= 0:
        return results

    now = time.time()

    for result in results:
        age_seconds = now - result["timestamp"]
        if age_seconds < 0:
            age_seconds = 0
        decay = math.exp(-0.693 * age_seconds / halflife)
        result["score"] = result["score"] * decay

    # Re-sort by boosted score
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
