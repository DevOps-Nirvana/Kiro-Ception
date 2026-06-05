"""Peer-to-peer search federation.

When peers are configured, search queries are fanned out to remote
kiro-ception instances in parallel. Results are merged with local
results and deduplicated by score.

Encryption is optional — when a shared secret is configured, all
payloads are encrypted with AES-256-GCM (key derived via Argon2id).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import get_config
from .peer_crypto import decrypt, derive_key, encrypt

logger = logging.getLogger(__name__)

# Content-Type for encrypted payloads
_ENCRYPTED_CONTENT_TYPE = "application/x-kiro-encrypted"
_JSON_CONTENT_TYPE = "application/json"

# Reusable thread pool for parallel peer requests
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="peer")
    return _executor


def get_peer_config() -> dict:
    """Get peer configuration from the main config.

    Returns:
        Dict with keys: enabled, nodes, secret, timeout_seconds, key (derived or None)
    """
    config = get_config()
    peer_cfg = {
        "enabled": getattr(config, "_peers_enabled", False),
        "nodes": getattr(config, "_peers_nodes", []),
        "secret": getattr(config, "_peers_secret", ""),
        "timeout_seconds": getattr(config, "_peers_timeout", 5),
        "key": None,
    }

    # Use the peers config if available
    if hasattr(config, "peers"):
        peer_cfg["enabled"] = config.peers.enabled
        peer_cfg["nodes"] = config.peers.nodes
        peer_cfg["secret"] = config.peers.secret
        peer_cfg["timeout_seconds"] = config.peers.timeout_seconds

    # Derive encryption key if secret is set
    if peer_cfg["secret"]:
        peer_cfg["key"] = derive_key(peer_cfg["secret"])

    return peer_cfg


def _send_to_peer(
    node: str,
    path: str,
    payload: dict,
    key: bytes | None,
    timeout: float,
) -> dict | None:
    """Send a request to a single peer, handling encryption if configured.

    Args:
        node: Peer address (host:port)
        path: HTTP path (e.g., "/search")
        payload: JSON-serializable request body
        key: Encryption key (None = plaintext)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response dict, or None on failure.
    """
    url = f"http://{node}{path}"

    try:
        body_bytes = json.dumps(payload).encode("utf-8")

        if key:
            # Encrypt the payload
            encrypted = encrypt(body_bytes, key)
            resp = requests.post(
                url,
                data=encrypted,
                headers={"Content-Type": _ENCRYPTED_CONTENT_TYPE},
                timeout=timeout,
            )
        else:
            # Plaintext
            resp = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )

        resp.raise_for_status()

        # Decrypt response if encrypted
        if resp.headers.get("Content-Type") == _ENCRYPTED_CONTENT_TYPE:
            if not key:
                logger.warning(f"Peer {node} sent encrypted response but no secret configured")
                return None
            decrypted = decrypt(resp.content, key)
            return json.loads(decrypted)
        else:
            return resp.json()

    except requests.Timeout:
        logger.debug(f"Peer {node} timed out after {timeout}s")
        return None
    except requests.ConnectionError:
        logger.debug(f"Peer {node} unreachable")
        return None
    except Exception as e:
        logger.debug(f"Peer {node} error: {e}")
        return None


def fan_out_search(request: dict) -> list[dict]:
    """Fan out a search request to all configured peers in parallel.

    Args:
        request: The search request dict (same format as local _search)

    Returns:
        List of result dicts from peers (may be empty if all fail/timeout)
    """
    peer_cfg = get_peer_config()

    if not peer_cfg["enabled"] or not peer_cfg["nodes"]:
        return []

    key = peer_cfg["key"]
    timeout = peer_cfg["timeout_seconds"]
    executor = _get_executor()

    futures = {}
    for node in peer_cfg["nodes"]:
        future = executor.submit(_send_to_peer, node, "/search", request, key, timeout)
        futures[future] = node

    results = []
    for future in as_completed(futures, timeout=timeout + 1):
        node = futures[future]
        try:
            result = future.result()
            if result and result.get("results"):
                results.append(result)
        except Exception as e:
            logger.debug(f"Peer {node} future error: {e}")

    return results


def merge_peer_results(local_response: dict, peer_responses: list[dict]) -> dict:
    """Merge local search results with peer results.

    Combines all results, deduplicates by UUID (keeps highest score),
    and sorts by score descending. Pagination is re-applied.

    Args:
        local_response: The local search response dict.
        peer_responses: List of response dicts from peers.

    Returns:
        Merged response dict with combined results.
    """
    if not peer_responses:
        return local_response

    # Collect all results
    all_results = list(local_response.get("results", []))

    for peer_resp in peer_responses:
        for result in peer_resp.get("results", []):
            all_results.append(result)

    # Deduplicate by matched_message UUID (keep highest score)
    seen_uuids: dict[str, dict] = {}
    for result in all_results:
        uuid = result.get("matched_message", {}).get("uuid", "")
        if not uuid:
            continue
        existing = seen_uuids.get(uuid)
        if existing is None or result.get("score", 0) > existing.get("score", 0):
            seen_uuids[uuid] = result

    # Sort by score descending
    merged = sorted(seen_uuids.values(), key=lambda r: r.get("score", 0), reverse=True)

    # Apply pagination from original request
    total = len(merged)
    offset = local_response.get("offset", 0)
    # Use original max_results hint from local response count
    max_results = len(local_response.get("results", [])) or 10
    paginated = merged[offset:offset + max_results]

    has_more = offset + len(paginated) < total

    return {
        "results": paginated,
        "query": local_response.get("query", ""),
        "total_matches": total,
        "offset": offset,
        "has_more": has_more,
        "hint": _generate_merged_hint(total, offset, len(paginated), has_more, len(peer_responses)),
    }


def _generate_merged_hint(
    total: int, offset: int, count: int, has_more: bool, peer_count: int
) -> str:
    """Generate a hint string for merged results."""
    source_note = f" (merged with {peer_count} peer{'s' if peer_count > 1 else ''})"
    if total == 0:
        return "No matches found locally or from peers."
    start, end = offset + 1, offset + count
    if has_more:
        return f"Showing {start}-{end} of {total}{source_note}. Use offset for more."
    if start == 1:
        return f"Showing all {total} matches{source_note}."
    return f"Showing {start}-{end} of {total} (final page){source_note}."


# --- Server-side: handling incoming encrypted requests ---


def decrypt_request_body(body: bytes, content_type: str) -> dict | None:
    """Decrypt an incoming request body if encrypted.

    Args:
        body: Raw request body bytes.
        content_type: The Content-Type header value.

    Returns:
        Parsed JSON dict, or None if decryption fails.

    Raises:
        PermissionError: If encryption is required but not provided, or vice versa.
    """
    peer_cfg = get_peer_config()
    key = peer_cfg["key"]

    if content_type == _ENCRYPTED_CONTENT_TYPE:
        if not key:
            raise PermissionError("Received encrypted request but no peer secret configured")
        try:
            plaintext = decrypt(body, key)
            return json.loads(plaintext)
        except Exception as e:
            raise PermissionError(f"Decryption failed (wrong secret?): {e}")
    else:
        # Plaintext request
        if key:
            # We have a secret configured but received plaintext — reject
            raise PermissionError("Peer secret configured but received unencrypted request")
        return json.loads(body)


def encrypt_response_body(data: dict) -> tuple[bytes, str]:
    """Encrypt a response body if a peer secret is configured.

    Args:
        data: JSON-serializable response dict.

    Returns:
        Tuple of (body_bytes, content_type)
    """
    peer_cfg = get_peer_config()
    key = peer_cfg["key"]

    body_bytes = json.dumps(data).encode("utf-8")

    if key:
        encrypted = encrypt(body_bytes, key)
        return encrypted, _ENCRYPTED_CONTENT_TYPE
    else:
        return body_bytes, _JSON_CONTENT_TYPE
