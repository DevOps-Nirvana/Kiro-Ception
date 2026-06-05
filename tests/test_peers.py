"""Tests for peer-to-peer federation and encryption.

Tests the crypto layer (Argon2id key derivation, AES-256-GCM encrypt/decrypt),
peer communication, result merging, and encrypted HTTP request handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from kiro_ception.config import Config, PeersConfig
from kiro_ception.peer_crypto import decrypt, derive_key, encrypt
from kiro_ception.peers import (
    decrypt_request_body,
    encrypt_response_body,
    fan_out_search,
    merge_peer_results,
)


# --- Key derivation ---


class TestDeriveKey:
    def test_produces_32_byte_key(self):
        key = derive_key("my-secret-passphrase")
        assert len(key) == 32

    def test_deterministic(self):
        """Same passphrase always produces same key."""
        key1 = derive_key("test-passphrase")
        key2 = derive_key("test-passphrase")
        assert key1 == key2

    def test_different_passphrases_produce_different_keys(self):
        key1 = derive_key("passphrase-one")
        key2 = derive_key("passphrase-two")
        assert key1 != key2

    def test_empty_passphrase(self):
        """Empty passphrase still produces a valid key (edge case)."""
        key = derive_key("")
        assert len(key) == 32


# --- Encrypt / Decrypt ---


class TestEncryptDecrypt:
    def test_round_trip(self):
        key = derive_key("test-secret")
        plaintext = b"Hello, this is sensitive search data!"

        ciphertext = encrypt(plaintext, key)
        decrypted = decrypt(ciphertext, key)

        assert decrypted == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        key = derive_key("test-secret")
        plaintext = b"sensitive data"

        ciphertext = encrypt(plaintext, key)
        assert ciphertext != plaintext

    def test_different_nonce_each_time(self):
        """Each encryption produces different ciphertext (random nonce)."""
        key = derive_key("test-secret")
        plaintext = b"same message"

        ct1 = encrypt(plaintext, key)
        ct2 = encrypt(plaintext, key)

        assert ct1 != ct2  # Different nonces

    def test_wrong_key_fails(self):
        key1 = derive_key("correct-passphrase")
        key2 = derive_key("wrong-passphrase")
        plaintext = b"secret data"

        ciphertext = encrypt(plaintext, key1)

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(ciphertext, key2)

    def test_tampered_data_fails(self):
        key = derive_key("test-secret")
        plaintext = b"important data"

        ciphertext = bytearray(encrypt(plaintext, key))
        # Flip a byte in the ciphertext
        ciphertext[20] ^= 0xFF

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(bytes(ciphertext), key)

    def test_too_short_data_raises(self):
        key = derive_key("test-secret")
        with pytest.raises(ValueError, match="too short"):
            decrypt(b"short", key)

    def test_large_payload(self):
        """Can handle large payloads (typical search responses)."""
        key = derive_key("test-secret")
        # Simulate a large search response (~100KB)
        plaintext = json.dumps({"results": [{"content": "x" * 1000}] * 100}).encode()

        ciphertext = encrypt(plaintext, key)
        decrypted = decrypt(ciphertext, key)

        assert decrypted == plaintext


# --- Request/Response encryption helpers ---


class TestDecryptRequestBody:
    def test_plaintext_without_secret(self):
        config = Config(peers=PeersConfig(enabled=True, secret=""))
        with patch("kiro_ception.peers.get_config", return_value=config):
            body = json.dumps({"query": "test"}).encode()
            result = decrypt_request_body(body, "application/json")
            assert result == {"query": "test"}

    def test_encrypted_with_secret(self):
        config = Config(peers=PeersConfig(enabled=True, secret="shared-secret"))
        key = derive_key("shared-secret")
        payload = json.dumps({"query": "encrypted search"}).encode()
        encrypted_body = encrypt(payload, key)

        with patch("kiro_ception.peers.get_config", return_value=config):
            result = decrypt_request_body(encrypted_body, "application/x-kiro-encrypted")
            assert result == {"query": "encrypted search"}

    def test_encrypted_request_without_secret_raises(self):
        config = Config(peers=PeersConfig(enabled=True, secret=""))
        with patch("kiro_ception.peers.get_config", return_value=config):
            with pytest.raises(PermissionError, match="no peer secret configured"):
                decrypt_request_body(b"encrypted-data", "application/x-kiro-encrypted")

    def test_plaintext_request_with_secret_raises(self):
        config = Config(peers=PeersConfig(enabled=True, secret="my-secret"))
        with patch("kiro_ception.peers.get_config", return_value=config):
            body = json.dumps({"query": "test"}).encode()
            with pytest.raises(PermissionError, match="received unencrypted"):
                decrypt_request_body(body, "application/json")


class TestEncryptResponseBody:
    def test_plaintext_without_secret(self):
        config = Config(peers=PeersConfig(enabled=True, secret=""))
        with patch("kiro_ception.peers.get_config", return_value=config):
            body, content_type = encrypt_response_body({"results": []})
            assert content_type == "application/json"
            assert json.loads(body) == {"results": []}

    def test_encrypted_with_secret(self):
        config = Config(peers=PeersConfig(enabled=True, secret="shared-secret"))
        key = derive_key("shared-secret")

        with patch("kiro_ception.peers.get_config", return_value=config):
            body, content_type = encrypt_response_body({"results": [{"score": 0.9}]})

        assert content_type == "application/x-kiro-encrypted"
        # Verify we can decrypt it
        decrypted = decrypt(body, key)
        assert json.loads(decrypted) == {"results": [{"score": 0.9}]}


# --- Result merging ---


class TestMergePeerResults:
    def test_no_peer_results_returns_local(self):
        local = {"results": [{"matched_message": {"uuid": "a"}, "score": 0.9}], "query": "test", "total_matches": 1}
        merged = merge_peer_results(local, [])
        assert merged == local

    def test_merges_and_deduplicates(self):
        local = {
            "results": [
                {"matched_message": {"uuid": "a"}, "score": 0.9},
                {"matched_message": {"uuid": "b"}, "score": 0.7},
                {"matched_message": {"uuid": "d"}, "score": 0.5},
            ],
            "query": "test",
            "total_matches": 3,
            "offset": 0,
        }
        peer = {
            "results": [
                {"matched_message": {"uuid": "a"}, "score": 0.85},  # Duplicate, lower score
                {"matched_message": {"uuid": "c"}, "score": 0.8},   # New result
            ],
        }
        merged = merge_peer_results(local, [peer])

        uuids = [r["matched_message"]["uuid"] for r in merged["results"]]
        # Should have a, c, b (sorted by score desc), d may be paginated out
        assert "a" in uuids
        assert "c" in uuids
        assert merged["total_matches"] == 4  # a, b, c, d (deduplicated)
        # Highest score for "a" should be 0.9 (not 0.85)
        a_result = next(r for r in merged["results"] if r["matched_message"]["uuid"] == "a")
        assert a_result["score"] == 0.9

    def test_sorted_by_score(self):
        local = {
            "results": [{"matched_message": {"uuid": "a"}, "score": 0.5}],
            "query": "test",
            "total_matches": 1,
            "offset": 0,
        }
        peer = {
            "results": [{"matched_message": {"uuid": "b"}, "score": 0.9}],
        }
        merged = merge_peer_results(local, [peer])

        scores = [r["score"] for r in merged["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_hint_mentions_peers(self):
        local = {
            "results": [{"matched_message": {"uuid": "a"}, "score": 0.9}],
            "query": "test",
            "total_matches": 1,
            "offset": 0,
        }
        peer = {"results": [{"matched_message": {"uuid": "b"}, "score": 0.8}]}
        merged = merge_peer_results(local, [peer])

        assert "peer" in merged["hint"]


# --- Fan-out search ---


class TestFanOutSearch:
    def test_disabled_returns_empty(self):
        config = Config(peers=PeersConfig(enabled=False))
        with patch("kiro_ception.peers.get_config", return_value=config):
            results = fan_out_search({"query": "test"})
            assert results == []

    def test_no_nodes_returns_empty(self):
        config = Config(peers=PeersConfig(enabled=True, nodes=[]))
        with patch("kiro_ception.peers.get_config", return_value=config):
            results = fan_out_search({"query": "test"})
            assert results == []

    def test_unreachable_peer_returns_empty(self):
        config = Config(peers=PeersConfig(
            enabled=True, nodes=["192.0.2.1:19742"], timeout_seconds=1
        ))
        with patch("kiro_ception.peers.get_config", return_value=config):
            results = fan_out_search({"query": "test"})
            assert results == []

    def test_successful_peer_response(self):
        config = Config(peers=PeersConfig(
            enabled=True, nodes=["fake-peer:19742"], timeout_seconds=5
        ))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "results": [{"matched_message": {"uuid": "peer-1"}, "score": 0.8}],
            "total_matches": 1,
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("kiro_ception.peers.get_config", return_value=config),
            patch("kiro_ception.peers.requests.post", return_value=mock_response),
        ):
            results = fan_out_search({"query": "test"})

        assert len(results) == 1
        assert results[0]["results"][0]["matched_message"]["uuid"] == "peer-1"
