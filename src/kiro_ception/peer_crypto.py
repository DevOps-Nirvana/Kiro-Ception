"""Peer encryption using Argon2id key derivation and AES-256-GCM.

When a shared secret is configured, all inter-peer HTTP payloads are encrypted.
Both peers derive the same symmetric key from the passphrase using Argon2id
(memory-hard KDF, resistant to brute force). Payloads are encrypted with
AES-256-GCM which provides both confidentiality and integrity.

Wire format for encrypted payloads:
    nonce (12 bytes) || ciphertext || auth_tag (16 bytes)

Without a secret configured, payloads are sent as plain JSON (no encryption).
"""

import os

from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Argon2id parameters — tuned for security while keeping startup reasonable
_ARGON2_TIME_COST = 3        # Iterations
_ARGON2_MEMORY_COST = 65536  # 64 MB
_ARGON2_PARALLELISM = 1      # Single thread (deterministic)
_ARGON2_HASH_LEN = 32        # 256-bit key
_ARGON2_SALT = b"kiro-ception-peer-key-v1"  # Fixed salt (both peers must derive same key)


def derive_key(secret: str) -> bytes:
    """Derive a 256-bit AES key from a passphrase using Argon2id.

    Uses a fixed salt so both peers independently derive the same key
    from the same passphrase without needing a key exchange.

    Args:
        secret: The shared passphrase.

    Returns:
        32-byte key suitable for AES-256-GCM.
    """
    return hash_secret_raw(
        secret=secret.encode("utf-8"),
        salt=_ARGON2_SALT,
        time_cost=_ARGON2_TIME_COST,
        memory_cost=_ARGON2_MEMORY_COST,
        parallelism=_ARGON2_PARALLELISM,
        hash_len=_ARGON2_HASH_LEN,
        type=Type.ID,
    )


def encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt plaintext with AES-256-GCM.

    Args:
        plaintext: Data to encrypt.
        key: 32-byte key from derive_key().

    Returns:
        nonce (12 bytes) || ciphertext || auth_tag (16 bytes)
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-GCM encrypted data.

    Args:
        data: nonce (12 bytes) || ciphertext || auth_tag (16 bytes)
        key: 32-byte key from derive_key().

    Returns:
        Decrypted plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag: If key is wrong or data is tampered.
    """
    if len(data) < 28:  # 12 nonce + 16 tag minimum
        raise ValueError("Encrypted data too short")
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
