"""Dependency-free API-key primitives shared by Jarvis services."""

from __future__ import annotations

import hashlib
import hmac
import secrets

DEFAULT_KEY_BYTES = 32
MIN_KEY_BYTES = 16


def generate_api_key(prefix: str = "", *, num_bytes: int = DEFAULT_KEY_BYTES) -> str:
    """Generate a URL-safe random API key with an optional operator prefix."""
    if num_bytes < MIN_KEY_BYTES:
        raise ValueError(f"num_bytes must be >= {MIN_KEY_BYTES}")
    return f"{prefix}{secrets.token_urlsafe(num_bytes)}"


def hash_api_key(raw_key: str) -> str:
    """Return the hex SHA-256 digest of a raw API key."""
    if not raw_key:
        raise ValueError("raw API key cannot be empty")
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def keys_equal(left: str | None, right: str | None) -> bool:
    """Constant-time comparison for credentials; missing values never match."""
    if left is None or right is None:
        return False
    return hmac.compare_digest(left, right)


__all__ = [
    "DEFAULT_KEY_BYTES",
    "MIN_KEY_BYTES",
    "generate_api_key",
    "hash_api_key",
    "keys_equal",
]
