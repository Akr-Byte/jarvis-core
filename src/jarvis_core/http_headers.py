"""Small HTTP header primitives shared by Jarvis services."""

from __future__ import annotations

import re

SAFE_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:@/-]{1,128}")


def bearer_token(value: str | None) -> str | None:
    """Extract a bearer credential, tolerating case and surrounding space."""
    if not value:
        return None
    match = re.match(r"(?i)^Bearer\s+(.+)$", value.strip())
    if not match:
        return None
    token = match.group(1).strip()
    return token or None


def safe_request_id(
    value: str | None,
    *,
    pattern: re.Pattern[str] = SAFE_REQUEST_ID_PATTERN,
    max_length: int = 128,
    fallback: str | None = None,
) -> str | None:
    """Return an echo-safe request id or ``fallback`` when it is unsafe."""
    if not value:
        return fallback
    candidate = value.strip()
    if len(candidate) <= max_length and pattern.fullmatch(candidate):
        return candidate
    return fallback


__all__ = ["SAFE_REQUEST_ID_PATTERN", "bearer_token", "safe_request_id"]
