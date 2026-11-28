"""Dependency-free secret redaction used at every Jarvis service boundary."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")

REDACTED = "[REDACTED]"
REDACTED_KEY = "[REDACTED_KEY]"

# Keys whose values are never safe to emit, regardless of their shape.
SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
        "webhook_secret",
        "private_key",
    }
)

_PLACEHOLDERS = frozenset({REDACTED, REDACTED_KEY, "***"})
_PLACEHOLDER_FOLDED = frozenset(value.casefold() for value in _PLACEHOLDERS)

_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|password|secret|authorization|cookie)\b)"
    r"(\s*[:=]\s*)(?!bearer\b)([^\s,;]+)"
)
_PROVIDER_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_GITHUB_TOKEN = re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")
_GITLAB_TOKEN = re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")

DEFAULT_PATTERNS = (
    _BEARER,
    _SECRET_ASSIGNMENT,
    _PROVIDER_KEY,
    _GITHUB_TOKEN,
    _GITLAB_TOKEN,
)


def _replace_assignment(match: re.Match[str]) -> str:
    value = match.group(3)
    if value.casefold() in _PLACEHOLDER_FOLDED:
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


def redact_text(
    text: str,
    *,
    secrets: Iterable[str] = (),
    replacement: str = REDACTED,
    patterns: Iterable[re.Pattern[str]] | None = None,
) -> str:
    """Scrub known secrets and common credential shapes from free-form text.

    Known secret values are removed first so pattern passes cannot partially
    match them. Already-masked placeholders are preserved, which keeps repeated
    redaction idempotent.
    """
    result = str(text)
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        result = result.replace(secret, replacement)

    for pattern in DEFAULT_PATTERNS if patterns is None else patterns:
        if pattern is _SECRET_ASSIGNMENT:
            result = pattern.sub(_replace_assignment, result)
        elif pattern is _BEARER:
            result = pattern.sub(f"Bearer {REDACTED}", result)
        elif pattern is _PROVIDER_KEY:
            result = pattern.sub(REDACTED_KEY, result)
        else:
            result = pattern.sub(REDACTED, result)
    return result


def redact_value(
    value: T,
    *,
    secrets: Iterable[str] = (),
    replacement: str = REDACTED,
    key_names: Iterable[str] = SECRET_KEY_NAMES,
    patterns: Iterable[re.Pattern[str]] | None = None,
) -> T:
    """Recursively redact a payload while preserving its container types."""
    secret_values = tuple(secrets)
    secret_keys = frozenset(name.casefold() for name in key_names)
    if isinstance(value, str):
        return redact_text(  # type: ignore[return-value]
            value, secrets=secret_values, replacement=replacement, patterns=patterns
        )
    if isinstance(value, dict):
        return {  # type: ignore[return-value]
            key: (
                replacement
                if str(key).casefold() in secret_keys
                else redact_value(
                    item,
                    secrets=secret_values,
                    replacement=replacement,
                    key_names=secret_keys,
                    patterns=patterns,
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [  # type: ignore[return-value]
            redact_value(
                item,
                secrets=secret_values,
                replacement=replacement,
                key_names=secret_keys,
                patterns=patterns,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(  # type: ignore[return-value]
            redact_value(
                item,
                secrets=secret_values,
                replacement=replacement,
                key_names=secret_keys,
                patterns=patterns,
            )
            for item in value
        )
    return value


def redact_strings(value: T, *, secrets: Iterable[str] = ()) -> T:
    """Redact string leaves without applying key-name masking."""
    secret_values = tuple(secrets)
    if isinstance(value, str):
        return redact_text(value, secrets=secret_values)  # type: ignore[return-value]
    if isinstance(value, dict):
        return {  # type: ignore[return-value]
            key: redact_strings(item, secrets=secret_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_strings(item, secrets=secret_values) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(redact_strings(item, secrets=secret_values) for item in value)  # type: ignore[return-value]
    return value


__all__ = [
    "DEFAULT_PATTERNS",
    "REDACTED",
    "REDACTED_KEY",
    "SECRET_KEY_NAMES",
    "redact_strings",
    "redact_text",
    "redact_value",
]
