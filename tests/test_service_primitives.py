from __future__ import annotations

import re
from typing import Any, cast

import pytest

from jarvis_core import (
    REDACTED,
    REDACTED_KEY,
    ErrorEnvelope,
    bearer_token,
    env_csv,
    env_flag,
    env_float,
    env_int,
    env_list,
    env_text,
    error_payload,
    generate_api_key,
    hash_api_key,
    keys_equal,
    redact_strings,
    redact_text,
    redact_value,
    retry_after_seconds,
    safe_request_id,
)


def test_env_helpers_parse_and_bound_values() -> None:
    environ = {
        "BOOL": "YES",
        "BAD_BOOL": "perhaps",
        "INT": " 12 ",
        "FLOAT": "1.5",
        "LIST": "a, b,, c",
        "TEXT": " value ",
    }
    assert env_flag("BOOL", False, environ=environ) is True
    assert env_flag("BAD_BOOL", True, environ=environ) is False
    assert env_flag("MISSING", True, environ=environ) is True
    assert env_int("INT", 0, minimum=1, maximum=20, environ=environ) == 12
    assert env_float("FLOAT", 0.0, minimum=0.0, maximum=2.0, environ=environ) == 1.5
    assert env_list("LIST", "", environ=environ) == ["a", "b", "c"]
    assert env_csv("LIST", "", environ=environ) == ("a", "b", "c")
    assert env_text("TEXT", "fallback", environ=environ) == "value"
    assert env_text("EMPTY", "fallback", environ=environ) == "fallback"
    with pytest.raises(ValueError):
        env_int("INT", 0, minimum=20, environ=environ)
    with pytest.raises(ValueError):
        env_float("FLOAT", 0.0, maximum=1.0, environ=environ)


def test_api_key_primitives_are_stable_and_constant_time() -> None:
    key = generate_api_key("forge_")
    assert key.startswith("forge_")
    assert len(hash_api_key(key)) == 64
    assert hash_api_key(key) == hash_api_key(key)
    assert keys_equal(key, key) is True
    assert keys_equal(key, "other") is False
    assert keys_equal(None, key) is False
    with pytest.raises(ValueError):
        hash_api_key("")
    with pytest.raises(ValueError):
        generate_api_key(num_bytes=8)


def test_redact_text_preserves_placeholders_and_hides_credentials() -> None:
    value = "Authorization: Bearer abcdefghijklmnop token=secret-value sk-123456789012"
    redacted = redact_text(value)
    assert "abcdefghijklmnop" not in redacted
    assert "secret-value" not in redacted
    assert "sk-123456789012" not in redacted
    assert "Bearer [REDACTED]" in redacted
    assert REDACTED_KEY in redacted
    assert redact_text("api_key=*** token=[REDACTED]") == "api_key=*** token=[REDACTED]"
    assert (
        redact_text("password=hunter2", secrets=["hunter2"]) == f"password={REDACTED}"
    )
    assert redact_text("ghp_" + "a" * 24) == REDACTED
    assert redact_text("glpat-" + "a" * 24) == REDACTED
    assert redact_text("x", patterns=[re.compile("x")]) == REDACTED


def test_redact_value_masks_secret_keys_and_recurses() -> None:
    payload = {
        "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
        "nested": ["Bearer abc.def.ghi", {"token": "abc"}],
        "tuple": ("Authorization: Bearer abcdefghijklmnop",),
    }
    redacted = redact_value(payload)
    nested = cast(list[Any], redacted["nested"])
    assert redacted["api_key"] == REDACTED
    assert REDACTED in nested[0]
    assert nested[1]["token"] == REDACTED
    assert "Bearer [REDACTED]" in cast(tuple[str, ...], redacted["tuple"])[0]
    assert redact_strings({"token": "abc"})["token"] == "abc"


def test_error_envelope_parses_both_service_shapes() -> None:
    payload = error_payload("QUEUE_FULL", "busy", retryable=True, request_id="req-1")
    parsed = ErrorEnvelope.from_dict(payload)
    assert parsed is not None
    assert parsed.code == "QUEUE_FULL"
    assert parsed.request_id == "req-1"
    assert payload["error"]["retryable"] is True
    assert ErrorEnvelope.from_dict({"detail": "plain failure"}) == ErrorEnvelope(
        "ERROR", "plain failure"
    )
    fastapi = ErrorEnvelope.from_dict({"detail": {"code": "BAD", "message": "nope"}})
    assert fastapi is not None
    assert fastapi.code == "BAD"
    assert ErrorEnvelope.from_dict({"unexpected": True}) is None
    assert retry_after_seconds("QUEUE_FULL") == 2
    assert retry_after_seconds("QUEUE_TIMEOUT") == 5
    assert retry_after_seconds("OTHER") is None


def test_http_header_primitives_are_bounded() -> None:
    assert bearer_token("Bearer abc.def") == "abc.def"
    assert bearer_token("bearer abc.def") == "abc.def"
    assert bearer_token("Basic abc") is None
    assert bearer_token(None) is None
    assert safe_request_id(" req-1 ", fallback="generated") == "req-1"
    assert safe_request_id("bad id", fallback="generated") == "generated"
    assert safe_request_id("x" * 129, fallback="generated") == "generated"
    assert safe_request_id(None) is None
