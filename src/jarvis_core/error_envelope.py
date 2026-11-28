"""Provider-neutral OpenAI-compatible error envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

RETRY_AFTER_SECONDS: dict[str, int] = {
    "QUEUE_TIMEOUT": 5,
    "QUEUE_FULL": 2,
}


@dataclass(frozen=True)
class ErrorEnvelope:
    """The canonical ``{"error": {...}}`` payload used by Jarvis services."""

    code: str
    message: str
    retryable: bool = False
    request_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.request_id is not None:
            error["request_id"] = self.request_id
        error.update(self.details)
        return {"error": error}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ErrorEnvelope | None:
        """Parse an OpenAI error or FastAPI detail shape into the envelope."""
        error = payload.get("error")
        if isinstance(error, Mapping):
            code = str(error.get("code") or error.get("type") or "INFERENCE_ERROR")
            message = str(error.get("message") or error.get("msg") or "")
            request_id = error.get("request_id")
            reserved = {"code", "type", "message", "msg", "retryable", "request_id"}
            return cls(
                code=code,
                message=message,
                retryable=bool(error.get("retryable", False)),
                request_id=str(request_id) if request_id is not None else None,
                details={str(k): v for k, v in error.items() if k not in reserved},
            )

        detail = payload.get("detail")
        if isinstance(detail, Mapping):
            code = str(detail.get("code") or "ERROR")
            message = str(detail.get("message") or detail.get("msg") or "")
            reserved = {"code", "message", "msg", "retryable", "request_id"}
            return cls(
                code=code,
                message=message,
                retryable=bool(detail.get("retryable", False)),
                request_id=(
                    str(detail["request_id"])
                    if detail.get("request_id") is not None
                    else None
                ),
                details={str(k): v for k, v in detail.items() if k not in reserved},
            )
        if isinstance(detail, str):
            return cls(code="ERROR", message=detail)
        return None


def error_payload(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    request_id: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Build a canonical error response body."""
    return ErrorEnvelope(
        code=code,
        message=message,
        retryable=retryable,
        request_id=request_id,
        details=details,
    ).to_dict()


def retry_after_seconds(code: str) -> int | None:
    """Return the standard ``Retry-After`` value for a queue rejection code."""
    return RETRY_AFTER_SECONDS.get(code)


__all__ = [
    "RETRY_AFTER_SECONDS",
    "ErrorEnvelope",
    "error_payload",
    "retry_after_seconds",
]
