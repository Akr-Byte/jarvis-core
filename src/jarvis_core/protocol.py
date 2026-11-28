"""Canonical Jarvis Agent and Inference protocol v1.

Transport-neutral, dependency-free contracts shared by jarvis-core,
jarvis-inference, ai-stack, jarvis CLI, and forge.

Design goals:
- explicit version negotiation
- stable error/usage/capability shapes
- correlation/idempotency/request attribution for durable execution
- JSON-serializable dataclasses; no framework dependencies
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

AGENT_PROTOCOL_VERSION = 1
INFERENCE_PROTOCOL_VERSION = 1
EVENT_SCHEMA_VERSION = 1

AGENT_PROTOCOL_HEADER = "X-Jarvis-Protocol-Version"
INFERENCE_PROTOCOL_HEADER = "X-Jarvis-Inference-Protocol-Version"
REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"
IDEMPOTENCY_KEY_HEADER = "X-Idempotency-Key"
TENANT_ID_HEADER = "X-Jarvis-Tenant-ID"
PROJECT_ID_HEADER = "X-Jarvis-Project-ID"
SERVICE_ID_HEADER = "X-Jarvis-Service-ID"

# Canonical error codes
ERR_INCOMPATIBLE_PROTOCOL = "INCOMPATIBLE_PROTOCOL"
ERR_INVALID_PROTOCOL_HEADER = "INVALID_PROTOCOL_HEADER"
ERR_INVALID_REQUEST = "INVALID_REQUEST"
ERR_UNAUTHORIZED = "UNAUTHORIZED"
ERR_FORBIDDEN = "FORBIDDEN"
ERR_NOT_FOUND = "NOT_FOUND"
ERR_CONFLICT = "CONFLICT"
ERR_IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
ERR_RATE_LIMITED = "RATE_LIMITED"
ERR_QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
ERR_CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
ERR_MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
ERR_PROVIDER_ERROR = "PROVIDER_ERROR"
ERR_QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
ERR_QUEUE_FULL = "QUEUE_FULL"
ERR_INTERNAL = "INTERNAL_ERROR"


class ProtocolKind(str, Enum):
    AGENT = "agent"
    INFERENCE = "inference"


@dataclass(frozen=True)
class ProtocolRange:
    minimum: int
    maximum: int

    def supports(self, version: int) -> bool:
        return self.minimum <= version <= self.maximum

    def overlaps(self, other: "ProtocolRange") -> bool:
        return max(self.minimum, other.minimum) <= min(self.maximum, other.maximum)


@dataclass(frozen=True)
class ProtocolNegotiation:
    kind: ProtocolKind
    current: int
    minimum: int
    maximum: int

    def supports(self, version: int) -> bool:
        return self.minimum <= version <= self.maximum

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "current": self.current,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


AGENT_PROTOCOL = ProtocolNegotiation(
    kind=ProtocolKind.AGENT, current=AGENT_PROTOCOL_VERSION, minimum=1, maximum=1
)
INFERENCE_PROTOCOL = ProtocolNegotiation(
    kind=ProtocolKind.INFERENCE,
    current=INFERENCE_PROTOCOL_VERSION,
    minimum=1,
    maximum=1,
)


@dataclass(frozen=True)
class Capability:
    name: str
    version: int = 1
    features: tuple[str, ...] = ()

    def supports(self, required: Sequence[str]) -> bool:
        return set(required).issubset(set(self.features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    estimated: bool = False
    cost_usd: float | None = None
    currency: str = "USD"

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "total_tokens": int(
                self.total_tokens or (self.input_tokens + self.output_tokens)
            ),
            "cached_input_tokens": int(self.cached_input_tokens),
            "estimated": bool(self.estimated),
            "currency": self.currency,
            "prompt_tokens": int(self.input_tokens),
            "completion_tokens": int(self.output_tokens),
        }
        if self.cost_usd is not None:
            payload["cost_usd"] = float(self.cost_usd)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "Usage | None":
        if not isinstance(value, Mapping):
            return None
        input_tokens = int(
            value.get("input_tokens", value.get("prompt_tokens", 0)) or 0
        )
        output_tokens = int(
            value.get("output_tokens", value.get("completion_tokens", 0)) or 0
        )
        total = int(value.get("total_tokens", input_tokens + output_tokens) or 0)
        cost = value.get("cost_usd")
        return cls(
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            total_tokens=max(0, total),
            cached_input_tokens=max(0, int(value.get("cached_input_tokens", 0) or 0)),
            estimated=bool(value.get("estimated", False)),
            cost_usd=float(cost) if cost is not None else None,
            currency=str(value.get("currency", "USD") or "USD"),
        )


@dataclass(frozen=True)
class ProtocolError:
    code: str
    message: str
    retryable: bool = False
    request_id: str | None = None
    correlation_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": bool(self.retryable),
        }
        if self.request_id:
            error["request_id"] = self.request_id
        if self.correlation_id:
            error["correlation_id"] = self.correlation_id
        error.update(self.details)
        return {"error": error}


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str | None = None
    organization_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    actor_id: str | None = None
    actor_type: str | None = None
    service_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class EventEnvelope:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = AGENT_PROTOCOL_VERSION
    schema_version: int = EVENT_SCHEMA_VERSION
    run_id: str | None = None
    event_id: str | int | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    tenant: TenantContext = field(default_factory=TenantContext)
    ts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tenant"] = self.tenant.to_dict()
        return payload


def negotiate(local: ProtocolNegotiation, requested: int | None) -> int:
    if requested is None:
        return local.current
    if not local.supports(requested):
        raise ValueError(
            f"Incompatible {local.kind.value} protocol {requested}; "
            f"server supports {local.minimum}..{local.maximum}"
        )
    return requested


def parse_protocol_header(value: str | None, *, header: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{header} must be an integer protocol version") from exc


def extract_capabilities(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize capabilities payloads across agent/inference shapes."""
    protocol = (
        payload.get("protocol") if isinstance(payload.get("protocol"), Mapping) else {}
    )
    features = payload.get("features")
    if not isinstance(features, list):
        # nested model capabilities map
        if isinstance(payload.get("capabilities"), list):
            features = payload.get("capabilities")
        else:
            features = []
    min_v = protocol.get("min_cli", protocol.get("min_client", protocol.get("minimum")))
    max_v = protocol.get("max_cli", protocol.get("max_client", protocol.get("maximum")))
    cur_v = protocol.get("current", protocol.get("version"))
    return {
        "current": cur_v,
        "minimum": min_v,
        "maximum": max_v,
        "features": [str(f) for f in features if isinstance(f, (str, int, float))],
    }


def check_feature(payload: Mapping[str, Any], feature: str) -> bool:
    return feature in extract_capabilities(payload).get("features", [])


__all__ = [
    "AGENT_PROTOCOL",
    "AGENT_PROTOCOL_HEADER",
    "AGENT_PROTOCOL_VERSION",
    "CORRELATION_ID_HEADER",
    "ERR_CONFLICT",
    "ERR_CONTEXT_OVERFLOW",
    "ERR_FORBIDDEN",
    "ERR_IDEMPOTENCY_CONFLICT",
    "ERR_INCOMPATIBLE_PROTOCOL",
    "ERR_INTERNAL",
    "ERR_INVALID_PROTOCOL_HEADER",
    "ERR_INVALID_REQUEST",
    "ERR_MODEL_NOT_AVAILABLE",
    "ERR_NOT_FOUND",
    "ERR_PROVIDER_ERROR",
    "ERR_QUOTA_EXCEEDED",
    "ERR_QUEUE_FULL",
    "ERR_QUEUE_TIMEOUT",
    "ERR_RATE_LIMITED",
    "ERR_UNAUTHORIZED",
    "EVENT_SCHEMA_VERSION",
    "IDEMPOTENCY_KEY_HEADER",
    "INFERENCE_PROTOCOL",
    "INFERENCE_PROTOCOL_HEADER",
    "INFERENCE_PROTOCOL_VERSION",
    "PROJECT_ID_HEADER",
    "REQUEST_ID_HEADER",
    "SERVICE_ID_HEADER",
    "TENANT_ID_HEADER",
    "Capability",
    "EventEnvelope",
    "ProtocolError",
    "ProtocolKind",
    "ProtocolNegotiation",
    "ProtocolRange",
    "TenantContext",
    "Usage",
    "check_feature",
    "extract_capabilities",
    "negotiate",
    "parse_protocol_header",
]
