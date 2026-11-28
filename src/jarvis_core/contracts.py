"""Protocol and capability contracts shared across Jarvis services.

Transport-neutral: defines compatibility semantics without depending on HTTP,
FastAPI, or deployment-specific frameworks.

Canonical protocol v1 shapes live in ``jarvis_core.protocol``; this module keeps
backward-compatible helpers used by existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .protocol import (  # noqa: F401
    AGENT_PROTOCOL_VERSION,
    EVENT_SCHEMA_VERSION,
    INFERENCE_PROTOCOL_VERSION,
    ProtocolRange,
)

# Re-export for compatibility
__all__ = [
    "AGENT_PROTOCOL_VERSION",
    "EVENT_SCHEMA_VERSION",
    "INFERENCE_PROTOCOL_VERSION",
    "CapabilityDescriptor",
    "ProtocolRange",
    "compatible",
    "require_features",
]


@dataclass(frozen=True)
class CapabilityDescriptor:
    """A model/service capability advertised by a component."""

    name: str
    version: int = 1
    features: frozenset[str] = frozenset()

    def supports(self, required: Iterable[str]) -> bool:
        return set(required).issubset(self.features)


def compatible(
    local: ProtocolRange,
    remote: ProtocolRange,
    *,
    version: int | None = None,
) -> bool:
    """Return whether two protocol ranges can interoperate."""
    if version is not None:
        return local.supports(version) and remote.supports(version)
    return local.overlaps(remote)


def require_features(
    capabilities: Mapping[str, object],
    required: Iterable[str],
) -> None:
    """Raise ValueError when required capabilities are missing."""
    raw = capabilities.get("features", [])
    if isinstance(raw, str):
        features = {raw}
    elif isinstance(raw, Iterable):
        features = {str(item) for item in raw}
    else:
        features = set()
    missing = [feature for feature in required if feature not in features]
    if missing:
        raise ValueError(f"missing capabilities: {', '.join(sorted(missing))}")
