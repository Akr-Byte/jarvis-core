"""Dependency-free environment parsing shared by Jarvis services."""

from __future__ import annotations

import os
from collections.abc import Mapping

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _raw(name: str, default: str, environ: Mapping[str, str]) -> str:
    return str(environ.get(name, default)).strip()


def env_text(
    name: str, default: str = "", *, environ: Mapping[str, str] | None = None
) -> str:
    """Return a stripped environment string or ``default`` when it is empty."""
    value = _raw(name, default, environ if environ is not None else os.environ)
    return value or default


def env_flag(
    name: str, default: bool = False, *, environ: Mapping[str, str] | None = None
) -> bool:
    """Parse a boolean environment variable.

    Unrecognized values are false. This is deliberate: an operator typo must
    fail closed for feature flags and security switches.
    """
    value = _raw(
        name, str(default), environ if environ is not None else os.environ
    ).casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return False


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Parse a bounded integer environment variable."""
    value = _raw(name, str(default), environ if environ is not None else os.environ)
    parsed = default if not value else int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Parse a bounded float environment variable."""
    value = _raw(name, str(default), environ if environ is not None else os.environ)
    parsed = default if not value else float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def env_list(
    name: str,
    default: str = "",
    *,
    sep: str = ",",
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Parse a separated environment list, dropping empty entries."""
    value = _raw(name, default, environ if environ is not None else os.environ)
    return [item.strip() for item in value.split(sep) if item.strip()]


def env_csv(
    name: str, default: str = "", *, environ: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Parse a comma-separated environment value as an immutable tuple."""
    return tuple(env_list(name, default, environ=environ))
