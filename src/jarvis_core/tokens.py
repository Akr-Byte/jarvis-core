"""Thread-safe token estimation, reservation, and usage accounting."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Mapping
from dataclasses import asdict, dataclass, field
from threading import RLock
from uuid import uuid4

_logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised before a call that would exceed a configured token budget."""


def estimate_tokens(value: object) -> int:
    """Conservative tokenizer-free estimate suitable for unknown open models."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if not value:
        return 0
    ascii_chars = sum(ord(char) < 128 for char in value)
    non_ascii = len(value) - ascii_chars
    return max(1, (ascii_chars + 2 * non_ascii + 2) // 3)


@dataclass(frozen=True)
class TokenBudget:
    max_run_input: int = 32_000
    max_run_output: int = 3_000
    max_turn_input: int = 16_000
    max_turn_output: int = 2_000
    max_agent_input: int = 24_000
    max_agent_output: int = 2_500


@dataclass
class Usage:
    agent: str
    model: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    tool_result_tokens: int = 0
    compaction_saved_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TokenReservation:
    id: str
    agent: str
    input_tokens: int
    output_tokens: int


@dataclass
class TokenLedger:
    """Atomic budget ledger safe for parallel agent calls."""

    budget: TokenBudget = field(default_factory=TokenBudget)
    entries: list[Usage] = field(default_factory=list)
    usage_events: list[dict[str, Any]] = field(default_factory=list)
    _reservations: dict[str, TokenReservation] = field(default_factory=dict, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @staticmethod
    def _sum(entries: list[Usage], agent: str | None = None) -> Usage:
        total = Usage(agent=agent or "run")
        for entry in entries:
            if agent is not None and entry.agent != agent:
                continue
            total.input_tokens += entry.input_tokens
            total.cached_input_tokens += entry.cached_input_tokens
            total.output_tokens += entry.output_tokens
            total.tool_result_tokens += entry.tool_result_tokens
            total.compaction_saved_tokens += entry.compaction_saved_tokens
        return total

    def totals(
        self, agent: str | None = None, *, include_reserved: bool = False
    ) -> Usage:
        with self._lock:
            total = self._sum(self.entries, agent)
            if include_reserved:
                for reservation in self._reservations.values():
                    if agent is None or reservation.agent == agent:
                        total.input_tokens += reservation.input_tokens
                        total.output_tokens += reservation.output_tokens
            return total

    def _validate(self, agent: str, input_tokens: int, output_tokens: int) -> None:
        if min(input_tokens, output_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        run = self.totals(include_reserved=True)
        current = self.totals(agent, include_reserved=True)
        if input_tokens > self.budget.max_turn_input:
            raise BudgetExceeded("turn input token budget exceeded")
        if output_tokens > self.budget.max_turn_output:
            raise BudgetExceeded("turn output token budget exceeded")
        if run.input_tokens + input_tokens > self.budget.max_run_input:
            raise BudgetExceeded("run input token budget exceeded")
        if run.output_tokens + output_tokens > self.budget.max_run_output:
            raise BudgetExceeded("run output token budget exceeded")
        if current.input_tokens + input_tokens > self.budget.max_agent_input:
            raise BudgetExceeded(f"{agent} input token budget exceeded")
        if current.output_tokens + output_tokens > self.budget.max_agent_output:
            raise BudgetExceeded(f"{agent} output token budget exceeded")

    def reserve(
        self, agent: str, input_tokens: int, output_tokens: int
    ) -> TokenReservation:
        with self._lock:
            self._validate(agent, input_tokens, output_tokens)
            reservation = TokenReservation(
                str(uuid4()), agent, input_tokens, output_tokens
            )
            self._reservations[reservation.id] = reservation
            return reservation

    def commit(
        self, reservation: TokenReservation, usage: Usage | None = None
    ) -> Usage:
        """Commit actual usage and release unused reserved capacity."""
        with self._lock:
            held = self._reservations.get(reservation.id)
            if held is None:
                raise ValueError("unknown or already closed reservation")
            actual = usage or Usage(
                agent=held.agent,
                input_tokens=held.input_tokens,
                output_tokens=held.output_tokens,
            )
            if actual.agent != held.agent:
                raise ValueError("usage agent does not match reservation")
            # Exclude this hold while validating the complete actual usage. Keep it
            # open if validation fails so callers can inspect or refund it.
            del self._reservations[reservation.id]
            try:
                self._validate(actual.agent, actual.input_tokens, actual.output_tokens)
            except BaseException:
                self._reservations[reservation.id] = held
                raise
            self.entries.append(actual)
            return actual

    def refund(self, reservation: TokenReservation) -> None:
        with self._lock:
            if self._reservations.pop(reservation.id, None) is None:
                raise ValueError("unknown or already closed reservation")

    def record(self, usage: Usage) -> None:
        reservation = self.reserve(usage.agent, usage.input_tokens, usage.output_tokens)
        try:
            # Validation can still fail at commit time; never leak the hold.
            self.commit(reservation, usage)
        except BaseException:
            with self._lock:
                self._reservations.pop(reservation.id, None)
            raise

    @staticmethod
    def usage_from_provider(
        agent: str, model: str, usage: Mapping[str, int] | None
    ) -> Usage | None:
        """Normalize common OpenAI/Hugging Face usage field names.

        Provider-reported values are sanitized (negative or non-numeric
        fields clamp to zero) and an all-zero mapping is treated as absent so
        callers fall back to their own estimate instead of recording zero
        tokens for a real prompt (0.18.1).
        """
        if not usage:
            return None

        def _count(key: str, fallback_key: str) -> int:
            raw = usage.get(key, usage.get(fallback_key, 0))
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                return 0

        input_tokens = _count("input_tokens", "prompt_tokens")
        output_tokens = _count("output_tokens", "completion_tokens")
        cached_input_tokens = _count("cached_input_tokens", "cached_input_tokens")
        if input_tokens == 0 and output_tokens == 0:
            return None
        return Usage(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            cached_input_tokens=min(cached_input_tokens, input_tokens),
            output_tokens=output_tokens,
        )

    def call(
        self,
        *,
        agent: str,
        model: str,
        prompt: object,
        max_output_tokens: int,
        invoke: Callable[[], object],
        extract_usage: Callable[[object], Mapping[str, int] | None] | None = None,
    ) -> object:
        input_tokens = estimate_tokens(prompt)
        reservation = self.reserve(agent, input_tokens, max_output_tokens)
        try:
            response = invoke()
            provider_usage = extract_usage(response) if extract_usage else None
            try:
                usage = self.usage_from_provider(agent, model, provider_usage) or Usage(
                    agent=agent,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=estimate_tokens(response),
                )
                self.commit(reservation, usage)
                return response
            except (BudgetExceeded, TypeError, ValueError) as exc:
                # The provider call already succeeded; accounting failures
                # (e.g. honestly over-budget usage) must not discard the
                # response. Fall back to the estimated usage that the
                # reservation already validated (0.18.1).
                _logger.warning(
                    "usage accounting failed after a successful provider call "
                    "(%s); recording the estimated usage instead",
                    exc,
                )
                # A failed commit restores the reservation, so it is still
                # valid for the estimate-based retry below.
                try:
                    self.commit(
                        reservation,
                        Usage(
                            agent=agent,
                            model=model,
                            input_tokens=input_tokens,
                            output_tokens=estimate_tokens(response),
                        ),
                    )
                except (BudgetExceeded, TypeError, ValueError):
                    with self._lock:
                        self._reservations.pop(reservation.id, None)
                return response
        except BaseException:
            # Extraction or validation can fail after the provider returns. Never
            # leave that failed call consuming reserved capacity.
            with self._lock:
                self._reservations.pop(reservation.id, None)
            raise

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "budget": asdict(self.budget),
                "totals": self.totals().to_dict(),
                "reserved": self.totals(include_reserved=True).to_dict(),
                "entries": [entry.to_dict() for entry in self.entries],
                "usage_events": self.usage_events,
            }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist the ledger state to a JSON file.

        This enables budget survival across process restarts (CORE-10).
        The file format is JSON with a ``"version"`` field for forward
        compatibility.  Calls to :meth:`load` from a newer version will
        raise ``ValueError``.
        """
        data = self.to_dict()
        data["version"] = 1
        # Ensure parent directory exists
        p = os.fspath(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "TokenLedger":
        """Load a persisted ledger state from a JSON file.

        The ledger is initialised with the saved budget, entries, and
        reservations; any missing fields default to empty/zero state.
        """
        p = os.fspath(path)
        if not os.path.isfile(p):
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.pop("version", 0)
        if version > 1:
            raise ValueError(
                f"Unsupported ledger version {version}; "
                "the ledger format has changed."
            )
        ledger = cls()
        # Restore budget
        budget_data = data.pop("budget", {})
        if budget_data:
            # Reconstruct TokenBudget from dict; fields with defaults are
            # automatically populated by the dataclass constructor.
            from .tokens import TokenBudget

            ledger.budget = TokenBudget(**budget_data)
        # Restore entries (list of Usage)
        entries_data = data.pop("entries", [])
        for entry_dict in entries_data:
            from .tokens import Usage

            ledger.entries.append(Usage(**entry_dict))
        # Restore reservations (dict of TokenReservation)
        reservations_data = data.pop("_reservations", {})
        for rid, res_dict in reservations_data.items():
            from .tokens import TokenReservation

            ledger._reservations[rid] = TokenReservation(**res_dict)
        # Restore usage events
        usage_events_data = data.pop("usage_events", [])
        ledger.usage_events = usage_events_data
        return ledger

    def record_usage_event(self, event: dict[str, Any]) -> None:
        """Record a usage event (token count, cost, latency) for observability.

        Events are appended to the internal list and persisted via
        :meth:`save`/ :meth:`load` for cross-process observability.
        """
        with self._lock:
            self.usage_events.append(event)
