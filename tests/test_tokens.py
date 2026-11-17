import pytest

from jarvis_core.tokens import BudgetExceeded, TokenBudget, TokenLedger, Usage


def test_ledger_enforces_run_budget():
    ledger = TokenLedger(TokenBudget(max_run_input=10, max_turn_input=10))
    ledger.record(Usage(agent="one", input_tokens=8))
    with pytest.raises(BudgetExceeded):
        ledger.record(Usage(agent="two", input_tokens=3))


def test_ledger_serializes_usage():
    ledger = TokenLedger()
    ledger.record(Usage(agent="explorer", input_tokens=10, output_tokens=2))
    assert ledger.to_dict()["totals"]["input_tokens"] == 10


def test_failed_commit_preserves_reservation_until_refunded():
    ledger = TokenLedger(TokenBudget(max_run_input=10, max_turn_input=10))
    reservation = ledger.reserve("agent", input_tokens=5, output_tokens=0)

    with pytest.raises(BudgetExceeded):
        ledger.commit(reservation, Usage(agent="agent", input_tokens=11))

    assert ledger.totals(include_reserved=True).input_tokens == 5
    ledger.refund(reservation)
    assert ledger.totals(include_reserved=True).input_tokens == 0


def test_call_refunds_reservation_when_usage_extraction_fails():
    ledger = TokenLedger(TokenBudget(max_run_input=10, max_turn_input=10))

    def fail_extraction(_response):
        raise RuntimeError("invalid provider usage")

    with pytest.raises(RuntimeError, match="invalid provider usage"):
        ledger.call(
            agent="agent",
            model="model",
            prompt="hello",
            max_output_tokens=1,
            invoke=lambda: "response",
            extract_usage=fail_extraction,
        )

    assert ledger.totals(include_reserved=True).input_tokens == 0


def test_failed_record_refunds_its_reservation(monkeypatch):
    """record() must not leak a reservation when commit-time validation fails."""
    ledger = TokenLedger(TokenBudget(max_run_input=10, max_turn_input=10))

    def failing_commit(reservation, usage):
        raise BudgetExceeded("run input token budget exceeded")

    monkeypatch.setattr(ledger, "commit", failing_commit)

    with pytest.raises(BudgetExceeded):
        ledger.record(Usage(agent="agent", input_tokens=5, output_tokens=0))

    # The hold was refunded automatically; no manual refund is needed.
    assert ledger.totals(include_reserved=True).input_tokens == 0
    assert ledger.totals().input_tokens == 0


def test_provider_usage_zero_is_treated_as_absent_for_the_estimate_fallback():
    """All-zero provider usage must not bypass the estimate fallback (0.18.1)."""
    ledger = TokenLedger()
    usage = TokenLedger.usage_from_provider(
        "agent",
        "model",
        {"prompt_tokens": 0, "completion_tokens": 0},
    )
    assert usage is None


def test_provider_usage_negative_and_none_values_are_sanitized():
    ledger = TokenLedger()
    usage = TokenLedger.usage_from_provider(
        "agent",
        "model",
        {"prompt_tokens": -1, "completion_tokens": None},
    )
    # Both fields sanitize to zero, which counts as "no usage reported" so
    # callers fall back to their own estimate.
    assert usage is None


def test_cached_input_tokens_never_exceed_input_tokens():
    usage = TokenLedger.usage_from_provider(
        "agent",
        "model",
        {"prompt_tokens": 5, "cached_input_tokens": 9, "completion_tokens": 2},
    )
    assert usage is not None
    assert usage.cached_input_tokens == 5


def test_call_survives_over_budget_provider_usage_after_success():
    """A successful call must not be discarded by post-call accounting."""
    ledger = TokenLedger(
        TokenBudget(
            max_run_input=100,
            max_run_output=100,
            max_turn_input=100,
            max_turn_output=100,
        )
    )
    big = {"prompt_tokens": 40_000, "completion_tokens": 5}

    def invoke():
        return "answer"

    response = ledger.call(
        agent="agent",
        model="model",
        prompt="x" * 30,
        max_output_tokens=10,
        invoke=invoke,
        extract_usage=lambda _response: big,
    )
    assert response == "answer"
    assert ledger.totals().output_tokens > 0
