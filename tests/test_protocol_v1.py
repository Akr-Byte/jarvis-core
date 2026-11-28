import pytest

from jarvis_core import (
    AGENT_PROTOCOL,
    AGENT_PROTOCOL_HEADER,
    AGENT_PROTOCOL_VERSION,
    CORRELATION_ID_HEADER,
    EVENT_SCHEMA_VERSION,
    IDEMPOTENCY_KEY_HEADER,
    INFERENCE_PROTOCOL,
    INFERENCE_PROTOCOL_HEADER,
    INFERENCE_PROTOCOL_VERSION,
    REQUEST_ID_HEADER,
    EventEnvelope,
    ProtocolError,
    TenantContext,
    Usage,
    check_feature,
    extract_capabilities,
    negotiate,
    parse_protocol_header,
)


def test_protocol_constants_and_negotiation():
    assert AGENT_PROTOCOL_VERSION == 1
    assert INFERENCE_PROTOCOL_VERSION == 1
    assert EVENT_SCHEMA_VERSION == 1
    assert negotiate(AGENT_PROTOCOL, None) == 1
    assert negotiate(AGENT_PROTOCOL, 1) == 1
    with pytest.raises(ValueError):
        negotiate(AGENT_PROTOCOL, 2)


def test_parse_protocol_header():
    assert parse_protocol_header(None, header=AGENT_PROTOCOL_HEADER) is None
    assert parse_protocol_header("1", header=AGENT_PROTOCOL_HEADER) == 1
    with pytest.raises(ValueError):
        parse_protocol_header("abc", header=AGENT_PROTOCOL_HEADER)


def test_extract_capabilities_agent_and_inference_shapes():
    agent = {"protocol": {"current": 1, "min_cli": 1, "max_cli": 1}, "features": ["durable_events"]}
    inf = {"protocol": {"current": 1, "min_client": 1, "max_client": 1}, "features": ["queue"]}
    assert extract_capabilities(agent)["minimum"] == 1
    assert extract_capabilities(agent)["maximum"] == 1
    assert check_feature(agent, "durable_events")
    assert extract_capabilities(inf)["features"] == ["queue"]


def test_usage_normalization_and_aliases():
    usage = Usage.from_mapping({"prompt_tokens": 10, "completion_tokens": 5, "cached_input_tokens": 2})
    assert usage is not None
    payload = usage.to_dict()
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 5
    assert payload["total_tokens"] == 15
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5


def test_event_and_error_envelope_shapes():
    tenant = TenantContext(tenant_id="t1", project_id="p1")
    event = EventEnvelope(
        event_type="run.started",
        payload={"run_id": "r1"},
        run_id="r1",
        request_id="req1",
        correlation_id="corr1",
        tenant=tenant,
    ).to_dict()
    assert event["protocol_version"] == 1
    assert event["schema_version"] == 1
    assert event["tenant"]["tenant_id"] == "t1"
    assert event["correlation_id"] == "corr1"

    err = ProtocolError(code="INVALID_REQUEST", message="bad", request_id="req1", correlation_id="corr1").to_dict()
    assert err["error"]["code"] == "INVALID_REQUEST"
    assert err["error"]["request_id"] == "req1"
    assert err["error"]["correlation_id"] == "corr1"


def test_header_constants():
    assert AGENT_PROTOCOL_HEADER == "X-Jarvis-Protocol-Version"
    assert INFERENCE_PROTOCOL_HEADER == "X-Jarvis-Inference-Protocol-Version"
    assert REQUEST_ID_HEADER == "X-Request-ID"
    assert CORRELATION_ID_HEADER == "X-Correlation-ID"
    assert IDEMPOTENCY_KEY_HEADER == "X-Idempotency-Key"
