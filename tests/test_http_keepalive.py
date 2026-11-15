"""CORE-12: persistent HTTP connections for the inference gateway.

Unit tests use a stubbed `http.client.HTTPConnection` factory; the final test
proves real TCP reuse end-to-end against a live HTTP/1.1 keep-alive server.
"""

from __future__ import annotations

import http.client
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from jarvis_core import InferenceClient, InferenceClientError, InferenceConfig
from jarvis_core.http_keepalive import KeepAliveOpener, _is_connection_dropped


class FakeSock:
    """Non-socket stand-in that satisfies the stale-detection probe."""

    def __init__(self, *, present=True, closed=False):
        self.present = present
        self.closed = closed
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value

    def fileno(self):
        return 3 if self.present else -1

    def recv(self, _size, _flags=0):
        if self.closed:
            return b""
        raise BlockingIOError


class FakeHTTPConnection:
    """Minimal `http.client.HTTPConnection` double driven by a script.

    When the script is exhausted (a pooled connection is reused), the most
    recent response factory is served again.
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[tuple] = []
        self.sock = FakeSock()
        self.closed = False
        self.timeout = None

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, dict(headers)))

    def getresponse(self):
        if self.script:
            self._current = self.script.pop(0)
        return self._current(self)

    def close(self):
        self.closed = True

    def isclosed(self):
        return self.closed


class FakeRawResponse:
    """Minimal `http.client.HTTPResponse` double."""

    def __init__(self, *, status=200, body=b"", lines=None, will_close=False):
        self.status = status
        self.reason = "OK" if status < 400 else "Error"
        self.body = body
        self.lines = lines if lines is not None else [b""]
        self.will_close = will_close
        self.headers = http.client.HTTPMessage()
        self.length = len(body) or None
        self._iter = iter(self.lines)
        self.closed = False

    def read(self, size=-1):
        if size is None or size < 0:
            data, self.body = self.body, b""
            self.length = 0
            return data
        data, self.body = self.body[:size], self.body[size:]
        self.length = len(self.body) or 0
        if len(data) < size:
            self.length = 0
        return data

    def close(self):
        self.closed = True

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._iter)
        if not line:
            self.length = 0
        return line


def _opener_with_connections(connections, **kwargs):
    created: list[FakeHTTPConnection] = []

    def factory(_key, _timeout):
        connection = connections.pop(0)
        created.append(connection)
        return connection

    return KeepAliveOpener(connection_factory=factory, **kwargs), created


def _config(**kwargs):
    defaults = {"base_url": "http://inference.test:8080/v1", "api_key": "secret"}
    defaults.update(kwargs)
    return InferenceConfig(**defaults)


def test_second_request_reuses_the_pooled_connection():
    ok = lambda conn: FakeRawResponse(body=b'{"data": []}')
    first, second = FakeHTTPConnection([ok]), FakeHTTPConnection([ok])
    opener, created = _opener_with_connections([first, second])

    client = InferenceClient(_config(), opener=opener)
    client.list_models()
    client.capabilities()

    assert len(created) == 1
    assert first.requests[0][0] == "GET"
    assert first.requests[1][1] == "/v1/capabilities"
    assert first.requests[1][3]["Authorization"] == "Bearer secret"


def test_partially_consumed_stream_closes_the_connection():
    # SSE iterator stops at [DONE] before draining the body -> not reusable.
    models = lambda conn: FakeRawResponse(body=b'{"data": []}')
    stream = lambda conn: FakeRawResponse(
        lines=[b'data: {"a": 1}\n', b"data: [DONE]\n", b"data: more\n"]
    )
    first, second = FakeHTTPConnection([stream]), FakeHTTPConnection([models])
    opener, created = _opener_with_connections([first, second])

    client = InferenceClient(_config(), opener=opener)
    events = list(
        client.stream(model="m", messages=[{"role": "user", "content": "hi"}])
    )
    assert len(events) == 1

    client.list_models()

    assert len(created) == 2
    assert second.requests[0][0] == "GET"


def test_will_close_response_is_not_reused():
    models = lambda conn: FakeRawResponse(body=b'{"data": []}', will_close=True)
    first, second = FakeHTTPConnection([models]), FakeHTTPConnection([models])
    opener, created = _opener_with_connections([first, second])

    client = InferenceClient(_config(), opener=opener)
    client.list_models()
    client.list_models()

    assert len(created) == 2


def test_http_error_raises_real_httperror_and_skips_reuse():
    failing = lambda conn: FakeRawResponse(status=503)
    ok = lambda conn: FakeRawResponse(body=b'{"data": []}')
    first = FakeHTTPConnection([failing])
    second = FakeHTTPConnection([ok])
    opener, created = _opener_with_connections([first, second])

    client = InferenceClient(_config(queue_retries=0), opener=opener)
    with pytest.raises(InferenceClientError, match="HTTP 503"):
        client.list_models()

    client.list_models()
    assert len(created) == 2


def test_stale_idle_connection_is_dropped_not_reused():
    ok = lambda conn: FakeRawResponse(body=b'{"data": []}')
    first, second = FakeHTTPConnection([ok]), FakeHTTPConnection([ok])
    opener, created = _opener_with_connections([first, second])

    client = InferenceClient(_config(), opener=opener)
    client.list_models()
    # Server half-closed the idle socket between calls.
    first.closed = True
    first.sock = FakeSock(closed=True)

    client.list_models()

    assert len(created) == 2


def test_reused_connection_race_retries_once_on_a_fresh_connection():
    def ok(conn):
        return FakeRawResponse(body=b'{"data": []}')

    def dead(conn):
        raise http.client.RemoteDisconnected("No response")

    first = FakeHTTPConnection([ok, dead])
    second = FakeHTTPConnection([ok])
    opener, created = _opener_with_connections([first, second])

    client = InferenceClient(_config(), opener=opener)
    client.list_models()  # Populates the pool.
    client.capabilities()  # Reuse dies; retry once on the fresh connection.

    assert len(created) == 2
    assert second.requests[0][0] == "GET"
    assert second.requests[0][1] == "/v1/capabilities"


def test_transport_error_surfaces_as_urlerror_not_httpexception():
    def dead(conn):
        raise http.client.BadStatusLine("x")

    connection = FakeHTTPConnection([dead])
    opener, _created = _opener_with_connections([connection])

    client = InferenceClient(_config(), opener=opener)
    with pytest.raises(InferenceClientError, match="Could not reach"):
        client.list_models()


def test_idle_pool_is_bounded_and_close_flushes_it():
    connections = []
    ok = lambda conn: FakeRawResponse(body=b'{"data": []}')

    def factory(_key, _timeout):
        connection = FakeHTTPConnection([ok])
        connections.append(connection)
        return connection

    opener = KeepAliveOpener(max_idle=1, connection_factory=factory)
    request = Request("http://inference.test:8080/v1/models")

    # Check out three connections concurrently, then read+release them
    # newest first: only the newest fits in the 1-slot idle pool.
    r3 = opener(request)
    r2 = opener(request)
    r1 = opener(request)
    r3.read(-1)
    r2.read(-1)
    r1.read(-1)
    r3.close()
    r2.close()
    r1.close()

    assert len(opener._idle[("http", "inference.test", 8080)]) == 1
    assert not connections[0].closed  # First-released connection stays pooled.
    assert connections[1].closed and connections[2].closed  # Overflow closed.

    opener.close()
    assert not opener._idle  # close() flushed every pooled connection.


def test_connection_dropped_detector_uses_socket_state():
    assert _is_connection_dropped(FakeHTTPConnection([])) is False
    connection = FakeHTTPConnection([])
    connection.sock = None
    assert _is_connection_dropped(connection) is True
    connection.sock = FakeSock(closed=True)
    assert _is_connection_dropped(connection) is True


# --- end-to-end: real HTTP/1.1 keep-alive server ----------------------------


def test_real_server_sees_one_connection_for_many_requests():
    connection_count = {"count": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            with lock:
                connection_count["count"] += 1
            super().setup()

        def do_GET(self):
            body = b'{"data": [{"id": "m1"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = KeepAliveOpener()
        client = InferenceClient(
            InferenceConfig(f"http://127.0.0.1:{server.server_port}/v1"),
            opener=opener,
        )
        for _ in range(4):
            models = client.list_models()
            assert models == [{"id": "m1"}]

        # Four requests, one TCP connection.
        assert connection_count["count"] == 1
    finally:
        opener.close()
        server.shutdown()
        server.server_close()
