"""Dependency-free HTTP keep-alive for the Jarvis inference gateway.

`urlopen` opens a fresh TCP (and TLS) connection for every request, which
wastes a full handshake per call and adds latency to every agent turn. This
module provides an `urlopen`-compatible opener backed by a small pool of
persistent `http.client` connections, so `InferenceClient` consumers get
connection reuse without any behavior change.

Design notes:

- The pool only returns connections that are provably clean: the previous
  response was fully drained (`data: EOF` was seen) and the server did not
  ask to close. Anything else is closed and discarded.
- Reused idle connections are checked for a server-side half-close with a
  non-blocking MSG_PEEK before use (the same stale-detection strategy as
  urllib3). A reused connection that still dies with RemoteDisconnected /
  BadStatusLine / ConnectionResetError is retried exactly once on a fresh
  connection, because the server closed the idle socket before it could
  read the request.
- Responses expose the `urlopen` surface the client relies on: context
  manager, ``read(n)``, line iteration (SSE), ``headers.get``. HTTP >= 400
  raises a real ``urllib.error.HTTPError`` so the existing error mapping in
  ``InferenceClient._http_error`` keeps working unchanged.
- Only the Python standard library is used; no new dependency for Core.
"""

from __future__ import annotations

from typing import Any

import http.client
import select
import socket
import ssl
import threading
from collections import deque
from os import getenv
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_MAX_IDLE_CONNECTIONS = 4

_STALE_RECV_ERRORS = (http.client.RemoteDisconnected, http.client.BadStatusLine)


def _is_connection_dropped(connection: http.client.HTTPConnection) -> bool:
    """Return True when the server has closed its side of an idle socket."""
    sock = connection.sock
    if sock is None:
        return True
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        return bool(sock.recv(1, socket.MSG_PEEK) == b"")
    except (BlockingIOError, InterruptedError):
        # No data and not closed; MSG_PEEK raced with the readiness report.
        return False
    except ssl.SSLWantReadError:
        # TLS layer needs more handshake data before it can peek; treat the
        # connection as alive and let the normal request path surface issues.
        return False
    except Exception:
        # A socket we cannot probe cannot be validated for reuse.
        return True


class _KeepAliveResponse:
    """`urlopen`-shaped response bound to one pooled connection.

    The connection is recycled into the pool on context exit only when the
    response body was fully drained; a partially consumed response (an SSE
    stream that stopped at ``[DONE]``) closes the connection instead, since
    unread bytes would corrupt the next response.
    """

    def __init__(self, response, connection, opener, key) -> None:
        self._response = response
        self._connection = connection
        self._opener = opener
        self._key = key
        self._drained = response.length == 0
        self.headers = response.headers

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            data = self._response.read()
            self._drained = True
            return data
        data = self._response.read(size)
        if len(data) < size:
            self._drained = True
        return data

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._response)
        if not line:
            self._drained = True
        return line

    def __enter__(self) -> "_KeepAliveResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        response, connection = self._response, self._connection
        try:
            if response.will_close or not self._drained:
                # The socket is gone or still holds unread body bytes; the
                # connection can never serve another clean response.
                connection.close()
            else:
                response.close()
                if _is_connection_dropped(connection):
                    connection.close()  # Server already dropped it.
                else:
                    self._opener._release(self._key, connection)
        finally:
            self._response = None
            self._connection = None


class KeepAliveOpener:
    """Reusable persistent-connection opener compatible with `urlopen`.

    `opener(request, timeout)` returns a response with the `urllib` surface
    used by `InferenceClient`. Redirects are intentionally not followed —
    the Jarvis inference API never redirects, and following one from a
    shared client would hide misconfiguration.
    """

    def __init__(
        self,
        *,
        max_idle: int = DEFAULT_MAX_IDLE_CONNECTIONS,
        default_timeout: float | None = None,
        connection_factory=None,
    ) -> None:
        self._max_idle = max_idle
        self.default_timeout = default_timeout
        self._connection_factory = connection_factory or self._default_factory
        self._idle: dict[tuple, deque] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _default_factory(key, timeout):
        scheme, host, port = key
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout)
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def _acquire(self, key: tuple, timeout: float | None):
        with self._lock:
            idle = self._idle.get(key)
            while idle:
                connection = idle.popleft()
                if not _is_connection_dropped(connection):
                    return connection, True
                # Stale (server half-closed while idle): drop silently.
                connection.close()
        return self._connection_factory(key, timeout), False

    def _release(self, key: tuple, connection) -> None:
        with self._lock:
            idle = self._idle.setdefault(key, deque())
            if len(idle) >= self._max_idle:
                connection.close()
                return
            idle.append(connection)

    def close(self) -> None:
        with self._lock:
            for idle in self._idle.values():
                while idle:
                    idle.popleft().close()
            self._idle.clear()

    def __call__(self, request: Request, timeout: float | None = None):
        parts = urlsplit(request.full_url)
        if parts.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported inference URL scheme: {parts.scheme}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        key = (parts.scheme, parts.hostname, port)
        effective_timeout = timeout if timeout is not None else self.default_timeout
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        method = request.get_method()
        body = request.data
        headers = dict(request.header_items())
        headers.setdefault("Connection", "keep-alive")
        headers.setdefault("Accept-Encoding", "identity")

        attempt = 0
        while True:
            connection, reused = self._acquire(key, effective_timeout)
            try:
                connection.timeout = effective_timeout
                if connection.sock is not None:
                    connection.sock.settimeout(effective_timeout)
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
            except _STALE_RECV_ERRORS as exc:
                # A reused idle connection lost the race with the server's
                # keep-alive timeout; the request was never processed, so a
                # single fresh-connection retry is safe. A brand-new
                # connection failing the same way is a real error.
                connection.close()
                if reused and attempt == 0:
                    attempt += 1
                    continue
                raise URLError(str(exc)) from exc
            except (OSError, http.client.HTTPException, TimeoutError) as exc:
                connection.close()
                if isinstance(exc, TimeoutError):
                    raise
                raise URLError(str(exc)) from exc
            if response.status >= 400:
                # Error bodies are read by the caller via HTTPError; this
                # connection is never reused.
                connection.close()
                raise HTTPError(
                    request.full_url,
                    response.status,
                    response.reason,
                    response.headers,
                    response,
                )
            return _KeepAliveResponse(response, connection, self, key)


def default_opener() -> Any:
    """Return the opener `InferenceClient` uses when none is injected.

    Keep-alive is on by default; setting ``INFERENCE_KEEPALIVE=0`` (or
    ``false``/``off``) restores per-request `urlopen` connections.
    """
    enabled = getenv("INFERENCE_KEEPALIVE", "1").strip().lower()
    if enabled in {"0", "false", "off", "no"}:
        return urlopen
    return KeepAliveOpener()
