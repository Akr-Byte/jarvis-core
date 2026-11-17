# Changelog

## 0.18.1

### Fixed

- **CORE-15**: `InferenceClient._http_error` now parses the OpenAI-compatible
  error envelope `{"error": {"code": ..., "message": ...}}` in addition to the
  legacy FastAPI `{"detail": ...}` shape. Against the jarvis-inference gateway
  (which emits the OpenAI shape), `error_code` was previously always `None`,
  so `QUEUE_TIMEOUT`/`QUEUE_FULL` queue-admission retries never fired.
- **CORE-16**: token accounting is now robust against misbehaving provider
  usage reports. Negative or non-numeric provider fields are sanitized to
  zero, an all-zero usage mapping is treated as absent so the caller's
  estimate fallback applies instead of recording zero tokens, cached input
  tokens are clamped to the input total, and a completed provider call is no
  longer discarded when post-call accounting fails (the estimated usage that
  the reservation validated is committed instead).
- **CORE-17**: `normalize_tool_call` treats the JSON literal `null` arguments
  string (emitted by several local OpenAI-compatible runtimes) as an empty
  argument mapping instead of raising `TypeError` and discarding the tool
  call.
- **CORE-18**: `IdempotencyLedger` is now thread-safe: the completed/in-flight
  check-then-add runs under a re-entrant lock, so two threads racing on the
  same key execute the side effect exactly once. `ProviderPool` mutations are
  lock-protected and it no longer converts `KeyboardInterrupt`/`SystemExit`
  into `RuntimeError("all configured providers failed")`.

## 0.18.0

### Performance

- **CORE-12**: `InferenceClient` now reuses persistent HTTP(S) connections
  instead of opening a fresh TCP+TLS handshake for every request. The new
  dependency-free `jarvis_core.http_keepalive.KeepAliveOpener` keeps a small
  pool (4 idle per host) of validated `http.client` connections: a pooled
  connection is reused only when the previous response was fully drained and
  the server has not half-closed it (non-blocking MSG_PEEK stale detection,
  as in urllib3). A reused connection that still dies with
  `RemoteDisconnected`/`BadStatusLine` before the server could read the
  request is retried exactly once on a fresh connection. Responses expose the
  full `urlopen` surface (`read`, SSE line iteration, `headers.get`,
  context-manager), HTTP >= 400 still raises `urllib.error.HTTPError`, and
  transport failures surface as `URLError` — so retry, error-mapping and
  stream behavior are unchanged. Injected openers keep working; set
  `INFERENCE_KEEPALIVE=0` to restore per-request connections.
- `InferenceClient.close()` releases pooled connections; stateless openers
  ignore it.

### Tests

- `tests/test_http_keepalive.py`: connection reuse, partially-consumed SSE
  stream discards the connection, `will_close` responses are not reused,
  HTTPError paths skip reuse, stale idle detection, one-time reuse-race
  retry, pool bounding + `close()`, and an end-to-end test proving a real
  HTTP/1.1 server sees one connection across four requests.

## 0.17.4

### Reliability

- `TokenLedger.record()` no longer leaks a reservation when commit-time validation raises `BudgetExceeded`; the hold is refunded automatically so one failed `record()` cannot permanently shrink the run or agent budget.
- `InferenceClient` caps every retry sleep (`retry_sleep_cap_seconds`, default 30) and enforces a total retry deadline (`retry_deadline_seconds`, default 60), so an upstream `Retry-After: 86400` can no longer block a caller for a day.
- `InferenceClient` rejects `model` and `stream` in caller options with a clear `TypeError` instead of silently letting options corrupt the transport contract or target model.
- `normalize_tool_call()` now decodes OpenAI's on-the-wire JSON-string tool-call arguments into a mapping instead of raising `TypeError`; malformed JSON and non-object JSON raise a descriptive `TypeError`.
- `InferenceConfig` validates the new retry bounds (`retry_sleep_cap_seconds >= 0`, `retry_deadline_seconds > 0`) at construction time.

### Tests

- Regression coverage: `record()` refund-on-failure, capped retry sleeps, deadline-terminated retries, options transport guard, JSON-string tool-call decoding (valid, empty, malformed, and non-object payloads).

## 0.17.3

### Reliability

- Retry only explicit `QUEUE_TIMEOUT` and `QUEUE_FULL` HTTP rejections, which indicate the gateway did not admit generation.
- Honor `Retry-After` and apply bounded jitter; cap the retry budget at three and keep ambiguous transport, 408, 504, and streaming read failures non-retryable even if an upstream body says otherwise.
- Expose structured inference error codes and retry metadata on `InferenceClientError`.
- Convert structured SSE error frames into `InferenceClientError` and keep all post-admission stream failures non-retryable.

### Tests

- Verify one bounded retry for explicit queue rejection and prove the retry budget cannot loop indefinitely.


## 0.17.2

### Reliability

- Mark non-streaming inference timeouts as non-retryable when request admission is ambiguous, preventing duplicate expensive generations in consumers.
- Preserve retryability for clear connection failures and explicitly retryable HTTP status codes.

### Tests

- Add regression coverage for non-streaming read timeouts.


## 0.17.1

### Fixes

- Treat HTTP 408/504 and ambiguous transport timeouts as non-retryable so consumers do not replay potentially admitted inference generations.
- Ensure stream cancellation closes the underlying HTTP response.

### Tests

- Cover timeout single-attempt behavior, HTTP retry classification and stream cancellation cleanup.


## 0.17.0

### Features

- Add a dependency-free shared inference gateway client for endpoint configuration, bearer authentication, model discovery, capabilities, chat completions, SSE streaming, and embeddings.
- Normalize HTTP and transport failures with status, request ID, and retryability metadata; deliberately avoid automatic retries for potentially expensive inference requests.
- Export the client and configuration types from the top-level package.

### Tests

- Cover configuration precedence, endpoint/auth headers, model discovery, completion payloads, streaming, and retryable HTTP failures.

## 0.16.2

### Features

- Add transport-neutral protocol compatibility primitives for Agent Protocol, Inference Protocol, and event schema versioning.
- Add protocol range overlap and capability validation helpers.


## 0.16.1

### Features

- Add provider-neutral cost-aware local/cheap/frontier tier routing.
- Add bounded failure-budget escalation from local to cheap and frontier tiers.
- Add cheapest eligible model selection using input, cached-input and output cost estimates.
- Add verification requirements for elevated-risk and security-sensitive routing decisions.
- Keep provider SDKs, credentials, pricing configuration and execution adapters outside Core.

### Documentation

- Document the local-first hybrid routing policy and tier ownership.
- Align README and PyPI project documentation with the new routing primitives.

### Maintenance

- Format the new routing implementation and tests with the repository's Black configuration.
- Bump the package version to 0.16.1.

## 0.16.0

### Features

- Centralize empirical route calibration as a provider-neutral Core primitive.
- Add runtime/benchmark observation metadata for quality, cache usage, source and timestamp.
- Add 30-day recency weighting, minimum sample safeguards and configurable quality floors.
- Keep cost, latency, correctness and tool failures in one reusable route utility score.
- Preserve compatibility with existing route observations and downstream consumers.

### Architecture

- `jarvis-core` owns the reusable calibration algorithm and observation contract.
- AI Stack owns model execution, provider telemetry and persistence adapters.
- Jarvis owns real workload definitions and task-level evaluation.
- Downstream repositories must reuse Core calibration rather than maintain a second routing algorithm.

## 0.15.0

### Features

- Add token-efficient agent runtime primitives for bounded context and model usage.
- Add context budgeting across stable instructions, task state, evidence and history.
- Add deterministic context compilation with deduplication and relevance ranking.
- Add an agent state ledger for inspected files, executed commands, failures and decisions to avoid repeated work.
- Add provider-neutral usage and cost accounting, including cache-aware token accounting.
- Add route budgets and adaptive model routing based on task complexity, risk and uncertainty.
- Add request token estimation before model execution.

### Architecture

- Keep token-efficiency policy and accounting provider-neutral in Core.
- Keep provider SDKs, transports, credentials and deployment-specific routing outside Core.
- Reuse existing context compaction, artifact handling, escalation and reliability primitives instead of duplicating them.
- Enable consumers to combine deterministic tooling, local models and cloud models while minimizing unnecessary model context.

### Documentation

- Document the token-efficient runtime capabilities and release in the README.

## 0.14.0

### Features

- Add provider-neutral background-process handles and lifecycle states.
- Add live steering and checkpoint contracts for pause/resume/cancel/redirect/rewind controls.
- Add deterministic checkpoint identities and a helper for translating real execution observations into Core evidence.

### Architecture

- Keep execution meaning and evidence semantics in Core while leaving process spawning, OS isolation, persistence and provider integrations to Core consumers.
- Treat real command and test observations as evidence inputs; model prose remains non-authoritative.

## 0.13.0

### Features

- Add provider-neutral capability and approval primitives for shared agent runtimes.
- Add portable sandbox requirements and a consumer-owned `SandboxExecutor` protocol.
- Keep platform enforcement in consumers while making capability, approval and isolation semantics common Core contracts.

### Architecture

- Establish Core as the common agent brain: contracts and policy semantics live here; CLI/server execution adapters remain consumer-owned.
- Avoid duplicating execution-state semantics: the existing autonomous execution lifecycle remains the canonical shared state machine.

## 0.12.0

### Features

- Extend the provider-neutral model boundary with reusable usage, tool-call and response normalization helpers.
- Export the normalization helpers from the public Core API so independent agent applications can share canonical model semantics.
- Keep the provider contract dependency-free and suitable for any application or integration.

### Architecture

- Core owns provider-neutral contracts and reusable runtime primitives only.
- Provider SDKs, HTTP transports, credentials, persistence, user interfaces, deployment and operating-system enforcement remain outside Core.
- Core documentation is standalone and does not depend on or name downstream/private applications.
