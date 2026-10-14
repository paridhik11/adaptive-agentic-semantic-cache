# Scoped response cache recovery runtime

The historical benchmark harnesses assess candidate decisions and sometimes
simulate upstream generation. `ScopedResponseCache` adds a separate answering
runtime: a miss calls a supplied provider, a complete successful response can
be stored, and a later eligible identical request returns that response.

## Safety contract

- A trusted tenant identity and the complete JSON request form the cache key.
  Every message, model, generation parameter, format and metadata field is
  included. Text is preserved verbatim.
- An explicit eligibility callback runs before lookup, after waiting for
  concurrent work, and before admission. Dynamic requests, revoked decisions,
  policy errors and invalid policy outputs bypass reuse.
- Temperature must be explicitly present and inside the configured cap.
  Provider-dependent defaults are not assumed to be deterministic.
- Tool requests and non-text messages bypass caching. Streaming is explicitly
  unsupported by this runtime rather than buffered incorrectly.
- Only complete successful assistant text responses with the expected choice
  count are admitted. Errors, partial outputs, tool calls, refusals and empty
  content are not cached.
- Expiry is checked on lookup. TTL starts after provider completion. Capacity
  is bounded by both entry count and serialized UTF-8 response payload bytes with least-recently-used eviction. Oversized responses are returned without admission.
- Identical concurrent eligible misses are coordinated. Invalidation prevents
  an already-running provider call from repopulating the invalidated cache.
- Caller/provider mutation cannot change stored requests or responses.

The callback is responsible for stability, personalization and creative-request
policy. This module does not certify the classifier or decide semantic
equivalence. Tenant isolation requires identity supplied by the authenticated
application; trusting an arbitrary request header is insufficient.

## Integration

Import `EligibilityDecision` and `ScopedResponseCache` from `src.policy`.
Supply an eligibility callable that returns a typed decision and a provider
callable that accepts an independent request snapshot and returns the complete
response mapping. Call `cache.process(request, tenant_id=trusted_tenant,
provider=provider)`.

The result includes the response, HIT/MISS/BYPASS status, reason, whether a
response was stored, provider invocation count and total elapsed time. Elapsed
time includes coordination, provider generation on misses and synchronous
admission. The provider adapter must account for its own network retries and
token usage. Usage in a reused response describes the original generation;
it is not fresh consumption.

Provider failures propagate without admitting an answer. No credentials are
required by this component itself; the provider adapter owns authentication.

## Verification

Run `python -m pytest tests/test_response_cache.py -q`.
The original result is ledgered as PA-RUNTIME-TEST and the expanded gateway suite as PB-GATEWAY-TEST in `FACTS.md`. Tests use fixture
providers and establish application invariants, not measured cost savings,
semantic precision or real-provider throughput. Default configuration is
ledgered as PA-RUNTIME-CONFIG.

## Remaining work

An authenticated HTTP gateway and provider adapter now wrap this runtime; see
`gateway.md`. Storage remains process-local. Persistence, cross-process
coordination, streaming and broader real-provider workload evaluation remain
outstanding. Canonical and semantic lookup are deliberately
absent from this runtime until validated. The historical normalizer/router and
their experiment outputs remain intact for comparison; this addition does not
make their unsafe paths safe or revise their measurements.
