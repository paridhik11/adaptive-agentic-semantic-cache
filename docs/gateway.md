# Runnable response gateway

This milestone exposes the scoped exact-response runtime through HTTP. It uses
complete answers on misses, authenticated tenant namespaces, explicit cache
eligibility, TTL, entry and payload-byte limits, and measured upstream attempts.
Semantic reuse remains disabled. Historical Phase A Gate A failed; this
engineering milestone does not declare Gate A or Gate B passed.

## Run the functional smoke test

From the repository root, with Python 3.11 or newer:

```bash
python -m pip install -e ".[dev,gateway]"
python scripts/smoke_gateway.py
```

The script starts a local uvicorn server with temporary credentials, exercises
authentication, tenant isolation, cache bypass, invalidation, and a barrier-
synchronized burst of 50 client threads, then shuts down. It always forces the
simulated fixture provider, makes no external model calls, and reports its
evaluation mode explicitly. Optional `--output new-report.json` saves a new
report without overwriting existing evidence. This is a functional check, not
a throughput, semantic-accuracy, or real-model savings benchmark.

## Start a server manually

PowerShell example (demo credentials for local use only):

```powershell
$env:GATEWAY_TENANTS_JSON='{"local-demo-credential-change-me":"demo"}'
$env:GATEWAY_ADMIN_TOKEN='local-admin-credential-change-me'
$env:GATEWAY_PROVIDER='fixture'
$env:GATEWAY_CACHE_POLICY='rules'
python -m uvicorn src.api.gateway:app_from_env --factory --host 127.0.0.1 --port 8000 --limit-concurrency 64
```

In another PowerShell terminal, submit the same request twice:

```powershell
$headers=@{Authorization='Bearer local-demo-credential-change-me'}
$body='{"model":"fixture-model","messages":[{"role":"user","content":"What is photosynthesis?"}],"temperature":0,"max_tokens":64}'
Invoke-WebRequest -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post -Headers $headers -ContentType application/json -Body $body
```

The first successful eligible request returns `X-Cache: MISS`; the repeat
returns `HIT`. Every fixture answer begins `SIMULATED FIXTURE:` and carries
`X-Evaluation-Mode: simulated_fixture`. Its token counts use whitespace as a
fixture convention, not a model tokenizer or billing measurement. Bash users
can set the same environment variables with `export NAME='value'`.

## Live provider adapter

Set `GATEWAY_PROVIDER=http`, `UPSTREAM_CHAT_URL` to the complete server-owned
chat-completions endpoint, `UPSTREAM_API_KEY` if required by that provider, and
`GATEWAY_MODELS` to a comma-separated allowed model list. An example endpoint
shape is `https://provider.example/v1/chat/completions`; localhost HTTP is also
supported for a local inference server. Client requests cannot override the
endpoint or pass upstream credentials. HTTPS is required off loopback;
redirects are rejected so credentials cannot follow another destination.

The adapter executes one HTTP attempt per cache miss/bypass. It does not retry.
Timeouts return 504, upstream rate limits 503, and other upstream failures 502.
Raw upstream error bodies and exception details are not exposed. The default
timeout is 60 seconds and maximum upstream response body is 4,194,304 bytes.
This adapter was exercised against a local HTTP fixture server; live external
provider quality, token billing, and latency have not been measured.

## HTTP contract

| Endpoint | Authentication | Behavior |
| --- | --- | --- |
| `POST /v1/chat/completions` | Tenant bearer credential | Supported non-streaming chat request subset; complete provider response |
| `DELETE /v1/cache` | Tenant bearer credential | Invalidate only that credential's tenant; blocks in-flight re-admission |
| `GET /metrics` | Separate administrator bearer credential | Prometheus aggregate metrics, no prompt/key/tenant labels |
| `GET /health` | None | Process health and explicit provider mode; not an upstream readiness probe |

The service derives tenant IDs from the server's token mapping. `X-Tenant-ID`
has no authority. Tokens sharing a configured tenant intentionally share its
namespace. Use independently generated credentials for real deployments and
HTTPS at the serving boundary. Credentials are configured at process startup;
rotation currently requires a restart. `.env` files are not automatically read.

Requests require an enabled `model`, nonempty text `messages`, and one explicit
output limit (`max_tokens` or `max_completion_tokens`). Supported roles are
system, developer, user, and assistant. Supported optional fields are
temperature, top_p, frequency_penalty, presence_penalty, seed, stop, n=1,
stream=false, and response_format of text or json_object. Unknown fields,
streaming, tools, media, unsupported formats, invalid parameters, non-finite
JSON, and duplicate object keys are rejected before provider execution. This
is an OpenAI-style subset, not full protocol compatibility.

The body limit is 65,536 actual bytes, independent of the Content-Length
header. Body reads have a 10-second timeout. Request content encodings other
than identity are rejected. Model and output limits are checked server-side.

Success headers: `X-Cache`, `X-Cache-Tier`, `X-Cache-Reason`,
`X-Provider-Calls`, and `X-Evaluation-Mode`. Tier EXACT denotes a hit; PROVIDER
denotes a miss or bypass. `Cache-Control: no-store` prevents shared HTTP caches
from retaining tenant answers. Errors do not falsely claim HIT/MISS/BYPASS.

## Eligibility and resource limits

`GATEWAY_CACHE_POLICY=bypass` is the default. Opting into `rules` enables the
existing deterministic stability rules without a trained fallback or model
downloads. The gateway restricts that policy to a single user message and
bypasses uncertain, contextual, long, creative, and basic email/phone-pattern
requests. These are conservative heuristics, not certified safety or a general
personal-data detector. Temperature must be explicitly supplied and no greater
than the runtime cap of 0.2. `X-Cache-Mode: bypass` opts out of lookup/admission;
clients cannot force an ineligible request into the cache.

The full request and trusted tenant participate in the exact key. A custom
application policy may support full-history requests through `create_app`,
but must explicitly evaluate their eligibility. Tool calls, incomplete or
refused outputs, malformed responses, and errors are never admitted.

| Setting | Default | Meaning |
| --- | --- | --- |
| `GATEWAY_TTL_SECONDS` | 3600 | Entry lifetime from completed generation |
| `GATEWAY_MAX_ENTRIES` | 1024 | LRU entry bound |
| `GATEWAY_MAX_CACHE_BYTES` | 67108864 | Serialized UTF-8 response payload budget; not an exact process-RAM bound |
| `GATEWAY_MAX_INFLIGHT` | 32 | Concurrent service-processing slots; excess work receives 503 |
| `GATEWAY_MAX_OUTPUT_TOKENS` | 4096 | Maximum explicitly requested output limit |
| `UPSTREAM_TIMEOUT_SECONDS` | 60 | Adapter socket timeout |

Cache coordination is thread-safe within one process. Run one worker for a
shared in-memory cache. Multiple processes have separate caches and counters.
Uvicorn's connection limit additionally bounds queued HTTP work. In-memory
entries and metrics are lost on restart; persistence is outstanding.

## Accounting and verification

`cache_gateway_upstream_attempts_total` counts real adapter invocations,
including failures. Token counters increase only from newly returned upstream
usage; cache hits do not add tokens. Missing or invalid usage is counted
separately. A cached response's `usage` describes its original generation and
must not be interpreted as new consumption. There is no claimed dollar-saving
or token-saving counter inferred from it.

`cache_gateway_processing_seconds` is a histogram by HIT/MISS/BYPASS/ERROR.
It includes cache coordination, upstream execution, and admission inside the
service, but excludes HTTP parsing/serialization and external client transit.
Invalid requests and busy rejections are outside that histogram. Busy rejection
counts have their own counter. `/metrics` includes provider mode explicitly.

Focused tests:

```bash
python -m pytest tests/test_gateway.py tests/test_http_provider.py tests/test_response_cache.py -q
```

See `audit/GATEWAY_CONCURRENT_SMOKE.json` and PB-GATEWAY rows in `FACTS.md` for
fixture evidence. No new sealed evaluation was run. Streaming, durable storage,
scoped invalidation beyond tenant, deployment packaging/dashboard, and broader
real-provider paired workloads remain incomplete. Gate B has not passed.
