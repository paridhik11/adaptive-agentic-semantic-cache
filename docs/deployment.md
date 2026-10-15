# Gateway deployment and operations

## Local container

The image installs only gateway dependencies, copies application source, runs as
an unprivileged user, and starts one worker with bounded concurrency. The Compose
example uses the deterministic fixture provider and exposes a loopback-only port.
It is a local development deployment, not an Internet-facing production setup.

Set `GATEWAY_TENANTS_JSON` to a JSON object mapping independently generated bearer
tokens to tenant IDs, and `GATEWAY_ADMIN_TOKEN` to a separate token. Generate tokens
locally with Python's `secrets.token_urlsafe`; do not commit them or put them in
command history. Then:

```bash
docker compose up --build -d
docker compose logs --tail=50 gateway
```

Visit `http://127.0.0.1:8000/dashboard`. Enter the admin token to fetch metrics.
The page itself is a public static shell; metrics require admin authentication.
No prompts, tenant names, or credentials are included in metric labels. The page
uses no external scripts, local storage or third-party requests. Clear its token
before leaving a shared computer.

The named volume stores the SQLite database. `docker compose down` preserves it;
`docker compose down -v` intentionally deletes it. Change the database path and
cache version when changing provider/policy/model semantics. See
[gateway.md](gateway.md) for expiry, plaintext storage, namespace guards and
invalidation behavior. Response payload limits do not bound physical disk usage.

For live serving, change the provider configuration following the gateway guide;
the committed Compose file deliberately remains fixture-only. Use a TLS reverse
proxy and appropriate authentication, key rotation and admission/rate limits
before exposing it beyond loopback. Gateway processing slots are not per-tenant
quotas. `/health` checks the process, not upstream availability. The image's base
tag and transitive dependency resolution are not a fully reproducible image lock;
a validated release image should be retained by digest.

## Operational interpretation

The dashboard displays per-worker lifetime counts: requests, exact hits, provider
attempts/errors, newly reported prompt/completion tokens and missing usage. A
fixture banner prevents those numbers being read as LLM consumption. Counters
reset after restart. Multiple workers require scraping each worker separately;
load-balancing a single `/metrics` endpoint does not aggregate them.

Prometheus histogram samples cover service processing, including provider time
and lock waiting, excluding request parsing/network transit. Successful cache
hits do not imply correct answers. Use the answer-review benchmark for quality.

## Validation status

Docker is unavailable in the implementation workspace. The GitHub Actions
[container smoke job](https://github.com/paridhik11/adaptive-agentic-semantic-cache/actions/runs/38799899946/job/116645894452)
passed on commit `2bd019453b59044d2e3731b4f3d45adfaf7f3710`: image build,
read-only container startup with a writable SQLite volume, authenticated fixture
MISS/HIT, health and dashboard delivery. This is a functional container check,
not a production load or Compose-specific deployment certification. Other CI
jobs were still running when this result was recorded. The Python wheel also
passed the separate local check outside the source directory.

Streaming, distributed storage/coordination, per-tenant rate limits, production
identity integration, and deployment-specific load/SLO validation remain work.

The dashboard and review-page JavaScript pass syntax checks; HTTP/authentication
checks cover dashboard delivery. Browser-rendered interaction testing could not
run locally because the browser executable download failed. This is separate
from passing backend tests and should not be reported as browser validation.
