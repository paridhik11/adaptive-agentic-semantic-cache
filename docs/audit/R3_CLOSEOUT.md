# R3 closeout: complete cache serving slice

## Decision

The cache lifecycle engineering gate is complete. The semantic-quality and
live-utility gates are still open. R4 bounded agent implementation is next.
This is the first accepted recommendation's functional serving deliverable,
not a passing result for the historical Phase A safety gate.

## What changed and why

Previously the gateway served exact responses while semantic experiments lived
in a separate serial benchmark. `GuardedSemanticCache` now extends eligible
exact misses without replacing exact persistence/admission. It supports cache
off, exact, shadow, and explicit guarded modes. No similarity-only mode is
available for serving. A semantic hit returns the stored actual response and
cannot become a freshly timed exact alias.

Trusted knowledge-base/revision/access/policy scope joins the full request and
tenant key. Context and generation parameters remain exact. The bounded local
candidate index has independent request/answer/vector payload accounting and
an embedding dimension bound. It starts cold on restart; SQLite exact responses
persist. Invalidation epochs revoke candidates across SQLite users and prevent
in-flight stale reuse. Encoding/approval failures preserve the exact path.

The first approver is an immutable actual-answer/pair allowlist. It binds both
full requests, the actual full response, tenant and source/access scope. Unknown
pairs abstain. It does not learn or generalize to unseen paraphrases and is not
an independent correctness judge. The authored demo approvals establish the
mechanism, not quality. Broader semantic approval remains R5 evidence work.
The optional pinned-context eligibility policy is an operator opt-in for fixed
static document evidence; its marker checks are heuristics, not certification.

The HTTP factory exposes the modes with required scope/approval configuration,
reports EXACT/SEMANTIC/PROVIDER tiers and keeps the default exact. Health
explicitly reports semantic quality unvalidated. Provider metrics count actual
new upstream work, including a programmatic approver's metered calls. The
semantic-stage counters describe returned requests; the overall upstream
counter also covers failures that prevent a final response. Unmetered calls
inside arbitrary custom operator callbacks violate the integration contract.
Streaming remains bounded passthrough with mandatory cache bypass.

The accepted roadmap revision is included in this coherent R3 delivery. R3-R6
now map to complete cache path, bounded agent, paired benchmark, then refinement
and publication. R1/R2 and historical failed evaluations remain intact.

## Validation

- Final default offline suite: **593 passed, 23 deselected**, in **113.55 seconds**.
  Real-model/API integration tests remain excluded; offline flags were enabled.
- New runtime module: **37 focused tests passed**. They cover exact/semantic
  lineage, malformed/changed approvals, full request/context/tenant scope,
  revision/permission changes, expiry, revocation during approval, SQLite peer
  invalidation/restart, bounded storage, encoder failure, incomplete outputs,
  provider/verifier work, real local upstream HTTP, and concurrent cold misses.
- Loopback R3 HTTP smoke: **17 completion requests**, **13 fixture upstream
  attempts**; **3 BYPASS, 10 MISS, 4 HIT**. The hits are **3 exact and 1 semantic**.
  These scenarios are functional checks, not a paired savings benchmark.
- The earlier gateway smoke still passes: **57 requests**, **5 fixture upstream
  attempts**, including streaming bypass and a **50-client** barrier burst.
- A fresh environment installed only `deploy/requirements-gateway.txt` and
  reproduced the same R3 smoke aggregate. No ML stack is needed for lexical
  serving. Docker is unavailable on this host, so no new container execution
  result is claimed.
- Code/evidence hashes, environment versions and checks are recorded in
  `R3_VALIDATION.json`. Standard numeric audit and changed-document consistency
  checks passed. An expanded audit found only pre-existing numeric-token gaps
  in older roadmap references/decision history; no new gaps were introduced.
- Working-tree and full-history secret scans are required before publication;
  the final results are bound in `R3_VALIDATION.json`.
- **External model calls made for this phase: 0.**

Commands:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python -m pytest tests/test_semantic_runtime.py -q
python scripts/smoke_semantic_gateway.py --output runs/r3-new-smoke.json
python scripts/smoke_gateway.py --output runs/r3-new-exact-smoke.json
python scripts/check_numeric_tokens.py
```

## Fixes and practical limits

Cache-off health metadata is explicit. An invalid embedding cannot poison the
semantic index, invalidation during approval is rechecked before release, and
policy/source/permission changes cannot reuse the old binding. Semantic hits
never renew the original source lifetime. The smoke explicitly resets inherited
mode configuration so it cannot accidentally enable a different deployment mode.

The approval registry remains narrow and requires actual-answer review. The
new smoke uses a lexical encoder, complete fixture responses and authored
bindings; it cannot establish semantic precision, natural hit rate, task
quality, provider latency or monetary savings. The dense adapter is available
but no new dense inference or model-quality comparison was performed. General
staticness of the eligibility policy and general paraphrase correctness are
not certified. The index is per-process; miss coalescing is process-local and
only for identical requests. Payload budgets are estimates, not process RSS
limits. SQLite wall-clock expiry and the local monotonic semantic TTL differ
under clock adjustments; maintain the host clock.

## Next phase

R4 must add a bounded structured supervisor plan, document search/read tools,
persistent task/checkpoint state and scoped memory, plus human approval/resume.
All model stages use this gateway. Include every stage/tool/verifier in later
cache-on/off accounting. No user inference run is needed to start that work.
