# Guarded semantic serving runtime

This R3 implementation connects candidate retrieval to the existing gateway and
exact cache. It provides a complete functional path for real provider responses,
but current recorded evidence uses simulated fixture answers only. No semantic
precision, live hit rate, real-model latency or cost saving is established.

## Run the reproducible lifecycle demonstration

```bash
python -m pip install -e '.[dev,gateway]'
python scripts/smoke_semantic_gateway.py --output runs/r3-new-smoke.json
```

Create `runs/` first if needed; choose an unused output path. The script starts
local uvicorn processes, makes loopback HTTP requests, exercises all cache modes,
checks SQLite restart and changes source revision. It forces a deterministic
fixture provider and makes no external model calls. Temporary credentials,
SQLite files and authored approval files are removed after the run. Reports
contain hashes, outcomes and counts, not source prompts, answers or credentials.
The committed aggregate is `audit/R3_SEMANTIC_SERVING_SMOKE.json`.

## Serving and approval contract

`GuardedSemanticCache` wraps `ScopedResponseCache`; it does not duplicate the
exact persistence/admission implementation. An eligible exact miss invokes a
trusted candidate resolver. A semantic hit returns the actual original response
and is never readmitted as a new exact alias or granted a refreshed TTL.

| Mode | Non-identical request behavior |
| --- | --- |
| `off` | Generate without cache lookup or admission |
| `exact` | Exact lookup only; no embeddings or approval calls |
| `shadow` | Retrieve/record candidates, then generate fresh; no approval calls |
| `guarded` | Require retrieval thresholds AND explicit actual-answer approval; otherwise generate |

There is no similarity-only serving mode. The lexical encoder is a deterministic
hashing baseline; optional dense retrieval uses a local immutable snapshot and
fails setup if it is missing. Neither encoder is an answer-correctness oracle.
The default threshold and margin are development configuration, not calibrated
safety guarantees. The demo uses an intentionally declared fixture threshold.

An `AnswerBoundApprovals` file has this shape:

```json
{
  "schema_version": "answer-bound-approvals-v1",
  "provenance": "declared_human_review",
  "approvals": []
}
```

Each row must contain `scope_sha256`, `incoming_request_sha256`,
`candidate_request_sha256`, `candidate_response_sha256` and boolean `approved`.
Use `approval_binding` from `src/policy/semantic_runtime.py` with the actual full
requests, actual complete cached response, trusted tenant and `KnowledgeScope`.
The result's local diagnostic `binding` contains these same fingerprints for
candidate decisions. Do not mark a row approved without checking that exact
answer against the incoming request; missing bindings abstain. A changed model
response ID/content/usage also changes the strict full-response binding and
requires a new approval. This is intentionally conservative.

The registry is an explicit decision allowlist, not a learned approval model or
general paraphrase solution. It provides no unseen-pair generalization, and a
file's human-review declaration is not proof of correctness or independence.
`authored_fixture` approvals can run only with the fixture provider through the
environment factory. A programmatically injected approver is trusted application
code and must preserve this distinction. Future broader policies require actual
reviewed development/evaluation evidence rather than simply lowering cosine.

Programmatic approvers return `ReuseDecision` and receive a metered `provider`
callable. Calls made through it, including failed verifier attempts, are included
in result/provider metrics even when the final result is a hit. Out-of-band
calls made by arbitrary operator code cannot be measured; they violate this
integration contract. A remote verifier is not required by the runtime.

## Trusted scope and configuration

Server-owned `KnowledgeScope` binds knowledge-base identity, immutable content
revision, permission/access scope and policy revision. Tenant identity is still
assigned by authentication. Every request field and preceding message remains
in the exact key; only final user text varies during candidate retrieval.
These scope fields are key material and are never forwarded as unsupported
provider parameters. Source references alone do not certify answer correctness.

Set `GATEWAY_CACHE_MODE` and the following operator configuration to enable the
wrapper: `GATEWAY_KNOWLEDGE_BASE`, `GATEWAY_KNOWLEDGE_REVISION`,
`GATEWAY_ACCESS_SCOPE`, and `GATEWAY_SCOPE_POLICY_VERSION`. For shadow or guarded
mode, `GATEWAY_SEMANTIC_ENCODER` selects `lexical` or `dense`; dense also requires
`GATEWAY_EMBEDDING_MODEL` and `GATEWAY_EMBEDDING_REVISION`. No hidden weight
download is performed. Guarded mode additionally requires an approval file at
`GATEWAY_SEMANTIC_APPROVALS`. An empty file registry permits no semantic reuse.

Existing `bypass` and `rules` eligibility policies remain available. The optional
`pinned_context` policy requires `GATEWAY_DOCUMENT_PROMPT_FILE`: incoming messages
must exactly match that fixed evidence-bearing system prompt plus a user
question. It bypasses other history and basic personal/dynamic/creative markers.
The operator must actually supply the referenced static evidence in the prompt;
this heuristic policy does not ground answers by itself or certify staticness.

`update_scope` invalidates existing exact responses before changing a tenant or
deployment scope. An operator restart with a different revision creates different
keys. Old entries may remain until eviction but cannot match the new scope.
Approvals are immutable for the process lifetime; revocation requires a restart
or explicit cache invalidation, and configuration/policy revisions must change
when their meaning changes.

## Bounds, persistence and observations

The existing exact-cache entry/response budgets apply unchanged. Separate
`GATEWAY_SEMANTIC_MAX_ENTRIES` and `GATEWAY_SEMANTIC_MAX_BYTES` bound the semantic
index. Its payload estimate includes request/response JSON, vector values and
scope text; it is not an exact Python-process RAM bound. Approval-file memory
and encoder/model memory are separate operator-controlled costs.

The semantic index is process-local and starts cold after restart. Existing
SQLite exact hits do not repopulate semantic candidates, which avoids silently
refreshing old response lifetimes. Index expiry uses the monotonic clock and a
conservative generation-start TTL; SQLite exact expiry uses its existing
wall-clock contract. Provider duration can shorten the semantic lifetime. Source
invalidation epochs are checked across SQLite users so another process's
invalidation also revokes local candidates. Identical generation misses coalesce
within a process; distinct requests/verifier calls are not globally single-flight.

Headers distinguish `EXACT`, `SEMANTIC` and `PROVIDER`; provider-call counts
include configured verifier work. Fixed-cardinality metrics count exact/semantic
hits, observed/shadow candidates, approval rejection, processing failures and
verifier attempts. Metrics do not expose prompt, tenant or key labels.
`/health` reports configured mode and `semantic_quality_validated: false`;
configuration and local approvals do not certify a quality gate.

Streaming continues to bypass all caches and never populates the semantic
index. Review candidate and answer lineage in R5; default exact behaviour and
functional fixture outcomes must not be promoted to general savings claims.
