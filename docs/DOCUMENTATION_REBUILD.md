# Documentation product — rebuilt durability group

2026-10-26. Reconstruction from preserved Stage B head `847d3f9`, under the
user's instruction to continue the bounded documentation product. Original
`f9fcc9b` and its backup remain unavailable. This is new implementation and
new evidence, not recovery of that exact commit. No new training experiment.

## Implemented

Opt-in durable semantic candidates live beside exact answers in the existing
SQLite database. Every new exact admission receives a random source token;
identical bytes replacing an old row do not inherit its token. Candidates bind
to that token, tenant, full request options/history, knowledge/access scope,
encoder identity, approver identity and retrieval thresholds. Expiry is the
source's persisted absolute time; hits create no aliases or TTL extensions.

Recovery joins candidates with their current source. After approval, a transaction
checks candidate identity, source token, tenant, expiry and invalidation epoch
together. This is the decision's linearization point, not a promise to revoke an
already returned answer. Expiry, eviction/replacement and invalidation revoke reuse.
Invalidation conservatively clears all candidates even for tenant-specific exact
removal. Unchanged candidate objects retain identity during concurrent recovery.
SQLite connections are short-lived and not shared across worker threads.

Bounds count logical request/vector/source payload bytes and rows globally across
namespaces; they do not cap SQLite database/journal disk size. Cleanup occurs on
candidate admission/recovery. Failed candidate processing abstains; generation
remains available. Default process-local behavior is preserved. Persistence does
not authorize reuse: guarded mode still needs actual-answer approval.

## Deployment

Configure `GATEWAY_CACHE_DB`, explicit `GATEWAY_CACHE_VERSION`, knowledge scope,
`GATEWAY_CACHE_MODE=shadow` and `GATEWAY_PERSIST_SEMANTIC=true` to observe durable
candidates. The new flag defaults to false. Guarded mode retains existing approval
requirements; fixture approvals cannot enable live serving. No deployment settings
or serving policy were changed by this rebuild.

**Stop old workers before opening an existing database with this version.** The
schema adds a nullable source token. Old writers use positional inserts and are
incompatible with the migrated schema. Alternatively use a fresh database path
and deployment version. Changed namespaces cannot open the old path. Legacy exact
rows remain readable but are not automatically promoted to semantic candidates.

## Evidence and reproduction

Evidence type: builder-authored deterministic functional fixtures, not independent
human truth or a population safety/accuracy estimate. Tests cover restart, TTL,
eviction/replacement/invalidation during approval, candidate deletion, concurrent
paraphrases, tenant/options/history/policy/scope isolation, bounds, legacy schema,
incomplete answers, storage failure, bypass and gateway configuration.

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_durable_semantic_rebuild.py tests/test_semantic_runtime.py tests/test_sqlite_response_cache.py tests/test_response_cache.py tests/test_gateway.py tests/test_streaming_gateway.py tests/test_document_agent.py tests/test_static_document_stage.py tests/test_workbench.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/check_documentation_rebuild.py --output NEW_RECEIPT.json
```

Result: **242 passed in 3.51s**, raw `artifacts/documentation_rebuild/group1-tests.txt`.
Earlier narrower run: 125 passed; repeated tests add no independent observations.
The receipt script refuses to overwrite an output file.

| Mode | Requests | Actual fixture provider invocations |
| --- | --- | --- |
| Off | 4 | 4 |
| Exact | 4 | 3 |
| Guarded semantic | 4 | 2 |

Receipt: `artifacts/documentation_rebuild/group1-work.json`. Sequence: seed,
restart, exact repeat, explicitly approved paraphrase, fresh different question.
Constant-vector test encoder and answer-bound authored approvals; no pretrained
inference. Full constructed answers, workload and source hashes are retained.
No real tokens, cost, live latency or general coverage measured. Old 243-test
evidence is neither reused nor added to these results.

## Still unfinished

- Before-agent qualified lookup and complete local/remote stage accounting (Group 2).
- Specialist/task-memory integration, unified interface and deployment.
- Useful unseen natural wording and independently reviewed answer sufficiency.
- Representative live savings. Live calls remain unauthorized.
- Missing follow-up artifacts/reports; no results from them are claimed here.

Existing training data, labels, thresholds, locked final records and historical
results were untouched. Historical parsed-call comparator remains unavailable.
No model downloads, provider calls, push, merge or final evaluation.

The milestone backup contains a Git bundle and the existing opaque Stage B
complete archive without inspecting its final records. It excludes pretrained
weights, secrets, environments and application databases. Keep the backup as well
as source: ignored learned weights are not preserved by Git alone.
