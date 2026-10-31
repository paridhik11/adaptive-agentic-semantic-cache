# Group 3 — bounded document roles and persistent task context

2026-10-31. One integration group on `integration/documentation-rebuild`, based on
`06a996c`. Source restored from its saved checkpoint and verified before editing.
No new training, pretrained model download, public dataset or live inference.

## Behavior

Opt-in orchestration routes the existing durable document states through planner,
source researcher, answer writer and reviewer roles. Planner/writer/reviewer are
separate model invocations with role-specific instructions; they can share the
same configured model. The source researcher dispatches the model-selected bounded
plan through typed, read-only `search(query)` and `read(document_id)` handlers.
Unknown tools, extra fields and invalid types/lengths are rejected before handlers.
Routing is a deterministic supervisor over the state machine; it adds no provider
calls. This is not LangGraph or a new independent multi-model agent platform.
The older generic `AgentWorkflow` demonstrator remains separate.

Contextual tasks retrieve from compatible completed tasks with approved workflow
review and read evidence. Required human decisions must have completed. Human
edits or approval of a failed model review do not silently become reviewed memory.
Cache-copy tasks do not multiply source memory. Scope includes tenant, principal,
source revision, instruction/runtime binding, generation settings, encoder policy
and revocation epoch. Prior answers are explicitly untrusted context; current read
documents remain the factual evidence. A snapshot is persisted at task creation
and used consistently in actual model inputs, so full stage cache keys also bind it.

The local vector index persists in the task SQLite database. Missing vectors are
built lazily from completed task questions. Answer fingerprints prevent reuse of
an obsolete vector. Retrieval considers at most the latest 64 task rows, returns
at most 3 suggestions and uses a fixed cosine cutoff .15. The index has global
logical bounds of 256 rows and 2 MiB of serialized vectors; source task storage
and physical SQLite/journal size are not bounded by that index limit. Deletion
removes orphan vector records. No approximate-index or distributed-memory claim.

The demo uses the existing lexical hashing encoder. The interface supports an
explicit identified local encoder, but no dense memory model was loaded or
validated in this group. These settings are engineering defaults, not calibrated
natural-language thresholds. Unknown/failed retrieval yields an empty snapshot,
records failure and continues the source workflow; it does not fall back secretly
to recent answers. Cold indexing occurs under the bounded task transaction and
can hold its SQLite writer lock; concurrent production throughput is unmeasured.

Standalone tasks skip task-memory lookup and retain whole-answer cache-first
behavior. They do not inherit hidden history. Context-dependent tasks execute
their model workflow; similarity to a memory does not authorize answer reuse.
Epoch changes after rejection, edits, deletion or explicit clearing stop pending
tasks with stale snapshots before another stage. Already displayed drafts still
require their own requested human decisions.

## Management and demo

```bash
python scripts/run_demo.py --whole-answer-cache --task-memory
```

This selects a separate offline profile. The flag cannot enable live execution.
The existing session capability and origin/host protections cover GET `/api/memory`
and DELETE `/api/memory`. Listing performs no inference; clearing retires future
context and conservatively cached answers while retaining task audit history.
Clearing is not a claim that every answer byte is physically deleted from the
audit database. Full memory controls and browser visual QA remain Group 4 work.

## Evidence and reproduction

Builder-authored functional fixtures; scripted approvals are NOT human labels.
293 targeted regression tests passed in 12.94s:
`artifacts/documentation_rebuild/group3/tests.txt`. Earlier 291-test development
output is retained separately; reruns add no independent observations. Initial
test collection failed because a shared fixture was imported without its package
prefix; corrected to `tests.test_document_agent`, without changing product policy.

Tests exercise source/principal/tenant/runtime isolation, unapproved/edited-task
exclusion, restart recovery, frozen actual inputs, explicit role traces, stale
snapshot revocation, memory failure fallback, authenticated HTTP list/clear,
typed tool rejection and unchanged zero-call standalone hits.

New integration receipt: `artifacts/documentation_rebuild/group3/work.json`.

| Constructed task | Actual scripted provider invocations |
| --- | --- |
| Fresh standalone source task | 3 |
| Exact repeat after restart | 0 |
| Contextual follow-up using retrieved task snapshot | 3 |

Total 6; no extra model call for routing or memory retrieval. First retrieval
computes 2 lexical embeddings (source+query); after restart it reuses the source
vector and computes 1 query embedding. This establishes persistence mechanics,
not memory usefulness, natural paraphrase precision, token/cost savings or
representative latency. Complete states, events and source hashes are retained.

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_document_orchestration_memory.py tests/test_before_agent_cache.py tests/test_durable_semantic_rebuild.py tests/test_semantic_runtime.py tests/test_sqlite_response_cache.py tests/test_response_cache.py tests/test_gateway.py tests/test_streaming_gateway.py tests/test_document_agent.py tests/test_static_document_stage.py tests/test_workbench.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/check_document_memory.py --output NEW_RECEIPT.json
node --check src/api/workbench.js
git log -3 --oneline
```

Receipt generation refuses overwrites. WorkLedger includes local memory embedding
and search attempts/failures; stage observations are not end-to-end latency.

## Remaining gates and preserved scope

Group 4 still needs product/visual QA, clean installation and deployment work,
independent actual-answer review and separately authorized budgeted live utility.
General semantic answer serving stays disabled by default. No accuracy or broad
memory benefit claim follows. No threshold tuned on spent data, final records
opened/evaluated, training started, provider calls made, quota reset, push, merge
or main rebase. Historical study artifacts remain unchanged. Learned weights stay
in the preserved opaque Stage B archive, not Git.

This milestone is committed locally as a coherent group. Git history is preserved
automatically across workspace resets; no recovery download is required from the
user. Local commits are not a claim that GitHub has been updated.
