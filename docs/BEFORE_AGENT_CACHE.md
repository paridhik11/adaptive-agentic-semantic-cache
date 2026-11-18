# Group 2 — lookup before the documentation agent

2026-10-26. Continues the preserved repository and rebuilt durability commit
`f65ea9b`. That milestone was restored from its saved Library backup, verified
against archive SHA-256 and Git bundle refs. No repeat of missing model studies.

## Result

For explicitly standalone documentation tasks, qualified exact/approved semantic
lookup precedes planning. A hit creates its own durable task, carries source
evidence marked as reused, and requires a new human decision when requested.
No model planning/generation/review or tool dispatch occurs on that hit. A miss
executes the bounded agent. This mode bypasses the inner draft cache so rejected
outer lookups do not silently reuse a draft anyway. Legacy tasks remain unchanged.

Automatic completed-task suggestions are absent from the actual inputs of a
standalone task. Explicit relevant memory remains in all actual stage inputs and
the full cache request. History-reference questions use the contextual workflow.
The key binds model/settings, actual document content revision, instruction/runtime
binding, trusted scope, principal, explicit memory, budgets, contract and durable
task namespace epoch. No model-visible context is hidden just to obtain a hit.

Only workflow-approved answers with read evidence are admitted, after required
human approval or explicit auto-completion. Pending/rejected workflow answers are
not promoted by a subsequent human approval. Human edits are not automatically
treated as model-reviewed answers. Hit approval creates no TTL alias or refresh.
The source invalidation epoch captured before work prevents delayed admission
after invalidation. Cache failures cannot undo an already durable human approval.

The task store increments a per-tenant/principal namespace epoch atomically with
rejection/edit/delete/invalidation. This blocks future reuse even if separate
cache invalidation fails; physical cache cleanup is conservative and best effort.
Pending lookup checks this epoch before returning an answer. An already displayed
draft is not retroactively withdrawn; it still needs its own requested approval.
Task and answer-cache database paths must differ to avoid nested writer locks.

Question eligibility is a conservative heuristic, not a general intent parser.
Guarded whole-answer lookup requires durable candidates and local-only approval;
attempts to use the supplied provider for a remote judge abstain without transport
work. This is a trusted local-policy contract, not isolation of arbitrary code.
No learned policy or general semantic serving has been promoted.

## Work accounting

`WorkLedger` is task-owned and records actual entered provider stages (planning,
generation, review), successful provider-reported usage where present, and failed
attempts. Cache misses are not counted as provider calls. Local observations cover
embedding, candidate search/recovery, combined applicability, tool dispatch and
admission. Returned snapshots are detached. Unknown usage stays missing.

`work_coverage` explicitly excludes classifier/veto internals, provider-internal
retries and complete queue/task-persistence/human-wait latency. Local timings are stage
observations, not a complete end-to-end latency decomposition. Provider failures
retain the legacy aggregate unknown marker; ledger events preserve actual entered
calls. No synthetic zeros are substituted for an uninstrumented stage.

## Functional evidence

**Builder-authored offline fixtures, not independent human ground truth.**
274 targeted regression tests passed in 4.72s. Raw output:
`artifacts/documentation_rebuild/group2/tests.txt`. Earlier development outputs
are preserved; repeated computation adds no observations. Tests include actual
HTTP workbench restart and fresh approvals, scope/context/options isolation,
stale evidence fallback, remote-judge refusal, failures, human-rejection epoch
surviving a cache DB error, delayed invalidation and no source TTL extension.

| Four whole tasks per mode | Actual fixture calls | Planning | Generation | Review |
| --- | --- | --- | --- | --- |
| Off | 12 | 4 | 4 | 4 |
| Exact | 9 | 3 | 3 | 3 |
| Guarded semantic | 6 | 2 | 2 | 2 |

Sequence: seed/approve, restart, exact repeat/approve, approved paraphrase/approve,
fresh different question/approve. Guarded per-task calls are `[3,0,0,3]`.
Approval actions are scripted fixture actions, NOT human labels. Constant-vector
encoder and one answer-bound authored paraphrase approval fixed before the arm;
no natural-language model inference. Inner stage caching is disabled for this
comparison. Full answers, states, events and source hashes:
`artifacts/documentation_rebuild/group2/comparison-final-v2/work.json`.

This shows call avoidance and task lifecycle mechanics only. No independent
natural-language precision, representative coverage, live token/cost reduction
or application latency improvement follows. Repeated comparison snapshots are
the same four fixture tasks, not additional evaluation data.

## Reproduce and try

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_before_agent_cache.py tests/test_durable_semantic_rebuild.py tests/test_semantic_runtime.py tests/test_sqlite_response_cache.py tests/test_response_cache.py tests/test_gateway.py tests/test_streaming_gateway.py tests/test_document_agent.py tests/test_static_document_stage.py tests/test_workbench.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/check_before_agent_work.py --output-dir NEW_EVIDENCE_DIRECTORY
node --check src/api/workbench.js
python scripts/run_demo.py --whole-answer-cache
```

The comparison refuses to overwrite an output directory. No key needed. Demo
uses the bundled offline reference collection and exact whole-answer caching;
general semantic serving is not enabled in its UI. Existing default demo remains
available in a separate profile. The new flag rejects live mode until budget
integration is completed. UI bindings distinguish answer/draft reuse and matched
questions; JavaScript syntax and backend HTTP are checked, not visual browser QA.

## Remaining work and boundaries

Group 3: specialist/tool and scoped task-memory integration. Group 4: full UI QA,
clean installation/deployment, independent actual-answer labels and separately
authorized live utility. Reviewer labels and live budget approval are still needed
before performance claims. Natural-wording acceptance and answer sufficiency
remain unproven; no new dataset or training is justified by these integration checks.

No provider inference, new model/public-data download, training, final-split read
or evaluation, push, merge, main rebase or quota reset. Frozen experiment source,
configs, datasets and historical results are unchanged. The historical parsed-call
comparator remains unavailable. Learned models stay in the preserved opaque Stage B
archive, outside Git. The milestone backup contains Git history and that archive;
it excludes credentials, Python environment and pretrained checkpoints.

## Milestone storage and reproduction scope

One local milestone commit on `integration/documentation-rebuild`; use
`git log -2 --oneline` to see it stacked over the durability reconstruction.
Changed source: DocumentAgent and CompletedDocumentCache, WorkLedger, service and
semantic runtime accounting, opt-in demo/launcher and honest UI bindings. Added
integration tests, whole-task comparison harness and committed evidence. README,
current status, workbench instructions and FACTS are updated.

After committing the coherent group, create the recovery package with:

```bash
python scripts/backup_documentation_product.py --output artifacts/documentation_rebuild/documentation-product-group2.zip
```

This requires a clean working tree and refuses to overwrite a backup. The ZIP
must be kept separately from Git because it includes ignored learned artifacts
inside the opaque prior archive. A saved Library copy supplies recovery without
requiring the laptop or a temporary sandbox link. No push or merge is performed.
