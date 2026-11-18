# R6 scoped agent-stage redesign closeout

Status: preregistered offline engineering criterion PASS. User authorized go on
2026-10-19. The original R6_AGENT_CACHE_PREREG.md remains unchanged, including its
original proposed-status header. Its committed criteria were accepted before code
changes; no gate/threshold was changed after viewing the result.

## Implementation

DocumentAgent has an explicit static_document_stage option, default off. Planning
still sees scoped task memory. The opt-in factual draft prompt excludes unrelated
completed-task suggestions, while including original question, full tool evidence,
source content/version, trusted scope, instructions/generation settings, explicit
relevant memory, prior draft and feedback. Exact keys still cover the complete
actual provider request. No model-visible inputs are omitted only from a key.

An explicit history reference retains prior-task memory and bypasses static draft
reuse. This lexical fallback is not exhaustive ambiguity detection. The trusted
caller must restrict opt-in to fixed-reference factual tasks and include every
answer-relevant memory/constraint; ambiguous tasks retain full context or bypass.
No safe automatic memory-relevance classifier is claimed. Planning/review remain
uncached, semantic thresholds/approvals unchanged, and human approval remains a
separate uncached decision. No tool-result memoization was added.

Scope/instruction/generation bindings persist in checkpoints and scope memory.
Old unbound checkpoints cannot supply memory after a permission/context change.
Default legacy behaviour remains available. CLI exposes the opt-in flag without
enabling live caching by default. See ../document_agent.md for the contract.

## Frozen, single-run result

R6_STAGE_FREEZE.json binds source/runtime/test/workload hashes, the original
protocol, environment and passing regression logs BEFORE the criterion run.
It records a dirty source snapshot based on the prior audit commit, not a
pretended pre-evaluation clean commit. The original R5 baseline archive was
verified against its saved hash and internal manifest; its actual answer texts
were available for comparison. No baseline reconstruction fallback was needed.

| Arm | Tasks | Fixture provider invocations | Actual local tool invocations |
| --- | ---: | ---: | ---: |
| Redesigned contract, cache off | 44 | 132 | 88 |
| Same contract, exact cache on | 44 | 110 | 88 |

All task statuses were completed and corresponding final answer texts were
byte-identical across arms and to the retained original R5 raw baseline.
Planner calls: 44 per arm. Reviewer calls: 44 per arm. Draft calls: 44 off,
22 exact, with 22 exact draft hits. Call counts reconcile against an outer provider
counter, not just the agent's own trace. Tool dispatches are independently counted.
No omitted stage, hidden retry, unknown call count or missing task produced the gain.
The evaluator ran once under the frozen criterion; all conjunctive checks passed.
See R6_AGENT_STAGE_RESULT.json and R6_STAGE_VALIDATION.json.

Historical R5 remains 132 versus 132; those artifacts and the preregistration were
not rewritten. R6 demonstrates 22 fewer simulated provider invocations on the
specified constructed replay. No application-level percentage saving is promoted.

## Validation and limits

- Final focused tests: 60 passed, covering changed question, document content/ID/
  revision, tenant/principal, access/policy/knowledge scope, model/settings,
  instructions, contract version, relevant memory and revision feedback.
- Final full default offline suite: 660 passed, 23 deselected. External model/API
  integrations remain excluded. Human approve/modify/reject, restart and budget
  checks passed. CLI fixture smoke paused for human approval as expected.
- Source hashes rechecked after execution; Git diff/targeted secret checks recorded
  separately. No raw model answer/private review mapping is publicly uploaded.

These are functional fixture counts, not independent safety trials. Scripted
plans/reviews and source-answer replay do not establish correct live planning,
answer quality or semantic precision. No external model calls or independent
answer reviews were made. No live latency, token or monetary saving was measured.
Meaningful memory can change a real plan/evidence set and legitimately prevent
reuse; this result does not show memory can be removed from arbitrary tasks.

## Reproduction / next work

No user commands are needed to finish this offline task. For an independent audit,
prepare the checksummed FAQ corpus as in ../R5_LIVE_RUN.md, run development tests,
then reproduce in a NEW directory (never overwrite the recorded result):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_static_document_stage.py tests/test_r6_criterion.py tests/test_document_agent.py tests/test_claim_hygiene_checkout.py -q > focused.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q > full.txt
.venv/bin/python scripts/run_r6_agent_stage.py freeze --corpus runs/r5-corpus/corpus.json --run-dir runs/r6-audit-replica --focused-tests focused.txt --full-tests full.txt
.venv/bin/python scripts/run_r6_agent_stage.py run --corpus runs/r5-corpus/corpus.json --run-dir runs/r6-audit-replica
```

If the verified original baseline archive is available, add --baseline-archive
PATH consistently to both commands. Otherwise the script explicitly labels its
baseline-answer comparison as reconstructed from pinned fixture references,
not verified against the original raw run. An independent replica is not a new
untouched evaluation or independent safety observation. Freeze/source mismatch
blocks execution; evaluation-directory reuse is rejected rather than retried.

The broader final delivery gate remains open: quota-bounded live collection,
independent actual-answer review, useful semantic approval, representative net
latency/work evaluation, and a fresh preregistered final cohort if performance/
quality claims are to be made. Existing R5 live collection can proceed on its
frozen implementation independently; do not update a partially executed pilot's
source mid-run. Publish an inconclusive final study if evidence remains inadequate.
