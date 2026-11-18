# R6 agent cache-stage development preregistration

Status: PROPOSED — awaiting explicit user go. Written before implementation and
before the named evaluation. No redesigned stage or new result exists yet.
This protocol is a development engineering check, not a fresh sealed safety test.
User requested publication and review before implementation.

## Hypothesis and fixed negative baseline

Prior-task memory changes the current draft prompt even for repeated static
questions, preventing exact reuse. A narrowly scoped documentation stage can
exclude orchestration-only memory from both its actual provider prompt AND its
full-request key, while retaining every answer-relevant input. This should reduce
simulated provider invocations without changing final answers on the R5 replay.

Retain parent `4fbe298`, docs/audit/R5_AGENT_SUMMARY.json and existing local R5
raw artifacts unchanged. Recorded negative baseline: 44 tasks per arm, 132
source-replay provider invocations and 88 actual local document-tool invocations
per arm; no provider-call reduction. Source: FACTS.md R5-DEVELOPMENT section.
No old result will be overwritten or relabeled.

## Smallest proposed change and input contract

An explicit static-document-stage option, default preserving existing behaviour.
Planning still receives scoped task memory. Search/read still execute normally.
Only drafting is eligible for this redesign; reviewer/planning caching, semantic
thresholds, approval policy, provider, and budgets are not changed in this test.
Do not strip values from a cache key while still sending them in the model body.
Exact caching continues to hash the complete actual generation request and trusted
scope. If context cannot be safely classified under this contract, retain it or
bypass caching. Do not use a model's relevance guess as a safety label.

| Input | Required handling |
| --- | --- |
| Original/resolved question and answer constraints | Preserve original question plus any explicit resolved constraints; a change must prevent reuse |
| Model, generation settings, complete instructions and stage contract version | Included in the actual request/key; changes prevent reuse |
| Document IDs, full consumed evidence and source revision/content | Included; changes to evidence or version prevent reuse even when question text matches |
| Tenant, principal and trusted access/knowledge/policy scope | Bound in trusted cache context or complete request; changed scope prevents reuse |
| Relevant memory | Explicitly typed/resolved answer constraints/context must enter the actual stage request/key; any change prevents reuse |
| Reviewer feedback, revision inputs and prior draft when used | Preserve in request; revised work must not reuse an earlier incompatible stage answer |
| Task IDs, progress counters, unrelated completed-task suggestions | May be omitted from static drafting only under an explicit contract that drafting does not use them; remain in orchestration/task state |
| Human decision/approval state | Never reuse or infer a human approval from a cached model answer/review |

Omitting prior-task suggestions changes the stage's model input contract; it is
not asserted semantically equivalent for arbitrary tasks. The opt-in scope is
fixed-reference factual documentation only. Ambiguous references to previous tasks,
preferences or user history require relevant memory/context or conservative bypass.
No generic memory deduplication/filter is claimed safe. No automatic broadening of
eligibility, question normalization, cross-tenant sharing or source freshness.

## Frozen experiment

After go, implement and run focused development/unit tests first. Freeze runtime,
fixture/provider implementation and workload hashes BEFORE one named criterion run.
Record Git revision/tree or exact dirty source hashes. Evaluation outputs use a new
exclusive directory; keep old artifacts untouched. No hidden retry or paid/live call.

Use the same pinned R2 corpus canonical digest
`0fe8140b1d3bb1bf752821d15e863b5552db324d17ed373b89952837620a7089`.
Only the same 22 development FAQ questions, each seed then constructed exact repeat:
44 tasks per arm. Preserve order, same scoped task-memory history, documents,
source-replay answers/plans/reviews, budgets, fixture completion policy and tool work.
Arm A: redesigned stage contract with cache off. Arm B: same contract with exact
cache on. Initial stores are empty and independent. Preserve original baseline
answers for a separate text comparison; no future answer may approve a prior hit.

A fixture replay cannot measure model correctness, real token use, monetary savings
or live latency. Scripted reviewer approval is not independent human review.

## Exact success criterion (conjunctive)

- Both arms produce all 44 expected task outcomes with identical terminal statuses
  and byte-identical final answer text per corresponding task. Compare text, not
  response IDs/usage/timing. No missing/unknown answer is treated as equal success.
- Both arms also retain the original R5 final answer texts for these fixture tasks.
  If archived raw answers are unavailable, reconstruct expected fixture text from
  pinned reference data and disclose that limitation; never claim a verified raw
  baseline match without access to it.
- Exact-cache arm uses strictly fewer total measured fixture provider invocations
  than off arm; all planner/drafter/reviewer/errors/retries count. Unknown attempt
  counts, skipped tasks or incomplete outcomes fail the criterion. Off-arm work
  must remain the original 132 invocations; do not remove stages to create savings.
- Tool work remains 88 actual local document-tool invocations per arm. No hidden
  tool/result memoization or reduced research is introduced.
- Isolation tests pass for changed question, source text/revision, tenant,
  principal/access/policy scope, instructions/model/settings and relevant memory.
  Changing irrelevant orchestration metadata alone may permit exact stage reuse.
- Human pause/approve/modify/reject and restart tests remain passing; task budgets
  remain enforced; no real external model call is made.

Failure of ANY item is FAIL. Pre-execution data/hash/environment mismatch is
BLOCKED/INVALID, not PASS; retain its evidence. A coding/runtime error during the
criterion run is recorded as failure/deviation, never silently rerun until green.
No threshold or criterion is changed after seeing the outcome. Any revised
experiment needs a new dated protocol version, explicit rationale and new run ID;
the original failure remains visible. Report rate/count uncertainty appropriately;
fixture call counts are deterministic engineering counts, not independent safety
observations. No new application-level savings percentage is preregistered.

## Report and scope limit

Output: per-task stage trace, attempts/statuses, tool counts, answer-text hashes,
source/config/runtime hashes, comparison against the retained baseline and explicit
PASS/FAIL/BLOCKED. Include all failed attempts and limitations. Keep raw answer and
private review mapping artifacts out of public publication; publish aggregates
and reproduction instructions in the authorized task branch.

Even PASS establishes only controlled fixture reuse for this scoped input contract.
R5 live collection/review, semantic approval quality, representative whole-task
latency and any new untouched final evaluation remain separate unresolved work.
Do not retune on a sealed final set; agree a fresh final protocol/cohort and stopping
rule before the last evaluation. A failed or underpowered final result can still
ship as an honest project, without invented improvement claims.
