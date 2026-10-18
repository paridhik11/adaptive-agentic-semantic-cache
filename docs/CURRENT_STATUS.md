# Current implementation and evidence

The original semantic-cache research goal is still unproven. The repository has
working runtime and evaluation components; passing functional tests does not
establish safe semantic reuse or real workload savings. A completion percentage
would hide that distinction.

| Area | Implemented | Still needed |
| --- | --- | --- |
| Exact response cache | Tenant/request scope, eligibility, TTL, memory bounds, SQLite, invalidation and concurrency handling | Representative live workload performance and answer review |
| Gateway | Authentication, metrics, local dashboard, HTTP provider, bounded SSE bypass | Live operational validation; streaming responses remain uncached |
| Semantic matching | Opt-in guarded serving wrapper, semantic shadow mode, bounded candidates, lexical/pinned-local dense adapters, strict actual-answer approval bindings | Reviewed development calibration, broader approval beyond an explicit pair allowlist, independent untouched quality/utility evaluation; default remains exact |
| Evaluation | Answer-bound trace/review/report, unknown-label handling, work accounting, uncertainty intervals, independent review reconciliation | Complete live answers and independent human labels across sources/groups |
| Agent integration | Bounded specialist/reviewer workflow and scope isolation checks | Broader task/tool workflow and task-level quality evaluation; this is not the full original agent roadmap |
| Zero-budget execution | Free-only routing, durable quota journal, resume rules, foreground progress page and single results bundle | Observation that the revised concise/reasoning-off profile completes on the live backend |

## Observed live results

| User-run check | Output limit | Attempts | Complete admissible answers | Outcome |
| --- | --- | --- | --- | --- |
| connection-check-v1 | 128 tokens | 2 | 0 | User-pasted journal diagnostics confirm length termination; all completion tokens spent on reasoning |
| connection-check-v2 | 1024 tokens | 2 | 0 | Uploaded trace confirms length termination, with partial answer text |

Evidence: `audit/OPENROUTER_CONNECTION_PILOT_V1.json` and
`audit/OPENROUTER_CONNECTION_PILOT_V2.json`. These are failed connection-pilot
attempts, not a benchmark. There are no independently reviewed live-answer
metrics to put on a resume yet. Historical Gate A remains failed; its spent test
stream must not become a tuning set or be rerun to manufacture a passing result.

The user then completed `guided-exact-v1`: six constructed development
requests produced two exact-repeat hits and four misses, with ten provider
attempts across cache and fresh arms. Both hits remain unreviewed and the
paired-latency interval spans zero. This confirms exact-hit wiring only; see
[`audit/OPENROUTER_GUIDED_EXACT_V1.json`](audit/OPENROUTER_GUIDED_EXACT_V1.json).

## Next evidence steps

The user accepted the revised Project 7 + bounded Project 15 sequence on
October 18, 2026. The active plan is [EXECUTION_ROADMAP.md](EXECUTION_ROADMAP.md):
R3 cache lifecycle engineering is complete with fixture evidence; R4 adds structured planning,
document tools, persistent task state/scoped memory and human approval/resume;
R5 compares the same requests/tasks with cache off/on; R6 refines and publishes.
This supersedes the earlier matcher-first sequence. A disclosed constructed
load test is valid controlled evidence; natural-traffic prevalence is reported
separately. A working implementation does not itself pass a quality gate.

Target: integrated demo October 22 and scoped engineering/report delivery
October 24. October 25-28 is a contingency estimate, not a changed deadline;
free quota and review can leave statistical evidence inconclusive. Commit/push
only after each complete phase, with the roadmap update included in the R3 closeout.
No user action is required for the offline build. The evidence workflow below
still applies when a concrete foreground live run or review packet is ready.

The earlier guided exact run confirms complete-answer admission on its small
constructed workload. Further live collection uses [free_pilot.md](free_pilot.md)
with explicit free quota planning. Inspect incomplete responses before expanding
traffic. The R3 lifecycle smoke uses local fixture responses and consumes no
external model calls; it does not replace paired live evaluation.

Then use a disclosed, diverse development workload with realistic repetition,
near misses, tenant/context separation and changing-information cases. Allocate
the shared free quota across days, preserving all failed and rejected attempts.
Compare exact, semantic-ablation and verified policies, counting verifier work.
Free-model availability and provider latency are external conditions to record.

Have independent reviewers label actual returned answers and reuse decisions.
Report coverage, incorrect reuse with uncertainty, provider/token work and
successful-answer latency separately. Missing labels stay unknown. If semantic
reuse is not useful at an acceptable error rate, retain exact caching and narrow
the semantic scope based on development evidence. Freeze a new protocol before
using an untouched evaluation stream. Only measured, scoped results become
resume claims.

Engineering, offline testing and repository updates run in the development
workspace. Live inference is user-initiated through the guided foreground
session, which removes per-query manual commands and separate evidence
exports. It does not run autonomously in the background.

## Recovery execution status

R1 diagnostics are complete: recorded eligibility, matching and verifier stages
now explain the reproduced development outcomes. See
[audit/R1_CLOSEOUT.md](audit/R1_CLOSEOUT.md) and
[EXECUTION_ROADMAP.md](EXECUTION_ROADMAP.md). This closes an instrumentation
phase, not the semantic safety/utility gate. R2 workload investigation is
complete: a pinned Python technical-reference corpus, an authored adversarial
suite and a separate OASST1 prevalence audit. The OASST1 result is descriptive
crowdsourced traffic only; it does not establish safe reuse or production
savings. WildChat remains excluded pending complete retrieval and license-file
reconciliation. R3 can test matcher development against these grouped sources
without changing serving defaults.

R3 adds the complete gateway candidate/approval path and source-revision scope
on top of the existing exact runtime. Off, exact, shadow and guarded modes are
exercised by `scripts/smoke_semantic_gateway.py`. Guarded mode's current
operator approval registry is an actual-answer/pair allowlist; it does not infer
correctness for unseen paraphrases. The larger benchmark-only semantic engine
remains separate. See [R3 closeout](audit/R3_CLOSEOUT.md) and
[serving contract](semantic_cache_runtime.md). Semantic quality, live utility
and the broader agent feature set remain unproven/unfinished.
