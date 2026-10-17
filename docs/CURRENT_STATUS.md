# Current implementation and evidence

The original semantic-cache research goal is still unproven. The repository has
working runtime and evaluation components; passing functional tests does not
establish safe semantic reuse or real workload savings. A completion percentage
would hide that distinction.

| Area | Implemented | Still needed |
| --- | --- | --- |
| Exact response cache | Tenant/request scope, eligibility, TTL, memory bounds, SQLite, invalidation and concurrency handling | Representative live workload performance and answer review |
| Gateway | Authentication, metrics, local dashboard, HTTP provider, bounded SSE bypass | Live operational validation; streaming responses remain uncached |
| Semantic matching | Experimental benchmark policies, lexical baseline, pinned dense encoder adapter, fail-closed verifier | Independent answer review, calibration on development data and a new untouched evaluation; production gateway integration depends on evidence |
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

First obtain complete responses and exact-cache admission using the guided local
session in [free_pilot.md](free_pilot.md). This deliberately constructed repeat
sequence is a functional check. Inspect failures before changing a profile;
do not expand traffic while responses are incomplete.

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
