# Current implementation and evidence

Single current status page. Updated 2026-10-19 (Asia/Calcutta).
Historical status/master reports are preserved with superseded banners. FACTS.md
is the evidence ledger, not a second implementation-status page.

The gateway and bounded agent are implemented. General semantic safety and useful
live end-to-end savings remain unproven. Historical Gate A is still failed.
Passing engineering tests does not change that gate or establish a completion rate.

| Area | Implemented | Still unknown or unfinished |
| --- | --- | --- |
| Exact cache | Full-request/tenant scope, eligibility, complete-response admission, TTL, memory bounds, SQLite, invalidation and process-local miss coordination | Representative live quality/performance; distributed miss coordination |
| Gateway | Authentication, HTTP adapter, work metrics, local dashboard, bounded SSE bypass and recorded packaging/container checks | Broader live operational validation; streaming remains uncached |
| Semantic path | Off/exact/shadow/guarded wrapper, bounded candidates, lexical/pinned-local dense adapters, actual-answer approval bindings | Generalizing approval beyond allowlist, calibrated retrieval/verifier quality, net utility |
| Documentation agent | Validated bounded plan, actual search/read tools, task checkpoints, scoped memory, review retries and human approval/resume | Live task correctness, memory benefit, guarded-verifier budget integration, efficient stage reuse |
| Evaluation | Answer-bound traces/reviews, missing-label handling, attempt/usage accounting, grouped-analysis tools and R5 comparisons | Independent actual-answer labels, useful reviewed semantic live comparison and fresh final evaluation |
| Free collection | Free-only routing, shared durable journal/caps, fail-closed pauses and foreground evidence export | Completed current R5 live pilot and review |

## Recorded evidence

- Historical Phase A: saved designated-pair labels failed its gate. Source audit
  found candidate-label attribution, verifier-path and reproducibility gaps. Do
  not treat its original error labels as independently reviewed actual answers.
  See [source audit](audit/PHASE_A_SOURCE_AUDIT.md).
- User-run connection pilots returned incomplete answers. Later guided exact demo:
  six constructed requests, two exact hits and four misses; four cache-arm versus
  six fresh-arm actual provider attempts, ten total. Hits unreviewed; latency
  uncertainty included no difference. See [live audit](audit/OPENROUTER_GUIDED_EXACT_V1.json).
- R3/R4: functional cache/agent lifecycle and restart checks passed, with scripted
  fixtures. See [R3](audit/R3_CLOSEOUT.md) and [R4](audit/R4_CLOSEOUT.md).
- R5 request source replay: 132 constructed events per arm; off 132 provider
  invocations, exact/guarded 110 each, with 22 exact hits and no guarded semantic
  hits. No real model-token or monetary claim follows.
- R5 agent source replay: 44 tasks per arm; both arms made 132 simulated provider
  invocations and 88 actual local tool invocations. No agent call reduction.
  Changing task memory changed drafting inputs. See [R5](audit/R5_CLOSEOUT.md).
- Last recorded full offline suite: 620 passed, 23 excluded. These are engineering
  tests, not accuracy observations. Source: [validation](audit/R5_VALIDATION.json).

Counts above cite FACTS.md recovery/R5 sections and linked raw aggregate artifacts.
No new model run or replay is implied by this documentation update.

## Active work and user action

Claim hygiene is complete on the audit branch; old wording is retained in marked
historical snapshots. See [claim audit](audit/CLAIM_HYGIENE_AUDIT.md).
The [R6 agent-stage preregistration](audit/R6_AGENT_CACHE_PREREG.md) is PROPOSED:
implementation and criterion evaluation have not started. User review/go is the
next checkpoint. The full-request key must not silently omit model-visible context.

Live evidence may be collected independently via [R5 live instructions](R5_LIVE_RUN.md):
eight development requests, at most 16 free-provider attempts, using the existing
shared journal. No key is available to this workspace. Human reviewers get only
the blinded template; model-assisted review is labeled separately. No paid service
or new dataset hunt is needed. A reconstructed run cannot certify live latency.

After approved offline refinement, retain negative baselines, collect/review actual
answers, and define fresh final evaluation criteria/data before running it. Stop
and ship a scoped honest report even if quality/utility remains inconclusive;
never manufacture a passing outcome by retuning a spent test.

Delivery targets in EXECUTION_ROADMAP.md are forecasts, not guarantees of positive
results. Commit/push completed tasks on branches under the current user instruction;
main is unchanged until an authorized merge. Work runs during active sessions,
not unattended between chat turns.
