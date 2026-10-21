# Current implementation and evidence

Single current status page. Updated 2026-10-21 (Asia/Calcutta).
Historical status/master reports are preserved with superseded banners. FACTS.md
is the evidence ledger, not a second implementation-status page.

## Active priority: general semantic answer reuse

The user's scope correction makes semantic answer caching the primary product,
with the bounded agent as its client. Follow the new
[evaluation contract](SEMANTIC_EVALUATION_CONTRACT.md). Do not resume failed live
pilots or expand the website before completing a generalizing local applicability
policy. Successful eligible semantic lookup must precede agent planning and make
zero remote calls. A question-pair score alone cannot authorize a hit.

The pretrained pinned dense and pair models are now available locally and both
executed on exposed authored development probes. Dense retrieval selected the
intended seed for all three authored positive paraphrases; the pair model gave
very low scores to two of them. This is a development finding, not answer
precision, a passing policy or representative coverage. The old fine-tuned
checkpoint remains missing. See `audit/SEMANTIC_PREPARATION_RESULT.json`.

An [answer-aware diagnostic policy](ANSWER_POLICY_DEVELOPMENT.md) now reads the
actual complete cached answer and fixed context, separately checking sufficiency
and source support. Both tested local baselines failed development: FLAN
recommended no reuse on 4 positives/11 negatives; Qwen recommended 4 positives
and 8 unsafe negatives. These are authored cases in 3 correlated intent groups,
not independently labeled quality observations. The diagnostic serving interface
always abstains, including on positive recommendations. No new general semantic
serving is enabled. Exact hits skip the local verifier; local check work is
reported separately from remote verifier calls. See `audit/ANSWER_POLICY_RESULT.json`.

A decomposed local NLI verifier now separates request preservation, source-span
coverage and answer support, with explicit requirement guards. On 45 authored
development cases it recommended reuse for 2/14 valid paraphrases and 0/31
negatives. The fixture cache served 2 semantic hits, with no unsafe semantic hit;
it also repeated one deliberately bad seed through exact matching. This exposes
the distinction between complete-response admission and factual correctness.
Coverage remains inadequate; production serving still abstains. These are
builder-authored diagnostics, not independent accuracy. See
[audit closeout](audit/DECOMPOSED_POLICY_CLOSEOUT.md).

[Day 3 research and policy measurements](audit/DAY3_CLOSEOUT.md) are now recorded.
Four question-conditioned QA variants were tested on 75 authored development
cases (24 positives, 51 negatives, 23 intent groups). At fixed initial settings:
strict spans accept 0/24 positives; parsed-call anchors 5/24 with 0 negatives;
function-question rewriting 13/24 with 4 unsafe negatives; the intent-gated hybrid
6/24 with 0 negatives. Hybrid fixture cache: 4 semantic hits, no unsafe semantic
hit; the separate exact path repeats one bad seeded answer. The old 45-case
cohort is unchanged, and the hybrid accepts only 2/14 positives on that cohort.
None of the fixed QA grid points combines 30% positive coverage and zero negative
accepts on these authored cases. This is not independent accuracy or a final gate.
Day 3 quality remains blocked; do not advance to Day 4 calibration or UI work.
Engineering fixes include full-score diagnostics, preserved evidence outputs and
word-boundary-safe span comparison. Full suite: 767 passed, 23 excluded; final
focused suite after boundary hardening: 79 passed. No general serving enabled.

The target is the tolerant group gate, at least 125 accepted groups with a
one-sided 95% exact lower safety bound at least 95%, plus separate answer
precision, coverage and hard-negative endpoints. Final cohort sizing and review
staffing remain unresolved; no independent human reviewers are confirmed.
Model-assisted diagnostics remain the fallback. No final evaluation has started.

Codex may commit/push substantial validated milestones on the task branch under
the latest user instruction; do not commit every small edit, push main or merge.

The gateway and bounded agent are implemented. General semantic safety and useful
live end-to-end savings remain unproven. Historical Gate A is still failed.
Passing engineering tests does not change that gate or establish a completion rate.

| Area | Implemented | Still unknown or unfinished |
| --- | --- | --- |
| Exact cache | Full-request/tenant scope, eligibility, complete-response admission, TTL, memory bounds, SQLite, invalidation and process-local miss coordination | Representative live quality/performance; distributed miss coordination |
| Gateway | Authentication, HTTP adapter, work metrics, local dashboard, bounded SSE bypass and recorded packaging/container checks | Broader live operational validation; streaming remains uncached |
| Semantic path | Off/exact/shadow/guarded wrapper, bounded candidates, lexical/pinned-local dense adapters, actual-answer approval bindings | Generalizing approval beyond allowlist, calibrated retrieval/verifier quality, net utility |
| Documentation agent | Validated bounded plan, actual search/read tools, task checkpoints, scoped memory, review retries and human approval/resume | Live task correctness, memory benefit, guarded-verifier budget integration, live validation of opt-in static-stage reuse |
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
- Last recorded full offline suite: 767 passed, 23 excluded. These are engineering
  tests, not accuracy observations. Source: [validation](audit/DAY3_POLICY_RESULT.json).

Counts above cite FACTS.md recovery/R5 sections and linked raw aggregate artifacts.
R6 added one explicitly frozen source-reference replay; no live model run is implied.

## Active work and user action

Claim hygiene is complete on the audit branch; old wording is retained in marked
historical snapshots. See [claim audit](audit/CLAIM_HYGIENE_AUDIT.md).
The [R6 preregistration](audit/R6_AGENT_CACHE_PREREG.md) was authorized by the user's
go on 2026-10-19. Its original document remains unchanged. The opt-in static drafting
contract is implemented, and its single engineering criterion run passed: 44 tasks
per arm, 132 off versus 110 exact fixture provider calls, 88 actual tool dispatches
per arm, identical final answers/statuses and original raw-baseline answer match.
Full-request keys still bind all actual model-visible stage inputs. No live savings
or general semantic safety follows. See [R6 closeout](audit/R6_AGENT_STAGE_CLOSEOUT.md).

The user-run R5 v1 live collection stopped on a token-limit response, reported
October 20. Preserve the original journal and failure; the hosted workspace has
not inspected that private journal. The explicit [v2 repair](R5_LIVE_V2_PROTOCOL.md)
adds a concise-answer instruction and larger output allowance, retaining strict
complete-response admission. Its combined predecessor/revision attempt budget
remains 16; both use the existing shared journal. See [v2 run instructions](R5_LIVE_V2_RUN.md).
The uploaded partial evidence has now been checked: v1 contains six attempts
(five complete responses and one truncated); v2 contains ten attempts, all complete.
The revised allowance was insufficient to finish the expected paired workload.
Antigravity returned a model-assisted review of the twelve items; all final
immutable bindings validate. Its submitted counts are two correct/supported, nine
correct/unsupported and one incorrect/unsupported. Some judgments remain disputed;
no human labels or accepted quality rate follow. See [review reconciliation](audit/R5_ANTIGRAVITY_RECONCILIATION.md). No completed paired result or live latency/savings claim
follows. See [partial live closeout](audit/R5_PARTIAL_LIVE_CLOSEOUT.md). Do not
resume stopped live runs or reset the journal.
Human reviewers get only the blinded template. Reconstruction invalidates live latency.

The approved offline refinement is complete. Retain negative baselines, collect/review actual
answers, and define fresh final evaluation criteria/data before running it. Stop
and ship a scoped honest report even if quality/utility remains inconclusive;
never manufacture a passing outcome by retuning a spent test.

Delivery targets in EXECUTION_ROADMAP.md are forecasts, not guarantees of positive
results. Commit/push completed tasks on branches under the current user instruction;
main is unchanged until an authorized merge. Work runs during active sessions,
not unattended between chat turns.

## Partial live-agent pilot and integrated workbench

The user-uploaded agent pilot stopped after six recorded attempts: five complete
responses and one upstream HTTP 502 during exact-arm workflow review. The off
seed reached AWAITING_APPROVAL; the exact seed drafted successfully before the
review failed. Neither repeat task started, so there is no live reuse/savings
result. Two real local document reads were observed. See [aggregate evidence](audit/LIVE_AGENT_PARTIAL_RESULT.json).
Do not resume that failed pilot or reset its allowance.

The integrated [Trace workbench](WORKBENCH.md) now connects the existing agent,
source tools, full-request exact cache, durable approvals and recent tasks through
a responsive local interface. Reference replay starts without a key and labels
its scripted answers. The real HTTP flow records three first-task scripted calls,
two on an identical repeat with a draft HIT, and three with caching off. These
are fixture mechanics, not live savings or semantic-quality measurements.

Optional interactive live mode requires an explicit --live flag, environment key
and existing shared journal; a fixed separate twelve-attempt demo allowance
survives app restarts. No live calls were made during interface development. The
launcher binds only to loopback. Multi-user hosting remains separate work.

The full offline suite passed with 701 tests and 23 exclusions; JavaScript syntax,
HTTP application flow and packaged asset inclusion passed. Browser visual QA was
unavailable here and is not claimed. See [validation](audit/WORKBENCH_VALIDATION.json).
