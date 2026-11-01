# Current implementation and evidence

## Active product milestone — 2026-11-01

Continue the existing bounded documentation product; no new training experiments
without a specific integration need. User requested branch push on 2026-11-01;
automatic approval review blocked it pending destination-specific sharing
confirmation. Local commits continue; no merge or force operations. No live inference or locked final evaluation.

Missing Group 1 was reconstructed from `847d3f9` on
`integration/documentation-rebuild`. Candidates now share their SQLite exact
source database. New evidence: **242 regression tests passed**; constructed
four-request restart fixture calls off/exact/guarded = **4/3/2**. These are
engineering fixtures, not human answer labels, live savings or a recovery of the
old unavailable 243-test result. General semantic serving stays disabled by default.

Group 2 is now implemented and fixture-verified: standalone whole-answer lookup
before agent planning, source-bound admission after required approval, durable
human-rejection namespace revocation, fresh approval tasks on hits, and actual
provider/local-stage work observations. **274 regression tests passed**. Four
tasks per arm: constructed fixture calls **off 12 / exact 9 / guarded 6**, with
restart after the approved seed. Workload, answers, stage events and source hashes
are retained; scripted approval actions are not human answer review.

Try the offline exact whole-answer demo with
`python scripts/run_demo.py --whole-answer-cache`. General semantic serving,
independent answer quality and live utility remain unproven/disabled. No model
training, downloads, live/provider calls or final evaluation. Timings are local
fixture observations, not representative latency. Combined applicability is
measured; classifier/veto internals and provider-internal retries are not.

Group 3 is implemented: explicit bounded document-role routing, typed search/read
registry, persistent scoped vector index and frozen contextual-task snapshots.
Local authenticated memory list/clear endpoints retain audit history; epoch
revocation blocks stale pending tasks. Standalone answer hits skip memory and all
model roles. **293 regression tests passed**. New fixture receipt: fresh/repeat/
contextual tasks use [3,0,3] scripted model calls; source+query embeddings 2 on
first retrieval versus 1 query embedding after restart. This is lexical retrieval
and lifecycle evidence, not natural paraphrase quality or a memory benefit study.

Next: Group 4 UI/installation/deployment QA, independent actual-answer review and
separately authorized budgeted live utility. No training detour, push or merge.
See [Group 3 report](DOCUMENT_ORCHESTRATION_MEMORY.md). Milestone commits are
preserved automatically; the user need not manage recovery downloads.

Group 4 is PARTIAL: data inventory, independent-review authoring packet/schema
validator and installed-wheel HTTP smoke are complete. 301 regression tests
passed. Wheel smoke uses existing dependencies and is not clean-machine install
or browser QA. No new independent labels exist. General approval beyond authored
bindings remains unresolved, separately from UI/deployment work. See
[dataset/review report](DATASETS_AND_REVIEW.md). No training or live calls.

Separate model review is now available: 30 model-authored/self-reviewed items,
policy-blind by instruction. Human labels are still zero; user will review a
shuffled form without model judgments. Those labels will be human review of
model-authored questions. No claim of independent human-authored wording, policy
accuracy or general serving promotion. See `data/document_product_review/human_v1/`.

Complete user human review is now recorded: 14 Y / 15 N / 1 uncertain across all
30 model-authored items. The model agrees on 29 definite judgments; H05 remains
human-uncertain versus model-yes. These are source-answer applicability labels,
not cache-policy predictions. No accuracy/savings claim or serving promotion.
The earlier zero-human-label statements above describe preparation snapshots.
See [review reconciliation](HUMAN_MODEL_REVIEW.md).

UI context management is now connected to its existing authenticated API, with
confirmation before clearing future context. Whole-answer mode captions and hit
labels distinguish it from draft reuse. 61 focused API/agent regressions and JS
syntax passed; visual browser QA is BLOCKED by missing browser executables.
A loopback-only Playwright smoke script is ready. General semantic approval and
representative live utility are still unresolved; no release completion claimed.
See [48-hour release checklist](RELEASE_48H.md).

Release preflight now explicitly reports BOTH approved MiniLM/Quora snapshots
missing in the current local cache. No fallback scores, model load or downloads.
176 focused product/preflight regressions passed. A manually triggered GitHub
workflow is prepared for minimal dependency installation and actual browser QA;
it has NOT run. GitHub metadata confirms repository ownership and push permission,
but automatic approval review again blocked publication because destination-
specific sharing authorization was considered unclear. No bypass attempted.
Core semantic decision quality remains unresolved; supporting infrastructure
completion must not be presented as completion of that goal.

The status below is historical Stage B context. Missing follow-up artifacts are
not recreated by repeating experiments; their results are not claimed here.

Single current status page. Updated 2026-10-21 (Asia/Calcutta).
Historical status/master reports are preserved with superseded banners. FACTS.md
is the evidence ledger, not a second implementation-status page.

## Current milestone: Stage B B0–B5 complete; stopped for review

Stage A is committed on three stacked local branches. Stage B has one local
commit per B0–B5 on `stage-b/classifier`. No push, merge or main rebase.
Full eight-part report: [Stage B closeout](audit/STAGE_B_REPORT.md).

Evidence: Codex-authored synthetic template-held-out diagnostics, not human truth.
Primary seed four-class agreement: training 100%, dev 46.31%, calibration 45.45%.
All three trained seeds fail the prespecified conditional calibration gate;
selected policies abstain on every semantic query. Their coverage is zero and
precision undefined. Retrieval recall@3 is separate from this decision failure.
The unchanged guard baseline at 0.7 safely reuses 128/256 dev and 1,014/1,024
calibration opportunities under the authored oracle, with zero observed unsafe
accepts. No baseline is promoted; these are not independent answer-quality labels.

| Roadmap item | Current evidence |
| --- | --- |
| Day 1 inventory/contracts | Recorded, with synthetic-label and artifact limitations |
| Day 2 local retrieval | Implemented and measured; natural wording unverified |
| Day 3 applicability quality | NOT passed; trained policy lacks useful calibrated coverage |
| Stage B B0–B5 authorized study | Complete; stopped after B5 |
| Day 4 onward integration/product/final claims | Not authorized in this scope; not complete |

General semantic serving remains DISABLED. Final records remain unevaluated;
independent human labels are zero. The 50-slot natural-review request is blank.
No public records were downloaded. A future study needs separately versioned,
mixed-domain training data and independent natural-wording review. Preserve this
study; do not tune it to the already observed held-out results.

Download `artifacts/stage_b/stage-b-complete-backup.zip` before reset. It includes
all three learned safetensors exports and the local Git bundle; ignored learned
weights are not preserved by a Git commit alone. No live/provider calls or UI,
agent, Stage C, final-evaluation or serving-promotion work was performed.

The older preparation findings below are historical context, not the active task.

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


## Stage A local publication complete (2026-10-21)

Latest authorization: LOCAL commits only; no pushing, merging, main rebasing or
force operations. Harness and data are committed on stacked branches. The user
approved the verified compressed decision log; the raw oversized duplicate is
gitignored. Results are committed next, followed by Stage B on its own branch.
Final regeneration matched by hash only; final question and answer stores remain
out of Git and in backups. General semantic serving remains disabled.

Day 1 inventory and Day 2 local retrieval experiments are recorded, with
limitations. Day 3 quality is unresolved: baseline unsafe reuse remains high.
Stage B trains and measures a candidate policy; completion of its implementation
will not itself prove generalization, human-reviewed quality or a passing gate.


## Stage B through B3 — quality remains blocked

B0 diverse synthetic cohorts, B1 three-seed LoRA fine-tuning and B2 question
veto are committed locally. B3 actual-candidate dev evaluation shows a major
train/held-out gap: 100% train four-class fit versus 46.3%–73.2% dev agreement
with synthetic labels. At fixed diagnostic threshold 0.5, primary seed 11
safely accepts 4/256 positive opportunities with zero observed unsafe accepts;
other seeds accept 60–66/256. Top-3 retrieval recall is 256/256. These are
synthetic diagnostics, not independent accuracy. Do not switch to the best seed.
B4 calibration follows the already written config without model/rule changes.
General semantic serving stays DISABLED. No Stage C or UI work is authorized.
