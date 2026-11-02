## Large-Scale Offline Metrics Campaign & Evidence Correction — 2026-11-02

This section records large-scale evaluation results across 25 documentation source groups and 125 base intents under two standardized profiles: Repeat-Rich (600 events per arm, executed on 12 groups / 60 base intents) and Lower-Repeat Breadth (400 events per arm, executed across all 25 groups / 125 base intents).
Harness: `scripts/evaluate_large_workload.py` | Telemetry: `artifacts/metrics_campaign/` | Detailed Report: `docs/METRICS_RESULTS.md`.

1. **Evidence Corrections & Terminology (FACTS MC-01, MC-07, MC-11, MC-16)**:
   - **Workload-Expected (Eligible) Reuse**: Avoided provider calls are labeled `calls_avoided_eligible` based on category matching (`is_eligible_reuse = is_hit and item["eligible"]`). Actual answer correctness, semantic suitability, and safe coverage remain **unverified pending independent answer-level review**.
   - **Executed vs Defined Scope**: The repeat-rich profile executed 60 intents across the first 12 source groups (modeling query redundancy). The lower-repeat profile executed **all 125 intents across all 25 source groups** (modeling breadth).
   - **Adversarial Classification**: The 0/120 result on repeat-rich represents **development regression evidence** following the Task D regex refinement, not untouched holdout precision.

2. **Workload Profile A: Repeat-Rich (600 Events per Arm; 12 Executed Groups, 60 Intents)**:
   - **`no_cache`**: 1,800 provider calls across 600 events, 201,667 synthetic tokens, p50: 130.86 ms.
   - **`exact_only` (Safe Release Default)**: 948 provider calls, 852 calls avoided on eligible reuse, 0 on ineligible reuse, 107,750 synthetic tokens (46.57% token reduction), 284 exact hits, p50: 140.80 ms, 0 false accepts (0/120).
   - **`verified_semantic` (Guarded Opt-In)**: 927 provider calls, 873 calls avoided on eligible reuse, 0 on ineligible reuse, 105,429 synthetic tokens (47.72% token reduction), 291 hits (273 exact, 18 verified semantic), p50: 137.81 ms, 0 false accepts (0/120 development regression).
   - **`shadow_semantic` (Diagnostic)**: 948 provider calls, 852 calls avoided on eligible reuse, 0 unproven answers substituted.

3. **Workload Profile B: Lower-Repeat Breadth (400 Events per Arm; ALL 25 Groups, 125 Intents)**:
   - **`no_cache`**: 1,200 provider calls, 133,964 synthetic tokens, p50: 89.70 ms.
   - **`exact_only` (Safe Release Default)**: 1,023 provider calls, 177 calls avoided on eligible reuse, 0 on ineligible reuse, 114,453 synthetic tokens (14.56% token reduction), 59 exact hits, p50: 168.42 ms, **0 false accepts across 125 challenges** (0/125, 0.00%, exact 95% Clopper-Pearson CI `[0.00%, 2.91%]`).
   - **`verified_semantic` (Guarded Opt-In, Untuned)**: 975 provider calls, 222 calls avoided on eligible reuse, 3 on ineligible reuse, 109,302 synthetic tokens (18.41% token reduction), 75 hits (57 exact, 18 semantic), p50: 180.60 ms, **1 false accept across 125 challenges** (1/125, 0.80%, exact 95% Clopper-Pearson CI `[0.02%, 4.38%]`) on `async_05:challenge_added_requirements`.
   - **`shadow_semantic` (Diagnostic)**: 1,023 provider calls, 177 calls avoided on eligible reuse, 0 on ineligible reuse, 59 exact hits, 0 unproven answers substituted, p50: 183.55 ms, 0 false accepts (0/125).

4. **Architectural Confirmation & Field Deprecation**:
   - The 1 false accept in untuned `verified_semantic` (`"benchmark using cProfile"` query matching basic asyncio answer) conclusively proves that exact caching (`mode="exact"`) must remain the default serving mode, and semantic caching must remain shadow-only by default and experimental opt-in.
   - Legacy field `is_correct_reuse` is deprecated across all documentation and telemetry: category eligibility tags do not denote actual answer-level correctness.

5. **Audited Real-Provider Pilot Plan & Preflight Safety (FACTS MC-10, MC-17, MC-18)**:
   - Prepared small frozen 4-task real-provider pilot (`scripts/evaluate_live_paired.py`) across `list_operations` and `dict_operations` comparing `no_cache` (12 calls) vs `exact_only` (6 calls) vs `shadow_semantic` (6 calls).
   - Exact proposed provider requests: **24 calls total** (complying with single-run limit of 30 and daily cap of 40 in `PilotJournal`).
   - Preflight status: **BLOCKED (live_authorized=False)**. Zero live API calls initiated. Shared quota enforcement, stable rate limits, zero paid fallback, free allowlist, and immutable receipts verified in `docs/LIVE_BENCHMARK_READINESS.md`.

## Final Release Evidence Correction & P2 Integration — 2026-11-02

This section supersedes earlier production claims. Independent frozen P2 neural evaluation was executed against checkpoint `4f9858e` by ChatGPT Work using real local neural inference (MiniLM DenseEncoder + MS MARCO passage CrossEncoder).

1. **Independent P2 Neural Evaluation Findings (FACTS P2-08–P2-15)**:
   - **Workload**: 144 requests evaluated across 12 documentation groups.
   - **Semantic Accepts**: 25 semantic answers accepted by cross-encoder threshold.
   - **Model-Assisted Review**: 16 / 25 provisionally sufficient (point estimate 64.00%, exact 95% Clopper-Pearson CI [42.52%, 82.03%]); 9 / 25 insufficient (36.00%, exact 95% Clopper-Pearson CI [17.97%, 57.48%]).
   - **Failure Breakdown**: 7 failures involving missing explicitly requested executable examples; 2 failures involving wrong-operation reuse (e.g. inverse operations).
   - **Scope Challenges**: 0 hits across 60 scope challenges (0.00%, exact 95% Clopper-Pearson CI [0.00%, 5.96%]).
   - **Status**: Failed safety qualification; not a production precision result. Evaluated prototype with experimental semantic serving. No claim of 98% precision, independent human truth, or live monetary savings.

2. **Safe Release Configuration & Serving Modes (FACTS P2-16)**:
   - **Default Serving Mode**: Exact caching is default (`mode="exact"`).
   - **Shadow Semantic Reuse**: Automatic neural semantic reuse operates in shadow mode by default (`mode="shadow"`), evaluating candidates and logging diagnostics without substituting unproven answers.
   - **Experimental Semantic Serving**: Guarded semantic serving (`mode="guarded"`) is enabled strictly via explicit opt-in for controlled demonstrations and evaluations.

3. **Targeted Semantic Guardrail Improvements**:
   - Added request/answer coverage checks for explicit executable-example requirements (`requires_executable_example`, `contains_executable_example`, `answer_satisfies_requirements`).
   - Added direction and inverse operation compatibility guardrails (`inverse_operation_mismatch`, `conversion_slots`).
   - Maintained fail-closed abstention for uncertain requests and preserved all atomic locking, race, scope, and revocation protections.

4. **Lifecycle Demo Clarification (FACTS P2-17)**:
   - Demo executes 7 independent lifecycle execution scenarios (fresh request, operator approval, exact repeat, experimental paraphrase reuse, changed requirement fallback, scope invalidation, restart persistence).
   - Criterion 8 is the evidence-reporting summary criterion across all 7 scenarios (not 8 independent scenarios).

5. **Live Runner Ceiling Safeguard (FACTS P2-18)**:
   - Live benchmark runner validates zero-output ceiling (`max_output_tokens > 0`); requests with non-positive ceilings are rejected before transport dispatch. Live calls remain disabled.

## P3–P6 Flagship Delivery Checkpoint — 2026-11-02

P3 (End-to-End Efficiency Evidence), P4 (Measured Latency Optimization), P5 (Working Demo & Release QA), and P6 (Flagship Release Documentation) are fully implemented, benchmarked, and verified (PASS).

1. **P3 Efficiency Benchmarks**:
   - **Realistic Mixed Workload (24 tasks)**: Verified semantic cache reduced provider calls from 72 (`no_cache`) to 45 (37.7% call & token reduction; exact: 57 calls). Whole-agent latency p50 improved from 133.4 ms to 122.4 ms (eligible traffic p50: 103.6 ms). Opportunity coverage: 83.3% exact, 66.7% semantic. Adversarial false hit rate: 0.0% (fail closed).
   - **Repeat-Rich Workload (24 tasks)**: Verified semantic cache reduced provider calls from 72 (`no_cache`) to 18 (75.1% call & token reduction; exact: 42 calls). Whole-agent latency p50 dropped from 144.2 ms to 43.9 ms (69.6% whole-agent latency reduction). Opportunity coverage: 100% exact, 88.9% semantic. Adversarial false hit rate: 0.0%.
   - Full environment provenance, model revisions (`all-MiniLM-L6-v2@1110a24` and `ms-marco-MiniLM-L-6-v2@233902d`), and per-task raw records published to `artifacts/resume/whole_agent_benchmark_results.json`.
   - Distinct scripted simulation vs opt-in live provider benchmarks preserved (`scripts/run_live_benchmark.py --check-only` enforces spend cap $0.00 and zero live calls unless explicitly authorized).

2. **P4 Latency Optimization**:
   - Diagnosed 28 ms DenseEncoder + 32 ms LocalPassageScorer per-candidate inference overhead on misses.
   - Implemented candidate presence pre-check before neural encoding, cheap structural guards (model match, history length, message structure) before scoring, and thread-safe in-memory embedding & passage score caching in `ConservativeHybridCache`.
   - Net whole-agent p50 improved from 165.6 ms down to 122.4 ms (mixed) and 43.9 ms (repeat-rich). Semantic hit p50 measured at 26.2–35.5 ms. Correctness strictly preserved (100% pass on acceptance suite).

3. **P5 Flagship Demo & QA**:
   - Built and validated `scripts/demo_flagship.py` (and `python scripts/run_demo.py --cli`) executing all 8 core lifecycle criteria: fresh request, operator approval, exact repeat (0 calls), safe paraphrase (0 calls), changed requirement miss (fail closed, 3 calls), scope invalidation (stale reuse blocked), restart persistence (durable SQLite reload), and per-request receipts (100% PASS).
   - Clean Windows temp directory handling verified in `scripts/smoke_workbench.py`.

4. **P6 Release Ready**:
   - Professional README.md with clear architecture diagram, setup instructions, comparative accounting tables, reproducibility commands, and honest evidence boundaries.

## P1.1 Final Closure Checkpoint — 2026-11-02 (`05ce156`)

All five supplemental independent audit blockers are closed with independently reproducible fixes (PASS).
1. Revocation/expiry during verification: atomic revalidation under lock in `resolve_semantic` & `admit_candidate`.
2. Live runner restart/reservation safety: explicit replay of completed/errored reservations, refusal to auto-repeat uncertain attempts.
3. Provider request sanitization: strict outbound allowlisting, model binding, disabled paid fallback.
4. Complete agent timing: boundary includes planning, tools, drafting, reviewer check, approval, admission, and SQLite terminal state persistence.
5. Evidence & documentation: corrected `measured-live` labels to `historical component measurement` / `simulated`, preserved 11/12 model review, disclosed token reductions as offline accounting.
281 regression tests passed (100%). P1.1 is closed; ready for P2 evaluation, P3 measurement, P5 demonstration, and P6 release workflow.

## Evidence correction — 2026-11-02 (supersedes publication/precision snapshots below)

GitHub integration branch was verified at `8a272ff`. All 1,100 historical events
were audited; all 129 semantic-hit answers recovered by full-response hash.
Corrected answer-bound benchmark no longer inherits generated family labels.
Model-assisted review of 12 answer-aware hits: 11 sufficient, 1 false behavioral
claim under strict whole-answer rubric; no independent human accuracy claim.
New offline no-cache/exact replay directly counts 220/198 scripted calls over
220 requests per arm; no failures, live calls or measured billable token costs.
Policy defaults/thresholds and agent architecture are unchanged. Neural/whole-agent
live utility remains unverified. See [evidence correction](BENCHMARK_EVIDENCE_CORRECTION.md).

# Current implementation and evidence

## Latest release checkpoint — 2026-11-02

This section supersedes the historical snapshots below. Groups 1–3 are
implemented and fixture-tested; Group 4 is PARTIAL. Human/model reviews and
resume component measurements are recorded. Automatic semantic applicability
and live savings remain unproven. No final split or new training was used.

User requested resuming publication to `integration/documentation-rebuild`.
Automatic approval review rejected the push because authorization to publish
recorded review data after the earlier pause was considered unclear. Nothing
was pushed; no alternate upload route was attempted after this rejection.

Fresh-environment release regressions: 174 passed, one Starlette/httpx deprecation
warning, 3.53s. This is software validation, not answer quality. Installed-wheel
verification and current release limitations are documented in RELEASE_48H.md.

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

Core-model update: both approved MiniLM/Quora revisions have now been restored
and actually executed locally. The earlier missing-weight preflight is historical.
A real CrossEncoder constructor incompatibility in SentenceTransformers 3.4.1
was repaired (supported identity activation argument, explicit safetensors).
Fixed-setting exposed diagnostic: MiniLM recommends 3/30 (2 human-sufficient,
1 human-insufficient); Quora and combined-margin recommend none. No serving
promotion or threshold changes. 120 related regressions passed. GitHub work
is paused by user instruction; no branch head was published. See
[real-model diagnostic](REAL_DOCUMENT_MODEL_DIAGNOSTIC.md).

Resume submission priority: reproducible local cache benchmark completed.
SQLite warm-hit p95 1.79 ms over 5,000 requests; three keys/five runs. Separate
64-request cold fixture made one stub invocation. 320 targeted regressions pass.
These are engineering metrics, not live savings or semantic quality. Resume
entry and demo instructions are ready in [Resume_Ready_Project.md](Resume_Ready_Project.md).
General semantic approval remains unresolved. GitHub stays postponed.

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
