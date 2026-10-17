# Recovery execution roadmap

Status: R1 and R2 workload investigations are complete and published. R2 has a
pinned domain reference corpus and a descriptive OASST1 prevalence audit.
Representative production traffic, answer quality and live utility remain
unestablished. R3 may proceed as development-only matcher evaluation; no
semantic quality or live utility gate is declared passed.
Latest instruction: finish, debug and validate a phase before committing and
pushing its coherent changes. This replaces the earlier small-commit workflow
for future work. Historical Phase A/B/C files and spent evaluation artifacts
remain unchanged. Recovery stages below use R names to avoid rewriting history.
This plan is included in the R1 phase closeout.

## Delivery window

User-requested deadline: a ready project within 5–7 days of October 17, 2026.
Target demo/report delivery October 22; maximum delivery window October 24.
This bounds scope, not evidence claims. R1 diagnosis first; R2 data and live
answer collection next; R3 matcher/calibration and R4 serving integration take
priority over broad agent features. R5 review/evaluation runs as soon as its
prerequisites pass; R6 hardening closes the window. If quality evidence remains
underpowered, deliver an honest scoped result and limitations, not invented
confidence. A validated exact-cache result takes precedence over unsafe semantic
coverage. No new paid dependency is authorized by this schedule.

## Engineering judgment

Continue the project as a workload-aware, quality-constrained LLM response
cache. Caching is an established technique; novelty alone is not the project's
strength. The value must come from measured safe reuse, request/context
isolation, reduced provider work, useful latency and reproducible evaluation.
A general cache cannot manufacture reusable answers in mostly unique traffic.
A simpler exact cache is a valid winning baseline if semantic verification
costs more than it saves. Adding agents does not repair an ineffective matcher.

The previous constructed development run bypassed 94.425% of requests before
matching. Only 223 of 4000 were eligible. The lexical matcher served 50 hits,
all with fixture answers and no independent review. These diagnose a restrictive
policy and a workload mismatch, not semantic accuracy. The current gateway
serves exact caching only; experimental semantic matching is benchmark-only.
Earlier live connection checks returned no admissible complete answers. A
successful guided repeat demo would establish wiring, not representative savings.
Sources and derived values: RECOVERY-PLAN-BYPASS and FREE-LIVE-* in FACTS.md.

## Objective and metric contract

Maximize net provider work saved and useful latency improvement, subject to
answer quality and isolation constraints. Report denominators, misses, bypasses,
failed attempts, verifier work, cache occupancy, cold/warm state and source mix.

- Reuse precision: independently judged safe accepted reuses / reviewed accepted
  reuses; semantic and exact results separate. Missing labels remain unknown.
- Answer correctness: returned answer versus task/reference or blinded human
  rubric. A cached answer can be equivalent to a wrong fresh answer; agreement
  alone is not correctness.
- Hit rate: served hits / all incoming requests. Also report hits / eligible
  requests, exact versus semantic, and correct-reviewed-hit coverage. Do not
  quietly exclude bypasses or cold starts to inflate the main figure.
- Opportunity capture: safe hits / independently audited prior answer-reuse
  opportunities within the same scope, TTL and available history. Pair-dataset
  paraphrase labels are only a proxy until actual answer substitutability is
  assessed. Never use future answers to approve a past hit.
- Net provider attempts and tokens: cache arm includes generation, all verifier
  calls, failures and retries. Report attempts and tokens separately: an LLM
  verifier can erase call savings even when it saves output tokens.
- Latency: whole request including retrieval, policy, verification and generation.
  Record mean, p50 and p95 separately, successful and failed request durations,
  hardware/concurrency and cold/warm state. Hit-only latency is not the headline.
- Streaming: time to first token and total completion time separately; a cached
  complete answer does not automatically have the same user experience as SSE.
- Monetary cost: free-provider observed bill is zero in both arms. Claim actual
  calls/tokens saved. Any paid-price projection must be explicitly hypothetical,
  dated, model-specific and include all verifier/provider work; local compute
  and memory overhead are disclosed rather than assumed free.

Proposed goals, not results: reuse precision point estimate at least 99%, with
the two-sided 95% exact upper bound on incorrect reuse at most 2%; paired
answer-quality difference CI lower bound at least -2 percentage points. Utility
aspirations on the disclosed target workload: at least 20% fewer total provider
attempts, 20% fewer provider tokens, and 15% lower mean whole-request latency.
Report p50/p95 even if unchanged or worse. These goals do not imply a universal
minimum hit rate. Feasibility and sample planning happen before preregistration;
criteria cannot change after seeing a new untouched evaluation outcome.

## R1 — Explain where reuse is lost

Work: add a reproducible diagnosis command and decision waterfall: eligibility
reason, complete-answer admission, exact opportunity, retrieval availability,
scoring rejection, verifier rejection/failure, eviction/expiry, final outcome.
Reconcile events and provider accounting to the trace. Audit the current
static-question rules against representative supported and deliberately unsafe
requests. Distinguish missing instrumentation from measured zero counts.

Deliverables: machine-readable diagnostics, readable source/domain breakdown,
policy-scope contract, regression checks for instrumentation accounting and an
error taxonomy. Incorporate the running guided pilot when its evidence arrives;
continue offline diagnosis without waiting for it.

Exit: every request in the audited development run has an explained terminal
outcome, arithmetic reconciles, and no safe/unsafe claim is inferred merely from
a rule or similarity score. Identify the dominant losses and interventions.
Debug and close this phase before its commit/push.

## R2 — Establish workload fit and honest reuse opportunities

Work: select one useful primary scope: versioned technical Q&A with fixed
instructions and content revision. Verify public data terms before acquisition.
Keep three separate suites: chronological natural traffic for prevalence,
domain QA for correctness/reuse, and adversarial stress for safety. QQP/PAWS
can expose matching errors but cannot alone prove answer reuse or deployment
savings. Existing exposed development data stays development-only.

Preserve tenant, conversation, source and topic grouping; split before mining
pairs or generating variants. Cluster related prompts/threads to prevent train,
calibration and test overlap. Record checkpoint training exposure and unknown
pretraining contamination. Mark authored repeat variants as constructed. Never
pool a manufactured repeat-heavy stream into the natural-traffic headline.
Audit representative hits AND misses to avoid accepted-only selection bias.
Keep a new untouched cohort unavailable to tuning; do not reuse historical A.

Deliverables: data/source manifest, download/preparation scripts, checksums,
grouped development splits, label rubric and sample of audited prior-answer
opportunities. Identify stable classes that could support broader eligibility
without broadening access to private, dynamic or unsupported content.

Exit: documented rights/provenance and split checks pass; observed opportunity
and accounting show whether meaningful savings could be feasible. If natural
traffic has too little reuse, report that and narrow to a real useful domain;
do not tune on the future test or fabricate a universal high-hit workload.

## R3 — Build a matcher that earns its overhead

Work: retain exact lookup as the first path. Compare cheap lexical, pinned dense
and hybrid retrieval, then a local pair reranker on candidates. Select using
calibration data with actual reuse labels and adversarial negatives: changed
entities, quantities, units, direction, negation, code/input, requested depth,
versions and temporal/context scope. Broader eligibility must have its own
labeled validation; deleting safeguards is not a matching improvement.

High-confidence, validated classes may reuse locally. Ambiguous cases use a
remote verifier only when its measured cost/latency is worthwhile; otherwise
miss and generate. Reject uncertain or incomplete decisions. Do not require an
LLM verification call on every semantic hit: that can save no provider calls.
Cache validated decisions only under complete scope and bounded freshness.
Do not equate paraphrase or shared topic with an answer that satisfies all
incoming instructions. Measure candidate recall separately from approval.

Compare no cache, exact, unsafe similarity-only ablation and guarded semantic
policies. Plot precision versus coverage and net work/latency rather than
maximizing one headline hit rate. Calibrate category thresholds only with enough
independent labels; abstain elsewhere. No automatic online threshold relaxation
based solely on noisy self-generated labels or accepted hits.

Exit: a frozen development candidate improves useful reuse over exact while
meeting the declared development quality constraint and net work accounting.
If not, close the investigation honestly and keep exact caching or narrow the
semantic domain. A failing matcher cannot silently become the serving default.

## R4 — Integrate and measure the actual serving path

Work: integrate only the validated policy behind opt-in configuration, with
shadow mode before serving semantic answers. Bind tenant, model identity,
instructions/history, generation settings, content revision and applicable
tool/environment state. Support fixed system prompts with exact context scope
where validated; keep unsupported history/dynamic requests bypassed.
Do not reuse contextual answers merely because the final question matches.

Profile embedding/reranking overhead, retrieval/index scale, memory budgets,
TTL/invalidation and concurrent duplicate misses. Optimize what profiles show;
add approximate indexing only when scale warrants it. Enforce an overhead
budget derived from measured generation time and achievable safe hit share.
Evaluate miss and tail latency as well as fast hits. Keep SSE bypass unless a
separate bounded cache-serving contract is implemented and tested.

Deliverables: feature-controlled gateway integration, isolation/invalidation and
concurrency regressions, reproducible load profiles, cold/warm measurements,
resource bounds and configuration documentation. Fixture load tests are marked
functional; they cannot establish free-provider performance.

Exit: serving and benchmark decisions agree under the same configuration,
unsafe/state-crossing reuse is rejected, failures preserve accounting, and the
profile supports a plausible whole-workload latency benefit. Debug before push.

## R5 — Independently review live answers and evaluate once

Work: finish a complete-response pilot first. Spend the shared free quota on
planned answer collection and paired cache/fresh comparisons, rather than
re-running unchanged sanity checks. Continue offline work while local calls
run. Record provider/model/profile/date, failed attempts, usage and hardware.
Freeze workload/group selection, order, policies, safety/utility gates, quality
rubrics, sample-size reasoning and analysis BEFORE the untouched evaluation.
Predeclare batching across days, interleave arms, and account for provider drift.
Use new journals for protocol identity, not to reset shared project quotas.

Obtain independent declared human reviews of returned answers, blinded to arm
where possible, with disagreement resolution and missing/conflicting labels
unknown. Automated verifier labels remain policy signals, never independent
human ground truth. Report reviewed coverage and sampling weights if review is
sampled. Review enough groups; many repeats of one answer are not independent.

Use exact binomial intervals for suitable independently sampled decisions and
group-aware intervals for clustered streams; paired group bootstrap for work,
latency and task quality. No early stopping when results happen to look good.
Zero errors in 183 independent reviewed hits would only put the two-sided 95%
upper incorrect-reuse bound below 2%; this illustrative calculation does not
ensure sample sufficiency for quality noninferiority, clusters or nonzero errors.
At a free quota, credible evidence may require multiple weeks. Choose precision
and power transparently; if insufficient, label the result inconclusive.

Exit: immutable artifacts reproduce all metrics and show pass, fail or
inconclusive against preregistered criteria. No post-test retuning and rerun of
that same cohort. Failed outcomes remain published; any revised model requires
a fresh future evaluation design and untouched data. A favorable point estimate
alone does not clear a confidence-bound gate.

## R6 — Agent utility, final hardening and defensible presentation

Work: extend the bounded specialist/reviewer workflow only after cache utility
is established. Use task repetition/shared subproblems with recorded prevalence,
strict input/tool/data-version scope and current correctness checks. Tool results
are not reusable without a freshness/state contract. Include reviewer and tool
work in the task comparison. Agents must demonstrate task-level benefit rather
than merely adding calls to create an impressive diagram.

Predeclare cache-on/off task evaluation and independent correctness checks;
compare task quality, calls, tokens and time. Do not inherit cache precision as
proof of agent task accuracy. Finish fresh-clone, integration, failure recovery,
security, packaging and end-to-end regressions; investigate remaining failures.
Publish the final report, demo, limitations and only supported resume claims.

Exit: a reproducible task report, working clean installation/demo and final
claim audit. If agent overhead erases benefits, keep it as an explicitly bounded
integration example; lead the project with the useful caching result instead.

## Phase delivery and failure policy

Each phase ends with a short closeout: what changed, raw evidence, tests,
failures/fixes, known limits, exit-gate decision and next action. Run appropriate
checks and secret scans before one coherent phase commit/push; no intermediate
micro-commits. A blocked closeout may be committed as completed investigation,
clearly labeled blocked, never as achieved performance. Track this distinction.
Historical reports remain intact. Update FACTS before metric claims elsewhere.

Debug throughout each phase and perform final cross-component hardening in R6.
Leaving all debugging until the end could invalidate measurements collected
with broken instrumentation. A dependency failure blocks deployment or live
claims, but not independent offline work on diagnostics, data or tests.

User action now: let the current local pilot finish and share its combined
results (or pause evidence). Later, independently reviewed labels are needed;
the assistant cannot invent human reviewers. Dataset research, implementation,
offline runs, debugging and phase-closeout publication remain engineering work.
No new key, paid service or dataset search is requested from the user now.

Primary background sources checked for this plan: MeanCache
https://arxiv.org/abs/2403.02694 and SCALM https://arxiv.org/abs/2406.00025
establish that semantic caching and workload/context limitations are existing
research topics; their reported gains are not transferable project metrics.
OpenRouter https://openrouter.ai/pricing documents free quota/failed-attempt
constraints; local project caps remain the conservative controls in free_pilot.md.
