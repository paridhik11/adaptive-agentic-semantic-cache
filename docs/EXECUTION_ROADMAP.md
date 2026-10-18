# Recovery execution roadmap

Status (2026-10-18, Asia/Calcutta): R1 and R2 are published. R3 cache-lifecycle
engineering is complete with fixture evidence; its quality/utility gate remains
open. See `audit/R3_CLOSEOUT.md`. R4 bounded documentation-agent engineering is implemented; see `audit/R4_CLOSEOUT.md`. R5 offline investigation is published as inconclusive; live collection/review remain pending (`audit/R5_CLOSEOUT.md`). R6 refinement must preserve the negative development baseline.
The user accepted the Project 7 + bounded Project 15 recommendation on this
date. The active remaining sequence below replaces the earlier matcher-first
R3-R6 sequence: R3 completes a working cache slice, R4 adds a bounded real agent
client, R5 benchmarks both paths, and R6 refines and publishes the results.
Historical numbered phases and Phase A/B/C artifacts remain unchanged. This
revision changes execution order; it does not declare any quality gate passed.

## Accepted scope and guide alignment

Build the semantic caching middleware from Project 7, used by a bounded
version-aware documentation agent drawn from Project 15. Start with the pinned
Python technical-reference corpus from R2. Preserve provider-agnostic request
contracts and make the corpus/domain replaceable. This is a scoped combination,
not delivery of the guide's entire general-purpose multi-agent platform.

The implementation must demonstrate an actual request/response lifecycle before
more standalone matcher research. The agent must add structured planning,
allowlisted document tools, persisted task state, scoped memory and resumable
human review; renaming a fixed specialist/reviewer prompt pair is insufficient.

Constructed repeat workloads are valid controlled experiments when labeled and
reported with their repeat shares. Natural-traffic prevalence is a separate
analysis and is not a prerequisite for building this vertical slice. Include
unique/cold traffic and hard near-misses so the workload can expose a loss as
well as a gain. Different users may share approved public knowledge within the
same tenant/access scope. A source user ID is not a deployment tenant; absence
of same-user duplicates alone does not show that shared caching is useless.

## Delivery window and estimates

All dates below are estimates in Asia/Calcutta, assuming continued active
working sessions. They do not schedule background execution. The original
user deadline remains October 24, 2026; it has not silently been extended.

| Recommendation | Active phase | Target | Definition of delivery |
| --- | --- | --- | --- |
| 1. Finish a complete cache path | R3 | October 19-20 | Working exact/semantic candidate path, storage, scope, invalidation and end-to-end functional evidence |
| 2. Connect a bounded agent client | R4 | October 21-22 | Structured plan, real document-tool interface, persisted task state, review and approval/resume; fixture demo plus explicit live mode |
| 3. Compare identical tasks cache off/on | R5 | October 22-23 | Frozen protocol, controlled offline workload, quota-bounded paired live sample and review status; pass/fail/inconclusive report |
| 4. Refine and publish | R6 | October 24 | Development fixes, appropriate fresh evaluation if policy changes, clean-install demo, limitations and supported claims |

October 22 is the integrated demo target; October 24 is the scoped four-step
engineering/report target. October 25-28 is a planning contingency if integration,
provider availability, review turnaround or failures prevent the target; this
is a forecast, not a new user-approved deadline. Broader domains, the full
Project 15 platform and a tightly certified safety bound are additional scope.

Independent review and a free-provider quota can make statistical validation
take longer than implementation. In particular, the existing strict safety goal
requires more evidence than a tiny demo provides. October 24 cannot guarantee
positive savings or a passing confidence-bound gate. Report an underpowered
result as inconclusive, with completed engineering and missing evidence listed
separately. Do not mark a pending evaluation passed to meet the date.

No paid service or inference is authorized. Keep the existing durable project
caps; preflight all live work, count failures and retries, and confirm current
free model availability before a run. User-initiated foreground collection may
begin during R3 development; the new final evaluation stays frozen and separate.
Do not spend the quota on repeated unchanged connection checks.

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
policy and a workload mismatch, not semantic accuracy. R3 subsequently integrated opt-in guarded semantic serving and shadow mode;
quality and utility remain unvalidated.
Earlier live connection checks returned no admissible complete answers. The
later guided exact-repeat demo establishes wiring, not representative savings.
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

## R3 — Finish the complete cache serving slice (recommendation 1)

Build one request path: receive a scoped request, try exact reuse, retrieve a
semantic candidate, approve or abstain, call the provider on a miss, and store
only a complete successful response. Store actual response/candidate lineage;
a reference answer or canned fixture must never masquerade as fresh generation.
Expose cache off, exact, semantic shadow and explicit experimental semantic modes.
The default serving policy remains conservative until development labels support
specific semantic reuse classes. A runnable functional semantic path is not a
safety certification and does not require silently enabling it for all traffic.

Use the existing gateway/cache/storage rather than a new parallel service.
Explicitly bind tenant, trusted access scope, model, instructions, preceding
context, generation settings, corpus/document revision and relevant tool state.
References make an answer inspectable, not automatically correct. An updated
revision must exclude stale entries. Different output requirements, entities,
quantities, negation, code input and requested scope require independent checks.

Compare lexical, pinned local dense and hybrid candidates only as needed to
choose a development baseline. Keep exact lookup first. Calibrate approval on
actual-answer reuse examples and adversarial negatives, distinguishing authored
intentions from independent human labels. Measure accepted and missed
opportunities. Use a remote verifier only if total measured work justifies it;
an extra model call on every hit can erase savings. No learned/adaptive
threshold claim is made without adequate feedback evidence.

Keep the existing bounded streaming bypass contract for this first slice.
Caching streaming output requires a separate completeness/cancellation contract;
it must not be implied by a working non-streaming cache. Keep persistent
storage, TTL, invalidation, payload bounds and concurrent identical-miss handling.

Deliverables: reproducible cache-off/miss/exact-hit/semantic-candidate demo,
provider and storage traces, scope/revision regression checks, explicit failure
paths and one coherent closeout. Fixtures prove functionality; an optional
capped live smoke confirms actual complete-response admission. Local models and
test doubles require no paid API calls.

Exit: end-to-end lifecycle and accounting tests pass; unsafe scope changes and
incomplete outputs cannot be admitted; opt-in semantic decisions retain
provenance. If semantic approval has insufficient quality evidence, deliver the
functional path with shadow/default abstention and record that quality remains
open. Do not describe this as validated semantic serving. Debug, then commit
and push this whole phase once.

## R4 — Connect the bounded documentation agent (recommendation 2)

Implement an application client of R3, not a competing cache. A supervisor
produces a schema-validated bounded plan with subtasks/dependencies. A research
specialist can invoke allowlisted local document search/read tools; an answer
specialist synthesizes evidence, and a reviewer checks the final deliverable.
In live mode the model must actually select registered tools where required;
scripted fixture tool sequences are labeled functional demonstrations.
Model-generated tool arguments are validated against schemas and the caller's
trusted scope. No unrestricted shell, arbitrary external writes or paid tools.

Persist task/checkpoint state, plan, tool outcomes, attempts and approval status
in SQLite using existing infrastructure. Supply task-local working context and
a small scoped store of prior completed tasks/evidence; retrieval is tied to
user/tenant permissions and source revision. Provide delete/invalidation.
Retrieved memories are untrusted suggestions, not correctness labels. Measure
whether they help rather than claiming the system learns simply by storing them.

Include bounded review feedback/retry behavior. Ambiguous evidence or a configured
approval requirement pauses at a durable checkpoint with task context and a
proposed next action. A human can approve, modify or reject and resume without
re-executing completed side effects. This is distinct from independent benchmark
answer labeling. A failure status alone is not a human-approval system.

All model calls go through the gateway with explicit budgets. Cache only stages
whose full inputs, evidence, tool state and freshness are defined. Read-only
tool-result memoization may use exact tool arguments plus source revision; it
is reported separately from saved LLM calls. Do not assume semantic similarity
of tool arguments makes their results interchangeable. Reviewer/model/tool work
always appears in task-level counts, even if it reduces the apparent benefit.

Deliverables: documented tool registry, bounded plan/execution loop, SQLite
checkpoints and scoped memory, review/resume interface, per-stage traces,
cache-on/off switch and reproducible fixture/live modes. A small browser page
or CLI is sufficient; a full general-purpose orchestration platform is outside
this delivery scope.

Exit: integration tests cover planning/schema failure, actual tool dispatch,
context/version isolation, memory invalidation, retry budgets, and restart after
human approval/rejection. Fixture replay is deterministic; any live validation
is explicitly labeled and budgeted. Commit/push only after the phase closeout.

## R5 — Compare identical requests and tasks (recommendation 3)

Freeze workload selection, group splits, stream order, policies, code/model/data
revisions, review rubric, endpoints, stopping rule and analysis before the new
final evaluation. Keep development, authored stress and untouched evaluation
separate. The existing reserved FAQ split is visible and is not automatically
an untouched sealed evaluation. Preserve historical failed cohorts unchanged.

Compare the same application and agent with cache disabled, exact caching and
guarded semantic caching. Include similarity-only as an unsafe offline ablation.
An optional no-agent baseline evaluates orchestration overhead separately; do
not attribute the benefit of adding/removing an agent to caching. Count the
full request/task lifecycle and every verifier, retry and tool invocation.

Use cold/unique traffic, exact repeats, reviewed semantic variants, changed
context, changed source revision and difficult non-reuse cases. Vary the
constructed repetition mix and publish its composition; do not select just
accepted hits. The guide's large repeated-query workload is a valid disclosed
stress/load test. It does not become natural production traffic because it
contains public questions. The OASST prevalence audit remains separate.

Run the larger functional/load suite locally with fixture or replay providers.
Mark simulated response times, replayed usage and source-answer replay explicitly;
these are not new live generation observations or paid savings. Use a planned,
stratified, quota-bounded paired live sample for real provider measurements.
Do not promise thousands of live baseline calls on the free quota. Preflight
cost/work and start collecting development answers early when a user foreground
session is available. Preserve incomplete attempts and interleave paired arms
where practical to reduce provider-time confounding.

Have independent declared reviewers judge actual answers and reuse decisions,
blinded to cache arm where possible. Review more than one topic/group; repeated
copies of one answer do not provide independent safety observations. Record
missing/disagreeing labels as unknown. Policy-judge output is never independent
human ground truth. Do not wait for human labels to run unrelated offline checks.

Report whole-workload hit and correct-hit coverage, exact/non-exact precision,
answer/task correctness, attempts/tokens, tool work, latency mean/p50/p95,
resource overhead, stale-answer reuse and scope violations. Measure first-token
and completion latency separately for streaming. Free-versus-free observed
inference charges are both zero; any priced projection is hypothetical and
must disclose dated prices and all extra calls. A high hit rate does not
necessarily improve tail latency.

Retain the metric goals above as aspirations, not guaranteed achievements.
Use grouped uncertainty for dependent tasks and paired samples. Zero errors in
183 independent reviewed hits is only an illustration for a two-sided 95%
upper incorrect-reuse bound below 2%; real clustering/nonzero errors need their
own sample plan. A small quota-limited pilot cannot certify that bound.

Deliverables/exit: immutable traces, review coverage, reproducible reports and
an explicit pass/fail/inconclusive decision against the frozen protocol. Pending
review or inadequate power is recorded, not replaced by self-labels. This phase
can close a completed inconclusive investigation; it cannot claim a safety gate
passed. Commit/push only the complete coherent evaluation closeout.

## R6 — Refine, harden and publish (recommendation 4)

Fix observed development bottlenecks rather than optimizing a headline hit rate.
Profile embedding/reranking, misses, concurrency and the reviewer/tool overhead.
Retain simpler exact caching if guarded semantics does not earn its overhead.
Keep unsupported classes abstaining. Optimize only within the declared scope.

If evaluation findings lead to policy changes, the revised version needs a new
untouched evaluation; do not rerun the spent test to manufacture a pass. If time
or quota prevents that, publish the earlier frozen result and clearly mark the
new implementation's validation pending. No comparative claim transfers to an
unmeasured version.

Finish clean-install/container smoke, integration/failure/restart checks,
resource/isolation checks, documentation and secret scans. Provide a short demo
showing fresh generation, exact reuse, semantic behavior, a source update,
a real tool-use task, and human pause/resume. Fixture/offline replay must be
obvious to the viewer. A useful minimal dashboard comes before optional extra
infrastructure; the guide's full Grafana stack is optional expanded scope.

Publish a report with workload construction, repeat shares, quality results,
failed attempts, overhead, uncertainty and limits. Correct any older resume
wording that overstates end-to-end savings. Link every measured performance
claim to the matching immutable run; do not promote proposed goals to results.
The final report may honestly conclude semantic utility remains unproven.

Exit: coherent clean-install demo and report reproduce the current claims,
known unresolved issues are explicit, and any remaining research gate is
labeled open. Commit and push this completed phase once.

## Phase delivery, commits and blockers

R1/R2 are completed history. Remaining R3-R6 correspond one-to-one to the four
accepted recommendations. Every phase receives implementation, focused tests,
debugging and an audit closeout before a single coherent commit/push. Do not
make intermediate micro-commits or commit this roadmap revision alone; include
it in the completed R3 cache-lifecycle phase. The closeout records scope, code/config, tests,
raw evidence, failures/fixes, gate decision, known limits and next action.
Run appropriate numeric/claim and secret checks before publication.

A completed blocked/inconclusive investigation can be committed with that
status; this is not delivery of a missing feature or a successful quality gate.
Offline engineering continues around missing quota or human review. Ask for
user assistance only when a concrete prepared live run, human-label packet or
other actual dependency is ready. Do not request new credentials, paid services
or an unguided dataset hunt. Never ask the user to open an unverified missing
artifact link.

Current user action: none required for the roadmap update or offline build.
When actual live generation or independent review is ready, present a working
command/review packet with its required effort and inference-attempt budget.
The assistant does not run foreground jobs on the user's laptop or work
unattended between chat sessions.

Primary background references retained from the earlier plan: MeanCache
https://arxiv.org/abs/2403.02694 and SCALM https://arxiv.org/abs/2406.00025.
Their results are not transferable project metrics. The uploaded BASWE guide's
Projects 7 and 15 motivate the build sequence; its advertised savings are not
a guarantee for this implementation.

R6 scoped agent-stage refinement was authorized on 2026-10-19 and passed its
preregistered offline engineering criterion; see `audit/R6_AGENT_STAGE_CLOSEOUT.md`.
The R5 live collection, independent review, semantic-quality gate and broader final
delivery/utility validation remain open. This is not full roadmap completion.
