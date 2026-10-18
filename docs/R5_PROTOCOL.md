# R5 development measurement protocol

Version r5-runtime-development-v1. Scope: development investigation and a
prospective quota-bounded live development pilot. This is NOT an untouched final
quality evaluation or a retrospective preregistration of the first replay run.
The initial replay findings were seen before this protocol was written. Historical
Phase A and the exposed reserved FAQ questions are not relabeled as new test data.

## Source, scenarios and policies

Use only the R2 pinned CPython v3.12.0 Programming FAQ preparation. Verify source
and LICENSE checksums using the existing manifest. The runner rejects a corpus
whose canonical digest differs from R2's recorded corpus. Development contains
22 source questions from two topic sections. Public pretraining exposure is
unknown. This is source documentation, not user traffic or answer-reuse gold.

Offline request stream: for each question, ordered seed, constructed exact repeat,
authored wording wrapper, authored negation/premise stress, different tenant and
changed revision marker. There are 132 events. The changed revision is a synthetic
scope-invalidation stress with unchanged reference text, not a real document edit.
Variant semantics and answer substitutability are unlabeled.

Three independent R3 GatewayService/cache arms: off, exact, guarded with an empty
actual-answer approval set. A fourth unsafe similarity-only approval ablation is
permitted exclusively with source replay, never live. Threshold .85, margin .05,
lexical hashing encoder, default bounded cache/TTL. These are development settings,
not calibrated parameters. Stream order is fixed; within-event arm order is
shuffled using seed 17. Each arm sees only its own past; no approvals are mined
from later answers. Empty approval coverage is deliberate and must be disclosed.

Offline source replay returns reference prose even for unsupported stress requests.
It does not simulate a correct fresh model response or validate the stress labels.
No invented token usage, network delay or inference cost is attached to replay.
Local wall-clock overhead is recorded but is not live latency evidence.

Agent experiment: same 22 development questions, seed then exact repeat, 44 tasks
in each of off/exact. Same document corpus, model instructions and scoped memory
behaviour in both arms. Actual R4 search/read/checkpoint loop runs. Planning and
review still run; all work is counted. Plans, draft answers and reviewer approval
are scripted source-replay fixtures. Auto completion is fixture-only and does not
stand for human task approval. No live agent or guarded agent comparison is claimed.
Agent arm order is fixed off then exact; fixture timings cannot establish latency
benefit. Preserve negative results if contextual memory prevents draft cache hits.

## Prospective live pilot and stopping

Select the first question from each development topic group in sorted group order.
For each: seed, constructed exact repeat, wording wrapper, negation/premise stress.
Eight events across off/exact arms, at most 16 generation attempts. The redundant
empty-approval guarded arm is omitted from this first live collection to conserve
quota; a reviewed-approval semantic live comparison is still pending. No remote verifier
or unsafe similarity-only live arm. This is a two-question/two-topic development
pilot and cannot certify precision or generalize across domains.

An explicitly selected `:free` model is required. FreeOpenRouterProvider enforces
zero maximum prices and disables paid fallback. Reasoning is requested off; output
limit 1024. Never silently substitute models or alter the profile after a failure.
A revised profile uses a new documented development run, retaining the old failure.
No call executes without --execute and an environment API key. Key values are
never written to evidence. Do not paste keys into chat.

Use the same project journal path as earlier pilots. Durable project caps remain
40 attempts/UTC day, 10/minute; this run cap is 16. Other account activity is not
visible locally. Reservations, failures and uncertain calls consume quota. No
automatic retry or sleep. Pause on quota, rate limit, incomplete output or uncertain
transport; resume the exact command. Unknown in-flight attempts require inspection.
A resumed run reconstructs caches from journaled calls; its timings are explicitly
invalid for live latency comparison. Reconstructed answers/work counts remain
inspectable. Do not reset run IDs or journal files to evade limits.

Runtime bytes, corpus/events digest, protocol bytes, model/profile, run ID and
output path are bound into the journal fingerprint. Output creation is exclusive.
After code changes, an old live run must use its frozen implementation to resume.

## Endpoints, review and gate

Report all incoming events, hits/eligible hits, exact/non-exact tiers, errors,
actual provider attempts and observed usage where present. Missing usage stays
unknown, especially failed calls. Include whole-request mean/p50/p95; distinguish
replay/reconstruction from uninterrupted live results and disclose transport scope.
Do not label local replay provider invocations as real model calls avoided.

Blinded answer packet binds request, revision and actual returned answer; private
mapping binds each arm and full response hash. Identical answer/question copies
share a review item. Reviewer fills correctness, reference support, declared name
and notes. Reviewer identity/independence is self-declared, not authenticated.
Workflow approval and policy/model output cannot substitute for independent human
labels. Reconcile labels against immutable answer bindings; altered/duplicate IDs
fail. Report missing labels as unknown, unique reviewed answers and topic groups
alongside event counts. Never count repeats as independent safety observations.

All existing precision/quality/net-utility aspirations remain unchanged. No gate
can pass without sufficient independent evidence. There is no safety confidence
bound for this tiny pilot. A performance gain in source replay does not pass live
utility. Free observed inference charges are zero in both arms; no monetary saving
or paid-price projection is reported. The R5 result may close as a completed
inconclusive/blocked investigation while quality/live validation remains pending.
