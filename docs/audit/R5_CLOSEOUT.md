# R5 development investigation and live-readiness closeout

Date: 2026-10-18 (Asia/Calcutta). Status: offline engineering/investigation
complete; **quality and live-utility gate INCONCLUSIVE / PENDING COLLECTION**.
This is a completed blocked investigation, not delivery of missing live validation.
Historical Gate A remains failed. No semantic-quality or savings gate is passed.

## What ran

The actual R3 gateway/runtime compares independent off, exact and guarded caches.
An unsafe similarity-only approval ablation is restricted to offline source replay.
The R4 agent executes actual search/read tools and SQLite task/memory transitions.
Official CPython FAQ source and license were recovered at the R2 pinned revision,
verified against their checksums, and reproduced R2's canonical corpus digest.
Only 22 development questions were used; no exposed test split was relabeled sealed.

The initial replay was exploratory. `../R5_PROTOCOL.md` was written after seeing
those findings and before a prospective live run. The final replay binds protocol,
source, events and runtime bytes. It is NOT retrospectively preregistered research.
Within-event request arm order uses seed 17. Task fixtures have fixed arm order.

## Request replay outcome

132 events per arm: 22 each of seed, constructed exact repeat, authored wording
variant, authored negation/premise stress, other tenant and revision-marker change.
Each provider invocation returns source reference prose, not fresh model output.
Stress answers can therefore be inappropriate: correctness remains unknown.

| Arm | Replay provider invocations | Exact hits | Semantic hits | Errors |
| --- | ---: | ---: | ---: | ---: |
| Off | 132 | 0 | 0 | 0 |
| Exact | 110 | 22 | 0 | 0 |
| Guarded, empty approvals | 110 | 22 | 0 | 0 |
| Unsafe similarity-only offline ablation | 109 | 22 | 1 | 0 |

Exact/guarded hit fraction is 22/132 = 16.67% on this disclosed constructed stress
mix. The 22 fewer replay invocations are functional work accounting, not observed
model/API savings. No real token usage is invented. The unsafe arm's one semantic
reuse is an authored wording wrapper; no independent label establishes correctness.
Other tenant/revision cases miss. Guarded semantics adds no accepted reuse under
an empty approval policy; this cannot diagnose semantic precision or approved-pair
coverage. It makes the absence of a generalizing approval mechanism explicit.
Local timing is in `R5_DEVELOPMENT_SUMMARY.json`, clearly marked non-live.

## Agent replay outcome

44 tasks per arm (22 questions, each repeated), memory enabled in both arms:

| Arm | Simulated provider calls | Actual local document-tool calls | Fixture-completed tasks |
| --- | ---: | ---: | ---: |
| Off | 132 | 88 | 44 |
| Exact | 132 | 88 | 44 |

No provider-call reduction was observed for this agent configuration. Prior-task
memory changes full drafting inputs, so an identical user question is not an
identical stage request. Planning and review also still run. This is a real
architectural limitation exposed by the replay, not an isolation bug to bypass.
Scripted model plans/reviews and source-answer replay do not establish real task
success. No real tool-result memoization or model-token saving is claimed.

## Deliverables and evidence

- Local generated evidence bundle: final trace, source-replay agent trace,
  blinded answer template, private arm mapping, summary and file-hash manifest.
  The raw bundle is not published. Automatic approval review rejected uploading
  it because answer traces and the private reviewer mapping were potentially
  sensitive. Public delivery contains aggregate results and reproduction code.
  Recreate the replay with `scripts/run_r5_paired.py` after preparing the source.
- `R5_DEVELOPMENT_SUMMARY.json`, `R5_AGENT_SUMMARY.json`, `R5_VALIDATION.json`.
- `scripts/run_r5_paired.py`: actual runtime replay, explicit free-only paired
  collection, shared durable journal/caps, exclusive exports and evidence archive.
- `scripts/review_r5_answers.py`: rejects altered/duplicate answer bindings;
  missing labels remain unknown and reviewer independence is self-declared.
- `../R5_LIVE_RUN.md`: verified plan command and user foreground collection steps.

## What is blocked and what happens next

External model calls: zero. Independent human labels: zero. The workspace has no
OpenRouter credential; the user's existing environment key is on their machine.
The prepared live pilot has eight events, two arms, at most 16 attempts, within
the unchanged shared 40/day and 10/minute project caps. Omitting the redundant
empty-approval guarded live arm conserves quota. Free-provider availability and
complete outputs are not guaranteed. Pause/reconstruction can invalidate latency.

User assistance is now concrete: run the published foreground command and return
its evidence archive; then provide actual independent review of the prepared
answer packet. No paid service, new key disclosure or dataset hunt is needed.
The two-topic/two-question pilot is underpowered and cannot yield certified safety.
A useful reviewed-approval semantic live comparison and live agent comparison
remain missing; final metrics and resume claims must wait for that evidence.

R6 development refinement should separate task orchestration/memory from stable,
fully scoped document-answer stages, preserve all answer-relevant context, and
measure its effect against this retained negative baseline. Broader semantic reuse
needs actual-answer approvals or a validated verifier, including its full cost.
Do not lower similarity thresholds or erase memory/context merely to inflate hits.
Any changed policy needs appropriately fresh evaluation; these spent development
runs are not promoted to untouched test evidence.
