# Prospective bounded live-agent development pilot

Defined October 20, 2026 before any calls in this experiment. This is a separate
agent experiment, not an extension/reset of the exhausted R5 request pilots.
The original R5 combined cap remains spent. No live calls are made by preparation.

## Question and comparison

Use the exposed development FAQ question about tuple/list conversion, its pinned
CPython 3.12 source revision, and one allowlisted document. This is development
material, not a fresh or representative quality evaluation. Public pretraining
exposure is unknown. Same model, source, task settings and human-approval rule in
both arms. Two tasks per arm: seed then exact repeated question. Interleave arms
in fixed off/exact order for each task. Fixed ordering is a confounder; no latency
confidence claim follows from this tiny workload.

Run the actual DocumentAgent planning, local search/read, drafting and workflow
review. Require human approval: stop tasks at AWAITING_APPROVAL, never pretend
model approval is independent human approval. No simulated live plans/reviews.
The static R6 drafting contract is opt-in in both arms; only exact arm enables
reuse. Full model-visible requests remain bound into cache keys. Planner/tool
variation may prevent draft reuse; preserve zero savings if that occurs.

## Budget and stopping contract

New experiment ID agent-live-v1, maximum 12 generation attempts: four tasks times
three calls (plan, draft, workflow review), zero revision retries. Budget for zero
cache hits; never rely on a predicted hit to fit the cap. Existing shared durable
journal required, so earlier usage remains in project accounting. Limits remain
40 per UTC day and 10 per rolling minute. Read-only preflight checks that daily
remaining allowance can cover all worst-case unsaved attempts before executing;
otherwise print NOT_STARTED and the next UTC-day time. Minute quota/rate-limit
pauses remain possible, and other account activity is invisible locally.

Explicit free model nvidia/nemotron-3-super-120b-a12b:free, zero maximum prices,
no provider fallback, reasoning requested off. Output allowance 4096; draft asks
for fewer than 180 words, source-only reasoning and document-ID citations. No paid
services, automatic retries, key logging or keys in command arguments.

Plan is default; --execute and the existing environment key are required for
inference. Corpus, code/protocol hashes, profile, selected document, output path
and allowance bind into the run fingerprint. Settings/source changes block resume.
Failed/uncertain/truncated recorded calls require inspection rather than reruns.
On a quota pause rebuild isolated local task databases and in-memory caches from
journaled calls in the same deterministic order. UUIDs are not model inputs.
Reconstruction timings are invalid for uninterrupted live-latency measurement.

DocumentAgent currently catches provider pauses into a failed task state. The
runner observes the original pause and immediately stops collection, retaining
that task state in its snapshot. A permissible quota resume starts a fresh local
reconstruction; it does not modify or falsely complete that failed checkpoint.

## Evidence and outcomes

Every completed, paused or failed collection exports a uniquely named private ZIP,
with task states, actual gateway requests/responses, saved provider responses,
usage when present, selected-run journal snapshot, source/license, configuration,
review packet and hash manifest. No API key or unrelated run payload is exported.
Do not send the whole archive to a blinded reviewer. No overwrite of earlier ZIPs.
Host/process termination or storage failure can still interrupt export; the shared
journal remains the durable source for spent attempts, including uncertain ones.

Review packets carry the actual drafting system instruction, source revision,
evidence and explicit relevant memory. They hide arm identity and judgments.
Correctness and source support are distinct; state permitted background knowledge
before any fresh final evaluation. This pilot packet asks for source-grounded
judgments, permits normal language/mathematical definitions as interpretive
background, and flags unsupported substantive domain claims separately. Do not
retroactively apply this rubric to the recorded Antigravity labels.

Engineering completion means all four tasks reach AWAITING_APPROVAL without
transport/protocol failures and within the budget. It does not mean their answers
are correct. Report per-stage generation/cache counts and actual local tools.
A reuse observation requires an actual draft HIT; do not retune to obtain one.
No independent accuracy/safety/population savings claim or quality-gate pass is
possible from these four dependent tasks on one exposed topic. Final evaluation
and broader guarded semantic approval remain separate unfinished work.
