# Zero-budget OpenRouter pilot

This path prepares a small real-answer development experiment using only explicit
OpenRouter free-model IDs. It defaults to planning; no requests are made until
`--execute` is supplied and `OPENROUTER_API_KEY` exists in the process environment.
Implementation regression tests used local fixtures. The first user-run live
connection check is recorded in `audit/OPENROUTER_CONNECTION_PILOT_V1.json`; it
yielded no admissible complete answers.

## One foreground session with visible progress

From the updated Windows checkout, with the key already in this PowerShell
session, run:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe scripts/run_zero_budget_session.py --execute
```

This opens `runs/guided-exact-v1/status.html` and prints progress in the terminal.
It runs six authored requests across two topics: seed, exact repeat and
paraphrase. The exact policy intentionally generates a fresh answer for the
paraphrase. You see each arm's status, model-call count and unreviewed answer
preview. Keep the foreground process running; it is not a background service.

The profile requests at most two concise sentences, a 512-token output limit
and OpenRouter's `reasoning: {"enabled": false}` setting. This setting is
[documented by OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens);
its effect on this live model still requires observation. Parameter support is
required and fallback remains disabled. The job stops on the first incomplete
answer or provider failure rather than spending the rest of the budget. It caps
total attempts at twelve; two exact hits would require ten generation attempts.
This constructed sequence demonstrates functionality, not real-traffic hit rate
or semantic safety.

Only a local rolling-minute quota pause can automatically wait and resume in
the same foreground job, for at most three minutes. Saved responses are replayed;
any reconstruction suppresses live latency comparisons. Daily limits, provider
errors and incomplete responses stop immediately. Run without `--execute` for a
plan; add `--no-browser` to keep progress in the terminal and local HTML only.

On completion share only `runs/guided-exact-v1/results.json` for analysis. It
packages the trace, unreviewed report and response diagnostics. It contains model
answers and local provenance paths, so inspect it before sharing publicly. On a
pause the page and `paused.json` give the reason. Rerunning a completed session
preserves evidence and makes no inference calls. Changed model/profile/code
requires a new run and directory; keep the shared quota journal.

The two earlier connection checks each used two live calls; all four were
truncated. The second produced partial text but still ended with `length`.
See `audit/OPENROUTER_CONNECTION_PILOT_V2.json`. Neither connection check
establishes successful-answer latency, cache savings or answer correctness.

## Quota and price controls

OpenRouter's [pricing page](https://openrouter.ai/pricing) documents the base free
allowance and notes that failed attempts count. Its
[provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection#max-price)
documents zero-price filtering. These provider rules can change; inspect the
current source and available model IDs before a real run.

The adapter permits explicit `provider/model:free` IDs only. It rejects automatic
model routing, caller-supplied fallback lists/provider settings, tools, web search
and plugins. Outgoing provider controls disable fallback and cap prompt,
completion and per-request prices at zero. Parameter support is required. This
can fail when no eligible provider exists; it never substitutes a paid model.
These are request-side restrictions, not an independent audit of upstream billing.

Defaults intentionally leave room below the documented allowance: a shared local
ledger caps project attempts per UTC day and rolling minute. Limits and run caps
are in `FACTS.md`. Every attempted call, including errors and uncertain outcomes,
is reserved transactionally before network I/O. All runs and models must share
one ledger. Usage from other apps, accounts or a different ledger is not visible;
this is not a query of your remaining OpenRouter quota.

On local quota exhaustion, the script prints a pause and earliest retry time;
the low-level pilot does not sleep or schedule itself. The guided wrapper above
can wait for a local minute window. A provider rate-limit response blocks new
project calls conservatively until the next local UTC day. Other provider errors
also pause immediately for inspection instead of automatically consuming the
rest of the pilot budget. Provider calls are never automatically retried.

## Prepare and inspect a plan

Use a development workload with provenance, source terms and group IDs. The pilot
selects the first development requests in input order and replaces their model
field with the explicitly selected free model. This is disclosed selection, not
a random or sealed sample. Use a representative hand-inspected pilot workload
before spending scarce calls on a larger development pool.

```bash
python scripts/run_free_pilot.py \
  --workload examples/benchmark/workload.fixture.json \
  --model PROVIDER/MODEL:free \
  --run-id pilot-dev-v1 \
  --output-dir runs/pilot-dev-v1
```

Replace the placeholder with a currently available explicit free model ID.
The existing project model `nvidia/nemotron-3-super-120b-a12b:free` appeared in
OpenRouter's public model catalog with zero prompt/completion prices when
checked during this implementation. That metadata check did not send a dataset
query, verify account access, or establish model availability by inference. The
plan command validates syntax and inputs without reading a key or creating the
ledger. It does not verify that the model is currently available. The bundled
fixture workload is for checking the pipeline; it is not representative real
traffic or a final scientific dataset.

Configure `OPENROUTER_API_KEY` securely in the environment that runs the command,
then repeat with `--execute`. Do not commit the key. On Windows PowerShell, read
it through a hidden prompt rather than including it literally in a command:

```powershell
$pilotSecret = Read-Host "OpenRouter API key" -AsSecureString
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new("", $pilotSecret).Password
```

The environment value is available to the local pilot process; configuring it
on your computer does not make it available to a separate ChatGPT workspace.
After running, remove it from the terminal session with
`Remove-Item Env:OPENROUTER_API_KEY`.

 Choose a fresh run ID for a
new experiment. The default journal is `runs/openrouter-free.sqlite`; keep this
same path across project runs so quotas remain shared.

Exact and similarity pilots can make one fresh and one cache-arm generation per
request. To test answer verification, add `--policy verified` and
`--verifier-model PROVIDER/JUDGE:free`; budget also covers a possible verifier
call. Verification is a cache-policy decision, not an independent review label.
The documented pilot limits keep the workload small and bound total attempts.
A run's total attempt cap persists across days and is not a daily allowance.

## Resume and interpret results

Repeat the identical command after a quota pause. The journal binds the run to
its source/selected workload hashes, policy, models, implementation-file hashes,
output location and attempt cap. Changed inputs/code are rejected. Previously
finished responses and failures replay at their original call ordinal; a changed
call sequence stops before any new request.

A call left reserved after a crash may already have reached OpenRouter. It stays
uncertain and is never automatically resent. Inspect it and provider activity
before deciding how to recover. There is deliberately no automatic "clear quota"
or "retry uncertain" switch. The journal contains plaintext model responses;
protect it and keep it out of Git. Keys are not stored in the journal.

A resumed run reconstructs the in-memory cache with replayed responses. It is
not a continuous real-time cache experiment across the intervening days; TTL
behavior follows reconstruction. Reports suppress all live latency comparisons
when any provider response was replayed. Record `execution_mode`, new calls in
this invocation, replayed calls and journal states alongside every result.
Logical provider-work counts describe the whole journaled experiment, not just
new calls in the final invocation. Keep separate uninterrupted runs for live
latency measurement.

Successful evidence is exported atomically to a new directory: trace, unreviewed
report, bound review template, offline review page and hash manifest. Existing
completed evidence is never overwritten. Independent human review is still
required before safe-reuse or answer-quality claims.

## Inspect an incomplete answer without another call

The first uploaded user-run connection check recorded live usage but no admissible
complete answers. Both attempts reached the workload's output-token cap. This is
consistent with truncation, but the original trace did not retain finish reasons;
confirm from the saved raw journal instead of guessing.

```bash
python scripts/inspect_pilot_journal.py --run-id connection-check-v1
```

This opens the journal read-only, requires no API key and makes no model calls.
It prints finish reasons, text/reasoning lengths, reported token counts and
response hashes; it does not export answer or reasoning text. The journal's
`complete` state means the HTTP response was saved, not that the answer passed
admission. New benchmark traces include this diagnostic metadata on rejected
answers and distinguish token-limit rejection from generic processing errors.

After inspecting the finish reason, an explicit `--max-output-tokens` can increase
the output limit for a new pilot run. Use a new run ID and output directory while
retaining the same quota journal. Old pilot evidence is preserved. Do not resume
an old run with changed request limits. This parameter also consumes reasoning
output capacity for models that include reasoning in their completion budget;
raising it does not guarantee a complete answer.
