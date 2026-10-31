# Trace: the integrated local documentation workbench

A single interface exercises the existing DocumentAgent, read-only source tools,
full-request exact cache and durable human approval. It displays actual work from
the backend. The first experience requires no API key and makes no external calls.
Its scripted source responses are prominently labeled Reference replay.

## Launch on the current Windows checkout

In your existing repository folder, using the Python environment from your pilots:

```powershell
git pull --ff-only
$py = "C:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache-pilot\.venv\Scripts\python.exe"
& $py scripts/run_demo.py
```

Open http://127.0.0.1:8765 in your browser. Keep the terminal running; Ctrl+C stops
the server. There is no model quota cost for this default reference-replay mode.
The source excerpts and license are bundled, so no corpus preparation is required.

If starting from a clean Python environment, install only the web runtime first:

```powershell
python -m pip install -r deploy/requirements-gateway.txt
python scripts/run_demo.py
```

The launcher binds to literal loopback, not a public network interface. This is
a local single-operator application; it is not a multi-user hosted deployment.
Use --port to choose another port and --directory to select another local data
parent. Reference and live tasks/caches use separate subdirectories.

## Try the complete flow

1. Choose the tuple/list question or type it into the question box.
2. Keep Reuse exact matches enabled and choose Find an answer.
3. Follow planning, real local search/read, drafting and workflow review.
4. Read the source and approve, edit or reject the draft. The decision is durable.
5. Choose Ask this again. The draft can come from the exact cache, while planning
   and review still run. Reference replay gives three then two scripted calls.
6. Turn off exact reuse and repeat. The reference-replay path uses three calls.

Counts are actual calls within the chosen mode, not token-cost projections. Timing
is local agent processing, excluding browser transit. The source library shows
three public CPython 3.12 excerpts with immutable source revision and bundled
license. This is exposed development material, not final evaluation data.

Recent questions and decisions persist in runs/workbench/reference/tasks.sqlite;
exact drafts persist in cache.sqlite with source/profile-specific namespaces.
The UI shows the latest twenty compatible tasks. The call counter is explicitly
current-process; restarting the server resets that counter, not stored history.
An interrupted active task can be selected from recent questions and continued.
The browser holds only a temporary local session capability, never the provider
key. Untrusted answer/source/question text renders as text, not injected HTML.

## Optional explicitly live mode

This is an interactive demo with a separate fixed twelve-attempt allowance under
product-demo-v1; it does not resume or reset spent R5/agent experiments. Launching
--live is an explicit decision to use that allowance. The existing shared journal
and environment key are required. No live execution was performed while building
this interface. It supports the same single-operator approval flow.

```powershell
$journal = "C:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache-pilot\runs\openrouter-free.sqlite"
& $py scripts/run_demo.py --live --journal $journal
```

Only the selected :free model is permitted, reasoning is requested off, maximum
prices are zero and provider fallback is disabled. Each new task checks remaining
run/daily capacity for three calls before starting. The shared journal enforces
rolling-minute/day limits per attempt. Restarts do not reset the allowance. Model
errors and incomplete output stop the task; no hidden model retry is made. Drafts
and spent work remain visible. Other external account usage remains unknown.

Live plans can vary, preventing a repeated draft hit. Human approval is a product
workflow action, not independent evaluation. No broad semantic quality, safety or
representative cost/latency benefit follows from this demo. Live deployment also
requires a dedicated hosting and multi-user authentication design.

## Developer verification

```bash
python -m pytest tests/test_workbench.py tests/test_document_agent.py tests/test_static_document_stage.py -q
python scripts/smoke_workbench.py
node --check src/api/workbench.js
```

The HTTP smoke uses FastAPI's test client (httpx from the development extras).
It tests actual backend transitions and cache storage with scripted source
responses. Browser visual QA was unavailable in this managed environment; no
browser screenshot or deployed public URL is claimed.


## Opt-in before-agent exact answer reuse (2026-10-26)

Use `python scripts/run_demo.py --whole-answer-cache` (PowerShell: `& $py
scripts/run_demo.py --whole-answer-cache`). This offline profile is stored under
`runs/workbench/reference-whole-answer`, separately from the original demo. Seed
a bundled documentation question, approve it, and repeat it. A qualified exact
whole-answer hit creates a fresh approval task without model planning, drafting
or review. Restart retains compatible approved answers. Human edits/rejections
retire the old answer namespace; edits are not automatically promoted as reviewed
model answers. Ineligible/context-dependent tasks run fresh work.

The UI distinguishes whole-answer and draft reuse, displays the matched question,
and retains reused evidence provenance. `/api/tasks/{id}` includes `work` and
`work_coverage` for the new profile. General semantic serving is not enabled by
this flag. The authored semantic comparison is offline harness evidence only.
Do not combine this option with `--live`; live budget integration remains pending.
Browser visual QA is still not claimed. See [Group 2 report](BEFORE_AGENT_CACHE.md).


## Scoped task context and role routing (2026-10-31)

`python scripts/run_demo.py --whole-answer-cache --task-memory` starts a separate
offline `reference-whole-answer-memory` profile. Source work is routed through
planner, source researcher, answer writer and reviewer roles; routing itself
makes no model calls. Ask and approve a source question. Its exact repeat skips
all model roles and still needs a new approval. A question referencing an earlier
answer uses a frozen snapshot of scoped prior tasks and runs fresh model work.
The default retrieval encoder is lexical hashing, not a demonstrated natural
paraphrase model. Prior answers remain untrusted context, not new source evidence.

With the local session capability, GET `/api/memory` lists compatible context
sources; DELETE `/api/memory` retires future context and cached-answer namespaces
while keeping task audit history. Clearing stops pending tasks with old snapshots;
create a new task. Deleting a task via the agent API removes its vector references.
Full memory management UI and visual QA remain Group 4 work. The flag rejects live
mode. Details: [Group 3 report](DOCUMENT_ORCHESTRATION_MEMORY.md).
