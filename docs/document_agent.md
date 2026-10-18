# Bounded documentation agent (R4)

This is an application client of GatewayService, not a second cache. The older
AgentWorkflow remains a compatibility demonstrator. DocumentAgent adds a model
planned sequence, actual local document reads, draft/review feedback and durable
human approval. Fixture approval is scripted, not an accuracy judgment.

## Run without API calls

From the repository root (use `.venv\\Scripts\\python.exe` on Windows):

```sh
.venv/bin/python scripts/demo_document_agent.py --database tasks.sqlite --documents examples/agent/documents.fixture.json start "Explain tuple immutability"
```

Copy the returned task ID. In a new process:

```sh
.venv/bin/python scripts/demo_document_agent.py --database tasks.sqlite --documents examples/agent/documents.fixture.json decide TASK_ID approve
.venv/bin/python scripts/demo_document_agent.py --database tasks.sqlite --documents examples/agent/documents.fixture.json inspect TASK_ID
```

`decide TASK_ID modify --answer "..."` edits the proposed answer; `reject`
terminates without an answer. `resume TASK_ID` advances a nonterminal checkpoint;
an awaiting task needs a decision. Completed tasks do not repeat tool/model work.
`delete TASK_ID` removes a task and its memory. CLI output contains document and
answer text: keep it private when operating on private sources.

The supplied document is an authored functional fixture, not an evaluation set.
For a real source, pass a JSON object mapping allowlisted IDs to document text.
Each corpus object is deployment-controlled for one access scope. Never populate
it with documents the invoking principal is not allowed to read. Tenant and
principal come from a trusted caller, not model output. This CLI is not an
authenticated multi-user web application; deployment authentication is external.

## Plan, tools and state

The supervisor returns strict JSON with 1–8 ordered steps, unique IDs, prior-only
dependencies and `search(query)` or `read(document_id)` arguments. Both tools are
read-only and operate on the supplied mapping. No filesystem path, URL, shell or
arbitrary function is available. A research execution loop dispatches validated
steps. Planning uses the bounded ID catalog; reads cannot interpolate future
search results into arguments in this version. Search provides a lexical ranking,
not semantic retrieval. The reviewer sees actual tool outcomes and the draft.

SQLite stores the plan, each tool result, model-call reservations, stage trace,
review feedback and human decision. At most one bounded revision is allowed by
default. Default task budget is six gateway requests; default successful path is
three: plan, draft, review. All stage calls go through GatewayService. Planning
and review bypass reuse. Drafting uses full evidence, source hash, principal,
model, instructions and feedback in its request key. A task-context byte cap
fails closed rather than silently trimming evidence. Usage and call counts are
reported per successful model stage; failures leave unknown upstream counts.

Only exact/shadow cache modes are supported here. Guarded semantic mode is
rejected because a verifier could exceed the one-call reservation. This is an
explicit R5 integration limitation, not automatic semantic reuse for agent stages.
The conservative live CLI policy bypasses contextual drafting until validated.
No tool-result memoization or tool-call savings claim is implemented.

Transactions serialize transitions across processes. Reservations are committed
before model work and survive a crash; an interrupted uncheckpointed model call
may be repeated and consumes another reservation. Exactly-once external inference
is not promised. Completed read steps are not rerun on ordinary restart. There
are no external write side effects. SQLite writes are serialized and not a
high-throughput distributed scheduling design.

## Human review and memory

Default workflow pauses before publishing its answer. An approve/modify/reject
transaction records the principal and result. Auto completion is an explicit
programmatic option; absent read evidence cannot be auto-approved. Model review
is fallible and is never independent benchmark labeling or proof of correctness.

A small memory returns up to three completed same-tenant/principal/source tasks
from the most recent twenty stored tasks. It includes answers and evidence IDs,
not correctness labels. Memories are marked untrusted in subsequent requests.
Document content hashes invalidate old task resumption and memory retrieval.
Changing a document requires a new task. `invalidate` deletes the principal's
stored tasks/memory. Cache administration is separate: deleting task memory does
not erase previously cached responses; response retention uses gateway TTL and
invalidation. Task storage itself has no automatic TTL; operators must delete
old tasks. No memory-benefit or learning claim is supported yet.

## Explicit live interface

`--gateway-url URL --model MODEL --allow-live-calls` sends requests to an
operator-controlled OpenAI-compatible gateway. Configure durable free-provider
quota enforcement and gateway authentication/transport before using this path;
the CLI does not implement an OpenRouter daily ledger. No automatic retries are
made. The per-task reservation is not a project-wide quota. No live requests
were made during R4 implementation. For authenticated gateways set `DOCUMENT_AGENT_GATEWAY_TOKEN` in the environment;
it is passed as a bearer token and is never included in task state. Do not put
keys in URLs. Use HTTPS for a remote gateway.

R5 must measure complete task work, correctness and overhead. Adding a planner
and reviewer can increase provider work relative to a direct answer; those calls
must not disappear from the comparison.
