# Adaptive Agentic Semantic Cache

A semantic caching gateway for a **bounded documentation assistant**, continuing
the existing Python gateway and document agent. Questions concern a selected,
versioned document collection. Suitable complete answers can be reused for exact
repeats or approved equivalent questions; misses execute the agent's planning,
source search/read, drafting and review workflow.

The goal is measurable useful reuse. Question wording is not restricted to an
allowlist, but a similar question must not receive an inapplicable answer. Cosine
similarity alone never authorizes semantic reuse. Unknown requests fall back to
fresh execution; unsupported sources still require an uncertainty response.

**2026-10-26 Group 2:** qualified lookup now precedes agent execution for
explicit standalone-document tasks. Fresh required human approval is preserved.
New offline functional evidence: **274 regression tests passed**; four whole-task
fixtures per mode observed **12/9/6** provider invocations for off/exact/approved
semantic caching, with restart after the seed. These are scripted mechanisms,
not live savings or natural-language accuracy. See [Group 2 report](docs/BEFORE_AGENT_CACHE.md).
[Group 1 durability evidence](docs/DOCUMENTATION_REBUILD.md) remains preserved.

**Current result:** exact-cache mechanics and agent workflow integration work in
functional checks. General semantic accuracy and useful live end-to-end savings
remain unproven. The original development agent replay showed no call reduction with its
memory-bearing prompts. R6 opt-in static drafting passes a controlled replay
criterion; its live utility remains unmeasured. Historical Gate A remains failed.

[Current status](docs/CURRENT_STATUS.md) is the single implementation/evidence
status page. See [design](DESIGN.md), [fact ledger](docs/FACTS.md),
[R5 investigation](docs/audit/R5_CLOSEOUT.md) and [claim audit](docs/audit/CLAIM_HYGIENE_AUDIT.md).

## Implemented

- Authenticated gateway, provider adapter, metrics and bounded streaming bypass.
- Scoped exact response caching, SQLite persistence, TTL, invalidation, payload
  bounds and process-local coordination of identical misses.
- Opt-in semantic candidate shadow/guarded paths. Guarded reuse needs explicit
  actual-answer approvals; the current registry does not generalize to unseen pairs.
- Bounded document plans, allowlisted search/read tools, SQLite checkpoints,
  scoped memory and durable human approve/modify/reject.
- Answer-bound review/accounting and free-only foreground collection with a
  durable shared quota journal. Missing labels remain unknown.

These are implemented contracts, not a semantic safety certification. Streaming
responses are uncached. Semantic indexes default to process-local storage.
Opt-in `GATEWAY_PERSIST_SEMANTIC=true` recovers candidates from the SQLite exact
database in shadow/guarded mode, bound to source expiry and admission identity.
The historical FAISS/cross-encoder research pipeline is separate from the current
lexical/pinned-local-adapter serving wrapper. No learned adaptation benefit is claimed.

## Completion groups

1. Durable source-bound candidates: rebuilt and verified with offline fixtures.
2. Qualified lookup before agent execution: implemented and fixture-verified,
   with provider-boundary and local-stage observations (see coverage limits).
3. Bounded role orchestration, typed read-only tools and persistent scoped task
   retrieval: implemented and fixture-verified. Demo retrieval is lexical, not
   evidence of broad semantic understanding.
4. Unified interface, reproducible deployment and independent answer review;
   separately authorized budgeted live comparison.

No new training experiment unless a specific integration failure requires one.
Compare off/exact/semantic modes on the same documentation workload: reviewed
answer applicability, actual calls, provider-reported tokens and valid timings.
There is no promised savings percentage. Zero useful coverage is a failed result.
Anyone can run the configured assistant; public hosting and unrestricted
question answering are not delivered. General semantic serving remains disabled
by default, and independent answer-quality review is still missing.

## Local workbench

Run the integrated question/source/approval interface without a model key:

```sh
python -m pip install -r deploy/requirements-gateway.txt
python scripts/run_demo.py
```

Open http://127.0.0.1:8765. Reference replay uses scripted answers with the real
agent, tools, exact cache and durable approval. The mode is clearly labeled.
Ask a source question, approve the draft and ask it again to inspect actual draft
reuse. See [Windows instructions and optional explicit live mode](docs/WORKBENCH.md).
This is a local application; no public deployment is implied.

## Before-agent demo (offline)

```sh
python scripts/run_demo.py --whole-answer-cache
```

Ask a bundled documentation question, approve its answer, then repeat it. The
opt-in reference demo reuses the exact whole answer before planning and requests
your approval again. It uses a separate local profile; the original demo remains
available. No external inference occurs. Guarded semantic whole-task lookup is
verified by authored answer-bound fixtures; it is not enabled generally in this
UI. This flag cannot be combined with live mode yet.

## Role workflow and task context

```sh
python scripts/run_demo.py --whole-answer-cache --task-memory
```

The offline profile exposes planner, source researcher, answer writer and reviewer
routes. Standalone questions keep the cache-first path. Contextual follow-ups use
locally retrieved prior tasks as untrusted context, frozen for the task. Listing
and clearing context are available through authenticated local memory endpoints;
clearing retains the audit history. The demo uses a lexical vector baseline.
New Group 3 evidence: **293 regression tests passed**; no natural-language memory
quality or live savings claim. See [Group 3 report](docs/DOCUMENT_ORCHESTRATION_MEMORY.md).

## Offline quickstart

Use Python in a virtual environment. The source tree is tested with the repository's
offline test configuration; optional model/API integration tests are excluded.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python scripts/smoke_document_agent.py
```

On Windows use `.venv\Scripts\python.exe` and set offline environment flags with
PowerShell. The smoke uses scripted fixture plans/answers/review, not a live model.
For the minimal gateway deployment see [deployment](docs/deployment.md).
For document CLI commands see [agent contract](docs/document_agent.md).

## Evidence and next work

[R5's controlled replay](docs/audit/R5_CLOSEOUT.md) counts all provider/tool work,
retains its negative agent result and makes no real token or monetary claim.
Live collection is explicit, free-only and quota-capped. The
[partial request evidence](docs/audit/R5_PARTIAL_LIVE_CLOSEOUT.md) and
[model-assisted review](docs/audit/R5_ANTIGRAVITY_RECONCILIATION.md) retain their
limitations. No key belongs in this repository or chat. Independent human answer
review remains pending.

The [R6 scoped stage-input redesign](docs/audit/R6_AGENT_STAGE_CLOSEOUT.md) is
implemented and passed its preregistered source-replay engineering criterion.
It keeps relevant memory/scope in the actual prompt and full-request key. This
does not validate live agent quality or net savings. Legacy documents are preserved and marked superseded. Historical
latency percentages are component comparisons against a remote judge, not
application savings against fresh generation.


## Which datasets are used?

See [data inventory and limitations](docs/DATASETS_AND_REVIEW.md). Historical
Stage B training used builder-authored synthetic Python intent pairs in
`data/intent_stage_b/v2`; its trained policies failed useful-coverage calibration
and were not promoted. Product regression tests use scripted fixtures. The demo
uses three bundled CPython 3.12 FAQ excerpts. No new training is running.

Group 4 has prepared a 30-slot independent wording/review request with blank
judgments, not a new labeled benchmark. Current general semantic approval remains
unresolved; the authored registry cannot establish unseen paraphrase recognition.
An installed-wheel smoke check passed using existing dependencies; clean-machine
installation, browser QA, deployment and live utility are not yet established.


## Actual-model check (2026-11-01)

Both pinned MiniLM and Quora models now execute locally after a constructor
compatibility fix. On the exposed 30-question diagnostic at fixed historical
settings, dense similarity recommends three reuses: two human-sufficient and one
insufficient. The pair and combined rules recommend none. These are failed
applicability/coverage results, not proof of useful general semantic serving.
No threshold was lowered to pass. [Details and commands](docs/REAL_DOCUMENT_MODEL_DIAGNOSTIC.md).


## Measured engineering benchmark

The local SQLite exact-hit path measured **1.79 ms p95 over 5,000 warm requests**
(three keys, five runs). A separate 64-request concurrent cold fixture made one
counted stub-provider invocation. These are cache-component measurements, not
live LLM latency/cost savings or semantic accuracy. Full raw samples and source
hashes: `artifacts/resume/cache-benchmark-v1.json`. Current targeted suite: 320
passed. [Resume wording, limitations and commands](docs/Resume_Ready_Project.md).
