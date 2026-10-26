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

**2026-10-26 reconstruction:** the unavailable durability milestone was rebuilt
from the preserved Stage B foundation. New offline functional evidence: **242
regression tests passed**; four-request restart fixtures observed **4/3/2**
provider invocations with off/exact/approved-semantic modes. This is not live
savings or natural-language accuracy. See [rebuild report](docs/DOCUMENTATION_REBUILD.md).

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
2. Qualified lookup before agent execution and complete local/remote work accounting.
3. Specialist/tool orchestration and scoped persistent task-memory integration.
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
