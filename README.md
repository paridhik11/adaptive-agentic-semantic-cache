# Adaptive Agentic Semantic Cache

A Python LLM response-cache gateway and bounded documentation agent. The project
investigates whether scoped exact and guarded semantic reuse can reduce model
work while preserving answer quality.

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
responses are uncached. Semantic indexes start cold after process restart.
The historical FAISS/cross-encoder research pipeline is separate from the current
lexical/pinned-local-adapter serving wrapper. No learned adaptation benefit is claimed.

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
[Live collection](docs/R5_LIVE_RUN.md) is explicit, free-only and quota-capped;
no key belongs in this repository or chat. Actual independent answer review is pending.

The [R6 scoped stage-input redesign](docs/audit/R6_AGENT_STAGE_CLOSEOUT.md) is
implemented and passed its preregistered source-replay engineering criterion.
It keeps relevant memory/scope in the actual prompt and full-request key. This
does not validate live agent quality or net savings. Legacy documents are preserved and marked superseded. Historical
latency percentages are component comparisons against a remote judge, not
application savings against fresh generation.
