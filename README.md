# adaptive-agentic-semantic-cache
Adaptive Semantic Response Reuse Engine

Reduces LLM token usage, cost, and latency by detecting when a query is about stable ("evergreen") information and safely reusing a previously generated, high-quality answer instead of calling the LLM again.

(Working name - rename freely.)

## The problem

A large fraction of LLM traffic asks the same kind of thing in different words:

> "Explain binary search" / "Can you explain how binary search works?"

Both deserve the same answer. Calling the LLM twice for this wastes tokens, money, and latency. But blindly caching everything is dangerous - queries about dynamic information (weather, prices, news, "current" anything) must never be served from a stale cache.

Evergreen's job: tell the two apart, reuse aggressively and safely on the first kind, and never touch the second kind.

## What this is
- A **stability classifier** that decides whether a query is about timeless/stable information or dynamic/time-sensitive information.
- A **semantic cache** (embeddings + vector search) that finds previously answered, semantically equivalent stable queries.
- A **reuse decision layer** - a lightweight agent that is invoked only when the signals are ambiguous (borderline similarity, borderline stability confidence), not on every request.
- An **adaptive policy engine** that learns per-category similarity thresholds from labeled reuse-outcome data, with statistical confidence bounds - not a single hand-picked global threshold.
- A **metrics/eval layer** reporting cache hit rate, tokens/cost saved, latency, and - most importantly - the incorrect-reuse rate, since a cache that saves money but serves wrong answers isn't a win.

## What this is NOT
- Not a general RAG / live-freshness / web-crawling system.
- Not a multi-agent framework for its own sake - the decision layer only fires on ambiguous cases; high-confidence cases are resolved by plain vector lookup with zero extra LLM calls.
- Reuse is verbatim by default - no "adaptation" LLM call on the hot path, since that would quietly cancel out the cost savings.

## Pipeline
```
                     User Query
                         │
                         ▼
              ┌─────────────────────┐
              │ Stability Classifier│
              │  stable? / dynamic? │
              └─────────────────────┘
                         │
              dynamic ───┴─── stable
                 │              │
                 ▼              ▼
           LLM Generate   Semantic Similarity Search
           (no caching)          │
                          high sim ─── ambiguous sim
                                  │         │
                          Auto-Reuse   Reuse Decision Agent
                                  │         │
                                  ▼         ▼
                              REUSE     REUSE / GENERATE
                                  │         │
                                  └────┬────┘
                                       │
                              Log outcome + signal
                                       │
                                       ▼
                          Adaptive Policy Engine
                        (per-category threshold tuning)
```

## Tech stack
| Component | Tool | Why |
| :--- | :--- | :--- |
| Language | Python 3.11+ | Ecosystem fit |
| Embeddings | `text-embedding-3-small` (or similar) | Cheap, fast, good enough |
| Vector store | Redis + RedisVL / Qdrant | Sub-ms lookups |
| API layer | FastAPI | Drop-in proxy |
| Stability classifier | Rules + small model fallback | Cheap first pass, model for edge cases |
| Stats / policy tuning | `scipy.stats` | Confidence intervals on threshold changes |
| Monitoring | Prometheus + Grafana | Real-time hit rate / cost dashboards |
| Containerization | Docker + docker-compose | One-command spin-up |

## Metrics tracked
- Cache hit rate (overall + per category)
- Tokens avoided / estimated cost saved
- Latency: cached vs. LLM-generated vs. agent-decided path (P50/P95/P99)
- Incorrect-reuse rate, with confidence intervals, per threshold setting
- Stability classifier precision/recall (reported separately for "dynamic misclassified as stable" - the costly error - vs. the reverse)
- Agent invocation rate (should be a small % of total traffic)

## Empirical Results

> **Two sets of numbers are reported below.** Phase 1-5 results were measured on datasets that informed the system's design and tuning at some point during development. Phase 6 was the first **sealed blind evaluation** — the dataset was committed to git before the pipeline ran against it, with zero prior exposure, and the run was executed exactly once. Both sets use real judge calls, real latency measurements, and the identical `evaluate_load_test.py` methodology.

### Phase 5 — Design-Informed Evaluation (N=338, corrected)

Dataset: `data/raw/new_dataset_v3.json` | Run: 2026-09-22 | See: [`docs/phase5_walkthrough.md`](docs/phase5_walkthrough.md)

| Metric | Value |
| :--- | :---: |
| **Overall Hit Rate** | 4.14% (14/338) |
| **IRR_cache (incorrect reuse rate)** | **0.00%** — k=0 FP out of n=14 hits |
| **95% Clopper-Pearson CI** | [0.00%, 23.16%] |
| **BYPASS rate** | 86.09% |
| **AMBIGUOUS judge calls** | 46 |
| **AUTO_REUSE speedup** | 124.6x over AMBIGUOUS path |

### Phase 6 — Sealed Blind Evaluation (N=175)

Dataset: `data/raw/phase6_blind_eval_dataset.json` (sealed in commit `fc35744` before run) | Run: 2026-09-28 | See: [`docs/phase6_walkthrough.md`](docs/phase6_walkthrough.md)

| Metric | Value |
| :--- | :---: |
| **Overall Hit Rate** | 4.57% (8/175) |
| **IRR_cache (incorrect reuse rate)** | **0.00%** — k=0 FP out of n=8 hits |
| **95% Clopper-Pearson CI** | [0.00%, 36.94%] — underpowered; n=8 < 36 required floor |
| **BYPASS rate** | 89.14% |
| **AMBIGUOUS judge calls** | 18 |
| **AUTO_REUSE speedup** | 622.7x over AMBIGUOUS path |

**Verdict:** The core safety property (IRR_cache = 0.00%, FP = 0) replicates exactly on blind data. The CI is wider because Phase 6 produced fewer hits (n=8 vs. n=14) due to Sub-stage A's conservative confidence override intercepting academic-phrasing paraphrase queries. No divergence on safety or hit rate was observed. See [`docs/phase6_walkthrough.md`](docs/phase6_walkthrough.md) Section 7 for the plain-language divergence statement.

## Project status

See `WORKFLOW.md` for the phased build plan and `ANTIGRAVITY_WORKFLOW.md` if building with an agentic dev tool.

## Quickstart
```bash
git clone <repo>
cd evergreen
docker-compose up --build
```
*(Fill in once Phase 2 scaffolding exists.)*

## Repo structure
```
adaptive-agentic-semantic-cache/
├── src/
│   ├── classifier/        # stability classifier (rules + fallback model)
│   ├── cache/             # embeddings, vector store, similarity search
│   ├── decision_agent/    # reuse decision agent + tools
│   ├── policy/            # adaptive threshold tuning
│   ├── api/               # FastAPI proxy
│   ├── evaluation/        # evaluation and metrics
│   └── utils/             # shared utilities
├── data/
│   ├── raw/               # raw datasets
│   └── processed/         # processed data
├── tests/                 # automated tests
├── notebooks/             # exploratory notebooks
├── .github/
│   └── workflows/ci.yml   # CI workflow
├── pyproject.toml
├── .gitignore
├── .env.example
└── README.md
```