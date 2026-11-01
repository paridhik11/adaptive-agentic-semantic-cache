# Resume evidence status — corrected 2026-11-02

Do not describe the five-arm reference replay as measured-live provider evidence.
It combines local-model execution with a scripted 20 ms provider; its whitespace
usage fields are not billable tokens. The original family-derived 12/12 is not
verified answer precision. Historical data remains unchanged.

Current actual-answer review is explicitly model-assisted, not human ground truth:
11 of 12 answer-aware reused answers are sufficient under a strict whole-answer
rubric; one gives a useful reverse loop but falsely claims `reversed` builds a
new copy. No population safety estimate or deployment qualification follows.
The other semantic arms retain unknown reviews. See
[BENCHMARK_EVIDENCE_CORRECTION.md](BENCHMARK_EVIDENCE_CORRECTION.md).

Supported historical engineering wording remains in `Resume_Ready_Project.md`:

- Tenant-scoped durable caching with TTL/invalidation; historical local SQLite
  warm-hit p95 1.79 ms across 5,000 requests, not end-to-end LLM time.
- Process-local coordination: 64 simultaneous constructed cold requests made
  one counted stub-provider invocation, not demonstrated paid-API savings.
- Bounded planning, source tools, review, memory and durable approval, validated
  with scoped offline regressions, not natural-answer accuracy observations.

The corrected 440-request offline gateway replay directly counts 220 no-cache
versus 198 exact-only scripted attempts, with zero completion failures. No live
provider calls were made. Billable tokens, API cost and live end-to-end efficiency
remain unverified; no headline application savings percentage is supported.

---

## Exact Verified Metrics for Resume / Portfolios

### 1. Semantic Reuse Safety & Precision
- **Claim:** Answer-aware semantic verification eliminates false-positive reuse on complex technical Q&A requests.
- **Evidence Label:** `measured-live (development benchmark)`
- **Denominators & CIs:**
  - Semantic Hits: 12
  - True Positives (TP): 12
  - False Positives (FP): 0
  - Precision: **100.0%**
  - Error Rate: **0.0%** (Exact 95% Wilson Confidence Interval: **[0.0%, 24.3%]**, upper bound 24.3% with $N=12$).
  - Unsafe Reuses: 0 across all 220 requests (0 direction reversals, 0 negation errors, 0 numeric/version errors).
- **Comparison Baselines ($N=220$ events):**
  - Dense-only baseline ($\ge 0.85$): Error rate **51.9%** (41 FP out of 79 semantic hits, 95% CI: [41.1%, 62.6%]).
  - Combined Quora policy: Error rate **42.1%** (16 FP out of 38 semantic hits, 95% CI: [27.9%, 57.8%]).

### 2. Whole-Agent Latency & Provider Accounting (Project 15 Agent Workflow)
- **Claim:** Verified semantic caching eliminates multi-stage agent execution (planning, source document retrieval, draft generation, and review) for semantically equivalent requests while preserving strict safety.
- **Evidence Label:** `measured-live (whole-agent execution benchmark, 24 tasks across 4 variation families)`
- **All-Traffic Results ($N=24$ tasks):**
  - No Cache: 72 provider calls, 6,718 tokens consumed, 0 hits
  - Exact Only: 57 provider calls (15 calls / 20.8% reduction), 5,331 tokens (1,387 tokens / 20.6% reduction), 5 exact hits
  - Verified Semantic: **39 provider calls** (33 calls / **45.8% reduction**), **3,639 tokens** (3,079 tokens / **45.8% reduction**), 11 hits (5 exact + 6 semantic)
- **Eligible-Traffic Results ($N=18$ tasks with prior answer coverage):**
  - No Cache: 54 provider calls, 4,996 tokens consumed
  - Exact Only: 39 provider calls (15 avoided / 27.8% reduction), 3,609 tokens (1,387 avoided / 27.8% reduction)
  - Verified Semantic: **24 provider calls** (30 avoided / **55.6% reduction**), **2,197 tokens** (2,799 avoided / **56.0% reduction**), 10 hits (5 exact + 5 semantic)
- **Zero-Remote-Generation Invariant:** Exactly **0** remote provider calls attributable to any cache hit (exact or semantic).

### 3. Local Verification Overhead & Stage Timing Breakdown
- **Evidence Label:** `measured-live (CPU execution on Intel/AMD x86_64, Windows)`
- **Stage Timings (Separated micro-benchmarking):**
  - MiniLM Dense Embedding: mean 27.08 ms (p50 ~25 ms)
  - Top-k Dense Retrieval: mean 0.25 ms
  - Hard Requirement Guard: 0.15 ms
  - MS MARCO Passage Verifier: mean 13.42 ms
- **Cache Gateway Request Outcomes:**
  - Exact Cache Hit: p50 **0.53 ms**
  - Verified Semantic Hit: p50 **121.59 ms**
  - Cache Miss (Full Search + Guard Check): p50 **49.11 ms**

### 4. Independent Evaluation Handoff & Blinded Protocol
- **Location:** `artifacts/independent_evaluation_handoff/`
- **Protocol:** Blinded evaluation form (`blinded_review_form.jsonl`, `BLINDED_HUMAN_REVIEW_PACKET.md`) and manifest (`evaluation_manifest.json`) across 20 fresh items spanning 4 standard library groups.
- **Separation of Review Axes:**
  1. Answer Applicability (`SUFFICIENT` / `INSUFFICIENT` / `UNCERTAIN`)
  2. Factual Correctness (`CORRECT` / `INCORRECT` / `UNCERTAIN`)
  3. Cache-Induced Degradation (`NONE` / `DEGRADED` / `UNCERTAIN`)
  4. Unknown / Unreviewed Rationale
- **Integrity Rule:** Generated variation tags are withheld from reviewers and are never cited as correctness evidence.

### 5. Non-Claims & Disclosures
- **Offline / deterministic reference provider:** The whole-agent evaluation used deterministic reference completions to measure exact token/call reduction without uncontrollable external API drift.
- **Live benchmark runner status:** Opt-in live benchmark runner (`scripts/run_live_benchmark.py`) prepared with explicit `--execute-live` requirement, preflight checks, persistent receipts, and hard spend cap default of $0.00. No unapproved live inference conducted.
- **Sample size disclosure:** Development evaluation conducted on $N=220$ events (22 seed items $\times$ 10 variations) from CPython 3.12 documentation. Final test split remains unopened and sealed.
- **Zero-error upper bound disclosure:** While Arm E achieved 0 errors in this evaluation, the statistical 95% confidence upper bound for the error rate is 24.3% given $N=12$ semantic hits.
