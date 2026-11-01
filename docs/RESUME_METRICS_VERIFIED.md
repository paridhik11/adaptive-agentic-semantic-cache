# RESUME_METRICS_VERIFIED.md

_Classification of experimental claims across the codebase._  
_Written: 2026-11-02_

---

## Metric Evidence Classification

| Category | Definition | Status / Traceability |
| :--- | :--- | :--- |
| **Measured-Live (Development Benchmark)** | Actual local CPU forward passes and full runtime execution on fixed, version-pinned test records. | **VERIFIED:** Phase 3 benchmark (`runs/semantic-dev/phase3_benchmark_results.json`). 220 events across 5 arms. |
| **Historical Component Measurement** | Isolated micro-benchmarks of individual encoders or models. | **VERIFIED:** Documented in `docs/FACTS.md` (e.g. MiniLM embedding ~30–60 ms, Quora latency ~40 ms). |
| **Simulated / Replay** | Synthetic provider replays simulating network latency or fixed response generation. | **VERIFIED:** Used for offline evaluation without paid API calls (`ReferenceProvider`). Not claimed as live API savings. |
| **Projected / Estimated** | Extrapolations of cost or latency savings under hypothetical live traffic distributions. | **CAUTION:** Must always be labeled as projections; cannot be cited as headline application benefits. |
| **Unproven / Disproven** | Hypotheses contradicted by empirical data. | **DISPROVEN:** Claim that Quora QQP cross-encoder is effective for technical Python semantic caching (disproven in Phase 1: logits −7 to −8.35). |

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

### 2. Local Verification Overhead
- **Evidence Label:** `measured-live (CPU execution on Intel/AMD x86_64)`
- **Breakdown:**
  - MiniLM Dense Encoder: 20–30 ms p50
  - Hard Requirement Guard: 0.15 ms p50
  - MS MARCO Passage Scorer: 25–45 ms p50
  - End-to-end Gateway Cache Processing p50: **47.62 ms** (Arm E) vs **21.51 ms** (Arm B exact cache).

### 3. Non-Claims & Disclosures
- **No live API cost savings claimed:** The development benchmark operated on local models and deterministic reference replay; no external provider tokens were saved during offline testing.
- **Sample size disclosure:** Development evaluation conducted on $N=220$ events (22 seed items $\times$ 10 variations) from CPython 3.12 documentation. Final test split remains sealed.
- **Zero-error upper bound disclosure:** While Arm E achieved 0 errors in this evaluation, the statistical 95% confidence upper bound for the error rate is 24.3% given $N=12$ semantic hits.
