# Phase 3 Fix — Adaptive Threshold Policy Engine: Refinement Walkthrough

**Document Purpose:** Execution record, empirical verification, and definitive evaluation for the Phase 3 ambiguous-band decision engine refinement. Documents the three scoped fixes:
1. Shifting the operating threshold from point estimate $\theta$ to the upper bound of the 95% confidence interval ($\theta_{\text{upper}} = \text{ci\_upper}$).
2. Benchmarking against the trivial "bypass all ambiguous" baseline.
3. Tightening the eligibility gate to enforce `MINIMUM_MINORITY_CLASS_N = 10` alongside `MINIMUM_CATEGORY_N = 20`.

> [!IMPORTANT]
> **Required Statement (Framing Rule):** These results are a **same-set comparison** on `query_pair_reuse_benchmark.json` ($N=120$), already used to establish Phase 2's baseline. They show whether the refined adaptive threshold engine improves outcomes on known pairs compared to the failed linear combination attempt and whether it beats the trivial bypass baseline, not whether it generalizes to unseen traffic. A fresh pair dataset would be required before claiming generalization.

---

## HEADLINE FINDING & VERDICT: DOES THE ADAPTIVE ENGINE BEAT THE TRIVIAL BASELINE?

> [!WARNING]
> ### HEADLINE VERDICT: NO — THE ADAPTIVE ENGINE DOES NOT BEAT THE TRIVIAL BYPASS BASELINE ON HAZARD
>
> 1. **Hazard Ceiling Failure:** The trivial "bypass all ambiguous" baseline achieves **$\text{IRR}_{\text{cache}} = 0.00\%$** (0 cache hazards / 0 false positives) by definition, trivially satisfying Phase 2's $< 10\%$ hazard ceiling. The refined adaptive engine achieves **$\text{IRR}_{\text{cache}} = 23.08\%$** (3 false positives out of 13 hits), which **violates the $< 10\%$ hazard ceiling**.
> 2. **The Ambiguous-Band Tradeoff:** The adaptive engine recovers **10 true reuses** ($TP=10$, ARR=10.83%) compared to only **2 true reuses** in the trivial baseline ($TP=2$, ARR=1.67%). However, because this project treats cache hazard as an asymmetric, catastrophic risk, recovering 8 additional hits in the ambiguous band at the cost of 3 cache hazards is an **unacceptable safety compromise**.
> 3. **The Semantic Limits of Embedding Similarity:** All 3 remaining false positives occur in `computer_science` (`PAIR-035` sim=0.9961, `PAIR-002` sim=0.9645, `PAIR-005` sim=0.8677) where syntactically identical queries contain inverted algorithmic intent (e.g. in-place sort vs. returning a copy). Cosine similarity cannot separate these pairs.
> 4. **Strategic Recommendation:** **The ambiguous-band statistical threshold engine is NOT sufficient on its own.** Unless paired with an LLM judge or structural AST diff checker, the ambiguous band must either use the trivial bypass baseline (pure safety, near-zero yield) or invoke an agentic LLM verification call. Statistical threshold heuristics on raw embeddings cannot solve semantic ambiguity in code.

---

## 1. Background & Rationale for the Refinement

### 1.1 Why Fix 1 was Needed: Asymmetric Risk vs. Bayes-Optimal Point
The initial per-category fit computed $\theta$ at $p^* = 0.50$ on the fitted logistic curve $P(\text{safe} \mid \text{sim})$. This is the Bayes-optimal threshold assuming symmetric loss (cache hazard and missed reuse are equally costly). However, this project has consistently enforced asymmetric conservative bias (Phase 1's `STABLE_CONFIDENCE_THRESHOLD = 0.65`, Phase 2's $\text{IRR}_{\text{cache}} < 10\%$ ceiling). Operating at $\theta$ produced thresholds of 0.6855 for CS and 0.6171 for Science/Medicine—substantially *more* permissive than the 0.85 fallback, leading to 8 false positives in the ambiguous band.
- **Fix:** Set `operating_threshold = ci_upper` ($\theta_{\text{upper}}$). This demands higher similarity before granting reuse, providing a rigorous statistical safety margin.

### 1.2 Why Fix 2 was Needed: The Trivial Benchmark to Beat
The failed linear combination had $\text{IRR}_{\text{cache}} = 52.17\%$. Beating a broken baseline is trivial; the meaningful question is whether the decision layer beats doing *nothing* in the ambiguous tier (unconditionally bypassing all ambiguous queries to cache MISS).

### 1.3 Why Fix 3 was Needed: Dual Sample-Size Gates (EPV Rule)
The Events-Per-Variable (EPV) rule of thumb for 2-parameter logistic models ($\beta_0, \beta_1$) requires at least 10 events per variable, which governs the **minority class count**, not just total $N$. A total-$N$ gate could admit a domain with 24 safe and 1 unsafe pair.
- **Fix:** Require both `MINIMUM_CATEGORY_N = 20` AND `MINIMUM_MINORITY_CLASS_N = 10`.

---

## 2. Dataset Distribution & Dual-Gate Verification

Verification directly against `data/raw/query_pair_reuse_benchmark.json` (120 pairs):

```text
computer_science:        54 total (26 safe / 28 unsafe) -> Minority count = 26
science_medicine:        27 total (17 safe / 10 unsafe) -> Minority count = 10
mathematics:             13 total (11 safe /  2 unsafe) -> Minority count =  2
system_operations:       11 total ( 2 safe /  9 unsafe) -> Minority count =  2
history_geography:        9 total ( 5 safe /  4 unsafe) -> Minority count =  4
finance_economics:        5 total ( 1 safe /  4 unsafe) -> Minority count =  1
realtime_news_weather:    1 total ( 0 safe /  1 unsafe) -> Minority count =  0
legal_compliance:         0 total ( 0 safe /  0 unsafe) -> Absent from dataset
Total:                  120 pairs (62 safe / 58 unsafe)
```

### 2.1 Per-Domain Strategy & Dual-Gate Evaluation

| Domain | Total $N$ | Minority $N$ | Total Gate ($N \ge 20$) | Minority Gate ($N_{\text{min}} \ge 10$) | Strategy | Operating Thresh ($\theta_{\text{upper}}$) | Point Est ($\theta$) | SE | 95% CI | Rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| `computer_science` | 54 | 26 | **PASS** | **PASS** | **PER-CATEGORY** | **0.7924** | 0.6855 | 0.0546 | [0.5785, 0.7924] | Both gates pass; $\beta_0=-3.6696, \beta_1=5.3535$. Operating threshold is CI upper bound. |
| `science_medicine` | 27 | 10 | **PASS** | **PASS** (exact) | **PER-CATEGORY** | **0.7245** | 0.6171 | 0.0548 | [0.5097, 0.7245] | Meets minority cutoff of 10 exactly; passes dual gate. Operating threshold is CI upper bound. |
| `mathematics` | 13 | 2 | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | $N=13 < 20$; minority=2 $< 10$. Falls back to 0.85. |
| `system_operations` | 11 | 2 | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | $N=11 < 20$; minority=2 $< 10$. Falls back to 0.85. |
| `history_geography` | 9 | 4 | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | $N=9 < 20$; minority=4 $< 10$. Falls back to 0.85. |
| `finance_economics` | 5 | 1 | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | $N=5 < 20$; minority=1 $< 10$. Falls back to 0.85. |
| `realtime_news_weather` | 1 | 0 | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | $N=1 < 20$; minority=0 $< 10$. Falls back to 0.85. |
| `legal_compliance` | 0 | 0 | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | $N=0$; domain absent from dataset. Falls back to 0.85. |
| *Unknown / Unseen* | — | — | FAIL | FAIL | **FALLBACK** | **0.8500** | 0.8500 | N/A | N/A | Defaults to global 0.85 safely without runtime error. |

**Empirical Finding on `science_medicine`:** Under the new minority gate, `science_medicine` has exactly 10 unsafe pairs ($N_{\text{min}} = 10$). Because the gate is $N_{\text{min}} \ge 10$, it **still qualifies** for a per-category fit. Its operating threshold shifts from point estimate 0.6171 to CI upper bound **0.7245**.

---

## 3. Four-Way Same-Set Benchmark Comparison

All four strategies evaluated on the exact same 120 pairs from `query_pair_reuse_benchmark.json`:

| Metric | Phase 2 Baseline (@ 0.85) | Phase 3 Linear Combo (Failed Attempt) | Phase 3 Adaptive Fix ($\theta_{\text{upper}}$) | Trivial Bypass-All Ambiguous Baseline |
|---|---|---|---|---|
| **ARR (Actual Reuse Rate)** | 20.00% | 19.17% | **10.83%** | **1.67%** |
| **CRR (Correct Reuse Precision)** | 79.17% | 47.83% | **76.92%** | **100.00%** |
| **IRR_cache (Cache Hazard Rate)** | **20.83%** | **52.17%** | **23.08%** | **0.00%** |
| **IRR_traffic (Traffic Hazard Rate)** | 4.17% | 10.00% | **2.50%** | **0.00%** |
| **FRR (False Rejection Rate)** | 69.35% | 82.26% | **83.87%** | **96.77%** |
| **TP (Correct Reuse)** | 19 | 11 | **10** | **2** |
| **FP (Cache Hazard)** | 5 | 12 | **3** | **0** |
| **FN (Missed Reuse)** | 43 | 51 | **52** | **60** |
| **TN (Correct Reject)** | 53 | 46 | **55** | **58** |

*(For reference: The intermediate point-estimate adaptive engine had ARR=15.00%, CRR=55.56%, IRR_cache=44.44%, TP=10, FP=8, FN=52, TN=50).*

---

## 4. Per-Tier Breakdown of the Refined Adaptive Fix

| Tier | Traffic Count | TP | FP | FN | TN | Tier ARR | Tier IRR_cache | Tier CRR |
|---|---|---|---|---|---|---|---|---|
| `AUTO_REUSE` | 2 (1.7%) | 2 | 0 | 0 | 0 | 100.0% | **0.00%** | 100.0% |
| `AMBIGUOUS` | 32 (26.7%) | 8 | 3 | 2 | 19 | 34.4% | **27.27%** | 72.73% |
| `BYPASS` | 86 (71.7%) | 0 | 0 | 50 | 36 | 0.0% | 0.00% | 100.0% |

### 4.1 Detailed Ambiguous Tier Analysis
- In the ambiguous band (32 pairs), shifting to $\theta_{\text{upper}}$ dropped false positives from **8 down to 3** (-62.5% reduction vs point estimate, -75% vs linear combo).
- The 5 unsafe pairs successfully eliminated by shifting to $\theta_{\text{upper}}$ were in `computer_science` (`PAIR-116`, `PAIR-011`, `PAIR-113`, `PAIR-023`, `PAIR-015`), all having similarities between 0.69 and 0.785 (above $\theta=0.6855$ but below $\theta_{\text{upper}}=0.7924$).
- The 3 remaining false positives are high-similarity semantic code variants:
  - `PAIR-035` (sim = 0.9961)
  - `PAIR-002` (sim = 0.9645)
  - `PAIR-005` (sim = 0.8677)
- In `science_medicine`, all 5 ambiguous pairs are safe and have similarities $> 0.73$, so all 5 remain correctly classified as REUSE ($TP=5, FP=0$).

---

## 5. Architectural Conclusions & Recommendations

1. **Safety First:** If the system is deployed under a strict requirement of $\text{IRR}_{\text{cache}} < 10\%$, **the trivial bypass-all policy is currently the only viable embedding-only baseline** (yielding $\text{IRR}_{\text{cache}} = 0.00\%$).
2. **The Role of the Adaptive Policy Engine:** The adaptive policy engine with $\theta_{\text{upper}}$ successfully cuts false positives by 75% compared to the ad-hoc linear combination, and demonstrates how per-category calibration with confidence bounds behaves properly. However, statistical thresholding on dense embeddings cannot overcome the fundamental limit of lexical vs. semantic distinction in code.
3. **Next Step (Phase 4):** Ambiguous-band pairs must be routed to an agentic LLM judge for intent verification rather than relying on scalar embedding similarity thresholds.
