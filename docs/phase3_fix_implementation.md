# Phase 3 Fix — Adaptive Threshold Policy Engine: Specification

**Document Purpose:** Technical specification for replacing the hand-tuned linear combination in `src/decision/decision_step.py` with an adaptive per-category threshold engine, as committed in `README.md`.

---

## 1. Background — Why the Linear Combination Failed

In Phase 3's initial implementation, ambiguous-band traffic was routed using a hand-tuned linear combination:

$$\text{score} = 0.50 \cdot \text{sim\_norm} + 0.30 \cdot \text{conf\_norm} + 0.20 \cdot \text{hist\_rate}$$

with a static decision threshold of 0.50 ($\text{score} \ge 0.50 \implies \text{REUSE}$).

When evaluated against `query_pair_reuse_benchmark.json` (same 120 pairs as Phase 2), this linear combination **worsened cache hazard**:
- Phase 2 baseline cache hazard rate ($\text{IRR}_{\text{cache}}$): **20.83%** (5 FP / 24 hits)
- Phase 3 linear combination $\text{IRR}_{\text{cache}}$: **52.17%** (12 FP / 23 hits)
- Ambiguous tier $\text{IRR}_{\text{cache}}$: **57.14%** (12 FP / 21 hits)

### Root Cause Analysis
1. **Uncalibrated Threshold:** The fixed decision threshold of 0.50 placed the decision boundary at the midpoint of the normalized ambiguous similarity band ($[0.50, 0.92)$). Phase 2's empirical sweep had already demonstrated that similarity in the range $0.55\text{--}0.85$ is a high-hazard zone ($27\%\text{--}40\%\; \text{IRR}_{\text{cache}}$).
2. **Ad-Hoc Blending:** The weights ($0.50, 0.30, 0.20$) were assigned based on intuition rather than fit to labeled data.
3. **Domain Agnosticism:** Different enterprise domains have vastly different semantic overlap distributions (e.g. `science_medicine` is separable above 0.65; `computer_science` contains semantic adversarial pairs with similarity up to 0.9961). A single global threshold or fixed linear weighting cannot capture domain-specific hazard profiles.

---

## 2. The Adaptive Policy Engine Design

`README.md` committed to:
> *"An adaptive policy engine that learns per-category similarity thresholds from labeled reuse-outcome data, with statistical confidence bounds — not a single hand-picked global threshold."*

The tech stack already includes `scipy.stats` for this purpose.

### 2.1 Mathematical Formulation
For each domain qualifying under the minimum sample size rule:
1. We model the empirical relationship between embedding similarity score $s \in [0, 1]$ and binary reuse safety $y \in \{0, 1\}$ using logistic regression:
   $$P(y = 1 \mid s) = \sigma(\beta_0 + \beta_1 s) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 s)}}$$
2. Model parameters $(\beta_0, \beta_1)$ are estimated via maximum likelihood estimation (BFGS minimization of Bernoulli negative log-likelihood).
3. The decision threshold $\theta$ is derived at the standard Bayes decision point $p^* = 0.50$:
   $$\sigma(\beta_0 + \beta_1 \theta) = 0.50 \implies \beta_0 + \beta_1 \theta = 0 \implies \theta = -\frac{\beta_0}{\beta_1}$$
   A valid threshold requires $\beta_1 > 0$ (similarity positively correlates with safety). If $\beta_1 \le 0$, the domain model is invalidated and falls back to global reference.

### 2.2 Statistical Confidence Bounds via `scipy.stats`
To quantify estimate uncertainty, confidence bounds on $\theta$ are derived using the **Delta Method**:
1. Let $g(\beta_0, \beta_1) = -\frac{\beta_0}{\beta_1}$.
2. The gradient is:
   $$\nabla g = \begin{bmatrix} \frac{\partial g}{\partial \beta_0} \\ \frac{\partial g}{\partial \beta_1} \end{bmatrix} = \begin{bmatrix} -\frac{1}{\beta_1} \\ \frac{\beta_0}{\beta_1^2} \end{bmatrix}$$
3. Let $\Sigma = \mathcal{I}^{-1}$ be the asymptotic parameter covariance matrix (the inverse Hessian of the negative log-likelihood at the optimum).
4. The asymptotic variance of $\theta$ is:
   $$\operatorname{Var}(\theta) \approx (\nabla g)^T \Sigma (\nabla g), \quad \text{SE}(\theta) = \sqrt{\operatorname{Var}(\theta)}$$
5. The two-sided $95\%$ confidence interval is computed using `scipy.stats.norm.ppf`:
   $$z_{0.975} = \text{scipy.stats.norm.ppf}(0.975) \approx 1.95996$$
   $$\text{CI}_{95\%} = \left[ \theta - z_{0.975} \cdot \text{SE}(\theta), \; \theta + z_{0.975} \cdot \text{SE}(\theta) \right]$$

---

## 3. Minimum Sample Size (`MINIMUM_CATEGORY_N = 20`) & Fallback Rule

### 3.1 Justification for `MINIMUM_CATEGORY_N = 20`
1. **Events-Per-Variable (EPV) Rule:** Standard statistical methodology (Peduzzi et al., 1996; Harrell, *Regression Modeling Strategies*) establishes that logistic regression requires 10–20 observations per parameter. Estimating $(\beta_0, \beta_1)$ requires at least 20 observations, with adequate representation in both classes.
2. **Sampling Error:** For $N < 20$, the standard error of a proportion or cutoff exceeds $0.15$, yielding a 95% margin of error greater than $\pm 0.30$. Fabricating a per-category threshold on 1, 5, or 11 data points gives an illusion of precision while having virtually no statistical power.
3. **Separation and Degeneracy:** In sparse categories (e.g. `system_operations` with 2 safe / 9 unsafe, `finance_economics` with 1 safe / 4 unsafe, `realtime_news_weather` with 0 safe / 1 unsafe), extreme imbalance causes quasi-complete separation or singular Fisher information matrices.

### 3.2 Explicit Minimum-N Fallback Rule
For any domain where:
$$\text{sample\_size} < \text{MINIMUM\_CATEGORY\_N} \; (20)$$
or for any unseen domain string, the engine **strictly falls back** to Phase 2's global reference threshold:
$$\text{FALLBACK\_THRESHOLD} = 0.85$$

This threshold is documented from Phase 2 as the least-bad global reference point (20.83% hazard, 24 hits). It is used rather than fabricating unreliable per-category numbers.

### 3.3 Domain Eligibility Summary on Benchmark Dataset ($N=120$)

| Domain | Total Pairs | Safe | Unsafe | Meets $N \ge 20$? | Policy Applied |
|---|---|---|---|---|---|
| `computer_science` | 54 | 26 | 28 | **YES** | Per-category fit ($\theta=0.6855$, 95% CI: [0.5785, 0.7924]) |
| `science_medicine` | 27 | 17 | 10 | **YES** | Per-category fit ($\theta=0.6171$, 95% CI: [0.5097, 0.7245]) |
| `mathematics` | 13 | 11 | 2 | **NO** | Fallback to Phase 2 global 0.85 |
| `system_operations` | 11 | 2 | 9 | **NO** | Fallback to Phase 2 global 0.85 |
| `history_geography` | 9 | 5 | 4 | **NO** | Fallback to Phase 2 global 0.85 |
| `finance_economics` | 5 | 1 | 4 | **NO** | Fallback to Phase 2 global 0.85 |
| `realtime_news_weather` | 1 | 0 | 1 | **NO** | Fallback to Phase 2 global 0.85 |
| `legal_compliance` | 0 | 0 | 0 | **NO** | Fallback to Phase 2 global 0.85 |
| *Unknown / Unseen* | — | — | — | **NO** | Fallback to Phase 2 global 0.85 |

---

## 4. Isolation of Stability Confidence

Phase 1's stability confidence remains an explicit, hard gate in `tier_router.py`:
- `bypass_conf_ceiling = 0.80`: queries below 0.80 confidence route immediately to `BYPASS`.
- `auto_reuse_conf_floor = 0.90`: combined with similarity $\ge 0.92$, routes to `AUTO_REUSE`.

Inside the `AMBIGUOUS` tier, queries have already passed the stability confidence floor ($\ge 0.80$).
**Stability confidence is deliberately kept OUT of the ambiguous policy engine model.**
- Re-blending confidence as a linear term (as the failed decision step did with $0.30 \cdot \text{conf\_norm}$) creates an arbitrary additive boost that incorrectly promoted low-similarity unsafe pairs to REUSE.
- Keeping the decision rule as a direct comparison between similarity score and the category-specific threshold ($\text{similarity\_score} \ge \theta_{\text{category}}$) ensures interpretability, auditability, and direct empirical grounding.

---

## 5. Software Architecture & Public API

### 5.1 `src/decision/adaptive_threshold_engine.py`
- `CategoryThreshold`: Immutable dataclass storing domain threshold metadata, sample size, fallback status, and `scipy.stats` confidence bounds.
- `ThresholdDecisionResult`: Dataclass capturing the decision (`REUSE` or `BYPASS`), applied threshold, domain, confidence bounds, and human-readable rationale.
- `AdaptiveThresholdEngine`:
  - Holds calibrated thresholds for all 8 taxonomy domains.
  - Exposes `decide(similarity_score: float, domain: str) -> ThresholdDecisionResult`.
  - Exposes `get_threshold(domain: str) -> CategoryThreshold`.
  - Exposes classmethod `fit(labeled_data) -> AdaptiveThresholdEngine` for training from arbitrary data.

### 5.2 Component Swap in `src/evaluation/decision_evaluator.py`
- Replaces `DecisionStep` with `AdaptiveThresholdEngine` for `Tier.AMBIGUOUS` routing.
- Emits detailed per-domain breakdown and a 3-way comparison table in `print_report()`.
- Protocol, tier boundaries, and same-set comparison framing remain untouched.
