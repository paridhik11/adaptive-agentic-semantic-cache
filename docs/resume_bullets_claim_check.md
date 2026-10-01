# Resume Bullets Claim Check and Verification Audit

> **Purpose:** Authoritative verification matrix mapping every metric, sample count, percentage, latency timing, ratio, and threshold appearing in [`docs/RESUME_BULLETS.md`](docs/RESUME_BULLETS.md) directly to its source row in [`docs/FACTS.md`](docs/FACTS.md).  
> **Strict Compliance Policy:** Zero unledgered numeric tokens permitted. 100% verified against [`docs/FACTS.md`](docs/FACTS.md).

---

## Audit Summary

- **Target Document:** [`docs/RESUME_BULLETS.md`](docs/RESUME_BULLETS.md)
- **Token Verification Result:** 0 unledgered numeric tokens (**PASS**)
- **Total Claims Audited:** 54 claims across 6 bullets and the "Do Not Claim" checklist
- **Verification Status:** 100% PASS

---

## 1. Bullet 1: System Latency Optimization & Gateway Architecture

| Claim ID | Metric / Token | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| B1-01 | `99.20% to 99.84%` | Latency reduction range across all load tests | [`LAT-05`](docs/FACTS.md) | 99.20% to 99.84% | **PASS** |
| B1-02 | `124.6x to 622.7x` | Speedup factor range across all load tests | [`LAT-04`](docs/FACTS.md) | 124.6x to 622.7x | **PASS** |
| B1-03 | `10.56–61.91 ms` | `AUTO_REUSE` mean latency range | [`LAT-01`](docs/FACTS.md), [`LAT-02`](docs/FACTS.md), [`LAT-03`](docs/FACTS.md) | 10.56 ms (Phase 6) to 61.91 ms (Phase 5 scaled) | **PASS** |
| B1-04 | `6,577.94–9,291.35 ms` | Remote LLM judge mean latency range | [`LAT-01`](docs/FACTS.md), [`LAT-03`](docs/FACTS.md), [`P5A-07`](docs/FACTS.md), [`P6-10`](docs/FACTS.md) | 6,577.94 ms (Phase 6) to 9,291.35 ms (Phase 5 pilot) | **PASS** |
| B1-05 | `86.39% to 89.71%` | Local resolution rate range | [`LRR-04`](docs/FACTS.md) | 86.39% to 89.71% | **PASS** |
| B1-06 | `70`, `338`, `175` | Workload query counts across pilot, scaled, and blind runs | [`D-07`](docs/FACTS.md), [`D-08`](docs/FACTS.md), [`D-09`](docs/FACTS.md) | 70 queries, 338 queries, 175 entries | **PASS** |
| B1-07 | `384` | Embedding dimension (`all-MiniLM-L6-v2`) | [`C-03`](docs/FACTS.md) | 384 output dimension | **PASS** |
| B1-08 | `0.92` | `TierRouter` `AUTO_REUSE` similarity floor | [`C-06`](docs/FACTS.md) | 0.92 | **PASS** |
| B1-09 | `0.90` | `TierRouter` `AUTO_REUSE` stability confidence floor | [`C-07`](docs/FACTS.md) | 0.90 | **PASS** |
| B1-10 | `n=1` | Scaled and blind runs single `AUTO_REUSE` hit caveat | [`LAT-06`](docs/FACTS.md), [`P5B-08`](docs/FACTS.md), [`P6-09`](docs/FACTS.md) | Exactly 1 `AUTO_REUSE` hit in each run | **PASS** |
| B1-11 | `n=20`, `19.18 ms` | Pilot run `AUTO_REUSE` hit count and mean latency | [`LAT-01`](docs/FACTS.md), [`P5A-06`](docs/FACTS.md) | n=20 hits, mean 19.18 ms | **PASS** |

---

## 2. Bullet 2: Sealed Blind Safety Evaluation Methodology

| Claim ID | Metric / Token | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| B2-01 | `0` false positives | Observed false positive count on blind evaluation | [`P6-03`](docs/FACTS.md), [`P6-04`](docs/FACTS.md) | k=0 false positives | **PASS** |
| B2-02 | `8` hits | Admitted cache hits (1 auto, 7 judge) | [`P6-03`](docs/FACTS.md) | 8 total cache hits | **PASS** |
| B2-03 | `0.00%` | Empirical cache hazard rate ($IRR_{\text{cache}}$) | [`P6-04`](docs/FACTS.md) | 0.00% | **PASS** |
| B2-04 | `[0.00%, 36.94%]` | Exact 95% Clopper-Pearson CI (k=0, n=8) | [`P6-05`](docs/FACTS.md) | [0.00%, 36.94%] | **PASS** |
| B2-05 | `175` | Blind evaluation query count | [`D-09`](docs/FACTS.md), [`P6-01`](docs/FACTS.md) | 175 entries | **PASS** |
| B2-06 | `7` domains | Category domains evaluated in blind benchmark | [`P6-01`](docs/FACTS.md) | 7 domains | **PASS** |
| B2-07 | `fc35744` | Cryptographic dataset seal commit hash | [`P6-02`](docs/FACTS.md) | `fc35744` | **PASS** |
| B2-08 | `10%` | Pre-stated safety ceiling ($IRR_{\text{cache}}$) | [`C-13`](docs/FACTS.md) | 0.10 (10%) safety ceiling | **PASS** |
| B2-09 | `n >= 36` | Minimum hits required to certify 10% ceiling at 95% CI | [`P4-04`](docs/FACTS.md), [`POOL-06`](docs/FACTS.md) | n >= 36 hits | **PASS** |
| B2-10 | `n=14`, `n=8`, `n=22` | Pooled sample counts (scaled + blind) | [`POOL-01`](docs/FACTS.md), [`POOL-02`](docs/FACTS.md) | 14 scaled + 8 blind = 22 hits | **PASS** |
| B2-11 | `15.44%` | Pooled exact 95% Clopper-Pearson upper bound | [`POOL-03`](docs/FACTS.md) | 15.44% exact upper bound | **PASS** |

---

## 3. Bullet 3: Ambiguous-Band Adjudication & Hazard Reduction

| Claim ID | Metric / Token | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| B3-01 | `20.83%` | Best fixed threshold hazard rate (5 FP / 24 hits) | [`P2-01`](docs/FACTS.md) | 20.83% (5 FP / 24 hits) | **PASS** |
| B3-02 | `0.85` | Best fixed cosine similarity threshold | [`P2-01`](docs/FACTS.md), [`P2-04`](docs/FACTS.md) | Threshold 0.85 | **PASS** |
| B3-03 | `[7.13%, 42.15%]` | Exact 95% Clopper-Pearson CI on fixed threshold | [`HR-01`](docs/FACTS.md) | [7.13%, 42.15%] | **PASS** |
| B3-04 | `8.33%` | OpenRouter Nemotron judge hazard rate (1 FP / 12 hits) | [`P3-06`](docs/FACTS.md) | 8.33% (1 FP / 12 hits) | **PASS** |
| B3-05 | `[0.21%, 38.48%]` | Exact 95% Clopper-Pearson CI on judge run | [`P3-06`](docs/FACTS.md) | [0.21%, 38.48%] | **PASS** |
| B3-06 | `12.50 percentage points` | Absolute hazard reduction (fixed vs judge) | [`HR-02`](docs/FACTS.md) | 12.50 percentage points | **PASS** |
| B3-07 | `60.00%` | Relative hazard reduction | [`HR-03`](docs/FACTS.md) | 60.00% relative reduction | **PASS** |
| B3-08 | `120` pairs | Calibration query pair benchmark size | [`D-01`](docs/FACTS.md), [`P2-06`](docs/FACTS.md) | 120 pairs | **PASS** |
| B3-09 | Non-significance | Confidence intervals overlap ([7.13%, 42.15%] vs [0.21%, 38.48%]) | [`HR-04`](docs/FACTS.md) | CI overlap; not statistically established | **PASS** |
| B3-10 | `52.17%` | Non-LLM decision step hazard rate (23 hits, 12 FP) | [`P3-01`](docs/FACTS.md) | 52.17% (12 FP / 23 hits) | **PASS** |
| B3-11 | Fail-closed | Judge fallback on timeout or error | [`C-16`](docs/FACTS.md) | `decision=BYPASS`, `confidence=0.0` | **PASS** |

---

## 4. Bullet 4: Volatility Interception & Stability Classification

| Claim ID | Metric / Token | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| B4-01 | `2.29 ms` | Classifier inference mean latency | [`P1-07`](docs/FACTS.md), [`P1-08`](docs/FACTS.md) | 2.29 ms mean latency | **PASS** |
| B4-02 | `91.67%` | Held-out challenge and final test accuracy (55/60) | [`P1-01`](docs/FACTS.md), [`P1-03`](docs/FACTS.md) | 91.67% (55/60 correct) | **PASS** |
| B4-03 | `0` dangerous FP | Dynamic query false positives on challenge set (24 queries) | [`P1-04`](docs/FACTS.md) | FP=0 on 24 dynamic queries | **PASS** |
| B4-04 | `0` dangerous FP | Dynamic query false positives on final test (23 queries) | [`P1-02`](docs/FACTS.md) | FP=0 on 23 dynamic queries | **PASS** |
| B4-05 | `60` | Benchmark size for challenge and final test sets | [`D-03`](docs/FACTS.md), [`D-04`](docs/FACTS.md) | 60 queries in each benchmark | **PASS** |
| B4-06 | `13.51%` | Final test conservative error rate (5/37 stable queries) | [`P1-05`](docs/FACTS.md) | 13.51% (5/37) | **PASS** |
| B4-07 | `13.89%` | Held-out challenge conservative error rate (5/36 stable) | [`P1-06`](docs/FACTS.md) | 13.89% (5/36) | **PASS** |
| B4-08 | `0.80` | Classifier confidence override floor | [`C-05`](docs/FACTS.md) | 0.80 | **PASS** |

---

## 5. Bullet 5: Benchmark Auditing & Ground-Truth Collision Correction

| Claim ID | Metric / Token | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| B5-01 | `10` collisions | Ground-truth labeling collisions identified | [`P5B-05`](docs/FACTS.md) | 10 collisions | **PASS** |
| B5-02 | `338` queries | Scaled load test dataset size | [`D-08`](docs/FACTS.md) | 338 queries (210 cold seeds + 128 authored) | **PASS** |
| B5-03 | `71.43%` | Scaled run pre-correction hazard rate (10 FP / 14 hits) | [`P5B-01`](docs/FACTS.md) | 71.43% (10 FP / 14 hits) | **PASS** |
| B5-04 | `[41.90%, 91.61%]` | Pre-correction 95% Clopper-Pearson CI | [`P5B-03`](docs/FACTS.md) | [41.90%, 91.61%] | **PASS** |
| B5-05 | `0.00%` | Scaled run post-correction hazard rate (0 FP / 14 hits) | [`P5B-02`](docs/FACTS.md) | 0.00% (0 FP / 14 hits) | **PASS** |
| B5-06 | `[0.00%, 23.16%]` | Post-correction 95% Clopper-Pearson CI | [`P5B-04`](docs/FACTS.md) | [0.00%, 23.16%] | **PASS** |
| B5-07 | `6` collisions | Uncorrected residual collisions disclosed | [`P5B-06`](docs/FACTS.md), [`KL-01`](docs/FACTS.md) | 6 collisions | **PASS** |
| B5-08 | `210`, `128` | StackExchange cold seeds and authored query breakdown | [`D-08`](docs/FACTS.md) | 210 cold seeds + 128 authored | **PASS** |

---

## 6. Bullet 6: Empirical Reproducibility, Fact Ledgering & CI Verification

| Claim ID | Metric / Token | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| B6-01 | `100%` | Empirical claim traceability and Gitleaks secret verification | [`SHA-01`..`SHA-04`](docs/FACTS.md) | 100% clean secret scanning | **PASS** |
| B6-02 | `0` leaks | Zero secret leaks across repository history | [`SHA-01`..`SHA-04`](docs/FACTS.md) | 0 leaks detected | **PASS** |
| B6-03 | `Python 3.14.5` | Verified runtime environment | [`ENV-01`](docs/FACTS.md) | Python 3.14.5 verified | **PASS** |

---

## 7. "Do Not Claim" Checklist Verification

| Claim ID | Metric / Constraint | Context in `RESUME_BULLETS.md` | FACTS.md Row(s) | Status |
| :---: | :--- | :--- | :--- | :---: |
| DNC-01 | In-memory store | FAISS operates in-memory without disk persistence | [`KL-07`](docs/FACTS.md) | **PASS** |
| DNC-02 | Synthetic traffic | Synthetic streams, not live enterprise production workloads | [`KL-08`](docs/FACTS.md) | **PASS** |
| DNC-03 | `50` RPD | Free-tier daily quota limit on OpenRouter judge | [`KL-03`](docs/FACTS.md) | **PASS** |
| DNC-04 | No dollar claims | Zero dollar cost calculations ledgered | All rows | **PASS** |
| DNC-05 | `n >= 36` | Ceiling certification requirement | [`P4-04`](docs/FACTS.md), [`POOL-06`](docs/FACTS.md) | **PASS** |
| DNC-06 | `7 categories`, `0.85` | All categories remained on fallback | [`P4-01`](docs/FACTS.md), [`KL-11`](docs/FACTS.md) | **PASS** |
| DNC-07 | Speedup baseline | Speedup measured against judge latency, not full generation | [`LAT-04`](docs/FACTS.md), [`LAT-05`](docs/FACTS.md) | **PASS** |
| DNC-08 | `8.04%` | Pooled 3-run sensitivity upper bound (44 hits, non-blind) | [`POOL-07`](docs/FACTS.md) | **PASS** |
