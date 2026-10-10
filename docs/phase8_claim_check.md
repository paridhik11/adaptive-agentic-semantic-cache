# Phase 8 — Claim Check and Number Verification Audit

> **Purpose:** Authoritative verification matrix mapping every numeric value, threshold, latency timing, token count, percentage, confidence interval, and ratio appearing in [`docs/phase8_walkthrough.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/phase8_walkthrough.md) and the [`README.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/README.md) Phase 8 results table row directly to its source row in [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md).
>
> **Strict Compliance Policy:**
> - Every number must originate from [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md).
> - Zero unverified metrics permitted.
> - Preserves Phases 1–7 results completely untouched.

---

## Audit Summary

- **Total Phase 8 Claims Verified**: 40 items across 6 categories (100% verified against [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md))
  - **Section 1**: System Architecture, Models, and Hardware (`P8-C01`..`P8-C10`)
  - **Section 2**: Public Datasets, Splits, and Checksums (`P8-D01`..`P8-D05`)
  - **Section 3**: Baseline Real LLM Generation Measurements (`P8-B01`..`P8-B06`)
  - **Section 4**: Four-Arm Simulation Results on Sealed Test Stream (`P8-R01`..`P8-R04`)
  - **Section 5**: Path Breakdown, Latencies, and Avoided Workload (`P8-P01`..`P8-P11`)
  - **Section 6**: Component Ablation Table (`P8-A01`..`P8-A04`)
- **Unledgered Numeric Tokens**: 0 (strictly verified by `scripts/check_numeric_tokens.py`)

---

## 1. System Architecture, Models, and Hardware Constants

| # | Metric / Claim | Walkthrough Location | FACTS.md Row | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :---: | :--- | :---: |
| CK-01 | Normalizer module | Step 1, Section 2.1 | [`P8-C01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `src/cache/normalizer.py` (canonical match = AUTO_REUSE) | **PASS** |
| CK-02 | Primary local CrossEncoder model | Step 2, Section 2.2 | [`P8-C02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `cross-encoder/quora-distilroberta-base` | **PASS** |
| CK-03 | Primary model revision | Step 2, Section 2.2 | [`P8-C03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `f62e7a4b20b97195c2868e53ec59126df5eac743` | **PASS** |
| CK-04 | Training data disclosure | Step 2, Section 2.2 | [`P8-C04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | QQP training split; zero-shot transfer evaluated on val & PAWS | **PASS** |
| CK-05 | Alternative local model | Step 2, Section 2.2 | [`P8-C05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **PASS** |
| CK-06 | Alternative model revision | Step 2, Section 2.2 | [`P8-C06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | **PASS** |
| CK-07 | Calibrated threshold $\theta_{\text{local}}$ | Step 2, Section 2.2 | [`P8-C07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `0.8500` (calibrated on validation split only) | **PASS** |
| CK-08 | Rule guards | Step 2, Section 2.2 | [`P8-C08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | Numbers, negation polarity, entity tokens | **PASS** |
| CK-09 | Remote judge fallback status | Step 2, Section 2.2 | [`P8-C09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | `fallback_to_llm=False` (disabled by default; 100% local resolution) | **PASS** |
| CK-10 | Evaluation hardware | Step 5, Section 2.5 | [`P8-C10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | Intel Core Ultra 7 155U (12 cores, 14 threads), 32 GB RAM, Win 11 | **PASS** |

---

## 2. Public Datasets, Splits, and Checksums

| # | Dataset / Split | Size | License | Checksum / Commit | FACTS.md Row | Status |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: |
| CK-11 | PAWS validation split | 8,000 rows | CC-BY-SA 4.0 | `7760d829453764ba342a6f562809a8ed21c2c3eec3fd9ffa544089f145d42f6d` | [`P8-D01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-12 | QQP validation split | 40,430 rows | Quora Terms / GLUE | `efd86a539c412d74874ee451573d7bd142f56c47fe36de033b9f367d8bb0fa71` | [`P8-D02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-13 | MS MARCO validation split | 10,047 rows | CC-BY 4.0 / MSR | `67aca2e1b2fffe418a6bd317142887c90973c9c98dd8d940eaae3f8521daed7c` | [`P8-D03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-14 | Tuning validation split | 160 pairs | Combined open | `3782fa16b1dc9d8469adfe2bb8e1a1e127394aa52ca1714bfa4911d35a51da03` | [`P8-D04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-15 | Sealed test stream | 230 queries | Sealed prior to run | Hash: `fbcf1437aa...`, committed at `325d4c6` | [`P8-D05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |

---

## 3. Baseline Real LLM Generation Measurements

| # | Metric | Sample | Model | Measured Value | FACTS.md Row | Status |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| CK-16 | Baseline mean latency | n=10 | `nvidia/nemotron-3-super-120b-a12b:free` | 1505.56 ms | [`P8-B01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-17 | Baseline median (p50) latency | n=10 | `nvidia/nemotron-3-super-120b-a12b:free` | 1235.5 ms | [`P8-B02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-18 | Baseline p95 latency | n=10 | `nvidia/nemotron-3-super-120b-a12b:free` | 2239.32 ms | [`P8-B03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-19 | Baseline prompt tokens | n=10 | `nvidia/nemotron-3-super-120b-a12b:free` | 36.3 tokens | [`P8-B04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-20 | Baseline completion tokens | n=10 | `nvidia/nemotron-3-super-120b-a12b:free` | 139.8 tokens | [`P8-B05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-21 | Baseline total tokens | n=10 | `nvidia/nemotron-3-super-120b-a12b:free` | 176.1 tokens | [`P8-B06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |

---

## 4. Four-Arm Simulation Results (N=230 Sealed Test Stream)

| # | Simulation Arm | Hits (n) | Hit Rate (95% CI) | IRR (95% CI) | Remote Calls | FACTS.md Row | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CK-22 | (a) No Cache | 0 | 0.0% [0.0%, 1.6%] | N/A (0 hits) | 230 LLM gen | [`P8-R01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-23 | (b) Fixed Cosine (0.85) | 90 | 39.1% [32.8%, 45.8%] | 34.4% [24.7%, 45.2%] (FP=31) | 140 LLM gen | [`P8-R02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-24 | (c) Current Pipeline | 16 | 7.0% [4.0%, 11.1%] | 18.8% [4.0%, 45.6%] (FP=3) | 21 judge calls | [`P8-R03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-25 | (d) New Pipeline | 50 | 21.7% [16.6%, 27.6%] | 6.0% [1.3%, 16.5%] (FP=3) | 0 remote calls | [`P8-R04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |

---

## 5. Path Breakdown, Latencies, and Avoided Workload

| # | Metric / Sub-path | Measured Value | FACTS.md Row | Status |
| :---: | :--- | :--- | :---: | :---: |
| CK-26 | Total cache hits | 50 hits (TP=47, FP=3) | [`P8-P01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-27 | Canonical exact hits | 41 hits (82.0% of cache hits) | [`P8-P02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-28 | Vector store AUTO_REUSE hits | 3 hits | [`P8-P03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-29 | Local CrossEncoder approved hits | 6 hits | [`P8-P04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-30 | CANONICAL_EXACT latency (n=41) | mean 0.02 ms, p50 0.01 ms, p95 0.02 ms | [`P8-P05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-31 | AUTO_REUSE latency (n=3) | mean 36.72 ms, p50 36.03 ms, p95 41.37 ms | [`P8-P06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-32 | LOCAL_EQUIVALENCE latency (n=14) | mean 390.07 ms, p50 68.00 ms, p95 1665.89 ms | [`P8-P07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-33 | BYPASS latency (n=172) | mean 40.38 ms, p50 39.15 ms, p95 58.03 ms | [`P8-P08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-34 | LLM calls avoided per 1,000 | 217.4 calls | [`P8-P09`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-35 | Tokens avoided per 1,000 | 38,282.6 tokens | [`P8-P10`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-36 | Cost avoided per 1,000 (estimate) | $0.011 (at $0.30/1M tokens, OpenAI Oct 2024 published price) | [`P8-P11`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |

---

## 6. Component Ablation Table

| # | Ablation Stage | Hits | Hit Rate | FP | IRR | p50 Latency | FACTS.md Row | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CK-37 | 1. Normalizer only | 41 | 17.8% | 0 | 0.0% | 0.01 ms | [`P8-A01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-38 | 2. + Embedding (Cosine 0.85) | 90 | 39.1% | 31 | 34.4% | 34.02 ms | [`P8-A02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-39 | 3. + Local Equivalence Model (Arm d) | 50 | 21.7% | 3 | 6.0% | 37.82 ms | [`P8-A03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |
| CK-40 | 4. + Remote Judge (Current pipeline) | 16 | 7.0% | 3 | 18.8% | 50.65 ms | [`P8-A04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) | **PASS** |

---

*Verified: 2026-10-11. Strict token check: 0 discrepancies against `docs/FACTS.md`.*
