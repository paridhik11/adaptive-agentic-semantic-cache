# AUDIT_PACK_A.md: Phase A Verification and Audit Package

> **Audit Object:** Phase A — Out-of-Sample Verification, Semantic Matcher Sweep & Sealed Stream Evaluation  
> **Repository:** `adaptive-agentic-semantic-cache`  
> **Evaluation Mode:** `sealed` out-of-sample single run ($N=2200$)  
> **Execution Date:** 2026-10-12  
> **Gate Status:** **GATE A FAILED — EXECUTION HALTED (PHASE B BLOCKED)**  
> **Target Audience:** Independent Auditor / Claude Review  

---

## 1. Executive Summary & Audit Purpose

This audit pack documents the complete results, cryptographic artifacts, statistical endpoints, and gate verification of **Phase A** of the `adaptive-agentic-semantic-cache` roadmap (`docs/ROADMAP_ABC.md`).

Phase A aimed to establish whether a local-first semantic cache architecture utilizing canonical normalization, fast-path vector routing, and local cross-encoder verification could achieve an incorrect-reuse rate ($\text{IRR}$) $\le 10.0\%$ on real and adversarial out-of-sample traffic without remote LLM judge calls.

### Primary Audit Verdict
- **Gate A Evaluation:** **FAILED**
- **Non-Exact Hits Achieved:** 75 (Required: $\ge 100$)
- **Non-Exact Incorrect-Reuse Rate:** **28.00% point estimate** with exact 95% Clopper-Pearson CI **[18.24%, 39.56%]** (Required: $\le 10.0\%$ point estimate)
- **Protocol Action:** Autonomous execution is **halted** per `ROADMAP_ABC.md`. Phase B has **not** started. No post-hoc tuning was performed on the sealed test set.

---

## 2. Cryptographic Integrity & Protocol Timeline

In accordance with strict scientific pre-registration principles, all configuration parameters, model choices, and test query streams were hashed and committed to git prior to the evaluation run:

1. **Pre-Registration Commit:** Git commit `d1f9d97` (`docs/phaseA_preregistration.md`).
2. **Frozen Pipeline Configuration:** `data/phaseA_frozen_config.json`  
   - SHA-256: `a1e0dc2a0a2e89e8c29a357597264f2d47fbf556f3f531b9a0b5c760f2fd8466`
3. **Sealed Test Stream:** `data/raw/phaseA_sealed_test_stream.json`  
   - SHA-256: `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991`  
   - Pre-run verification: Stream checksum re-verified by `scripts/run_phaseA_sealed_evaluation.py` immediately prior to inference.
4. **Single Execution Invariant:** The sealed evaluation runner was executed exactly once on local CPU hardware (`Intel Core Ultra 7 155U`, total duration: 66.64 s).
5. **Telemetry Persistence:** Raw per-query decisions saved to `data/phaseA_sealed_evaluation_telemetry.json`; aggregated metrics saved to `data/phaseA_sealed_evaluation_summary.json`.

---

## 3. Pre-Sealed Validation Work & Component Preparation

### Subsection A — Phase A0 Baseline Re-Measurement
Before pipeline evaluation, local LLM generation latency was re-measured ($n=100$) on `Qwen/Qwen2.5-0.5B-Instruct` (CPU, temp 0.0, max_new_tokens 64) to anchor simulated savings:
- Mean latency: **8997.49 ms**; Median (p50): **8153.50 ms**; p95: **12763.96 ms**.
- Mean tokens: Prompt 11.05, completion 61.87, total **72.92 tokens**.
- Auxiliary remote baseline on OpenRouter (`nvidia/nemotron-3-super-120b-a12b:free`, $n=20$): mean 2002.11 ms, p50 1346.64 ms, p95 3759.78 ms, mean tokens 164.25.

### Subsection B — Phase A1 Semantic Matcher Sweep ($N=160$ validation)
To address Phase 8 false positives, three candidate matcher architectures were tested:
- **Candidate (a) Off-the-shelf DistilRoBERTa (no guards):** 79 hits, 34 FP (all 34 from PAWS), IRR = 43.04% [31.94%, 54.67%].
- **Candidate (b) Fine-tuned on QQP+PAWS (no guards):** Fine-tuned on 2000 balanced pairs (1 epoch, CPU 449.57 s). 59 hits, 3 FP (0 PAWS), IRR = 5.08% [1.06%, 14.15%].
- **Candidate (c) Fine-tuned + Rule & Generative Guards:** 40 hits, 39 TP, 1 FP (0 PAWS), Precision = 97.50% [86.84%, 99.94%], Recall = 65.00% [51.60%, 76.87%], IRR = **2.50% [0.06%, 13.16%]**. Candidate (c) was frozen for the sealed evaluation.

### Subsection C — Phase A2 CQADupStack Calibration
Using `mteb/cqadupstack-retrieval` (CC BY-SA 3.0 / 4.0), 12 subforums were calibrated:
- All 12 subforums cleared sample size ($N=50 \ge 20$) and minority class ($25 \ge 10$) gates. Subforums using fallback (0.85): 0.
- Operating thresholds spanned from 0.1087 (physics) to 0.3854 (programmers).

### Subsection D — Phase A3 WildChat Natural Repeat Rates
Analyzing 5000 first turns from `allenai/WildChat` (ODC-By):
- Exact string repeats: 490 / 5000 (9.80%).
- Canonical normalizer repeats: 529 / 5000 (10.58%), an incremental lift of +0.78%.
- Guarded semantic reuse at similarity $\ge 0.85$: 986 / 5000 (19.72%) vs 1398 (27.96%) raw ungoverned.
- Guarded semantic reuse at similarity $\ge 0.92$: 777 / 5000 (15.54%) vs 1063 (21.26%) raw ungoverned.
- 100 queries exported to `data/review/pa_human_audit.csv` for human inspection.

---

## 4. Sealed Evaluation Results & Confusion Matrix

### Overall Stream Performance ($N=2200$)
- **Total Cache Hits:** 86
- **Overall Hit Rate:** 3.91% [3.14%, 4.81%]
- **Total True Positives (TP):** 60
- **Total False Positives (FP):** 26
- **Total False Negatives (FN):** 476
- **Total True Negatives (TN):** 438
- **Overall Incorrect-Reuse Rate (IRR):** 30.23% [20.79%, 41.08%]
- **Exact Canonical Matches:** 11 (11 TP, 0 FP)

### Primary Endpoint: Non-Exact Matches
- **Non-Exact Cache Hits ($n$):** 75
- **Non-Exact True Positives:** 54
- **Non-Exact False Positives ($k$):** 21
- **Non-Exact Incorrect-Reuse Point Estimate:** **28.00%**
- **Exact 95% Clopper-Pearson Confidence Interval:** **[18.24%, 39.56%]**

### Source Breakdown
- **`paws_test` (600 queries):** 43 hits (22 TP, 21 FP). Non-exact hits: 34, non-exact FP: 16. Non-exact IRR = **47.06% [29.84%, 64.89%]**.
- **`qqp_heldout` (1000 queries):** 42 hits (37 TP, 5 FP). Non-exact hits: 40, non-exact FP: 5. Non-exact IRR = **12.50% [4.19%, 26.80%]**.
- **`cqadupstack_heldout` (400 queries):** 1 hit (1 TP, 0 FP). Non-exact hits: 1, non-exact FP: 0.
- **`wildchat_traffic` (200 queries):** 0 hits (cold traffic).

---

## 5. Latencies, Workload Avoided & Break-Even Analysis

### Path Latency Distribution
- `CANONICAL_EXACT` ($n=30$): Mean 0.06 ms, median (p50) **0.04 ms**, p95 **0.16 ms**.
- `AUTO_REUSE` ($n=43$): Mean 14.28 ms, median (p50) **14.36 ms**, p95 **17.75 ms**.
- `LOCAL_EQUIVALENCE` ($n=181$): Mean 33.72 ms, median (p50) **34.54 ms**, p95 **66.03 ms**.
- `BYPASS` ($n=1946$): Mean 16.65 ms, median (p50) **15.58 ms**, p95 **26.07 ms**.
- **Overall Pipeline Decision Overhead:** Median (p50) **15.81 ms**, p95 **37.11 ms**.
- **End-to-End Latency (Simulated):** Median (p50) **8169.03 ms**, p95 **8184.04 ms** (vs No-Cache baseline p50 8153.50 ms, p95 12763.96 ms).

### Workload Avoided & Break-Even
- **Calls Avoided per 1000 Queries:** 39.1 calls
- **Tokens Avoided per 1000 Queries:** 2850.5 tokens
- **Break-Even Hit Rate:** **0.1976%** (0.00198). The cache pays for its latency overhead at any repeat hit rate above 0.1976%.

---

## 6. Gate A Compliance Verification

| Gate Requirement | Target Value | Measured Value (`sealed`) | Result |
| :--- | :--- | :--- | :---: |
| Sealed stream pre-registered & committed | Committed in git before run | Pre-registered in commit `d1f9d97` | **PASS** |
| Cryptographic hash check | Verified identical SHA-256 | `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991` | **PASS** |
| Run count | Exactly once | Single run (66.64 s) | **PASS** |
| Repository unit tests | All tests pass | 281 / 281 passing | **PASS** |
| Strict token check | 0 missing tokens vs FACTS.md | Verified clean (0 errors) | **PASS** |
| Gitleaks security scans | 0 secrets / clean scan | Verified clean | **PASS** |
| **Statistical Non-Exact Sample Size** | **$\ge 100$ non-exact hits** | **75 non-exact hits** | **FAIL** |
| **Safety Ceiling (Primary Endpoint)** | **$\text{IRR} \le 10.0\%$ point estimate** | **28.00% point estimate [18.24%, 39.56%]** | **FAIL** |

**Gate Outcome:** **FAILED**. As mandated by `ROADMAP_ABC.md`, execution stops immediately.

---

## 7. Traceability Index & Audit Artifacts

Every claim in this audit package is traceable to reproducible code and data:
- Authoritative Fact Ledger: [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L400-L490) (Section 17: rows `PA-B01` through `PA-RC05`, and `C-17`).
- Claim Verification Matrix: [`docs/claim_check_A.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/claim_check_A.md).
- Full Walkthrough & Ablations: [`docs/phaseA_walkthrough.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/phaseA_walkthrough.md).
- Blocked Gate Documentation: [`docs/BLOCKED_A.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/BLOCKED_A.md).
- Raw Summary Data: `data/phaseA_sealed_evaluation_summary.json`.
- Raw Telemetry Data: `data/phaseA_sealed_evaluation_telemetry.json`.
