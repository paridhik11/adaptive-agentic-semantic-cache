# Phase A Walkthrough: Rigorous Verification, Matcher Sweep & Sealed Stream Evaluation

> **Phase Status:** Evaluated (Gate A Failed — execution halted per protocol).  
> **Evaluation Mode:** `sealed` out-of-sample single run ($N=2200$).  
> **Author:** Antigravity agent.  
> **Date:** 2026-10-12.  
> **Authoritative Facts:** All reported figures are strictly ledgered in [FACTS.md](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L400-L470).

---

## Section 1 — Executive Summary & Gate A Decision

Phase A was designed to transition the semantic caching framework from small synthetic benchmarks to rigorous, reproducible verification on public out-of-sample datasets under pre-registered frozen parameters.

Every evaluation in Phase A strictly adhered to the `GLOBAL RULES` of `docs/ROADMAP_ABC.md`:
1. **Pre-registration:** The evaluation protocol and pipeline configuration were committed to git (`docs/phaseA_preregistration.md`, commit `d1f9d97`) before executing the sealed test stream.
2. **Cryptographic Sealing:** The sealed evaluation dataset (`data/raw/phaseA_sealed_test_stream.json`, SHA-256: `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991`) was verified against the frozen configuration prior to inference.
3. **Single Execution:** Executed exactly once on local CPU hardware (`Intel Core Ultra 7 155U`, 66.64 s total runtime).
4. **No Post-Hoc Tuning:** When the out-of-sample error rate exceeded the safety target, no hyperparameter adjustments were made on the test stream.

### Gate A Outcome

| Gate A Criterion | Required Threshold | Measured Result (`sealed`) | Status |
| :--- | :--- | :--- | :---: |
| Pre-registration committed before run | Committed in git before execution | Commit `d1f9d97` | **PASS** |
| Cryptographic hash match | Identical SHA-256 | Verified `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991` | **PASS** |
| Strict token check & gitleaks | 0 missing tokens; 0 secrets | Verified clean | **PASS** |
| Unit tests green | All tests pass | 281 / 281 passing | **PASS** |
| **Statistical Non-Exact Volume** | **$\ge 100$ non-exact hits** | **75 non-exact hits** | **FAIL** |
| **Safety Ceiling (Primary Endpoint)** | **$\text{IRR}_{\text{non-exact}} \le 10.0\%$ point estimate** | **28.00% point estimate [18.24%, 39.56%]** | **FAIL** |

**Gate Decision:** **GATE A FAILED**. In accordance with the protocol, Phase B is **NOT** started. The raw findings are documented below and in `docs/BLOCKED_A.md`.

---

## Section 2 — Phase A0: Preflight, Errata & Baseline Measurements

### Subsection A — Reproduction & Root-Cause Analysis of Phase 8 False Positives
Phase 8 simulation results were reproduced identically to [FACTS.md](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md) (`P8-R04`, `P8-P01`..`P8-P04`).
Root-cause analysis revealed that all 3 false positives (`P8-ADV-PROBE-001`, `P8-ADV-PROBE-015`, `P8-ADV-PROBE-025`) occurred in the `ADVERSARIAL_TRAP` category (PAWS word scrambles) via the vector store `AUTO_REUSE` fast path (cosine similarity $\ge 0.92$, stability confidence $\ge 0.90$). Because the fast path assumed high bi-encoder similarity was safe, it bypassed both the CrossEncoder and the rule guards, allowing lexical bag-of-words anagrams to be incorrectly reused.

### Subsection B — Validation vs Test Stream Divergence Explained
The Phase 8 validation set ($N=160$) had an error rate of 39.62% [26.5%, 54.0%] due to 40 PAWS word scrambles (25% of the validation set, 40% of true negatives) whose identical token multisets deceive bag-of-words bi-encoders. On the Phase 8 test stream, 41 of 50 hits (82%) were exact string or canonical normalizer folds with zero errors. The non-exact test IRR was 3/9 = 33.33%, which closely matched the validation error rate.

### Subsection C — Errata & Workload Normalization
- **Ceiling Erratum:** The Phase 8 point estimate of 6.0% carried a 95% Clopper-Pearson CI of [1.3%, 16.5%]. Because the upper confidence bound (16.5%) exceeded 10.0%, Phase 8 did not formally clear the 10% safety ceiling.
- **Cost Normalization:** The unanchored $0.011/1k dollar figure was retracted. Workload savings are strictly reported as **38,282.6 tokens** and **217.4 calls** avoided per 1000 queries.

### Subsection D — Empirical Real LLM Baseline ($n=100$)
Measured on local hardware (`Intel Core Ultra 7 155U`, 32 GB RAM, Windows 11) using `Qwen/Qwen2.5-0.5B-Instruct` (PyTorch CPU, temperature 0.0, max_new_tokens 64, do_sample False):
- **Mean Latency:** 8997.49 ms
- **Median (p50) Latency:** 8153.50 ms
- **95th Percentile (p95) Latency:** 12763.96 ms
- **Token Usage:** Mean prompt 11.05 tokens, mean completion 61.87 tokens, mean total 72.92 tokens.
- **Auxiliary Remote Baseline ($n=20$ live OpenRouter calls to `nvidia/nemotron-3-super-120b-a12b:free`):** mean 2002.11 ms, p50 1346.64 ms, p95 3759.78 ms, mean tokens 164.25.

---

## Section 3 — Phase A1: Candidate Semantic Matcher Comparison

We benchmarked three candidate matcher architectures on the 160-pair validation set (`data/phase8_tuning_val.json`):
1. **Candidate (a):** Off-the-shelf `cross-encoder/quora-distilroberta-base` without rule guards (decision threshold 0.85).
2. **Candidate (b):** Fine-tuned `quora-distilroberta-base` on 2000 balanced QQP + PAWS pairs (1 epoch, batch size 16, lr 2e-05, trained on CPU in 449.57 s), without rule guards.
3. **Candidate (c):** Fine-tuned model + rule guards (number, negation, entity) + long-prompt guard + generative-request guard (threshold 0.85).

### Candidate Comparison Table (`validation`, $N=160$)

| Metric | Candidate (a) Off-the-Shelf | Candidate (b) Fine-Tuned | Candidate (c) Fine-Tuned + Guards (Selected) |
| :--- | :---: | :---: | :---: |
| Total Hits | 79 | 59 | 40 |
| Hit Rate | 49.38% [41.39%, 57.38%] | 36.88% [29.39%, 44.85%] | 25.00% [18.50%, 32.45%] |
| True Positives (TP) | 45 | 56 | 39 |
| False Positives (FP) | 34 | 3 | 1 |
| PAWS False Positives | 34 | 0 | 0 |
| Precision | 56.96% [45.33%, 68.06%] | 94.92% [85.85%, 98.94%] | 97.50% [86.84%, 99.94%] |
| Recall | 75.00% [62.14%, 85.28%] | 93.33% [83.80%, 98.15%] | 65.00% [51.60%, 76.87%] |
| **Incorrect-Reuse Rate (IRR)** | **43.04% [31.94%, 54.67%]** | **5.08% [1.06%, 14.15%]** | **2.50% [0.06%, 13.16%]** |

**Selection:** Candidate (c) reduced the validation IRR from 43.04% to 2.50% by completely eliminating PAWS errors and suppressing generative requests, and was selected for the frozen pipeline.

---

## Section 4 — Phase A2 & A3: Domain Calibration & Natural Repeat Rates

### Subsection A — Domain-Level Calibration on CQADupStack
Using `mteb/cqadupstack-retrieval` (CC BY-SA 3.0 / 4.0), we constructed balanced 50-pair validation sets for 12 subforums:
- All 12 subforums passed the sample size gate ($N=50 \ge 20$) and minority class gate ($25 \ge 10$).
- Subforums relying on the global fallback (0.85): 0.
- Operating thresholds spanned from 0.1087 (physics) to 0.3854 (programmers).

### Subsection B — Natural Repeat Rates in Real Chat Traffic (WildChat)
Using `allenai/WildChat` (ODC-By), we analyzed 5000 first user turns:
- **Exact String Repeats:** 490 / 5000 queries (9.80%).
- **Canonical Normalizer Repeats:** 529 / 5000 queries (10.58%), yielding a +0.78% incremental lift.
- **Guarded Semantic Matches (sim $\ge 0.85$):** 986 / 5000 queries (19.72%) guarded vs 1398 (27.96%) raw ungoverned.
- **Guarded Semantic Matches (sim $\ge 0.92$):** 777 / 5000 queries (15.54%) guarded vs 1063 (21.26%) raw ungoverned.
- 100 queries were exported to `data/review/pa_human_audit.csv` for human inspection.

---

## Section 5 — Phase A4: Sealed Out-of-Sample Evaluation Results

### Subsection A — Execution Summary
- **Stream Size:** 2200 queries (1000 QQP held-out, 600 PAWS test, 400 CQADupStack held-out, 200 WildChat; 536 safe non-exact targets, 464 adversarial negatives, 1200 cold seeds).
- **Stream Checksum:** `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991` (verified identical).
- **Execution Runtime:** 66.64 s on local CPU.

### Subsection B — Metrics and Confusion Matrix (`sealed`)

| Metric | Measured Value | 95% Clopper-Pearson CI |
| :--- | :---: | :---: |
| Total Queries ($N$) | 2200 | — |
| Total Cache Hits | 86 | — |
| Overall Hit Rate | 3.91% | [3.14%, 4.81%] |
| Total True Positives (TP) | 60 | — |
| Total False Positives (FP) | 26 | — |
| Total False Negatives (FN) | 476 | — |
| Total True Negatives (TN) | 438 | — |
| Overall Incorrect-Reuse Rate (IRR) | 30.23% | [20.79%, 41.08%] |
| Exact Canonical Hits | 11 | — |
| **Non-Exact Hits (Primary Endpoint)** | **75** | — |
| **Non-Exact True Positives** | **54** | — |
| **Non-Exact False Positives** | **21** | — |
| **Non-Exact IRR (Primary Endpoint)** | **28.00%** | **[18.24%, 39.56%]** |

### Subsection C — Breakdown by Source Dataset

| Dataset Source | Queries in Stream | Hits | TP | FP | Non-Exact Hits | Non-Exact FP | Non-Exact IRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `paws_test` | 600 | 43 | 22 | 21 | 34 | 16 | 47.06% [29.84%, 64.89%] |
| `qqp_heldout` | 1000 | 42 | 37 | 5 | 40 | 5 | 12.50% [4.19%, 26.80%] |
| `cqadupstack_heldout` | 400 | 1 | 1 | 0 | 1 | 0 | 0.0% |
| `wildchat_traffic` | 200 | 0 | 0 | 0 | 0 | 0 | 0.0% |

### Subsection D — Path Latencies and Decision Overhead

| Execution Path | Query Count | Mean Latency | Median (p50) | 95th Percentile (p95) |
| :--- | :---: | :---: | :---: | :---: |
| `CANONICAL_EXACT` | 30 | 0.06 ms | 0.04 ms | 0.16 ms |
| `AUTO_REUSE` | 43 | 14.28 ms | 14.36 ms | 17.75 ms |
| `LOCAL_EQUIVALENCE` | 181 | 33.72 ms | 34.54 ms | 66.03 ms |
| `BYPASS` | 1946 | 16.65 ms | 15.58 ms | 26.07 ms |
| **Overall Pipeline Decision Overhead** | **2200** | — | **15.81 ms** | **37.11 ms** |
| **End-to-End Latency (Simulated)** | **2200** | — | **8169.03 ms** | **8184.04 ms** |

---

## Section 6 — Workload Avoided & Repeat-Share Projection Curve

### Subsection A — Workload Avoided
- **Calls Avoided per 1000 Queries:** 39.1 calls
- **Tokens Avoided per 1000 Queries:** 2850.5 tokens (based on baseline mean of 72.92 tokens/query)
- **Break-Even Hit Rate:** **0.1976%** (0.00198). At any hit rate above 0.1976%, the 15.81 ms cache lookup overhead is smaller than the 8153.50 ms LLM latency saved.

### Subsection B — Projected Hit Rate & Latency across Repeat Shares

| Scenario | Repeat Share | Projected Hit Rate | Projected End-to-End p50 Latency |
| :--- | :---: | :---: | :---: |
| Zero Repeats (Cold Cache) | 0.0% | 0.0% | 8169.31 ms |
| Low Chat Traffic | 10.0% | 0.86% | 8099.19 ms |
| Typical SaaS Traffic | 20.0% | 1.72% | 8029.07 ms |
| High Repeat Agent Traffic | 40.0% | 3.44% | 7888.82 ms |
| High Congestion Traffic | 60.0% | 5.16% | 7748.58 ms |

---

## Section 7 — Deep Dive: Why Gate A Failed

Gate A failed on two separate criteria:
1. **Statistical Volume:** 75 non-exact hits were achieved instead of the required $\ge 100$.
2. **Safety Ceiling:** The non-exact IRR point estimate was 28.00% [18.24%, 39.56%], exceeding the 10.0% ceiling.

### Root Cause Analysis
1. **Generalization Gap on Adversarial Scrambles:** While fine-tuning on 2000 training pairs eliminated PAWS false positives on the validation set (down from 34 to 0), the model did not generalize to unseen grammatical permutations in `paws_test`. Out of 21 non-exact false positives on the sealed stream, **16 originated from `paws_test`**.
2. **Failure of Bi-Encoder Vector Routing:** When word-scrambled negatives had high cosine similarity ($\ge 0.50$), they entered the `LOCAL_EQUIVALENCE` stage. The fine-tuned cross-encoder assigned scores $> 0.85$ to 16 adversarial pairs, resulting in incorrect reuses.
3. **QQP Generalization:** On standard paraphrase queries (`qqp_heldout`), the pipeline performed reasonably well (37 TP, 5 FP; non-exact IRR = 12.50%), but this alone was still above the 10.0% safety boundary.

### Engineering Integrity Note
In strict accordance with the scientific pre-registration protocol, **no post-hoc threshold tuning or re-filtering was performed on the sealed test set**. The result stands as a verified finding regarding the limits of small cross-encoder models on adversarial syntactic variations.

Per protocol instructions, Phase B is **halted** and documented in `docs/BLOCKED_A.md`.
