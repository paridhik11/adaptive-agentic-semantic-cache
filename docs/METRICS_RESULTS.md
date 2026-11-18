# Offline Metrics Campaign Results: Large-Scale Evaluation & Evidence Correction

**Campaign Version:** 1.1  
**Date:** 2026-11-02  
**Harness:** `scripts/evaluate_large_workload.py`  
**Evaluated Workloads:**
- **Profile A (Repeat-Rich):** 600 events per arm (2,400 events total across 4 matched arms). Executed on 12 groups / 60 base intents. Telemetry in `artifacts/metrics_campaign/repeat_rich/` and `artifacts/metrics_campaign/`.
- **Profile B (Lower-Repeat):** 400 events per arm (1,600 events total across 4 matched arms). Executed across all 25 groups / 125 base intents. Telemetry in `artifacts/metrics_campaign/lower_repeat/`.

---

## 1. Executive Summary & Evidence Corrections

Following independent review, this report incorporates three essential methodological corrections:

1. **Workload-Expected (Eligible) Reuse vs Actual Answer Correctness:**
   - Raw records track `is_eligible_reuse`. The legacy field `is_correct_reuse` is **DEPRECATED** as a misleading name: category eligibility tags never imply answer-level correctness.
   - This records strictly whether a cache hit occurred on a benchmark-defined eligible query type (seed repeat or safe paraphrase), **not** that the answer was independently adjudicated as factually correct.
   - Avoided calls are labeled **workload-expected eligible reuse**. Actual answer correctness, semantic suitability, and safe coverage remain **unverified pending independent answer-level review**.
2. **Defined vs Executed Coverage Disclosed:**
   - The workload framework defines **25 documentation source groups** and **125 distinct base intents**.
   - The **repeat-rich profile** executes **60 base intents across the first 12 source groups** (`intents[:60]`), modeling high query redundancy (4x exact repeats, 3x paraphrases).
   - The **lower-repeat profile** executes **all 125 base intents across all 25 source groups**, modeling exploratory breadth and cross-domain coverage.
3. **Classification of Adversarial Evidence:**
   - The inverse-operation guard regex was modified in `src/policy/decomposed_applicability.py` during development after observing an initial failure on `str_02:challenge_inverse_operation`.
   - The 0/120 adversarial result on the repeat-rich run is therefore classified as **development regression evidence**, not untouched holdout precision.
   - When evaluated without further tuning on the fresh 400-event lower-repeat profile across all 25 groups, `verified_semantic` produced **1 false accept out of 125 adversarial challenges** (0.80% error rate, exact 95% Clopper-Pearson CI `[0.02%, 4.38%]`), while `exact_only` achieved **0 false accepts** (0/125, 0.00%, CI `[0.00%, 2.91%]`).

---

## 2. Workload Profile A: Repeat-Rich Benchmark (600 Events per Arm)

- **Executed Scope:** 12 source groups, 60 base intents.
- **Composition:** 60 seeds, 240 exact repeats (4x), 180 natural paraphrases (3x), 120 adversarial challenges.
- **Artifacts:** `artifacts/metrics_campaign/repeat_rich/` (and root `artifacts/metrics_campaign/`).

| Metric | Arm 1: `no_cache` *(Baseline)* | Arm 2: `exact_only` *(Safe Default)* | Arm 3: `verified_semantic` *(Guarded Opt-In)* | Arm 4: `shadow_semantic` *(Diagnostic)* |
| :--- | :---: | :---: | :---: | :---: |
| **Serving Configuration** | `mode="off"` | `mode="exact"` | `mode="guarded"` | `mode="shadow"` |
| **Total Evaluation Events** | 600 | 600 | 600 | 600 |
| **Total Provider Invocations** | 1,800 | 948 | **927** | 948 |
| **Calls Avoided on Eligible Reuse** | 0 | **852** | **873** | **852** |
| **Calls Avoided on Ineligible Reuse** | 0 | **0** | **0** | **0** |
| **Synthetic Token Volume** *(Offline)* | 201,667 | 107,750 | **105,429** | 107,750 |
| **Synthetic Token Reduction** | 0.0% | 46.57% | **47.72%** | 46.57% |
| **Cache Hits Total** | 0 | 284 | **291** | 284 |
| - *Exact Hits (Tier 0)* | 0 | 284 | 273 | 284 |
| - *Verified Semantic Hits (Tier 1)* | 0 | 0 | **18** | 0 |
| **Exact Repeat Coverage** | 0.0% | **98.33%** | **98.33%** | **98.33%** |
| **Paraphrase Opportunity Coverage** | 0.0% | 26.67% | **29.44%** | 26.67% |
| **Adversarial Challenges Tested** | 120 | 120 | 120 | 120 |
| **Adversarial False Accepts** | 0 | **0** | **0** | **0** |
| **Adversarial Error Rate (Point Est.)** | 0.00% | **0.00%** | **0.00%** | **0.00%** |
| **Exact 95% Clopper-Pearson CI** | `[0.00%, 3.03%]` | `[0.00%, 3.03%]` | `[0.00%, 3.03%]` | `[0.00%, 3.03%]` |
| **Adversarial Evidence Type** | Baseline | Regression check | Development regression | Diagnostic only |
| **Whole-Agent Latency p50** | 130.86 ms | 140.80 ms | 137.81 ms | 141.58 ms |
| **Whole-Agent Latency p90** | 146.15 ms | 227.63 ms | 232.99 ms | 232.59 ms |
| **Whole-Agent Latency p95** | 151.58 ms | 233.14 ms | 237.73 ms | 241.28 ms |
| **Total Wall Clock Time** | 79.54 s | 81.56 s | 80.83 s | 81.91 s |

---

## 3. Workload Profile B: Lower-Repeat Breadth Benchmark (400 Events per Arm)

- **Executed Scope:** **All 25 documentation source groups, all 125 base intents**.
- **Composition:** 125 seeds (1 per intent across all 25 groups), 60 exact repeats, 90 natural paraphrases, 125 adversarial challenges (rotating 6 challenge types across all 25 groups).
- **Execution Invariant:** Executed without further policy tuning.
- **Artifacts:** `artifacts/metrics_campaign/lower_repeat/summary.json`, `manifest.json`, `offline_results.jsonl`.

| Metric | Arm 1: `no_cache` *(Baseline)* | Arm 2: `exact_only` *(Safe Default)* | Arm 3: `verified_semantic` *(Guarded Opt-In)* | Arm 4: `shadow_semantic` *(Diagnostic)* |
| :--- | :---: | :---: | :---: | :---: |
| **Serving Configuration** | `mode="off"` | `mode="exact"` | `mode="guarded"` | `mode="shadow"` |
| **Total Evaluation Events** | 400 | 400 | 400 | 400 |
| **Total Provider Invocations** | 1,200 | 1,023 | **975** | 1,023 |
| **Calls Avoided on Eligible Reuse** | 0 | **177** | **222** | **177** |
| **Calls Avoided on Ineligible Reuse** | 0 | **0** | **3** | **0** |
| **Synthetic Token Volume** *(Offline)* | 133,964 | 114,453 | **109,302** | 114,453 |
| **Synthetic Token Reduction** | 0.0% | 14.56% | **18.41%** | 14.56% |
| **Cache Hits Total** | 0 | 59 | **75** | 59 |
| - *Exact Hits (Tier 0)* | 0 | 59 | 57 | 59 |
| - *Verified Semantic Hits (Tier 1)* | 0 | 0 | **18** | 0 |
| **Exact Repeat Coverage** | 0.0% | **98.33%** | **98.33%** | **98.33%** |
| **Paraphrase Opportunity Coverage** | 0.0% | 0.00% | **10.00%** | 0.00% |
| **Adversarial Challenges Tested** | 125 | 125 | 125 | 125 |
| **Adversarial False Accepts** | 0 | **0** | **1** | **0** |
| **Adversarial Error Rate (Point Est.)** | 0.00% | **0.00%** | **0.80%** | **0.00%** |
| **Exact 95% Clopper-Pearson CI** | `[0.00%, 2.91%]` | `[0.00%, 2.91%]` | `[0.02%, 4.38%]` | `[0.00%, 2.91%]` |
| **Whole-Agent Latency p50** | 89.70 ms | 168.42 ms | 180.60 ms | 183.55 ms |
| **Whole-Agent Latency p90** | 101.70 ms | 190.42 ms | 205.24 ms | 206.12 ms |
| **Whole-Agent Latency p95** | 103.71 ms | 194.46 ms | 219.77 ms | 224.81 ms |
| **Total Wall Clock Time** | 35.98 s | 58.38 s | 63.39 s | 59.81 s |

---

## 4. Stage-by-Stage Provider Breakdown

Every cold miss executes the complete 3-stage agent workflow (supervisor planning, specialist draft generation, and reviewer verification):

### Repeat-Rich Workload (600 events):
| Arm | Planning Calls | Generation Calls | Review Calls | Total Remote Calls |
| :--- | :---: | :---: | :---: | :---: |
| **`no_cache`** | 600 | 600 | 600 | 1,800 |
| **`exact_only`** | 316 | 316 | 316 | 948 |
| **`verified_semantic`** | **309** | **309** | **309** | **927** |
| **`shadow_semantic`** | 316 | 316 | 316 | 948 |

### Lower-Repeat Workload (400 events):
| Arm | Planning Calls | Generation Calls | Review Calls | Total Remote Calls |
| :--- | :---: | :---: | :---: | :---: |
| **`no_cache`** | 400 | 400 | 400 | 1,200 |
| **`exact_only`** | 341 | 341 | 341 | 1,023 |
| **`verified_semantic`** | **325** | **325** | **325** | **975** |
| **`shadow_semantic`** | 341 | 341 | 341 | 1,023 |

---

## 5. False Accept Analysis & Architectural Validation

### 5.1 Repeat-Rich Run (Task D Optimization)
During initial testing of the repeat-rich profile, an adversarial false accept occurred on `str_02:challenge_inverse_operation` (`"How do I split a comma-separated string into a list in Python?"` matching seed `"How do I join a list of strings with a comma separator?"`).
- **Root Cause:** The regex `r"\b(?:split\w*|break\w*|separat\w*)\b"` matched `"comma separator"` in the seed query.
- **Refinement:** The regex was updated in `src/policy/decomposed_applicability.py` to `r"\b(?:split\w*|break\w*|partition\w*|divid\w*)\b"`.
- **Classification:** The resulting 0/120 is **development regression evidence** verifying the fix on that specific workload, not holdout precision.

### 5.2 Lower-Repeat Breadth Run (1 False Accept in `verified_semantic`)
When the untuned system was evaluated across all 25 source groups in the lower-repeat profile, a false accept was observed:
- **Event ID:** `async_05:challenge_added_requirements`
- **Query:** `"How do I optimize asynchronous programming with asyncio performance and benchmark using cProfile?"`
- **Candidate Reused:** Seed `async_01` answer: `"Answer grounded in documentation: Asynchronous tasks use async and await syntax. Run coroutines using asyncio.run(coro()). Run multiple tasks concurrently using asyncio.gather(*tasks)."`
- **Root Cause:** The query contained an unhandled added requirement (`"benchmark using cProfile"`). The cross-encoder scored the topical asyncio match above the 0.85 verification threshold.
- **Outcome:** Resulted in 3 avoided calls on ineligible reuse.
- **Architectural Confirmation:** This finding validates why:
  1. **`mode="exact"` is the mandatory default serving configuration.** Under `exact_only`, this query correctly missed and executed fresh agent work (0 false accepts, 0/125).
  2. **`mode="shadow"` is the default neural mode.** Diagnostic scoring logs candidate evaluation without substituting unproven answers into live serving.
  3. **Guarded semantic serving is strictly an experimental opt-in.**
  4. **Semantic suitability and safe coverage remain unverified without independent answer adjudication.**

---

## 6. Actual Selected Answers & Quality Review Status

- Every request is logged in the JSONL telemetry files with:
  - `selected_cached_answer`: The exact answer returned on hit (`null` on miss).
  - `matched_question`: The question associated with the admitted candidate.
  - `actual_response_answer`: The final answer returned by the agent.
  - `is_eligible_reuse`: Whether the hit matched benchmark-defined eligible query types.
  - `is_ineligible_reuse`: Whether a hit occurred on an ineligible challenge query.
- **Answer-Quality Review Status:** Labeled `unadjudicated_workload_expected_reuse`. The evaluation does not claim independent human truth or certified answer correctness.

---

## 7. Artifact Manifest & Reproduction

- **Python Version:** 3.14.5
- **Dense Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- **Passage Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- **Artifact Directories:**
  - Repeat-Rich (600 events): `artifacts/metrics_campaign/repeat_rich/` (and root `artifacts/metrics_campaign/`)
  - Lower-Repeat (400 events): `artifacts/metrics_campaign/lower_repeat/`
- **Reproduction Commands:**
  ```powershell
  # 1. Run repeat-rich profile (600 events x 4 arms = 2,400 events)
  python scripts/evaluate_large_workload.py --profile repeat_rich --output-dir artifacts/metrics_campaign/repeat_rich

  # 2. Run lower-repeat profile (400 events x 4 arms = 1,600 events)
  python scripts/evaluate_large_workload.py --profile lower_repeat --output-dir artifacts/metrics_campaign/lower_repeat
  ```
