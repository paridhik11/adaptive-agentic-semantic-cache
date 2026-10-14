# Phase A — Claim Check and Number Verification Audit

> **Purpose:** Authoritative verification matrix mapping every numeric value, threshold, latency timing, count, percentage, confidence interval, and ratio appearing in Phase A documentation directly to its source row in [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md).
>
> **Strict Compliance Policy:**
> - Every number must originate from [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md).
> - Zero unverified metrics permitted.
> - Verified strictly using `python scripts/check_numeric_tokens.py`.

---

## Audit Summary

- **Total Phase A Claims Audited**: All audited rows verified against [`docs/FACTS.md`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md)
- **Gate A Verification**: Evaluated and audited — **GATE A FAILED** (documented in `docs/BLOCKED_A.md`)
- **Unledgered Numeric Tokens**: 0 (strictly verified by `scripts/check_numeric_tokens.py`)

---

## 1. Baseline Re-Measurements & Preflight (Phase A0)

| # | Value / Metric | Location in Docs | Context / Claim | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| AC_01 | 281 | Preflight summary | Total passing unit tests in suite | [`C-17`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L45) | 281 tests passing | **PASS** |
| AC_02 | 8997.49 ms | Baseline section | Qwen2.5-0.5B local CPU mean latency ($n=100$) | [`PA-B02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L407) | 8997.49 ms | **PASS** |
| AC_03 | 8153.50 ms | Baseline section | Qwen2.5-0.5B local CPU median (p50) latency ($n=100$) | [`PA-B03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L408) | 8153.50 ms | **PASS** |
| AC_04 | 12763.96 ms | Baseline section | Qwen2.5-0.5B local CPU p95 latency ($n=100$) | [`PA-B04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L409) | 12763.96 ms | **PASS** |
| AC_05 | 11.05 tokens | Baseline section | Qwen2.5-0.5B local CPU mean prompt tokens | [`PA-B05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L410) | 11.05 tokens | **PASS** |
| AC_06 | 61.87 tokens | Baseline section | Qwen2.5-0.5B local CPU mean completion tokens | [`PA-B06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L411) | 61.87 tokens | **PASS** |
| AC_07 | 72.92 tokens | Baseline section | Qwen2.5-0.5B local CPU mean total tokens | [`PA-B07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L412) | 72.92 tokens | **PASS** |
| AC_08 | 2002.11 ms | Baseline section | Auxiliary remote OpenRouter mean latency ($n=20$) | [`PA-B08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L413) | 2002.11 ms | **PASS** |
| AC_09 | 1346.64 ms | Baseline section | Auxiliary remote OpenRouter p50 latency ($n=20$) | [`PA-B08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L413) | 1346.64 ms | **PASS** |
| AC_10 | 3759.78 ms | Baseline section | Auxiliary remote OpenRouter p95 latency ($n=20$) | [`PA-B08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L413) | 3759.78 ms | **PASS** |
| AC_11 | 164.25 tokens | Baseline section | Auxiliary remote OpenRouter mean total tokens | [`PA-B08`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L413) | 164.25 tokens | **PASS** |

---

## 2. Matcher Candidate Sweep & Validation (Phase A1)

| # | Value / Metric | Location in Docs | Context / Claim | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| AC_12 | 79 hits, 34 FP | Matcher table | Candidate (a) off-the-shelf hits and false positives | [`PA-M01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L427) | Hits: 79, FP: 34 (all PAWS) | **PASS** |
| AC_13 | 43.04% [31.94%, 54.67%] | Matcher table | Candidate (a) incorrect-reuse rate on validation set | [`PA-M01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L427) | IRR: 43.04% [31.94%, 54.67%] | **PASS** |
| AC_14 | 59 hits, 3 FP | Matcher table | Candidate (b) fine-tuned hits and false positives | [`PA-M02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L428) | Hits: 59, FP: 3 (0 PAWS) | **PASS** |
| AC_15 | 5.08% [1.06%, 14.15%] | Matcher table | Candidate (b) incorrect-reuse rate on validation set | [`PA-M02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L428) | IRR: 5.08% [1.06%, 14.15%] | **PASS** |
| AC_16 | 40 hits, 1 FP | Matcher table | Candidate (c) fine-tuned + guards hits and FP | [`PA-M03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L429) | Hits: 40, FP: 1 (0 PAWS) | **PASS** |
| AC_17 | 2.50% [0.06%, 13.16%] | Matcher table | Candidate (c) incorrect-reuse rate on validation set | [`PA-M03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L429) | IRR: 2.50% [0.06%, 13.16%] | **PASS** |
| AC_18 | 449.57 s | Matcher section | Candidate (b) fine-tuning execution duration | [`PA-M04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L430) | Duration: 449.57 s | **PASS** |

---

## 3. CQADupStack Calibration & WildChat Traffic (Phase A2 & A3)

| # | Value / Metric | Location in Docs | Context / Claim | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| AC_19 | 12 subforums | Calibration section | Total subforums analyzed and fitted | [`PA-C01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L436), [`PA-C02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L437) | 12 subforums fitted | **PASS** |
| AC_20 | 0.1087 to 0.3854 | Calibration section | Domain operating thresholds range | [`PA-C03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L438) | 0.1087 (physics) to 0.3854 (programmers) | **PASS** |
| AC_21 | 490 / 5000 (9.80%) | WildChat section | Exact string repeat count and percentage | [`PA-W02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L445) | 490 / 5000 (9.80%) | **PASS** |
| AC_22 | 529 / 5000 (10.58%) | WildChat section | Canonical normalizer repeat count and percentage | [`PA-W03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L446) | 529 / 5000 (10.58%) | **PASS** |
| AC_23 | 986 / 5000 (19.72%) | WildChat section | Guarded semantic reuse at sim 0.85 | [`PA-W04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L447) | 986 / 5000 (19.72%) | **PASS** |
| AC_24 | 777 / 5000 (15.54%) | WildChat section | Guarded semantic reuse at sim 0.92 | [`PA-W05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L448) | 777 / 5000 (15.54%) | **PASS** |
| AC_25 | 100 queries | WildChat section | Exported audit sample size | [`PA-W06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L449) | 100 queries | **PASS** |

---

## 4. Sealed Test Stream Single-Run Evaluation (Phase A4)

| # | Value / Metric | Location in Docs | Context / Claim | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| AC_26 | 2200 queries | Sealed results | Total sealed stream query count | [`PA-S02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L456) | 2200 queries | **PASS** |
| AC_27 | 66.64 s | Sealed results | Single execution duration | [`PA-S03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L457) | 66.64 s | **PASS** |
| AC_28 | 86 hits (TP=60, FP=26) | Sealed results | Overall cache hits and breakdown | [`PA-S04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L458) | 86 hits, 60 TP, 26 FP | **PASS** |
| AC_29 | 3.91% [3.14%, 4.81%] | Sealed results | Overall stream hit rate | [`PA-S04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L458) | 3.91% [3.14%, 4.81%] | **PASS** |
| AC_30 | 30.23% [20.79%, 41.08%] | Sealed results | Overall incorrect-reuse rate across all hits | [`PA-S04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L458) | 30.23% [20.79%, 41.08%] | **PASS** |
| AC_31 | 75 hits | Primary endpoint | Non-exact cache hits count | [`PA-S05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L459) | 75 hits | **PASS** |
| AC_32 | 54 TP, 21 FP | Primary endpoint | Non-exact hit correctness breakdown | [`PA-S05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L459) | 54 TP, 21 FP | **PASS** |
| AC_33 | 28.00% [18.24%, 39.56%] | Primary endpoint | Non-exact incorrect-reuse rate | [`PA-S05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L459) | 28.00% [18.24%, 39.56%] | **PASS** |
| AC_34 | 11 hits | Secondary endpoint | Exact canonical matches count | [`PA-S07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L461) | 11 hits | **PASS** |
| AC_35 | 47.06% [29.84%, 64.89%] | Source breakdown | PAWS test stream non-exact IRR | [`PA-SB01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L477) | 47.06% [29.84%, 64.89%] | **PASS** |
| AC_36 | 12.50% [4.19%, 26.80%] | Source breakdown | QQP held-out non-exact IRR | [`PA-SB02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L478) | 12.50% [4.19%, 26.80%] | **PASS** |

---

## 5. Latency Overhead, Workload Avoided & Projections

| # | Value / Metric | Location in Docs | Context / Claim | FACTS.md Row(s) | Verified Value in FACTS.md | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| AC_37 | 0.04 ms | Path latencies | Canonical exact path p50 latency | [`PA-L01`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L467) | 0.04 ms | **PASS** |
| AC_38 | 14.36 ms | Path latencies | Vector store auto-reuse path p50 latency | [`PA-L02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L468) | 14.36 ms | **PASS** |
| AC_39 | 34.54 ms | Path latencies | CrossEncoder local equivalence path p50 latency | [`PA-L03`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L469) | 34.54 ms | **PASS** |
| AC_40 | 15.58 ms | Path latencies | Pipeline bypass path p50 latency | [`PA-L04`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L470) | 15.58 ms | **PASS** |
| AC_41 | 15.81 ms | Decision overhead | Overall pipeline decision overhead median (p50) | [`PA-L05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L471) | 15.81 ms | **PASS** |
| AC_42 | 37.11 ms | Decision overhead | Overall pipeline decision overhead p95 | [`PA-L05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L471) | 37.11 ms | **PASS** |
| AC_43 | 8169.03 ms | E2E latency | End-to-end simulated p50 latency | [`PA-L05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L471) | 8169.03 ms | **PASS** |
| AC_44 | 8184.04 ms | E2E latency | End-to-end simulated p95 latency | [`PA-L05`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L471) | 8184.04 ms | **PASS** |
| AC_45 | 39.1 calls | Workload avoided | Calls avoided per 1000 queries | [`PA-L06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L472) | 39.1 calls | **PASS** |
| AC_46 | 2850.5 tokens | Workload avoided | Tokens avoided per 1000 queries | [`PA-L06`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L472) | 2850.5 tokens | **PASS** |
| AC_47 | 0.1976% | Workload avoided | Break-even hit rate | [`PA-L07`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L473) | 0.1976% | **PASS** |
| AC_48 | 8099.19 ms | Repeat curve | Projected p50 at 10% repeat traffic | [`PA-RC02`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/docs/FACTS.md#L487) | 8099.19 ms | **PASS** |
