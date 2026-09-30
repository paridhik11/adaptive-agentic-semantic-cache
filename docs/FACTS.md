# FACTS.md — Verified Fact Ledger

> **Purpose:** Single authoritative ledger for `adaptive-agentic-semantic-cache`.
> Every later document must cite a row from this file by row label.
> **Do NOT cite** `comprehensive_project_status_report.md` or
> `Adaptive_Agentic_Semantic_Cache_Master_Report.pdf` — those are known stale.

---

## Reading Guide

| Column | Meaning |
| :--- | :--- |
| **Claim / Value** | The exact numeric or categorical fact being recorded |
| **Source file + location** | The file and section/line where this was read |
| **How to re-run** | Command or step to reproduce the measurement |
| **DISPUTED** | Rows where two sources give different numbers |

Constants read from source code are marked **`[CODE]`**.
Facts read exclusively from walkthrough docs are marked **`[DOC]`**.
Facts verified by recomputation are marked **`[RECOMPUTED]`**.

---

## 1. System Constants (read from source code, not docs)

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| C-01 | Embedding model name | `all-MiniLM-L6-v2` **`[CODE]`** | `src/cache/embedding.py` line 52 | `grep DEFAULT_MODEL_NAME src/cache/embedding.py` |
| C-02 | Embedding model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` **`[CODE]`** | `src/cache/embedding.py` line 63 | `grep MODEL_REVISION src/cache/embedding.py` |
| C-03 | Embedding output dimension | 384 **`[CODE]`** | `src/cache/embedding.py` line 53 | `grep EMBEDDING_DIM src/cache/embedding.py` |
| C-04 | FAISS index type | `faiss.IndexFlatIP` (inner-product on L2-normalised vectors = cosine similarity) **`[CODE]`** | `src/cache/vector_store.py` line 54 | `grep IndexFlatIP src/cache/vector_store.py` |
| C-05 | Stability confidence threshold (Sub-stage A) | 0.80 **`[CODE]`** | `src/classifier/stability_classifier.py` line 71 | `grep STABLE_CONFIDENCE_THRESHOLD src/classifier/stability_classifier.py` |
| C-06 | TierRouter AUTO_REUSE similarity floor | 0.92 **`[CODE]`** | `src/decision/tier_router.py` line 210 | `grep auto_reuse_sim_floor src/decision/tier_router.py` |
| C-07 | TierRouter AUTO_REUSE confidence floor | 0.90 **`[CODE]`** | `src/decision/tier_router.py` line 211 | `grep auto_reuse_conf_floor src/decision/tier_router.py` |
| C-08 | TierRouter BYPASS similarity ceiling | 0.50 **`[CODE]`** | `src/decision/tier_router.py` line 213 | `grep bypass_sim_ceiling src/decision/tier_router.py` |
| C-09 | TierRouter BYPASS confidence ceiling | 0.80 **`[CODE]`** | `src/decision/tier_router.py` line 214 | `grep bypass_conf_ceiling src/decision/tier_router.py` |
| C-10 | Adaptive engine minimum category N gate | 20 **`[CODE]`** | `src/decision/adaptive_threshold_engine.py` line 78 | `grep MINIMUM_CATEGORY_N src/decision/adaptive_threshold_engine.py` |
| C-11 | Adaptive engine minimum minority class N gate | 10 **`[CODE]`** | `src/decision/adaptive_threshold_engine.py` line 79 | `grep MINIMUM_MINORITY_CLASS_N src/decision/adaptive_threshold_engine.py` |
| C-12 | Adaptive engine global fallback threshold | 0.85 **`[CODE]`** | `src/decision/adaptive_threshold_engine.py` line 80 | `grep FALLBACK_THRESHOLD src/decision/adaptive_threshold_engine.py` |
| C-13 | Safety ceiling IRR_cache | 0.10 (10%) **`[CODE]`** | `src/decision/adaptive_threshold_engine.py` line 82 | `grep SAFETY_CEILING_IRR src/decision/adaptive_threshold_engine.py` |
| C-14 | Production LLM judge model (Phase 3 onward) | `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter **`[CODE]`** | `src/decision/judge_call.py` line 34: `DEFAULT_JUDGE_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"` | `grep DEFAULT_JUDGE_MODEL src/decision/judge_call.py` |
| C-15 | Initial judge model attempted then abandoned | `gemini-2.5-flash` (Google AI Studio free tier; 17/32 pairs evaluated before daily RPD=20 quota exhausted) **`[CODE]`** | `src/decision/judge_call.py` line 36: `GEMINI_DEFAULT_JUDGE_MODEL: str = "gemini-2.5-flash"` | `grep GEMINI_DEFAULT_JUDGE_MODEL src/decision/judge_call.py` |
| C-16 | Judge fail-closed fallback | On ANY failure, returns `decision=BYPASS`, `is_safe=False`, `confidence=0.0`, `fallback_triggered=True`. Never fails open. **`[CODE]`** | `src/decision/judge_call.py` lines 8–14 (docstring invariant); line 356 (runtime contract); lines 391–405 (missing-key return path) | `grep -n "fallback_triggered=True" src/decision/judge_call.py` |

---

## 2. Dataset Sizes

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| D-01 | Pair reuse benchmark (Phase 2 baseline) | 120 pairs (62 SAFE, 58 UNSAFE) **`[DOC]`** | `docs/phase2_walkthrough.md` ss2 table | `python -c "import json; d=json.load(open('data/raw/query_pair_reuse_benchmark.json')); print(len(d))"` |
| D-02 | Stability benchmark development (non-pristine) | 160 queries **`[DOC]`** | `docs/phase2_walkthrough.md` ss2 table | `python -c "import json; d=json.load(open('data/raw/query_stability_benchmark.json')); print(len(d))"` |
| D-03 | Stability benchmark held-out challenge | 60 queries (31 STABLE, 24 DYNAMIC, 5 CONDITIONALLY_STABLE) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1B.2 | `python -c "import json; d=json.load(open('data/raw/query_stability_benchmark_heldout.json')); print(len(d))"` |
| D-04 | Stability benchmark final test (pristine) | 60 queries (37 STABLE, 23 DYNAMIC) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1.1 | `python -c "import json; d=json.load(open('data/raw/query_stability_benchmark_final_test.json')); print(len(d))"` |
| D-05 | Human credibility set (Phase 1 dev/diagnostic) | 85 queries **`[DOC]`** | `docs/phase2_walkthrough.md` ss2 table | `python -c "import json; d=json.load(open('data/raw/query_stability_human_credibility.json')); print(len(d))"` |
| D-06 | Synthetic query pair feedback (Phase 4) | 108 pairs **`[DOC]`** | `docs/phase4_walkthrough.md` ss3.1 | `python -c "import json; d=json.load(open('data/raw/synthetic_query_pair_feedback.json')); print(len(d))"` |
| D-07 | Phase 5 synthetic load test stream | 70 queries **`[DOC]`** | `docs/phase5_walkthrough.md` ss2.1 | `python -c "import json; d=json.load(open('data/raw/load_test_query_stream.json')); print(len(d))"` |
| D-08 | Phase 5 scaled validation dataset (new_dataset_v3.json) | 338 queries (210 StackExchange cold seeds + 128 hand-authored) **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.1 | `python -c "import json; d=json.load(open('data/raw/new_dataset_v3.json')); print(len(d))"` |
| D-09 | Phase 6 blind evaluation dataset | 175 entries across 7 domains **`[DOC]`** | `docs/phase6_walkthrough.md` ss1.1 | `python -c "import json; d=json.load(open('data/raw/phase6_blind_eval_dataset.json')); print(len(d))"` |

---

## 3. Phase 1 and Phase 2 — Classifier and Threshold Sweep

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P1-01 | Classifier accuracy on final test set (N=60, pristine) | **91.67%** (55/60 correct) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1.3 | `python scripts/step0_final_test_eval.py` |
| P1-02 | Dangerous error count on final test set | **0** (FP=0 on 23 dynamic queries) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1.4 and ss1.6 | `python scripts/step0_final_test_eval.py` |
| P1-03 | Classifier accuracy on held-out challenge set (N=60, pristine) | **91.67%** (55/60 correct) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1B.3 | `python scripts/step0b_heldout_test_eval.py` |
| P1-04 | Dangerous error count on held-out challenge set | **0** (FP=0 on 24 dynamic queries) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1B.4 and ss1B.6 | `python scripts/step0b_heldout_test_eval.py` |
| P1-05 | Conservative error rate — final test set | 13.51% (FN=5/37 stable queries routed to DYNAMIC) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1.6 | `python scripts/step0_final_test_eval.py` |
| P1-06 | Conservative error rate — held-out challenge set | 13.89% (FN=5/36 stable queries routed to DYNAMIC) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1B.6 | `python scripts/step0b_heldout_test_eval.py` |
| P2-01 | Best fixed-threshold IRR_cache and its threshold | **20.83%** (5 FP / 24 hits) at threshold **0.85** — lowest IRR_cache where any hits exist **`[DOC]`** | `docs/phase2_walkthrough.md` ss5 coarse sweep table threshold=0.85 row and ss6 | `python scripts/run_cache_threshold_sweep.py` |
| P2-02 | Safety criterion (pre-stated before sweep) | IRR_cache < 10% **`[DOC]`** | `docs/phase2_walkthrough.md` ss4 | — pre-stated not a measurement |
| P2-03 | Does any fixed threshold satisfy IRR_cache < 10%? | **NO** — at every threshold where hits > 0, IRR_cache >= 20.83% **`[DOC]`** | `docs/phase2_walkthrough.md` ss6 | `python scripts/run_cache_threshold_sweep.py` |
| P2-04 | Reference threshold for comparison (best CRR) | **0.85** (CRR=79.17%, ARR=20.00%, IRR_cache=20.83%) — reference only, not production-safe **`[DOC]`** | `docs/phase2_walkthrough.md` ss6 and ss7 | `python scripts/run_cache_threshold_sweep.py` |
| P2-05 | Mean embed + search latency per pair | **13.9–14.9 ms** **`[DOC]`** | `docs/phase2_walkthrough.md` ss5 header | `python scripts/run_cache_threshold_sweep.py` |
| P2-06 | Dataset N for Phase 2 sweep | 120 pairs **`[DOC]`** | `docs/phase2_walkthrough.md` ss2 | — |

---

## 4. Phase 3 — Judge IRR and ARR

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P3-01 | Phase 3 non-LLM decision step overall IRR_cache (same-set N=120) | **52.17%** (12 FP / 23 hits) — non-LLM linear combination proved inadequate **`[DOC]`** | `docs/phase3_walkthrough.md` ss3.2 table | re-run phase 3 evaluation |
| P3-02 | Phase 3 AMBIGUOUS-tier IRR_cache (non-LLM) | **57.1%** (12 FP / 21 hits in AMBIGUOUS band) **`[DOC]`** | `docs/phase3_walkthrough.md` ss3.3 | Same as P3-01 |
| P3-03 | Phase 3 AUTO-REUSE tier IRR_cache | **0.00%** (TP=2, FP=0) **`[DOC]`** | `docs/phase3_walkthrough.md` ss3.3 | Same as P3-01 |
| P3-04 | Phase 3 Gemini judge (gemini-2.5-flash) overall IRR_cache (partial: 17/32 evaluated, N=120) | **0.00%** (FP=0 / 5 hits; 15 pairs fell back to BYPASS on daily quota) **`[DOC]`** | `docs/phase3_judge_call_walkthrough.md` ss2 and ss6.1 | Blocked by RPD=20 daily quota |
| P3-05 | Phase 3 Gemini judge overall ARR (N=120) | **4.17%** (5 hits / 120 pairs) **`[DOC]`** | `docs/phase3_judge_call_walkthrough.md` ss2 comparison table | — |
| P3-06 | Phase 3 OpenRouter judge (nemotron-3-super-120b:free) overall IRR_cache (full 32/32, N=120) | **8.33%** (1 FP / 12 hits) — clears <10% ceiling; 95% CP CI **[0.21%, 38.48%]** (k=1, n=12, recomputed); same-set point estimate, not blind **`[DOC][RECOMPUTED]`** | `docs/phase3_judge_call_walkthrough.md` ss2 comparison table and ss7.8; CI recomputed via scipy | `python -c "from scipy.stats import beta; print(beta.ppf(0.025,1,12)*100, beta.ppf(0.975,2,11)*100)"` |
| P3-07 | Phase 3 OpenRouter judge overall ARR (N=120) | **10.00%** (12 hits / 120 pairs) **`[DOC]`** | `docs/phase3_judge_call_walkthrough.md` ss2 comparison table | Same as P3-06 |
| P3-08 | Tier distribution on 120 pairs (Phase 3) | AUTO_REUSE=2 (1.7%), AMBIGUOUS=32 (26.7%), BYPASS=86 (71.7%) **`[DOC]`** | `docs/phase3_walkthrough.md` ss3.1 | Same as P3-01 |

---

## 5. Phase 4 — Adaptive Policy Outcome

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P4-01 | Number of categories that moved off 0.85 fallback after Phase 4 | **0 of 7** — all remain on 0.85 global fallback **`[DOC]`** | `docs/phase4_walkthrough.md` ss7.3 Final Authoritative Table and ss11.1 | `python scripts/run_phase4_calibration.py` |
| P4-02 | Synthetic pairs labeled with genuine judge decisions | **108 of 108** (100%, across 3 sessions over 3 quota days) **`[DOC]`** | `docs/phase4_walkthrough.md` ss4.3 Session 2 post-run and ss10 DoD table | Inspect `data/openrouter_synthetic_feedback_telemetry.json` |
| P4-03 | Final operating threshold — all 7 categories | **0.8500** (global fallback) **`[CODE]`** | `src/decision/adaptive_threshold_engine.py` line 80; confirmed by `docs/phase4_walkthrough.md` ss7.3 | `grep FALLBACK_THRESHOLD src/decision/adaptive_threshold_engine.py` |
| P4-04 | Mathematical minimum hits to prove IRR_cache <= 10% (CP 95% CI, k=0) | **n >= 36 hits** **`[DOC]`** | `docs/phase4_walkthrough.md` ss8 equation | `python -c "import math; print(math.log(0.025)/math.log(0.90))"` (approx 35.01 so need 36) |
| P4-05 | Phase 4 contamination bug — 86 fallback stubs consumed as genuine labels in Session 0 | **BUG FIXED** — load_combined_data() now filters fallback_triggered=True records **`[DOC]`** | `docs/phase4_walkthrough.md` ss4.4 | `grep fallback_triggered scripts/run_phase4_calibration.py` |
| P4-06 | Dual role disclosure | Same Nemotron model used as both production judge and label generator — known methodological limitation **`[DOC]`** | `docs/phase4_walkthrough.md` ss4.2 | — |

---

## 6. Phase 5 — Load Test Results

### 6.1 — N=70 Pilot Run

Telemetry source: `data/load_test_telemetry.json` (raw, reconciled). AUTO_REUSE TP=20, AMBIGUOUS TP=2, total TP=22, FP=0.

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P5A-01 | Total queries | 70 **`[DOC]`** | `docs/phase5_walkthrough.md` ss3.2 | `python scripts/run_load_test.py` |
| P5A-02 | Cache hits (n); breakdown by path | **22** total (AUTO_REUSE TP=20, AMBIGUOUS judge-approved TP=2), FP=0 **`[RECOMPUTED]`** | `data/load_test_telemetry.json` outcome_type and tier counts | `python -c "import json; d=json.load(open('data/load_test_telemetry.json')); from collections import Counter; print(Counter((e['tier'],e['outcome_type']) for e in d if e['outcome_type']=='TP'))"` |
| P5A-03 | Empirical IRR_cache | **0.00%** (k=0 FP / n=22 hits) **`[DOC]`** | `docs/phase5_walkthrough.md` ss4.5 | Same |
| P5A-04 | 95% Clopper-Pearson CI on IRR_cache | **[0.00%, 15.44%]** (k=0, n=22) **`[DOC][RECOMPUTED]`** | `docs/phase5_walkthrough.md` ss4.5; scipy: beta.ppf(0.975, 1, 22) = 0.1544 | `python -c "from scipy.stats import beta; print(f'{beta.ppf(0.975,1,22)*100:.2f}%')"` |
| P5A-05 | Local resolution rate | **87.14%** (61/70 without remote judge call) **`[DOC]`** | `docs/phase5_walkthrough.md` ss4.2 | Same |
| P5A-06 | AUTO_REUSE mean latency | **19.18 ms** (n=20) **`[DOC]`** | `docs/phase5_walkthrough.md` ss4.3 | Same |
| P5A-07 | AMBIGUOUS judge mean latency | **9,291.35 ms** **`[DOC]`** | `docs/phase5_walkthrough.md` ss4.3 | Same |
| P5A-08 | Latency speedup (AUTO_REUSE vs AMBIGUOUS) | **484.5x** (99.79% reduction; 1 − 19.18/9291.35) **`[DOC][RECOMPUTED]`** | `docs/phase5_walkthrough.md` ss4.3 | Same |

### 6.2 — N=338 Scaled Run (new_dataset_v3.json)

Telemetry source: `data/scaled_load_test_telemetry_corrected.json` (raw, reconciled). AUTO_REUSE TP=1, AMBIGUOUS TP=13, total TP=14, FP=0.

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P5B-01 | IRR_cache before collision fix (as originally labeled) | **71.43%** (10 FP / 14 hits) **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.6 table row Empirical IRR_cache | `python scripts/evaluate_load_test.py --telemetry data/scaled_load_test_telemetry.json` |
| P5B-02 | IRR_cache after ground-truth collision correction | **0.00%** (0 FP / 14 hits; 10 annotation collisions relabeled MISS->HIT) **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.6 corrected column | `python scripts/evaluate_load_test.py --telemetry data/scaled_load_test_telemetry_corrected.json` |
| P5B-03 | 95% CP CI as originally labeled | **[41.90%, 91.61%]** (k=10, n=14) **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.6 | Same as P5B-01 |
| P5B-04 | 95% CP CI after correction | **[0.00%, 23.16%]** (k=0, n=14) **`[DOC][RECOMPUTED]`** | `docs/phase5_walkthrough.md` ss9.6; scipy: beta.ppf(0.975, 1, 14) = 0.2316 | `python -c "from scipy.stats import beta; print(f'{beta.ppf(0.975,1,14)*100:.2f}%')"` |
| P5B-05 | Number of annotation collisions relabeled | **10 of 10** FPs relabeled; all were genuine semantic equivalences **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.7 audit table | See ss9.7 |
| P5B-06 | Additional near-duplicate collisions NOT surfaced during execution | **6** (masked by BYPASS or StabilityClassifier) — remain uncorrected in new_dataset_v3.json **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.8 | `python scripts/run_new_dataset_load_test.py --dataset data/raw/new_dataset_v3.json --check-internal-collisions` |
| P5B-07 | Local resolution rate (N=338) | **86.39%** (292/338 without remote judge) **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.3 | — |
| P5B-08 | AUTO_REUSE mean latency (N=338, n=1 only) | **61.91 ms** **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.4 | — |
| P5B-09 | AMBIGUOUS judge mean latency (N=338) | **7,715.62 ms** (46 calls) **`[DOC]`** | `docs/phase5_walkthrough.md` ss9.4 | — |
| P5B-10 | Latency speedup (N=338) | **124.6x** (99.20% reduction; 1 − 61.91/7715.62) **`[DOC][RECOMPUTED]`** | `docs/phase5_walkthrough.md` ss9.4 | — |

---

## 7. Phase 6 — Blind Evaluation

Telemetry source: `data/phase6_telemetry.json` (raw, reconciled). AUTO_REUSE TP=1, AMBIGUOUS judge-approved TP=7, total TP=8, FP=0.
AMBIGUOUS tier invoked 18 times. **DISPUTED:** walkthrough ss6 says AMBIGUOUS breakdown TP=6, TN=10, FN=2, FP=0; raw telemetry shows TP=7, TN=9, FN=2, FP=0. Total hit count n=8 is consistent in both. Row P6-08 reflects raw telemetry.

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P6-01 | N | **175** entries across 7 domains **`[DOC]`** | `docs/phase6_walkthrough.md` ss1.1 | `python -c "import json; d=json.load(open('data/raw/phase6_blind_eval_dataset.json')); print(len(d))"` |
| P6-02 | Seal commit hash (dataset committed before run) | `fc35744` **`[DOC]`** | `docs/phase6_walkthrough.md` ss1.1 | `git show fc35744 --stat` |
| P6-03 | Hit count (n); breakdown by path | **8** total (AUTO_REUSE TP=1 → p6_cs_fol_01; AMBIGUOUS judge-approved TP=7), FP=0 **`[RECOMPUTED]`** | `data/phase6_telemetry.json` outcome_type and tier counts | `python -c "import json; d=json.load(open('data/phase6_telemetry.json')); print(sum(1 for e in d if e['outcome_type']=='TP'))"` |
| P6-04 | Empirical IRR_cache | **0.00%** (k=0 FP / n=8 hits) **`[DOC]`** | `docs/phase6_walkthrough.md` ss3.5 | Same as P6-03 |
| P6-05 | 95% CP CI on IRR_cache | **[0.00%, 36.94%]** (k=0, n=8) **`[DOC][RECOMPUTED]`** | `docs/phase6_walkthrough.md` ss3.5; scipy: beta.ppf(0.975, 1, 8) = 0.3694 | `python -c "from scipy.stats import beta; print(f'{beta.ppf(0.975,1,8)*100:.2f}%')"` |
| P6-06 | Local resolution rate | **89.71%** (no judge call) **`[DOC]`** | `docs/phase6_walkthrough.md` ss3.1 | Same as P6-03 |
| P6-07 | Judge calls invoked (AMBIGUOUS tier entries) | **18** (raw telemetry: 18 AMBIGUOUS-tier entries; of 27 potential AMBIGUOUS pairs) **`[RECOMPUTED]`** | `data/phase6_telemetry.json` tier counts | `python -c "import json; d=json.load(open('data/phase6_telemetry.json')); print(sum(1 for e in d if e['tier']=='AMBIGUOUS'))"` |
| P6-08 | Judge-tier outcome breakdown (18 calls) **DISPUTED** | Raw telemetry: TP=7, TN=9, FN=2, FP=0. Walkthrough ss6: TP=6, TN=10, FN=2, FP=0. Hit count n=8 consistent. **DISPUTED between raw and walkthrough on TP/TN split.** **`[RECOMPUTED]`** | `data/phase6_telemetry.json` vs `docs/phase6_walkthrough.md` ss6 | See tier reconciliation above |
| P6-09 | AUTO_REUSE mean latency | **10.56 ms** (n=1; query p6_cs_fol_01) **`[DOC][RECOMPUTED]`** | `docs/phase6_walkthrough.md` ss3.3; confirmed by `data/phase6_telemetry.json` | Same as P6-03 |
| P6-10 | AMBIGUOUS judge mean latency | **6,577.94 ms** (n=18) **`[DOC]`** | `docs/phase6_walkthrough.md` ss3.3 | Same as P6-03 |
| P6-11 | Latency speedup (AUTO_REUSE vs AMBIGUOUS) | **622.7x** (99.84% reduction; 1 − 10.56/6577.94) **`[DOC][RECOMPUTED]`** | `docs/phase6_walkthrough.md` ss3.3 | — |

---

## 8. Pooled Phase 5 + Phase 6 CI

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| POOL-01 | Pooled k (total FP: Phase 5 corrected N=338 + Phase 6 blind) | **k = 0** **`[DOC]`** | `docs/phase6_walkthrough.md` ss7.2; `docs/phase5_walkthrough.md` ss9.6 corrected | — |
| POOL-02 | Pooled n (Phase 5 corrected n=14 + Phase 6 n=8) | **n = 22** **`[DOC]`** | `docs/phase6_walkthrough.md` ss7.2 | — |
| POOL-03 | Pooled 95% CI upper bound (Clopper-Pearson exact, k=0, n=22) | **15.44%** **`[RECOMPUTED]`** | scipy.stats.beta.ppf(0.975, 1, 22) = 0.15438. Computed 2026-10-01. | `python -c "from scipy.stats import beta; print(f'{beta.ppf(0.975,1,22)*100:.2f}%')"` |
| POOL-05 | CI method | Exact two-sided 95% Clopper-Pearson (Beta distribution inversion). CI_upper = Beta_quantile(0.975, k+1, n-k). CI_lower = 0.00% trivially since k=0. **`[DOC]`** | `docs/phase4_walkthrough.md` ss5.1 formula | — |
| POOL-06 | Does pooled CI clear the 10% safety ceiling? | **NO** — 15.44% > 10%. n >= 36 hits with k=0 required to reach the ceiling. **`[DOC][RECOMPUTED]`** | `docs/phase4_walkthrough.md` ss8; `docs/phase5_walkthrough.md` ss4.5 | — |
| POOL-07 | Sensitivity: all three runs pooled (N=70 pilot + N=338 corrected + Phase 6 blind), k=0, n=44 | **CI_upper = 8.04%** (clears 10% ceiling) **NOT BLIND: pilot was development-stage; not a headline figure.** **`[RECOMPUTED]`** | `data/load_test_telemetry.json` (n=22) + `data/scaled_load_test_telemetry_corrected.json` (n=14) + `data/phase6_telemetry.json` (n=8) | `python -c "from scipy.stats import beta; print(f'{beta.ppf(0.975,1,44)*100:.2f}%')"` |

---

## 9. Latency Reduction Range Across All Three Runs

Reduction % computed as `1 - (AUTO_REUSE_mean_ms / AMBIGUOUS_mean_ms)` from raw walkthrough means. N=338 raw means yield 99.20%.

| # | Run | AUTO_REUSE mean | AMBIGUOUS mean | Speedup | Reduction % | Source |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| LAT-01 | Phase 5 N=70 pilot | 19.18 ms (n=20) | 9,291.35 ms | **484.5x** | **99.79%** | `docs/phase5_walkthrough.md` ss4.3 |
| LAT-02 | Phase 5 N=338 scaled | 61.91 ms (n=1) | 7,715.62 ms | **124.6x** | **99.20%** | `docs/phase5_walkthrough.md` ss9.4 |
| LAT-03 | Phase 6 blind N=175 | 10.56 ms (n=1) | 6,577.94 ms | **622.7x** | **99.84%** | `docs/phase6_walkthrough.md` ss3.3 |
| LAT-04 | Speedup range across all three runs | — | — | **124.6x to 622.7x** | — | Rows LAT-01 to LAT-03 |
| LAT-05 | Latency reduction % range (corrected) | — | — | — | **99.20% to 99.84%** | Recomputed from LAT-01..03 raw means |
| LAT-06 | Caveat: AUTO_REUSE n=1 in scaled runs | N=338 and N=175 each produced exactly 1 AUTO_REUSE hit (confirmed by raw telemetry). N=70 produced n=20 AUTO_REUSE hits and is the most reliable latency estimate. | — | — | — | `data/scaled_load_test_telemetry_corrected.json`; `data/phase6_telemetry.json`; `docs/phase5_walkthrough.md` ss9.3 |

---

## 10. Local Resolution Rate Range

| # | Run | Local Resolution Rate | Source |
| :---: | :--- | :---: | :--- |
| LRR-01 | Phase 5 N=70 pilot | **87.14%** | `docs/phase5_walkthrough.md` ss4.2 |
| LRR-02 | Phase 5 N=338 scaled | **86.39%** | `docs/phase5_walkthrough.md` ss9.3 |
| LRR-03 | Phase 6 blind N=175 | **89.71%** | `docs/phase6_walkthrough.md` ss3.1 |
| LRR-04 | Range across all three runs | **86.39% to 89.71%** | Rows LRR-01 to LRR-03 |

---

## 11. Known Limitations

| # | Limitation | Detail | Source |
| :---: | :--- | :--- | :--- |
| KL-01 | **6 uncorrected near-duplicate collisions in new_dataset_v3.json** | Phase 5 ss9.8 identified 27 pairwise collision candidates. 10 were surfaced as FPs and relabeled. Six additional near-duplicates remained undetected during execution (masked by BYPASS or StabilityClassifier uncertainty override): cs_552/cs_005, fin_548/fin_006, fin_550/fin_003, sys_548/sys_004, sci_544/sci_002, fin_546/fin_005. These were classified TN and not relabeled. The uncorrected records remain in data/raw/new_dataset_v3.json. | `docs/phase5_walkthrough.md` ss9.8 |
| KL-02 | **Underpowered CIs throughout** | Minimum n >= 36 hits with k=0 FP is required to prove IRR_cache <= 10% at 95% Clopper-Pearson confidence. No single blind run achieved this: N=338 corrected has n=14, Phase 6 has n=8, pooled blind-only has n=22. All blind CIs are underpowered. The three-run sensitivity (POOL-07, n=44) clears the ceiling but includes the development-stage N=70 pilot and is not a headline figure. | `docs/phase4_walkthrough.md` ss8; `docs/phase5_walkthrough.md` ss4.5; `docs/phase6_walkthrough.md` ss3.5 |
| KL-03 | **Free-tier judge model** | All production LLM judge calls use nvidia/nemotron-3-super-120b-a12b:free via OpenRouter free tier. Subject to: (a) 50 RPD account limit, (b) provider-level edge caching, (c) possible model version updates at OpenRouter without notice. Paid-tier or self-hosted models were not evaluated. | `docs/phase3_judge_call_walkthrough.md` ss7.3; `docs/phase4_walkthrough.md` ss4.3; `docs/phase5_walkthrough.md` ss9.2 |
| KL-04 | **Dual-role bias** | nvidia/nemotron-3-super-120b-a12b:free was used both as the production ambiguous-tier decision judge (Phase 3 onward) AND as the ground-truth label generator for Phase 4 calibration data. Any systematic inductive bias propagates into both production inference and the training signal, without an independent external referee. | `docs/phase4_walkthrough.md` ss4.2 |
| KL-05 | **No blind held-out set for the cache-reuse pipeline** | The StabilityClassifier has a held-out final test set (n=60). The cache-reuse decision pipeline (TierRouter thresholds, similarity threshold, LLMJudge logic) had never been evaluated on data structurally unavailable to its designers before Phase 6. Phase 6 is the first genuinely blind evaluation but uses structured synthetic benchmarks, not organic query logs. | `docs/phase5_walkthrough.md` ss8; `docs/phase6_walkthrough.md` ss7.3 |
| KL-06 | **Same-set evaluation for Phases 2 to 4** | All of Phase 2, Phase 3, and Phase 4 were evaluated on query_pair_reuse_benchmark.json (N=120), which also informed boundary selection and calibration decisions. Results describe within-sample consistency, not generalization. | `docs/phase3_walkthrough.md` ss2 framing rule; `docs/phase3_judge_call_walkthrough.md` ss1 framing rule |
| KL-07 | **FAISS index is non-persistent, in-memory only** | FlatVectorStore resets between pair evaluations and does not persist across process restarts. Production deployments require index serialization. | `src/cache/vector_store.py` line 107-110 |
| KL-08 | **Synthetic workload distributions, not organic traffic** | All load tests (N=70, N=338, N=175) use hand-crafted or StackExchange-sourced queries with intentional splits. Phase 6 ss7.3 states: "whether these distributions match any specific production deployment is unknown and unverified." | `docs/phase6_walkthrough.md` ss7.3 |
| KL-09 | **Leakage between development set and pair dataset** | 3 query strings overlap between query_stability_benchmark.json (dev/diagnostic, N=160) and query_pair_reuse_benchmark.json. Zero overlap with any pristine evaluation set. Documented and tested but not corrected. | `docs/phase2_walkthrough.md` ss2 Leakage Check |

---

## Appendix A — Recomputed Clopper-Pearson Summary

All values computed 2026-10-01 with `scipy.stats.beta.ppf`.
Method: exact two-sided 95% CI, upper bound = `Beta_quantile(0.975, k+1, n-k)`.

```text
scipy.stats.beta.ppf(0.975, k+1, n-k) with k=0:

  n= 8  -> CI_upper = 36.94%  (Phase 6 blind)
  n=14  -> CI_upper = 23.16%  (Phase 5 N=338 corrected)
  n=22  -> CI_upper = 15.44%  (pooled: Phase 5 corrected n=14 + Phase 6 n=8)
  n=44  -> CI_upper =  8.04%  (three-run sensitivity: all runs pooled; NOT BLIND)

P3-06 Clopper-Pearson CI for k=1, n=12 (OpenRouter same-set):
  lower = beta.ppf(0.025, k,   n-k+1) = beta.ppf(0.025, 1, 12) = 0.21%
  upper = beta.ppf(0.975, k+1, n-k  ) = beta.ppf(0.975, 2, 11) = 38.48%
  CI = [0.21%, 38.48%]
```

> [!NOTE]
> The "pooled n=22" follows Phase 6 ss7.2 definition: Phase 5 N=338 corrected (n=14 hits)
> + Phase 6 blind (n=8 hits) = 22 total hits. This is NOT the same experiment as the
> Phase 5 N=70 pilot (which also happened to produce n=22 hits). Both give CI_upper=15.44%
> because the formula depends only on n and k, but they are distinct experiments.
> The N=70 pilot was a development-stage run (design-informed, not blind) and is NOT
> included in the headline pooled CI (POOL-02/03).

---

*Last updated: 2026-10-01. Author: Antigravity agent. Commit: see git log.*
