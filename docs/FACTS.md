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
| C-17 | Preflight Unit Test Suite Size | 281 tests passing **`[CODE]`** | `tests/` | `pytest tests/` |

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
| P1-07 | Classifier latency percentiles (N=60 final test) | mean **2.29 ms** (2.2875 ms), median **0.25 ms** (0.2517 ms), P95 **8.70 ms** (8.7019 ms) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1.8 | `python scripts/step0_final_test_eval.py` |
| P1-08 | Classifier routing resolution (N=60 final test) | Stage 1 Rule Engine **40** (66.7%), Stage 2 Fallback (`TrainedLexicalClassifier`) **20** (33.3%) **`[DOC]`** | `docs/phase5_walkthrough.md` ss1.7 | `python scripts/step0_final_test_eval.py` |
| P2-01 | Best fixed-threshold IRR_cache and its threshold | **20.83%** (5 FP / 24 hits) at threshold **0.85** — lowest IRR_cache where any hits exist **`[DOC]`** | `docs/phase2_walkthrough.md` ss5 coarse sweep table threshold=0.85 row and ss6 | `python scripts/run_cache_threshold_sweep.py` |
| P2-02 | Safety criterion (pre-stated before sweep) | IRR_cache < 10% **`[DOC]`** | `docs/phase2_walkthrough.md` ss4 | — pre-stated not a measurement |
| P2-03 | Does any fixed threshold satisfy IRR_cache < 10%? | **NO** — at every threshold where hits > 0, IRR_cache >= 20.83% **`[DOC]`** | `docs/phase2_walkthrough.md` ss6 | `python scripts/run_cache_threshold_sweep.py` |
| P2-04 | Reference threshold for comparison (best CRR) | **0.85** (CRR=79.17%, ARR=20.00%, IRR_cache=20.83%) — reference only, not production-safe **`[DOC]`** | `docs/phase2_walkthrough.md` ss6 and ss7 | `python scripts/run_cache_threshold_sweep.py` |
| P2-05 | Mean embed + search latency per pair | **13.9–14.9 ms** **`[DOC]`** | `docs/phase2_walkthrough.md` ss5 header | `python scripts/run_cache_threshold_sweep.py` |
| P2-06 | Dataset N for Phase 2 sweep | 120 pairs **`[DOC]`** | `docs/phase2_walkthrough.md` ss2 | — |
| P2-07 | False Rejection Rate (FRR) at threshold 0.85 | **69.35%** (FN=43/62 missed safe reuses) **`[DOC]`** | `docs/phase2_walkthrough.md` ss5 table; `docs/phase3_fix_walkthrough.md` ss3 table | `python scripts/run_cache_threshold_sweep.py` |

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
| P3-09 | Phase 3 adaptive threshold engine ($\theta_{\text{upper}}$) IRR_cache | **23.08%** (3 FP / 13 hits) — violates <10% safety ceiling (same-set N=120) **`[DOC]`** | `docs/phase3_fix_walkthrough.md` ss3 table | `docs/phase3_fix_walkthrough.md` |
| P3-10 | Phase 3 adaptive threshold engine ($\theta_{\text{upper}}$) ARR & CRR | **10.83%** ARR (13 hits / 120 pairs; TP=10, FP=3), CRR = **76.92%**; same-set N=120 **`[DOC]`** | `docs/phase3_fix_walkthrough.md` ss3 table | `docs/phase3_fix_walkthrough.md` |
| P3-11 | Trivial bypass-all ambiguous baseline IRR_cache | **0.00%** (0 FP / 2 hits; AUTO_REUSE tier only; same-set N=120) **`[DOC]`** | `docs/phase3_fix_walkthrough.md` ss3 table | `docs/phase3_fix_walkthrough.md` |
| P3-12 | Trivial bypass-all ambiguous baseline ARR & CRR | **1.67%** ARR (2 hits / 120 pairs; TP=2, FP=0), CRR = **100.00%**; same-set N=120 **`[DOC]`** | `docs/phase3_fix_walkthrough.md` ss3 table | `docs/phase3_fix_walkthrough.md` |

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
| P4-07 | Phase 4 multi-session labeling progress | Session 0 (22 genuine, 86 fallback stubs); Session 1 (47 genuine, cumulative 69/108, 39 stubs); Session 2 (39 genuine, cumulative 108/108 genuine, 0 stubs) **`[DOC]`** | `docs/phase4_walkthrough.md` ss4.3 | Inspect `data/openrouter_synthetic_feedback_telemetry.json` |

---

## 6. Phase 5 — Load Test Results

### 6.1 — N=70 Pilot Run

Telemetry source: `data/load_test_telemetry.json` (raw, reconciled). AUTO_REUSE TP=20, AMBIGUOUS TP=2, total TP=22, FP=0.

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P5A-01 | Total queries | 70 **`[DOC]`** | `docs/phase5_walkthrough.md` ss3.2 | `python scripts/run_load_test.py` |
| P5A-02 | Cache hits (n); breakdown by path | **22** total (AUTO_REUSE TP=20, AMBIGUOUS judge-approved TP=2), FP=0 **`[RECOMPUTED]`** | `data/load_test_telemetry.json` outcome_type and tier counts | `python -c "import json; d=json.load(open('data/load_test_telemetry.json')); from collections import Counter; print(Counter((e['tier'],e['outcome_type']) for e in d if e['outcome_type']=='TP'))"` |
| P5A-02b | Pilot cache hits breakdown | **20 AUTO_REUSE + 2 judge-approved** (TP=22, FP=0) **`[RECOMPUTED]`** | `data/load_test_telemetry.json` outcome_type and tier counts | `python -c "import json; from collections import Counter; d=json.load(open('data/load_test_telemetry.json')); print(Counter((e['tier'],e['outcome_type']) for e in d if e['outcome_type']=='TP'))"` |
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
| P5B-02b | Scaled run corrected cache hits breakdown | **1 AUTO_REUSE + 13 judge-approved** (TP=14, FP=0) **`[RECOMPUTED]`** | `data/scaled_load_test_telemetry_corrected.json` outcome_type and tier counts | `python -c "import json; from collections import Counter; d=json.load(open('data/scaled_load_test_telemetry_corrected.json')); print(Counter((e['tier'],e['outcome_type']) for e in d if e['outcome_type']=='TP'))"` |
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
AMBIGUOUS tier invoked 18 times. **RESOLVED:** raw telemetry is source of truth with TP=7, TN=9, FN=2, FP=0 (walkthrough ss6 originally stated TP=6, TN=10; corrected via Erratum (2026-10-01) in `docs/phase6_walkthrough.md`). Total hit count n=8 is consistent in both.

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P6-01 | N | **175** entries across 7 domains **`[DOC]`** | `docs/phase6_walkthrough.md` ss1.1 | `python -c "import json; d=json.load(open('data/raw/phase6_blind_eval_dataset.json')); print(len(d))"` |
| P6-02 | Seal commit hash (dataset committed before run) | `fc35744` **`[DOC]`** | `docs/phase6_walkthrough.md` ss1.1 | `git show fc35744 --stat` |
| P6-03 | Hit count (n); breakdown by path | **8** total (AUTO_REUSE TP=1 → p6_cs_fol_01; AMBIGUOUS judge-approved TP=7), FP=0 **`[RECOMPUTED]`** | `data/phase6_telemetry.json` outcome_type and tier counts | `python -c "import json; d=json.load(open('data/phase6_telemetry.json')); print(sum(1 for e in d if e['outcome_type']=='TP'))"` |
| P6-04 | Empirical IRR_cache | **0.00%** (k=0 FP / n=8 hits) **`[DOC]`** | `docs/phase6_walkthrough.md` ss3.5 | Same as P6-03 |
| P6-05 | 95% CP CI on IRR_cache | **[0.00%, 36.94%]** (k=0, n=8) **`[DOC][RECOMPUTED]`** | `docs/phase6_walkthrough.md` ss3.5; scipy: beta.ppf(0.975, 1, 8) = 0.3694 | `python -c "from scipy.stats import beta; print(f'{beta.ppf(0.975,1,8)*100:.2f}%')"` |
| P6-06 | Local resolution rate | **89.71%** (no judge call) **`[DOC]`** | `docs/phase6_walkthrough.md` ss3.1 | Same as P6-03 |
| P6-07 | Judge calls invoked (AMBIGUOUS tier entries) | **18** (raw telemetry: 18 AMBIGUOUS-tier entries; of 27 potential AMBIGUOUS pairs) **`[RECOMPUTED]`** | `data/phase6_telemetry.json` tier counts | `python -c "import json; d=json.load(open('data/phase6_telemetry.json')); print(sum(1 for e in d if e['tier']=='AMBIGUOUS'))"` |
| P6-08 | Judge-tier outcome breakdown (18 calls) | **TP=7, TN=9, FN=2, FP=0** (**RESOLVED:** raw telemetry is source of truth; walkthrough ss6 corrected via Erratum (2026-10-01)) **`[RECOMPUTED]`** | `data/phase6_telemetry.json` and `docs/phase6_walkthrough.md` Erratum | See tier reconciliation above |
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
| LAT-01 | Phase 5 N=70 pilot | 19.18 ms (n=20) | 9,291.35 ms | **484.5x** | **99.79%** | `docs/phase5_walkthrough.md` ss 4.3 |
| LAT-02 | Phase 5 N=338 scaled | 61.91 ms (n=1) | 7,715.62 ms | **124.6x** | **99.20%** | `docs/phase5_walkthrough.md` ss 9.4 |
| LAT-03 | Phase 6 blind N=175 | 10.56 ms (n=1) | 6,577.94 ms | **622.7x** | **99.84%** | `docs/phase6_walkthrough.md` ss 3.3 |
| LAT-04 | Speedup range across all three runs | — | — | **124.6x to 622.7x** | — | Rows LAT-01 to LAT-03 |
| LAT-05 | Latency reduction % range (corrected) | — | — | — | **99.20% to 99.84%** | Recomputed from LAT-01..03 raw means |
| LAT-06 | Caveat: AUTO_REUSE n=1 in scaled runs | N=338 and N=175 each produced exactly 1 AUTO_REUSE hit (confirmed by raw telemetry). N=70 produced n=20 AUTO_REUSE hits and is the most reliable latency estimate. | — | — | — | `data/scaled_load_test_telemetry_corrected.json`; `data/phase6_telemetry.json`; `docs/phase5_walkthrough.md` ss 9.3 |

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
| KL-02 | **Underpowered CIs throughout** | Minimum n >= 36 hits with k=0 FP is required to prove IRR_cache <= 10% at 95% Clopper-Pearson confidence. No single blind run achieved this: N=338 corrected has n=14, Phase 6 has n=8, pooled design-informed + blind has n=22. All blind CIs are underpowered. The three-run sensitivity (POOL-07, n=44) clears the ceiling but includes the development-stage N=70 pilot and is not a headline figure. | `docs/phase4_walkthrough.md` ss8; `docs/phase5_walkthrough.md` ss4.5; `docs/phase6_walkthrough.md` ss3.5 |
| KL-03 | **Free-tier judge model** | All production LLM judge calls use nvidia/nemotron-3-super-120b-a12b:free via OpenRouter free tier. Subject to: (a) 50 RPD account limit, (b) provider-level edge caching, (c) possible model version updates at OpenRouter without notice. Paid-tier or self-hosted models were not evaluated. | `docs/phase3_judge_call_walkthrough.md` ss7.3; `docs/phase4_walkthrough.md` ss4.3; `docs/phase5_walkthrough.md` ss9.2 |
| KL-04 | **Dual-role bias** | nvidia/nemotron-3-super-120b-a12b:free was used both as the production ambiguous-tier decision judge (Phase 3 onward) AND as the ground-truth label generator for Phase 4 calibration data. Any systematic inductive bias propagates into both production inference and the training signal, without an independent external referee. | `docs/phase4_walkthrough.md` ss4.2 |
| KL-05 | **No blind held-out set for the cache-reuse pipeline** | The StabilityClassifier has a held-out final test set (n=60). The cache-reuse decision pipeline (TierRouter thresholds, similarity threshold, LLMJudge logic) had never been evaluated on data structurally unavailable to its designers before Phase 6. Phase 6 is the first genuinely blind evaluation but uses structured synthetic benchmarks, not organic query logs. | `docs/phase5_walkthrough.md` ss8; `docs/phase6_walkthrough.md` ss7.3 |
| KL-06 | **Same-set evaluation for Phases 2 to 4** | All of Phase 2, Phase 3, and Phase 4 were evaluated on query_pair_reuse_benchmark.json (N=120), which also informed boundary selection and calibration decisions. Results describe within-sample consistency, not generalization. | `docs/phase3_walkthrough.md` ss2 framing rule; `docs/phase3_judge_call_walkthrough.md` ss1 framing rule |
| KL-07 | **FAISS index is non-persistent, in-memory only** | FlatVectorStore resets between pair evaluations and does not persist across process restarts. Production deployments require index serialization. | `src/cache/vector_store.py` line 107-110 |
| KL-08 | **Synthetic workload distributions, not organic traffic** | All load tests (N=70, N=338, N=175) use hand-crafted or StackExchange-sourced queries with intentional splits. Phase 6 ss7.3 states: "whether these distributions match any specific production deployment is unknown and unverified." | `docs/phase6_walkthrough.md` ss7.3 |
| KL-09 | **Leakage between development set and pair dataset** | 3 query strings overlap between query_stability_benchmark.json (dev/diagnostic, N=160) and query_pair_reuse_benchmark.json. Zero overlap with any pristine evaluation set. Documented and tested but not corrected. | `docs/phase2_walkthrough.md` ss2 Leakage Check |
| KL-10 | **Classifier final test set "seen" once by pytest assertion before formal evaluation** | The classifier final test set's accuracy was "seen" once by a pytest assertion (`assert metrics.accuracy >= 0.85`) in `tests/test_stability_evaluation.py` (line 74) before the formal evaluation in `scripts/step0_final_test_eval.py`. The dataset was clean for hyperparameter tuning (no model weights or thresholds were fitted to it), but it was not strictly blind. | `tests/test_stability_evaluation.py` line 74; `docs/phase5_walkthrough.md` ss1 |
| KL-11 | **Adaptive threshold mechanism never fired** | All 7 categories stayed on the 0.85 fallback throughout calibration and evaluation. The per-category threshold fitting mechanism never lowered or altered an operating threshold in production (cite P4-01). | `docs/phase4_walkthrough.md` ss7.3, ss11.1; `docs/FACTS.md` P4-01 |

---

## 12. Dataset Roles

| # | Dataset Role | Dataset File | N | Composition / Breakdown | Evaluation Nature | Source / Citations |
| :---: | :--- | :--- | :---: | :--- | :--- | :--- |
| DROLE-01 | **Classifier Train / Dev** | `data/raw/query_stability_benchmark.json` + `data/raw/query_stability_human_credibility.json` | **245** | 160 dev queries + 85 human credibility queries | Same-set development and diagnostic tuning | D-02, D-05; `docs/phase2_walkthrough.md` ss2 table |
| DROLE-02 | **Held-Out Challenge** | `data/raw/query_stability_benchmark_heldout.json` | **60** | 31 STABLE, 24 DYNAMIC, 5 CONDITIONALLY_STABLE | Held-out challenge (unseen during prompt/rule tuning) | D-03, P1-03, P1-04; `docs/phase5_walkthrough.md` ss1B.2 |
| DROLE-03 | **Final Test** | `data/raw/query_stability_benchmark_final_test.json` | **60** | 37 STABLE, 23 DYNAMIC | Clean for tuning; seen once by pytest (KL-10) | D-04, P1-01, P1-02, KL-10; `docs/phase5_walkthrough.md` ss1.1 |
| DROLE-04 | **Cache Pipeline Calibration** | `data/raw/query_pair_reuse_benchmark.json` | **120 pairs** | 62 SAFE, 58 UNSAFE | Same-set baseline sweep & Phase 3 evaluation | D-01, P2-01..P2-06, P3-01..P3-08, KL-06; `docs/phase2_walkthrough.md` ss2 table |
| DROLE-05 | **Cache Pipeline Calibration** | `data/raw/synthetic_query_pair_feedback.json` | **108 pairs** | 108 targeted synthetic pairs across 7 domains | Same-set calibration for adaptive threshold sweep | D-06, P4-02; `docs/phase4_walkthrough.md` ss3.1 |
| DROLE-06 | **Design-Informed Pilot** | `data/raw/load_test_query_stream.json` | **70 queries** | Hand-crafted pilot stream | Same-set / design-informed pilot run | D-07, P5A-01..P5A-08; `docs/phase5_walkthrough.md` ss2.1 |
| DROLE-07 | **Design-Informed Pilot** | `data/raw/new_dataset_v3.json` | **338 queries** | 210 StackExchange cold seeds + 128 hand-authored | Scaled pilot (design-informed, contains collisions) | D-08, P5B-01..P5B-10, KL-01; `docs/phase5_walkthrough.md` ss9.1 |
| DROLE-08 | **Blind Evaluation** | `data/raw/phase6_blind_eval_dataset.json` | **175 entries** | 175 entries across 7 domains (sealed commit `fc35744`) | Strictly blind evaluation (unseen by pipeline/prompts) | D-09, P6-01..P6-11; `docs/phase6_walkthrough.md` ss1.1 |

---

## 13. Interactive Demonstration Telemetry & Runtime Environment

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| ENV-01 | Fresh-clone test runtime Python version | Python 3.14.5 verified during clean virtualenv installation and hermetic test execution | `README.md`; `pyproject.toml` | `python --version` |
| DEMO-T01 | Query 3 (LT-CS-003) demonstration similarity | 0.957 (cosine similarity to cached seed LT-CS-001) **`[CODE]`** | `demo/run_demo.py` line 26; `data/load_test_telemetry.json` | `python demo/run_demo.py` |
| DEMO-T02 | Query 3 (LT-CS-003) stability classifier confidence | 1.00 (STABLE) **`[CODE]`** | `demo/run_demo.py` line 26; `data/load_test_telemetry.json` | `python demo/run_demo.py` |
| DEMO-T03 | Query 3 (LT-CS-003) demo run latency | about 50 ms in the demo run (59.95 ms measured run; 61.80 ms AUTO_REUSE demo mean; within 10.56-61.91 ms FACTS range) **`[CODE]`** | `demo/run_demo.py` summary; rows LAT-01..LAT-03 | `python demo/run_demo.py` |
| DEMO-T04 | Query 6 (LT-CS-010) demonstration similarity | 0.688 (cosine similarity to cached seed LT-CS-004) **`[CODE]`** | `demo/run_demo.py` line 38; `data/load_test_telemetry.json` | `python demo/run_demo.py` |
| DEMO-T05 | Walkthrough script target duration & cadence | 2 minutes (0:00 to 0:30, 0:30 to 1:00, 1:00 to 1:30, 1:30 to 2:00) | `demo/SCRIPT.md` lines 4, 11, 17, 25, 33 | Read `demo/SCRIPT.md` |

---

## 14. Hazard Reduction Analysis (Same-Set N=120 pairs)

Computed by `scripts/compute_hazard_reduction.py` comparing Phase 2 best fixed threshold (0.85; 5/24 hits) against Phase 3 OpenRouter LLM judge (1/12 hits) on the exact same 120 query pairs from `data/raw/query_pair_reuse_benchmark.json`.

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| HR-01 | Phase 2 threshold 0.85 95% Clopper-Pearson CI | **[7.13%, 42.15%]** (5/24 hits = 20.83%) **`[RECOMPUTED]`** | `scripts/compute_hazard_reduction.py`; `docs/phase2_walkthrough.md` ss5 | `python scripts/compute_hazard_reduction.py` |
| HR-02 | Absolute difference in hazard rate | **12.50 percentage points** (20.83% - 8.33%) **`[RECOMPUTED]`** | `scripts/compute_hazard_reduction.py` | `python scripts/compute_hazard_reduction.py` |
| HR-03 | Relative hazard reduction | **60.00%** ((20.83% - 8.33%) / 20.83%) **`[RECOMPUTED]`** | `scripts/compute_hazard_reduction.py` | `python scripts/compute_hazard_reduction.py` |
| HR-04 | Statistical significance status | **same-set, CIs overlap** ([7.13%, 42.15%] vs [0.21%, 38.48%]); difference is not statistically established at 95% confidence **`[RECOMPUTED]`** | `scripts/compute_hazard_reduction.py` | `python scripts/compute_hazard_reduction.py` |

---

## 15. Digit-Only 7-Character Commit Hashes

In accordance with strict token extraction rules, four 7-character hexadecimal commit hashes consist entirely of decimal digits (`0-9`), colliding with the token check's digit regex. Rather than altering or artificially lengthening these hashes, they are ledgered here as verified commit identifiers:

| # | Commit SHA | Phase / Context | Commit Title / Purpose |
| :---: | :---: | :--- | :--- |
| SHA-01 | `4681031` | Phase 7 | docs: reconcile README metrics with FACTS ledger |
| SHA-02 | `5914533` | Phase 7 | docs: add comprehensive system architecture DESIGN.md |
| SHA-03 | `7194993` | Phase 7 | docs: record Phase 7c claim check documentation |
| SHA-04 | `9591101` | Phase 5 | test: evaluate Step 0 pristine final test set |


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

## 16. Phase 8 — Local Equivalence Layer & Real-Data Evaluation

### 16.1 — System Architecture, Models, and Hardware Constants

| # | Fact | Claim / Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| P8-C01 | Normalizer module | `src/cache/normalizer.py` (lowercase, punctuation, interchangeable question frames) **`[CODE]`** | `src/cache/normalizer.py` | `python tests/test_normalizer.py` |
| P8-C02 | Primary local pairwise CrossEncoder model | `cross-encoder/quora-distilroberta-base` **`[CODE]`** | `src/decision/local_equivalence.py` line 33 | `grep DEFAULT_CROSS_ENCODER_MODEL src/decision/local_equivalence.py` |
| P8-C03 | Primary local model revision | `f62e7a4b20b97195c2868e53ec59126df5eac743` **`[CODE]`** | `src/decision/local_equivalence.py` line 34 | `grep DEFAULT_CROSS_ENCODER_REVISION src/decision/local_equivalence.py` |
| P8-C04 | Primary local model training data | Quora Question Pairs (QQP) training split; zero-shot transfer evaluated on validation split & PAWS | Hugging Face model card (`cross-encoder/quora-distilroberta-base`) | Inspect model card |
| P8-C05 | Alternative local pairwise CrossEncoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` **`[CODE]`** | `src/decision/local_equivalence.py` line 35 | `grep MS_MARCO_CROSS_ENCODER_MODEL src/decision/local_equivalence.py` |
| P8-C06 | Alternative local model revision | `233902d25c440f23af6f7d6e94d2946bac0bee0a` **`[CODE]`** | `src/decision/local_equivalence.py` line 36 | `grep MS_MARCO_CROSS_ENCODER_REVISION src/decision/local_equivalence.py` |
| P8-C07 | Calibrated local equivalence threshold $\theta_{\text{local}}$ | **0.8500** (calibrated on validation split only) **`[CODE]`** | `scripts/tune_local_threshold.py` | `python scripts/tune_local_threshold.py` |
| P8-C08 | Deterministic rule guards | Strict number equality, negation polarity match, entity token match **`[CODE]`** | `src/decision/local_equivalence.py` | `python tests/test_local_equivalence.py` |
| P8-C09 | Remote LLM judge fallback status | Flag `fallback_to_llm=False` (disabled by default; 100% local resolution) **`[CODE]`** | `src/decision/local_equivalence.py` line 184 | `grep fallback_to_llm src/decision/local_equivalence.py` |
| P8-C10 | Evaluation hardware | Intel(R) Core(TM) Ultra 7 155U (12 cores, 14 logical processors), 32 GB RAM, Windows 11 **`[DOC]`** | Local system benchmark | Inspect system properties |

### 16.2 — Public Datasets and Splits

| # | Dataset / Split | Source / License | Size / Composition | SHA-256 Checksum |
| :---: | :--- | :--- | :--- | :--- |
| P8-D01 | PAWS validation split | Google Research / CC-BY-SA 4.0 (`google-research-datasets/paws`, `labeled_final`) | 8,000 rows | `7760d829453764ba342a6f562809a8ed21c2c3eec3fd9ffa544089f145d42f6d` |
| P8-D02 | QQP validation split | GLUE / NYU MLL / Quora Terms (`nyu-mll/glue`, `qqp`) | 40,430 rows | `efd86a539c412d74874ee451573d7bd142f56c47fe36de033b9f367d8bb0fa71` |
| P8-D03 | MS MARCO validation split | Microsoft Research License / CC-BY 4.0 (`microsoft/ms_marco`, `v1.1`) | 10,047 rows | `67aca2e1b2fffe418a6bd317142887c90973c9c98dd8d940eaae3f8521daed7c` |
| P8-D04 | Tuning validation split | `data/phase8_tuning_val.json` | 160 pairs (60 QQP pos, 40 PAWS neg, 30 QQP neg, 30 MS MARCO neg) | `3782fa16b1dc9d8469adfe2bb8e1a1e127394aa52ca1714bfa4911d35a51da03` |
| P8-D05 | Sealed evaluation test stream | `data/raw/phase8_sealed_test_stream.json` | **230 queries** (80 safe reuse, 150 miss/unsafe); committed before run at commit `325d4c6` | `fbcf1437aa91797942363f8faa58d6b6eef51d9df7f1d776fce556221dd900af` |

### 16.3 — Baseline Real LLM Generation Measurements

Measured on sample $n = 10$ queries from sealed stream using `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter (temp=0.0, max_tokens=128):

| # | Metric | Measured Value | Source file + location |
| :---: | :--- | :--- | :--- |
| P8-B01 | Baseline LLM mean latency | **1505.56 ms** (1505.6 ms) | `data/baseline_llm_measurements.json` |
| P8-B02 | Baseline LLM median (p50) latency | **1235.5 ms** | `data/baseline_llm_measurements.json` |
| P8-B03 | Baseline LLM p95 latency | **2239.32 ms** (2239.3 ms) | `data/baseline_llm_measurements.json` |
| P8-B04 | Baseline LLM mean prompt tokens | **36.3 tokens** | `data/baseline_llm_measurements.json` |
| P8-B05 | Baseline LLM mean completion tokens | **139.8 tokens** | `data/baseline_llm_measurements.json` |
| P8-B06 | Baseline LLM mean total tokens | **176.1 tokens** | `data/baseline_llm_measurements.json` |

### 16.4 — Four-Arm Simulation Results (N=230 Sealed Real-Query Stream)

Executed once on sealed test stream (`fbcf1437aa91797942363f8faa58d6b6eef51d9df7f1d776fce556221dd900af`):

| # | Arm | Hits ($n$) | Hit Rate (95% CI) | IRR (95% CI) | LLM Calls | p50 Latency | p95 Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| P8-R01 | (a) No Cache | 0 | **0.0%** [0.0%, 1.6%] | N/A (0 hits) | 230 | 1235.5 ms | 2239.3 ms |
| P8-R02 | (b) Fixed Cosine (0.85) | 90 | **39.1%** [32.8%, 45.8%] | **34.4%** [24.7%, 45.2%] (FP=31) | 140 | 48.4 ms | 74.2 ms |
| P8-R03 | (c) Current Pipeline | 16 | **7.0%** [4.0%, 11.1%] | **18.8%** [4.0%, 45.6%] (FP=3) | 21 (remote judge) | 50.6 ms | 6928.7 ms |
| P8-R04 | (d) New Pipeline (Phase 8) | **50** | **21.7%** [16.6%, 27.6%] | **6.0%** [1.3%, 16.5%] (FP=3) | **0** (100% local) | **37.8 ms** | **62.2 ms** |

### 16.5 — New Pipeline Breakdown, Path Latencies, and Workload Avoidance

| # | Metric / Sub-path | Measured Value | Notes / Citations |
| :---: | :--- | :--- | :--- |
| P8-P01 | Arm (d) Total Hits | **50 hits** (TP=47, FP=3) | `data/phase8_simulation_telemetry.json` |
| P8-P02 | Arm (d) Canonical Exact Hits | **41 hits** (82.0% of all cache hits) | Stage 0 normalizer folded exact/canonical frames |
| P8-P03 | Arm (d) Vector Store AUTO_REUSE Hits | **3 hits** | Stage 1-3 embedding similarity $\ge 0.92$, conf $\ge 0.90$ |
| P8-P04 | Arm (d) Local Equivalence Model Approved | **6 hits** | Stage 3 borderline band approved by CrossEncoder ($\ge 0.85$) |
| P8-P05 | CANONICAL_EXACT latency (n=41) | mean **0.02 ms**, p50 **0.01 ms**, p95 **0.02 ms** | Instant in-memory lookup without embedding |
| P8-P06 | AUTO_REUSE latency (n=3) | mean **36.72 ms**, p50 **36.03 ms**, p95 **41.37 ms** | Embedding + FAISS vector search |
| P8-P07 | LOCAL_EQUIVALENCE latency (n=14) | mean **390.07 ms**, p50 **68.00 ms**, p95 **1665.89 ms** | CrossEncoder CPU pairwise inference |
| P8-P08 | BYPASS latency (n=172) | mean **40.38 ms**, p50 **39.15 ms**, p95 **58.03 ms** | Missed / rejected query path |
| P8-P09 | LLM calls avoided per 1,000 queries | **217.4 calls** | (50 hits / 230 queries) * 1,000 |
| P8-P10 | Tokens avoided per 1,000 queries | **38,282.6 tokens** | 217.4 calls * 176.1 tokens/generation |
| P8-P11 | Estimated cost avoided per 1,000 queries | **$0.011** (at $0.30/1M tokens, OpenAI published price Oct 2024, estimate) | Optional cited estimate: 38,282.6 * $0.30 / 1,000,000 |

### 16.6 — Component Ablation Table

| # | Ablation Stage | Hits | Hit Rate | FP | IRR | p50 Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| P8-A01 | 1. Normalizer only | 41 | 17.8% | 0 | 0.0% | 0.01 ms |
| P8-A02 | 2. + Embedding (Cosine 0.85) | 90 | 39.1% | 31 | 34.4% | 34.02 ms |
| P8-A03 | 3. + Local Equivalence Model (Arm d) | 50 | 21.7% | 3 | 6.0% | 37.82 ms |
| P8-A04 | 4. + Remote Judge (Current pipeline) | 16 | 7.0% | 3 | 18.8% | 50.65 ms |

---

## 17. Phase A — Ground-Truth Facts, Baseline Re-measurements & Errata

### 17.1 — Real LLM Generation Baseline (n=100 Re-measured on Local Hardware)

Hardware: Intel(R) Core(TM) Ultra 7 155U (12 cores, 14 logical processors), 32 GB RAM, Windows 11.
Model: Qwen/Qwen2.5-0.5B-Instruct (local PyTorch CPU inference, temp=0.0, max_new_tokens=64, do_sample=False). Date: 2026-10-12.

| # | Fact | Measured Value | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-B01 | Baseline LLM local model | **Qwen/Qwen2.5-0.5B-Instruct** (CPU, n=100) | `data/baseline_llm_measurements_n100.json` | `python scripts/measure_real_llm_baseline.py` |
| PA-B02 | Baseline LLM local mean latency | **8997.49 ms** | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['primary_baseline_n100']['mean_latency_ms'])"` |
| PA-B03 | Baseline LLM local median (p50) latency | **8153.50 ms** (8153.5 ms) | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['primary_baseline_n100']['median_latency_ms'])"` |
| PA-B04 | Baseline LLM local p95 latency | **12763.96 ms** | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['primary_baseline_n100']['p95_latency_ms'])"` |
| PA-B05 | Baseline LLM local mean prompt tokens | **11.05 tokens** | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['primary_baseline_n100']['mean_prompt_tokens'])"` |
| PA-B06 | Baseline LLM local mean completion tokens | **61.87 tokens** | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['primary_baseline_n100']['mean_completion_tokens'])"` |
| PA-B07 | Baseline LLM local mean total tokens | **72.92 tokens** | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['primary_baseline_n100']['mean_total_tokens'])"` |
| PA-B08 | Auxiliary remote baseline (n=20 live OpenRouter) | mean **2002.11 ms**, p50 **1346.64 ms**, p95 **3759.78 ms**, mean tokens **164.25** | `data/baseline_llm_measurements_n100.json` | `python -c "import json; print(json.load(open('data/baseline_llm_measurements_n100.json'))['auxiliary_remote_baseline']['mean_latency_ms'])"` |

### 17.2 — Phase 8 Repairs, Category Breakdown, and Errata

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-E01 | Phase 8 Ceiling Erratum | Point estimate **6.0%** IRR carried 95% Clopper-Pearson CI **[1.3%, 16.5%]**. Because the upper bound (16.5%) exceeds 10%, the data did not formally establish the 10% safety ceiling. Retracted unanchored $0.011/1k estimate. Avoided workload strictly: **38,282.6 tokens** and **217.4 calls** per 1k. | `docs/DECISIONS.md` D-20261012-01; `docs/STATUS.md` | `python scripts/check_numeric_tokens.py` |
| PA-FP01 | Phase 8 3 False Positives Source | All 3 false positives (`P8-ADV-PROBE-001`, `P8-ADV-PROBE-015`, `P8-ADV-PROBE-025`) originated in category **ADVERSARIAL_TRAP** (PAWS validation) via vector store **AUTO_REUSE** (cosine sim >= 0.92, conf >= 0.90), completely bypassing CrossEncoder and rule guards. | `data/phase8_simulation_telemetry.json` | `python scripts/analyze_phase8_stream.py` |
| PA-VAL01 | Validation Set vs Test Stream Divergence | Validation set (N=160, IRR=**39.62%** [26.5%, 54.0%]) had 40 PAWS word scrambles (25% of all pairs, 40% of true negatives) with identical token multisets that defeat bag-of-words bi-encoders. In test stream, 41/50 hits (82%) were exact/canonical folds with 0% error; non-exact test IRR was 3/9 = **33.33%**, aligning with validation error rate. | `data/phase8_tuning_val.json` | `python scripts/analyze_validation_guards.py` |
| PA-G01 | Generative-Request Guard Benchmark | Evaluated on hand-built labeled benchmark (`data/generative_prompts_benchmark.json`, N=100: 50 generative, 50 factual). Achieved **100.0% accuracy**, **100.0% precision**, **100.0% recall**, **0 false alarms**. | `data/generative_prompts_benchmark.json` | `python scripts/check_generative_guard.py` |

### 17.3 — Candidate Semantic Matchers Comparison (Validation Set N=160)

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-M01 | Candidate a (Off-the-shelf DistilRoBERTa, no guards) | Hits: **79**, Hit Rate: **49.38%** [41.39%, 57.38%], TP: **45**, FP: **34** (all 34 PAWS), Precision: **56.96%** [45.33%, 68.06%], Recall: **75.0%** (75.00%) [62.14%, 85.28%], IRR: **43.04%** [31.94%, 54.67%] | `data/phaseA_candidate_comparison.json` | `python -c "import json; print(json.load(open('data/phaseA_candidate_comparison.json'))['validation_benchmark']['candidate_a_off_the_shelf_no_guards'])"` |
| PA-M02 | Candidate b (Fine-tuned on QQP+PAWS, no guards) | Hits: **59**, Hit Rate: **36.88%** [29.39%, 44.85%], TP: **56**, FP: **3** (0 PAWS, 3 QQP), Precision: **94.92%** [85.85%, 98.94%], Recall: **93.33%** [83.80%, 98.15%], IRR: **5.08%** [1.06%, 14.15%] | `data/phaseA_candidate_comparison.json` | `python -c "import json; print(json.load(open('data/phaseA_candidate_comparison.json'))['validation_benchmark']['candidate_b_finetuned_no_guards'])"` |
| PA-M03 | Candidate c (Fine-tuned + Rule & Generative Guards) | Hits: **40**, Hit Rate: **25.0%** (25.00%) [18.50%, 32.45%], TP: **39**, FP: **1** (0 PAWS, 1 QQP), FN: **21**, TN: **99**, Precision: **97.5%** (97.50%) [86.84%, 99.94%], Recall: **65.0%** (65.00%) [51.60%, 76.87%], IRR: **2.5%** (2.50%) [0.06%, 13.16%] | `data/phaseA_candidate_comparison.json` | `python -c "import json; print(json.load(open('data/phaseA_candidate_comparison.json'))['validation_benchmark']['candidate_c_finetuned_with_guards'])"` |
| PA-M04 | Candidate b Fine-Tuning Execution | Training pairs: **2000**, Epochs: **1**, Batch size: **16**, Learning rate: **2e-05**, Duration: **449.57 s**, Hardware: Intel Core Ultra 7 155U CPU | `data/phaseA_candidate_comparison.json` | `python -c "import json; print(json.load(open('data/phaseA_candidate_comparison.json'))['training'])"` |

### 17.4 — Domain-Level Calibration on CQADupStack (12 Subforums)

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-C01 | CQADupStack Retrieval Dataset & License | **mteb/cqadupstack-retrieval**, License: **CC BY-SA 3.0 / 4.0**. 12 subforums analyzed, 50 pairs/subforum (25 positive, 25 negative). | `data/phaseA_cqadupstack_calibration.json` | `python -c "import json; print(json.load(open('data/phaseA_cqadupstack_calibration.json'))['license'])"` |
| PA-C02 | Subforum Logistic Gate Clearance | All **12** subforums passed sample size gate ($N=50 \ge 20$) and minority class gate ($25 \ge 10$). Subforums using global fallback (0.85): **0**. | `data/phaseA_cqadupstack_calibration.json` | `python -c "import json; print(json.load(open('data/phaseA_cqadupstack_calibration.json'))['subforums_fitted'])"` |
| PA-C03 | Domain Operating Thresholds Range | Spanned from **0.1087** (physics, point estimate 0.0076) to **0.3854** (programmers, point estimate 0.3236). | `data/phaseA_cqadupstack_calibration.json` | `python -c "import json; print(json.load(open('data/phaseA_cqadupstack_calibration.json'))['domain_thresholds']['physics'])"` |

### 17.5 — Real Chat Traffic Natural Repeat Rates on WildChat (N=5,000)

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-W01 | WildChat Dataset & License | **allenai/WildChat**, License: **Open Data Commons Attribution License (ODC-By)**. Sample: **5000** first user turns. | `data/phaseA_wildchat_repeat_rates.json` | `python -c "import json; print(json.load(open('data/phaseA_wildchat_repeat_rates.json'))['license'])"` |
| PA-W02 | WildChat Exact String Repeats | **490 / 5000** queries (**9.80%**, 9.8%, 0.098). | `data/phaseA_wildchat_repeat_rates.json` | `python -c "import json; print(json.load(open('data/phaseA_wildchat_repeat_rates.json'))['exact_string_repeats'])"` |
| PA-W03 | WildChat Canonical Folded Repeats | **529 / 5000** queries (**10.58%**, 10.6%, 0.1058). Incremental canonical lift: 39 queries (+0.78%). | `data/phaseA_wildchat_repeat_rates.json` | `python -c "import json; print(json.load(open('data/phaseA_wildchat_repeat_rates.json'))['canonical_repeats'])"` |
| PA-W04 | WildChat Guarded Semantic Matches (sim >= 0.85) | **986 / 5000** queries (**19.72%**, 19.7%, 0.1972) guarded vs **1398** (**27.96%**, 28.0%, 0.2796) raw ungoverned. | `data/phaseA_wildchat_repeat_rates.json` | `python -c "import json; print(json.load(open('data/phaseA_wildchat_repeat_rates.json'))['semantic_guarded_counts']['0.85'])"` |
| PA-W05 | WildChat Guarded Semantic Matches (sim >= 0.92) | **777 / 5000** queries (**15.54%**, 15.5%, 0.1554) guarded vs **1063** (**21.26%**, 21.3%, 0.2126) raw ungoverned. | `data/phaseA_wildchat_repeat_rates.json` | `python -c "import json; print(json.load(open('data/phaseA_wildchat_repeat_rates.json'))['semantic_guarded_counts']['0.92'])"` |
| PA-W06 | Human Audit Sample Export | **100** sampled queries exported to `data/review/pa_human_audit.csv` with empty human_label column. | `data/phaseA_wildchat_repeat_rates.json` | `python -c "import json; print(json.load(open('data/phaseA_wildchat_repeat_rates.json'))['audit_sample_size'])"` |

### 17.6 — Sealed Test Stream Single-Run Evaluation (N=2,200)

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-S01 | Sealed Stream Cryptographic Hashes | Stream SHA-256: `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991`. Frozen config SHA-256: `a1e0dc2a0a2e89e8c29a357597264f2d47fbf556f3f531b9a0b5c760f2fd8466`. Git commit: `d1f9d97`. | `docs/phaseA_preregistration.md` | `python scripts/run_phaseA_sealed_evaluation.py` |
| PA-S02 | Sealed Stream Query Composition | Total: **2200** queries. Sources: **1000** QQP held-out, **600** PAWS test, **400** CQADupStack held-out, **200** WildChat. Roles: **536** safe non-exact, **464** adversarial negatives, **1200** cold seeds. | `data/phaseA_frozen_config.json` | `python -c "import json; print(json.load(open('data/phaseA_frozen_config.json'))['sealed_stream'])"` |
| PA-S03 | Sealed Execution Duration | Single execution: **66.64 s** (1.11 min) on local CPU. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['timestamp'])"` |
| PA-S04 | Total Cache Hits and Confusion Matrix | Total Hits: **86** (Hit Rate: **3.91%** [3.14%, 4.81%], 0.0391). TP: **60**, FP: **26**, FN: **476**, TN: **438**. Overall IRR: **30.23%** [20.79%, 41.08%] (0.3023). | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['secondary_endpoints'])"` |
| PA-S05 | Primary Endpoint: Non-Exact Matches | Non-exact Hits: **75** / 86 (TP: **54**, FP: **21**). Point estimate non-exact IRR: **28.0%** (28.00%, 0.28) with 95% Clopper-Pearson CI **[18.24%, 39.56%]**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['primary_endpoint_non_exact_irr'])"` |
| PA-S06 | Gate A Verification Outcome | **GATE A FAILED**. Fails Criteria: (1) Non-exact hits = 75 (< 100 target); (2) Non-exact IRR point estimate = 28.00% (> 10.0% target). Upper CI bound = 39.56%. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['gate_a'])"` |
| PA-S07 | Exact Canonical Matches on Sealed Stream | Canonical hits: **11** / 30 exact routing queries. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['secondary_endpoints']['exact_canonical_hits'])"` |

### 17.7 — Path Latencies, Workload Avoided, and Break-Even Curve

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-L01 | CANONICAL_EXACT Path Latency (n=30) | Mean: **0.06 ms**, p50: **0.04 ms**, p95: **0.16 ms**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['path_resolution']['latencies_ms']['CANONICAL_EXACT'])"` |
| PA-L02 | AUTO_REUSE Path Latency (n=43) | Mean: **14.28 ms**, p50: **14.36 ms**, p95: **17.75 ms**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['path_resolution']['latencies_ms']['AUTO_REUSE'])"` |
| PA-L03 | LOCAL_EQUIVALENCE Path Latency (n=181) | Mean: **33.72 ms**, p50: **34.54 ms**, p95: **66.03 ms**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['path_resolution']['latencies_ms']['LOCAL_EQUIVALENCE'])"` |
| PA-L04 | BYPASS Path Latency (n=1946) | Mean: **16.65 ms**, p50: **15.58 ms**, p95: **26.07 ms**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['path_resolution']['latencies_ms']['BYPASS'])"` |
| PA-L05 | Decision Overhead vs Baseline LLM | Decision p50: **15.81 ms**, p95: **37.11 ms**. End-to-end simulated latency: p50 **8169.03 ms** (vs baseline 8153.50 ms), p95 **8184.04 ms** (vs baseline 12763.96 ms). | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['latency_comparison_vs_baseline'])"` |
| PA-L06 | Workload Avoided per 1,000 Queries | Avoided calls: **39.1 calls** / 1k. Avoided tokens: **2850.5 tokens** / 1k (based on baseline mean 72.92 tokens). | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['secondary_endpoints']['calls_avoided_per_1000'])"` |
| PA-L07 | Break-Even Hit Rate | **0.1976%** (0.00198). Point where decision path overhead (15.81 ms) equals avoided LLM latency (8153.50 ms). | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['secondary_endpoints']['break_even_hit_rate'])"` |

### 17.8 — Source Breakdown on Sealed Stream

| # | Fact | Value / Finding | Source file + location | How to re-run |
| :---: | :--- | :--- | :--- | :--- |
| PA-SB01 | PAWS Test Stream Breakdown | Hits: **43**, TP: **22**, FP: **21**. Non-exact hits: **34**, Non-exact FP: **16** (Non-exact IRR: **47.06%** [29.84%, 64.89%]). | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['source_breakdown']['paws_test'])"` |
| PA-SB02 | QQP Held-Out Stream Breakdown | Hits: **42**, TP: **37**, FP: **5**. Non-exact hits: **40**, Non-exact FP: **5** (Non-exact IRR: **12.5%** (12.50%) [4.19%, 26.80%]). | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['source_breakdown']['qqp_heldout'])"` |
| PA-SB03 | CQADupStack Held-Out Stream Breakdown | Hits: **1**, TP: **1**, FP: **0**. Non-exact hits: **1**, Non-exact FP: **0**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['source_breakdown']['cqadupstack_heldout'])"` |
| PA-SB04 | WildChat Traffic Breakdown | Hits: **0**, TP: **0**, FP: **0**, Non-exact hits: **0**, Non-exact FP: **0**. | `data/phaseA_sealed_evaluation_summary.json` | `python -c "import json; print(json.load(open('data/phaseA_sealed_evaluation_summary.json'))['source_breakdown']['wildchat_traffic'])"` |

### 17.9 — Projected Hit-Rate vs Repeat-Share Curve

| # | Fact | Repeat Share | Projected Hit Rate | Projected End-to-End p50 |
| :---: | :--- | :---: | :---: | :---: |
| PA-RC01 | Zero Repeat Baseline | **0.0%** (0.0) | **0.0%** (0.0) | **8169.31 ms** |
| PA-RC02 | 10% Repeat Traffic | **10.0%** (10.0, 0.1) | **0.86%** (0.0086) | **8099.19 ms** |
| PA-RC03 | 20% Repeat Traffic | **20.0%** (20.0, 0.2) | **1.72%** (0.0172) | **8029.07 ms** |
| PA-RC04 | 40% Repeat Traffic | **40.0%** (40.0, 0.4) | **3.44%** (0.0344) | **7888.82 ms** |
| PA-RC05 | 60% Repeat Traffic | **60.0%** (60.0, 0.6) | **5.16%** (0.0516) | **7748.58 ms** |

---

*Last updated: 2026-10-12. Author: Antigravity agent. Commit: see git log.*


---

*Historical ledger update: 2026-10-11. Author: Antigravity agent. Recovery additions follow below.*

---

## Recovery Runtime — Application Invariants

These rows describe code contracts and hermetic regression tests. They are not
held-out semantic-accuracy results, real-provider measurements, or workload
savings. Historical experiment rows above are preserved.

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| PA-RUNTIME-CONFIG | Exact-response runtime defaults | TTL **3600.0 seconds**, capacity **1024 entries**, explicit temperature cap **0.2**, **64** bounded coordination lock stripes; **code configuration**, not calibrated operating values | `src/policy/response_cache.py`, `ScopedResponseCache.__init__` | Inspect the constructor defaults and stripe construction |
| PA-RUNTIME-TEST | Exact-response runtime regression suite | **50 passed**; **hermetic test fixtures**; no real provider/model execution and no semantic-safety inference | `tests/test_response_cache.py` | `python -m pytest tests/test_response_cache.py -q` |
| PA-RUNTIME-SUITE | Default regression suite after recovery additions and mocked-runner isolation repair | **329 passed**, **22 deselected**; model/API integration tests excluded by the existing configuration; executed with offline model flags, not a new workload evaluation | `tests/`; `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |


## Source-level Phase A Reconciliation (2026-10-15)

The following are offline artifact-audit findings, not a new sealed inference run
or actual-answer safety measurements. Historical results remain unchanged.

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| PA-AUDIT-COUNTS | Saved artifacts reconcile | 2,200 events; 1,200 seeds; 1,000 probes; 60 TP, 26 FP, 476 FN, 438 TN by designated-pair labels | `docs/audit/PHASE_A_ARTIFACT_RECONCILIATION.json` | `python scripts/audit_phase_a_artifacts.py` |
| PA-AUDIT-CANDIDATES | Accepted candidate partitions | 67 designated-seed matches, 10 identical-text other candidates, 9 unlabeled nonidentical other candidates among 86 accepted probe hits | Same | Same |
| PA-AUDIT-CANONICAL | Canonical-path reconciliation | 30 path events: 19 seeds and 11 probe hits; probe label counts 6 TP and 5 FP. All 5 FP-labeled records retrieve identical text from another candidate; these are label-attribution contradictions, not demonstrated wrong answers | Same | Same |
| PA-AUDIT-OVERLAP | Split overlap under strip/casefold normalization | 0 identical train pairs and 0 identical validation pairs; 13 stream events share training text and 1 shares validation text. Does not certify thread/group isolation | Same | Same |
| PA-AUDIT-SUITE | Integrated default offline regression suite | 354 passed, 22 deselected; CPU torch 2.5.1, sentence-transformers 3.4.1, transformers 4.48.3; model/API integration excluded; no sealed inference rerun | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |

## Gateway Foundation — Functional Evidence (2026-10-15)

These rows describe the exact-response gateway foundation. Gate A remains failed
and Gate B is not complete. Functional fixture work must not be presented as
real-model latency savings or semantic accuracy.

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| PB-GATEWAY-CONFIG | Gateway defaults | Entry TTL 3600 seconds; 1024 entries; 67108864 serialized UTF-8 response bytes; 32 processing slots; maximum requested output 4096 tokens; 65536 request bytes; 10-second body-read timeout; upstream socket timeout 60 seconds; upstream response bound 4194304 bytes. Default cache policy bypass. Configuration, not calibrated values | `src/api/gateway.py`, `src/api/providers.py`, `src/api/service.py`, `src/policy/response_cache.py` | Inspect constructors and environment factory |
| PB-GATEWAY-TEST | Focused gateway/runtime regressions | 104 passed; includes real loopback HTTP adapter fixtures, authentication, isolation, error accounting, byte limits, concurrent coordination and no implicit retries | `tests/test_gateway.py`, `tests/test_http_provider.py`, `tests/test_response_cache.py` | `python -m pytest tests/test_gateway.py tests/test_http_provider.py tests/test_response_cache.py -q` |
| PB-GATEWAY-SUITE | Full default offline suite | 408 passed, 22 deselected; live model/API integration excluded. Tested with FastAPI 0.143.1, Starlette 1.8.0 and uvicorn 0.54.0 | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |
| PB-GATEWAY-SMOKE | Running HTTP gateway functional smoke | **simulated_functional_smoke**: 56 authenticated completion requests; 50 client threads released through a barrier (server peak concurrency not measured); 3 MISS, 52 HIT, 1 BYPASS; 4 fixture upstream attempts; 0 external model calls. Tenant isolation/invalidation and authentication passed | `docs/audit/GATEWAY_CONCURRENT_SMOKE.json` | `python scripts/smoke_gateway.py` |

## SQLite Persistence — Functional Evidence (2026-10-15)

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| PB-SQLITE-FOCUSED | Persistence and gateway regression checks | 118 passed; separate-interpreter reopen, TTL, tenant isolation, shared invalidation epoch, LRU/UTF-8 budgets, namespace guard, storage errors and provider call accounting. Fixture responses only | `tests/test_sqlite_response_cache.py`, gateway/provider/runtime tests | `python -m pytest tests/test_sqlite_response_cache.py tests/test_gateway.py tests/test_http_provider.py tests/test_response_cache.py -q` |
| PB-SQLITE-SUITE | Full default offline suite after persistence | 422 passed, 22 deselected; no live model/API integration or sealed inference rerun | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |
| PB-SQLITE-SMOKE | Existing loopback HTTP smoke remains passing | 56 authenticated fixture completions, 50 barrier-released clients, 4 fixture provider calls, 0 external model calls; default in-memory mode. Persistence separately checked in regression tests | `scripts/smoke_gateway.py` | `python scripts/smoke_gateway.py` |

## Answer-Bound Evaluation and Operations (2026-10-16 Asia/Calcutta)

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| RECOVERY-ANSWER-SUITE | Full default offline suite | 448 passed, 22 deselected; no live model/API or sealed inference execution | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |
| RECOVERY-ANSWER-FOCUSED | Answer evaluator, gateway, provider and storage regressions | 144 passed; includes stale/wrong-candidate labels, missing-label denominators, source admission lineage, split leakage, provider errors, work accounting and dashboard auth boundary | Relevant files under `tests/` | `python -m pytest tests/test_answer_benchmark.py tests/test_gateway.py tests/test_http_provider.py tests/test_response_cache.py tests/test_sqlite_response_cache.py -q` |
| RECOVERY-ANSWER-FIXTURE | Small functional workload | 24 requests; 4 MISS, 18 BYPASS, 2 HIT; 22 cache-arm fixture attempts versus 24 fresh-arm fixture attempts; 0 human-reviewed hits and 0 external model calls | `docs/audit/ANSWER_BENCHMARK_FUNCTIONAL_EVIDENCE.json` | Fixture command in `docs/answer_evaluation.md` |
| RECOVERY-ANSWER-DEV | Exposed training-pair development exercise | 4000 constructed requests from 2000 existing QQP/PAWS pairs; 3777 BYPASS, 219 MISS, 4 HIT; 3996 cache-arm fixture attempts versus 4000 fresh-arm fixture attempts. All 4 hits unreviewed; 0 accepted non-exact hits; 0 external model calls. Not organic traffic or a quality/savings benchmark | Same evidence JSON; `data/candidate_b_train_pairs.json` | Development conversion/run commands in `docs/answer_evaluation.md` |
| RECOVERY-PACKAGING | Installed wheel check | Built and installed wheel outside source checkout; authenticated fixture MISS/HIT and dashboard delivery passed with ML imports blocked. Docker unavailable locally; browser executable download failed; no container/browser-render validation claimed | `pyproject.toml`, `docs/deployment.md`, container CI job | Build/install wheel and run gateway; Docker/browser checks require suitable environment |
| RECOVERY-CONTAINER-CI | Remote container functional smoke | Passed on commit `2bd019453b59044d2e3731b4f3d45adfaf7f3710`; build, read-only startup with SQLite volume, fixture MISS/HIT, health and dashboard delivery. Other CI jobs were still running when recorded | [Run 38799899946, job 116645894452](https://github.com/paridhik11/adaptive-agentic-semantic-cache/actions/runs/38799899946/job/116645894452) | Gateway container fixture smoke CI job |

## Semantic, Streaming and Agent Integration — Functional Evidence

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| RECOVERY-INTEGRATION-SUITE | Default offline suite | 496 passed, 23 deselected. Includes no paid/live generation or new sealed inference | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |
| RECOVERY-INTEGRATION-FOCUSED | Focused regression suite | 192 passed, 1 model integration test deselected; semantic scope/verification accounting, streaming parser/transport/cancellation, agent workflow and review agreement plus prior runtime tests | Relevant files under `tests/` | Run semantic, answer, review, agent, streaming, gateway, provider and cache test modules |
| RECOVERY-SEMANTIC-FIXTURE | Same development workload, experimental policy comparison | 4000 requests per arm. Exact: 4 hits. Lexical similarity at threshold 0.85 and margin 0.05: 50 hits, including 46 non-exact. Always-abstaining fixture verifier: 4 exact hits, 0 non-exact. All hits unreviewed; incorrect-reuse rates null. No dense-model or live-answer quality claim | `docs/audit/SEMANTIC_POLICY_FUNCTIONAL_EVIDENCE.json` | Commands in `docs/semantic_streaming_agents.md` |
| RECOVERY-STREAM-SMOKE | Running gateway SSE and existing concurrent smoke | 57 authenticated completions: 3 MISS, 52 HIT, 2 BYPASS; 5 fixture provider attempts, 0 external calls; SSE reaches valid terminal marker. 50-client barrier burst retained | `docs/audit/STREAMING_FUNCTIONAL_SMOKE.json` | `python scripts/smoke_gateway.py` |
| RECOVERY-STREAM-BOUNDS | Stream configuration | 65536 bytes per SSE event; configured upstream body limit defaults to 4194304 bytes; 180-second stream read deadline checked between chunks, with possible socket-timeout overshoot. Streaming never reads or admits cache entries | `src/api/streaming.py`, `src/api/providers.py` | Inspect stream parser and transport |
| RECOVERY-AGENT-FIXTURE | Scoped workflow fixture | 3 runs: initial tenant 2 provider calls; same tenant repeat 1; different tenant 2. Reviewer always bypasses caching; scripted fixture approval, not model-quality evidence | `docs/audit/AGENT_WORKFLOW_FUNCTIONAL_SMOKE.json` | `python scripts/demo_agent_workflow.py` |
| RECOVERY-STREAM-CONTAINER | Updated container CI smoke | Passed on commit `ed32ba14069ec4ee3b5cd428d1ba3f186985fcbb`; image build, exact fixture MISS/HIT, terminal SSE completion, health and dashboard delivery. Other CI jobs still running when recorded | [Run 38945826011, job 117124563501](https://github.com/paridhik11/adaptive-agentic-semantic-cache/actions/runs/38945826011/job/117124563501) | Gateway container fixture smoke CI job |

## Zero-Budget Pilot Configuration

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| FREE-PILOT-CAPS | Conservative local caps | At most 10 selected development requests; default and maximum 30 attempted calls per run, 40 attempts per local UTC day and 10 per rolling minute across runs sharing the ledger. Exact/similarity worst case 2 calls per request; verified worst case 3. Attempts persist before network I/O | `src/benchmark/free_pilot.py`, `scripts/run_free_pilot.py` | Inspect constructors and run the plan-only command in `docs/free_pilot.md` |
| FREE-PILOT-PRICE | Request-side zero-price restriction | Explicit `:free` IDs only, fallback disabled, max prompt/completion/request price 0. No tools, web search or plugins; no live provider/billing verification during implementation | Same | `tests/test_free_pilot.py` |
| FREE-PILOT-CATALOG | Metadata-only model check | Existing project model `nvidia/nemotron-3-super-120b-a12b:free` listed with prompt and completion prices 0 in the public model catalog during implementation. No authenticated inference or account-quota check | `https://openrouter.ai/api/v1/models` | Fetch public catalog and inspect this exact model ID and pricing |
| FREE-PILOT-TESTS | Quota/resume regression evidence | 19 focused tests passed: free-only routing, shared transactional quotas, failed/uncertain attempts, UTC rollover, resume bindings, provider/ledger pauses, atomic export and latency suppression | `tests/test_free_pilot.py` | `python -m pytest tests/test_free_pilot.py -q` |
| FREE-PILOT-SUITE | Full offline suite after free pilot implementation | 515 passed, 23 deselected; no live OpenRouter model calls | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |

## First User-Run OpenRouter Connection Pilot

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| FREE-LIVE-CONNECTION-V1 | User-supplied live development observation | 1 authored question, 2 recorded new live attempts, 40 reported prompt tokens and 256 reported completion tokens; both attempts used the 128-token output cap. 0 admissible complete answers; no cache admission. Failed-attempt times approximately 3256.88 ms and 2451.53 ms, not successful-answer latency | `docs/audit/OPENROUTER_CONNECTION_PILOT_V1.json`; uploaded trace/report content binding verified | Inspect uploaded trace and report; raw finish reasons require read-only journal diagnostics |
| FREE-LIVE-DIAGNOSTIC-LIMIT | New explicit output-limit override | Optional 1–4096 tokens; changes the selected request hash and requires a new run ID for changed limits. No inference performed while implementing the override | `scripts/run_free_pilot.py` | Plan-only command with `--max-output-tokens`; local fixture regression |
| FREE-LIVE-DIAGNOSTIC-TESTS | Rejection diagnostics regressions | 68 focused tests passed, 1 model integration test excluded; full offline suite 522 passed, 23 deselected. Read-only journal inspection hash check passes and no model calls are made by inspection | `tests/test_response_diagnostics.py`, benchmark/pilot tests | `python -m pytest tests/test_response_diagnostics.py tests/test_free_pilot.py tests/test_answer_benchmark.py tests/test_experimental_semantic.py -q`; full default suite |

## Guided Zero-Budget Session and Second Live Connection Check

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| FREE-LIVE-CONNECTION-V2 | User-supplied live development observation | 1 authored question; 2 new live attempts with 1024-token caps, both finish reason length. Partial answer text present, 0 admissible complete answers, no cache admission. Across the two connection checks: 4 attempts and 0 complete answers. Not quality, successful-answer latency or savings evidence | `docs/audit/OPENROUTER_CONNECTION_PILOT_V2.json`; uploaded trace/report binding verified | Compare canonical trace digest with report trace hash |
| FREE-GUIDED-PROFILE | Foreground session configuration | 6 authored requests across 2 topics; exact policy, 12-attempt cap, 512 output tokens, requested reasoning disabled and at most 2 concise sentences. Live effectiveness unverified. HTML refresh every 2 seconds; local minute-quota waiting bounded by 180 seconds | `scripts/run_zero_budget_session.py`, `src/benchmark/progress.py` | Plan command in `docs/free_pilot.md` |
| FREE-GUIDED-FOCUSED | Offline guided-run regressions | 8 tests passed: operator-owned reasoning toggle, first-incomplete stop with durable response, escaped previews, full bundle and crash recovery, plan without key, observer isolation and minute-window resume. Constructed complete-response run has 2 exact hits and 10 fixture attempts versus 12 fresh attempts; 0 external model calls | `tests/test_guided_pilot.py` | `python -m pytest tests/test_guided_pilot.py -q` |
| FREE-GUIDED-SUITE | Default offline regression run | 529 passed, 23 deselected in 125.61 seconds. The additional minute-window resume case was added afterward and passed in the 8-test focused run above; it was not part of this full-suite invocation. No external model calls | `tests/`, `pyproject.toml` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` |

## Recovery Roadmap Proposals and Workload Diagnosis (2026-10-17 Asia/Calcutta)

These are derived diagnostics or proposed engineering goals, not achieved live results.

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| RECOVERY-PLAN-BYPASS | Derived development eligibility bottleneck | 3777 / 4000 bypassed = 94.425%; 223 / 4000 eligible = 5.575%. Lexical similarity 50 / 4000 hits = 1.25%, all unreviewed and fixture-backed. This is not a real-traffic opportunity estimate | `docs/audit/SEMANTIC_POLICY_FUNCTIONAL_EVIDENCE.json` | Read similarity run; divide BYPASS and HIT counts by requests |
| RECOVERY-PLAN-TARGETS | Proposed recovery success criteria, not measured outcomes | Reuse precision point estimate at least 99%, two-sided 95% exact CI upper bound on incorrect reuse at most 2%; paired answer-quality difference CI lower bound at least -2 percentage points. Utility goals on a disclosed workload: at least 20% fewer total provider attempts, 20% fewer provider tokens and 15% lower mean end-to-end latency; p50/p95 reported separately. Test these only after development feasibility. No universal hit-rate promise | `docs/EXECUTION_ROADMAP.md`, proposed by engineering plan | Inspect declared criteria; freeze analysis and sample size before new untouched evaluation |
| RECOVERY-PLAN-SAMPLE | Independent-binomial illustration, not sample-size assurance | With 0 errors in 183 independently reviewed hits, two-sided 95% Clopper-Pearson upper bound is below 2%. Dependence, topic clustering, nonzero errors and other endpoints require separate planning | Exact bound 1 - 0.025 ** (1/n) | `python -c "import math; n=math.ceil(math.log(.025)/math.log(.98)); print(n,1-.025**(1/n))"` |

## R1 Reuse-Loss Diagnosis — Offline Phase Evidence

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| R1-POLICY | Reproduced development eligibility reasons | Each 4000-request fixture-backed policy run: 3738 not_rule_confirmed_static, 39 personal_or_generative_request, 223 eligible. These are heuristic decisions, not safe/unsafe labels | `docs/audit/R1_REUSE_DIAGNOSIS.json` | `python scripts/diagnose_reuse.py --pairs data/candidate_b_train_pairs.json --license-note 'Existing exposed development QQP/PAWS pairs; source terms recorded in docs/FACTS.md, no new licensing claim.' --output-dir runs/r1-recompute` |
| R1-STAGES | Reproduced lexical decision stages | Similarity-only: 3777 eligibility bypasses, 1 no-candidate, 172 threshold rejections, 46 semantic hits, 4 exact hits. Abstaining fixture verification: 42 verifier rejections, 4 margin rejections, 172 threshold rejections, 1 no-candidate, 4 exact hits and 3777 bypasses. Different admissions change later candidate pools; these are not identical verifier opportunities | Same | Same |
| R1-ACCOUNTING | Terminal and work accounting | Exact: 3996 fixture generation attempts versus 4000 fresh; similarity: 3950 versus 4000; abstaining fixture verifier: 3996 versus 4000 with 0 verifier API calls. Each run's 4000 terminal outcomes reconcile. No correctness or live savings claim | Same | Same |
| R1-CONTRACT | Authored policy boundary audit | 11 authored examples match expected current policy behavior; unrecognized stable wording and fixed system context are excluded. These are contract examples, not safety calibration data | `docs/audit/R1_POLICY_SCOPE.json` | `python scripts/audit_policy_scope.py` |
| RECOVERY-DELIVERY | User deadline and delivery target | User asked for a ready project in 5–7 days from October 17, 2026; target demo/report October 22, maximum window October 24. Scope and statistical limitations must be disclosed | User instruction; `docs/EXECUTION_ROADMAP.md` | Inspect schedule; not a measured result |
| R1-REGRESSIONS | Phase-closeout verification | 538 passed, 23 deselected in 121.06 seconds; focused diagnosis module 8 passed. Numeric-token ledger check and working-tree secret scan passed. No external model inference | `tests/`, phase closeout | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q`; `python -m pytest tests/test_reuse_diagnosis.py -q`; `python scripts/check_numeric_tokens.py` |
| FREE-GUIDED-LIVE-EXACT | User-run live exact-cache demonstration | 6 authored development requests across 2 topics: 2 exact-repeat HIT, 4 MISS, 0 non-exact reuse. 10 OpenRouter attempts (4 cache arm, 6 fresh arm), 2 attempts avoided, 79 prompt and 125 completion tokens avoided. Cache/fresh mean attempt latency 1414.84/1639.94 ms; paired mean difference -225.10 ms, group-bootstrap 95% interval [-562.96, 112.76] ms across 6 comparisons. 0 of 2 hits reviewed; safety not certified. Constructed demonstration only, not natural workload or resume-performance evidence | `docs/audit/OPENROUTER_GUIDED_EXACT_V1.json`; original user-provided result package hash is recorded there; raw answers omitted | Re-run `python scripts/audit_pilot_package.py --input <local pilot-result-package-v1 JSON> --output <new audit JSON>`; source bundle must be retained by its owner |
| R1-PILOT-AUDIT-REGRESSIONS | Final R1 phase checks after live-pilot ingestion | 545 passed, 23 deselected; focused pilot/diagnosis/guided/evaluation checks 48 passed; strict numeric-token check passed; working-tree and full-history secret scans clean. Model/API integration remained excluded from offline test suite | `tests/`, `docs/audit/R1_CLOSEOUT.md` | `PYTHONPATH=<review-venv-site-packages>:<runtime-deps> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q`; `python scripts/check_numeric_tokens.py`; Gitleaks v8.18.4 detect commands recorded in phase closeout |

## R2 Workload Fit and Source Preparation (2026-10-17)

These records distinguish a versioned QA reference set and authored stress
cases from natural traffic. No R2 semantic-answer or deployment-savings result
has been measured.

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| R2-CPYTHON-SOURCE | CPython Programming FAQ source | CPython tag `v3.12.0`, peeled revision `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`; Programming FAQ SHA-256 `648bc2d476230c98f61571e9de969811150875996de2552d5c7656368b2df184`; 78,511 bytes. Python documentation license: PSF-2.0; code examples are dual licensed with 0BSD. Model pretraining contamination unknown | `examples/benchmark/python_faq.sources.json`, `data/licenses/CPYTHON_PSF_LICENSE.txt` | `python scripts/prepare_python_faq.py --source-dir runs/r2-python-source-recompute --output-dir runs/r2-python-recompute --download` |
| R2-CPYTHON-GROUP-SPLIT | Section-grouped reference corpus | 67 FAQ questions: train 24, dev 22, reserved test 21; entire source sections stay within one split. The reserved section split is visible and is not a sealed test | `docs/audit/R2_PREPARATION_SUMMARY.json` | Same preparation command; inspect `question_counts_by_split` and `group_split_check` |
| R2-CPYTHON-CONSTRUCTED | Constructed exact-repeat workload | 134 requests: one source-question seed and one authored exact repeat per FAQ; repeat share is constructed, not a natural prevalence estimate. `answer_reuse_gold` is null; no live answers were collected | `docs/audit/R2_PREPARATION_SUMMARY.json` | Same preparation command; inspect `request_count_by_split`, `nature` and `answer_reuse_gold` |
| R2-ADVERSARIAL-AUTHORED | Handwritten matcher stress suite | 12 request pairs across changed entity, negation, quantity, unit, direction, code input, version, temporal scope, requested depth, context and generative categories. These expected non-reuse cases are authored test intentions, not independent safety labels | `examples/benchmark/python_faq_adversarial.json` | Read case categories and expected reuse fields; no model calls |
| R2-OASST-SOURCE | OpenAssistant conversation source | `OpenAssistant/oasst1` pinned to `fdf72ae0827c1cda404aff25b6603abec9e3399b`, license Apache-2.0. Train Parquet SHA-256 `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e`, 39,516,251 bytes; 84,437 message records in the shard. This is crowdsourced site data, not production API traffic | `examples/benchmark/oasst1.sources.json`, `data/licenses/OPENASSISTANT_APACHE2.txt` | `pip install '.[data]'`; `python scripts/prepare_oasst_natural.py --source-dir runs/r2-oasst-source-recompute --output-dir runs/r2-oasst-recompute --limit 9845 --download` |
| R2-OASST-REPETITION | Chronological first-turn repeat audit | All 9,845 eligible root prompts in the pinned shard, Jan 16–Apr 10, 2023. 66 raw exact repeats (0.6704%); 83 NFKC/casefold/whitespace repeats (0.8431%); earlier assistant reply records preceded 49 exact and 57 normalized repeat timestamps; 0 repeats within the source user ID grouping. No answer text was reviewed, so reply availability is not a safe hit. Descriptive only: not tenant-compatible labels and no general production prevalence claim | `docs/audit/R2_OASST_NATURAL_REPETITION.json` | Same OASST preparation command; source size, sample selection, normalization, reply timing and aggregate counts are recorded in the audit summary |
| R2-WILDCHAT-BLOCKED | Natural-traffic prevalence attempt | Two public viewer requests targeted 5,000 first-turn rows; observed progress reached about 1,000 and 2,000 rows before HTTP 429, and neither produced a complete export. No repeat rate is reported. The current dataset README says ODC-BY retroactively applies; the retained LICENSE.md still contains AI2 ImpACT text, so no new rows are admitted pending reconciliation | `docs/audit/R2_COLLECTION_ATTEMPTS.json`; official card and license links recorded there | Inspect attempt and license-source decision; no dataset request is retried |
| R2-REGRESSIONS | R2 offline corpus tests | 11 focused tests passed for FAQ parsing/group leakage, natural-root filtering, reply chronology, privacy-safe aggregation and authored stress cases; no model/API calls | `tests/test_faq_corpus.py`, `tests/test_natural_repetition.py` | `PYTHONPATH=<review-venv-site-packages>:<runtime-deps> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_faq_corpus.py tests/test_natural_repetition.py -q` |
| R2-PHASE-VALIDATION | Final R2 phase verification | 556 passed, 23 deselected; full default offline suite; numeric-token check passed; working-tree and full-history Gitleaks scans clean; no OpenRouter calls | `docs/audit/R2_CLOSEOUT.md`, `tests/`, audit artifacts | `PYTHONPATH=<review-venv-site-packages>:<runtime-deps> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q`; `python scripts/check_numeric_tokens.py`; Gitleaks v8.18.4 detect commands recorded in closeout |
| DELIVERY-20261018-REVISION | Accepted build-order revision and estimates, not measured results | On October 18, 2026 the user accepted four steps mapped to R3-R6: complete cache path (target October 19-20), bounded agent (October 21-22), paired benchmark/report (October 22-23), refine/publish (October 24). Integrated demo target October 22; original October 24 user deadline retained. October 25-28 is a contingency forecast, not a new accepted deadline. Estimates assume active sessions; independent review/free quota can extend evidence collection. No positive savings or safety-bound result is guaranteed | User instruction; `docs/EXECUTION_ROADMAP.md`; `docs/DECISIONS.md` D-20261018-DELIVERY | Inspect scope, delivery table, quality/utility gates and phase commit policy; no model calls for this planning revision |

## R3 Cache Serving Lifecycle (2026-10-18)

These are functional engineering observations. Approvals and responses in the
recorded smoke are authored/simulated; no new semantic-quality or savings claim
has been established.

| ID | Fact | Value and evidence | Source | Recompute |
| --- | --- | --- | --- | --- |
| R3-RUNTIME | Exact-first gateway extension and scope contract | Off/exact/shadow/guarded modes; no similarity-only serving. Actual-answer approval binds full candidate/incoming request, full response, tenant and knowledge/revision/access/policy scope. Exact SQLite persistence remains; semantic candidates are process-local and start cold on restart. Default exact; quality unvalidated | `src/policy/semantic_runtime.py`, `docs/semantic_cache_runtime.md`, `docs/audit/R3_CLOSEOUT.md` | `python scripts/smoke_semantic_gateway.py --output runs/r3-recompute.json` |
| R3-CONFIG | Development configuration and bounds, not calibrated thresholds | Retrieval threshold 0.85, runner-up margin 0.05; semantic index 256 entries, 8388608 estimated payload bytes and 4096 maximum vector dimensions; pinned-context question bound 8000 characters. The fixture smoke uses threshold 0.5. Payload estimates are not process RAM, approval/model memory is separate. Dense snapshots require immutable revisions and do not auto-download | Runtime/factory defaults and fixture script | Inspect constructor/factory configuration; no accuracy is inferred from a threshold |
| R3-HTTP-SMOKE | Loopback lifecycle smoke with scripted complete responses and authored approvals | 17 completion requests; 13 fixture provider attempts; 3 BYPASS, 10 MISS, 4 HIT (3 exact, 1 semantic). Off/exact/shadow/guarded paths, SQLite exact restart, invalidation, authentication, tenant and source-revision isolation passed. No paired workload savings estimate, 0 external model calls | `docs/audit/R3_SEMANTIC_SERVING_SMOKE.json` | Same R3 smoke command; compare aggregate JSON |
| R3-REGRESSIONS | Final offline phase verification | 593 passed, 23 deselected in 113.55 seconds; 37 new runtime tests passed focused. Full suite offline flags enabled; real-model/API integrations excluded. No new dense inference or external model calls | `docs/audit/R3_VALIDATION.json`, `tests/test_semantic_runtime.py`, `docs/audit/R3_CLOSEOUT.md` | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q`; focused runtime command in closeout |
| R3-DEPLOYMENT-SMOKES | Existing HTTP/SSE path and clean gateway-only install | Existing exact gateway smoke: 57 completion requests, 5 fixture upstream attempts, 3 MISS / 52 HIT / 2 BYPASS; streaming and 50-client barrier burst passed. Fresh environment installed only gateway requirements and reproduced the full R3 smoke. No new Docker execution on this host; no external model calls | `docs/audit/R3_EXACT_REGRESSION_SMOKE.json`, `docs/audit/R3_VALIDATION.json` | `python scripts/smoke_gateway.py`; fresh venv install `deploy/requirements-gateway.txt` then run R3 smoke |


## R4 functional agent evidence (2026-10-18)

R4-FUNCTIONAL: `audit/R4_FUNCTIONAL_SMOKE.json` is reproduced with
`python scripts/smoke_document_agent.py`. The first authored-fixture task makes
three fixture provider calls and two actual local document-tool calls. Approval
survives process restart and completed resume adds no model/tool work. External
model calls: zero. Independent reviews: zero. No task-quality, memory-benefit,
semantic-precision or live-savings measurement is established by this smoke.
Test execution and source binding: `audit/R4_VALIDATION.json`.
