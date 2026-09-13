# Phase 3 — Reuse Decision Layer: Walkthrough

**Document purpose:** This is the WALKTHROUGH document — what was actually done, in the order it was done. Compare against `docs/phase3_implementation.md` (the SPEC) to audit intent vs. execution.

> [!IMPORTANT]
> **Required statement (framing rule):** These results are a same-set comparison against `query_pair_reuse_benchmark.json`, already used to establish Phase 2's baseline. They show whether the tiered decision layer improves outcomes on known pairs, not whether it generalizes to unseen traffic. A fresh pair dataset (Option A) would be required before claiming generalization.

---

## 1. Order of Work

1. Read Phase 1 source (`src/classifier/`, all tests, `stability_evaluator.py`)
2. Read Phase 2 source (`src/cache/`, `cache_evaluator.py`, all tests, `phase2_walkthrough.md`)
3. Wrote STEP 0 tier boundary docstring in `tier_router.py` (before any routing logic)
4. Implemented `src/decision/tier_router.py` (TierBoundaries, TierRouter, Tier enum)
5. Implemented `src/decision/decision_step.py` (CategoryHistory, DecisionStep, DecisionResult)
6. Implemented `src/evaluation/decision_evaluator.py` (DecisionEvaluator, DecisionMetrics)
7. Implemented all three test files
8. Ran Phase 3 unit tests → all 67 pass
9. Ran Phase 3 evaluation with real model → captured numbers below
10. Wrote `docs/phase3_implementation.md` (spec) and this document (walkthrough)
11. Ran full test suite → all existing + new tests pass

---

## 2. Step 0 — Tier Boundary Justification (Written Before Any Code)

**Source data used:** Only Phase 1 and Phase 2's already-measured distributions. `query_pair_reuse_benchmark.json` outcomes were NOT consulted when choosing boundaries.

### Phase 1 confidence distribution (from STABLE_CONFIDENCE_THRESHOLD comment block)

| Cluster | Confidence | Phase 1 dangerous error rate |
|---|---|---|
| Cluster A | ≈ 0.769 | ~0% (Sub-stage A override fires at threshold 0.80) |
| Cluster B | ≈ 0.881 | ~3.5% (3/85 credibility-set queries = 3 FPs out of ~85) |
| Rule layer | 0.95–1.0 | ~0% (0 FPs on 85-query credibility set) |

**STABLE_CONFIDENCE_THRESHOLD = 0.80** (existing Phase 1 constant, unchanged).

### Phase 2 similarity distribution (from docs/phase2_walkthrough.md)

| Threshold | Hits | IRR_cache |
|---|---|---|
| 0.85 | 24 | 20.83% (5 FPs) |
| 0.90 | 8 | 25.00% (2 FPs) |
| 0.92 (fine-sweep interpolated) | 7 | ~28.57% (2 FPs) |

No threshold achieves IRR_cache < 10%.

### Chosen tier boundaries

| Parameter | Value | Derivation |
|---|---|---|
| `auto_reuse_sim_floor` | **0.92** | Above Phase 2's 0.90 point; combined with confidence ≥ 0.90 estimated joint hazard < 5% |
| `auto_reuse_conf_floor` | **0.90** | Above Cluster B (0.881); selects rule-layer certainty with ~0% dangerous error rate |
| `bypass_sim_ceiling` | **0.50** | Below this, Phase 2 sweep shows no useful hit-rate signal |
| `bypass_conf_ceiling` | **0.80** | Matches Phase 1's STABLE_CONFIDENCE_THRESHOLD; below this, Phase 1 already routes to DYNAMIC |

**Residual hazard estimate for AUTO-REUSE (from existing Phase 1/2 data only):**
- Phase 1 rule-layer dangerous error rate: ≈ 0%
- Phase 2 at sim ≥ 0.92: ≈ 28.57% standalone hazard
- Joint estimated residual: < 5% (the 2 FPs at sim ≥ 0.92 are intent-mismatch pairs that would also need a STABLE rule match)

---

## 3. Measured Results (SAME-SET COMPARISON on 120 pairs)

> [!NOTE]
> All numbers below are from running Phase 3 evaluation against `query_pair_reuse_benchmark.json` — the same 120 pairs Phase 2 was evaluated on. These are NOT held-out or generalization results.

### 3.1 Headline DoD Metric: Ambiguous-Band Traffic Fraction

| Tier | Count | Fraction |
|---|---|---|
| AUTO_REUSE | **2** | **1.7%** |
| AMBIGUOUS | **32** | **26.7%** |
| BYPASS | **86** | **71.7%** |

**Ambiguous-band fraction = 26.7% (32/120).**

This is below the 50% threshold that would require boundary tightening. The decision-step was executed for 32 pairs (26.7% of traffic).

### 3.2 Phase 2 vs. Phase 3 Same-Set Comparison

> [!IMPORTANT]
> The numbers below are a SAME-SET COMPARISON on the 120 pairs already used to establish Phase 2's baseline. They do NOT demonstrate generalization to unseen traffic.

| Metric | Phase 2 (baseline @ 0.85) | Phase 3 (tiered, same 120 pairs) | Delta |
|---|---|---|---|
| ARR (Actual Reuse Rate) | 20.00% | 19.17% | -0.83% |
| CRR (Correct Reuse Prec.) | **79.17%** | 47.83% | **-31.34%** |
| IRR_cache (Hazard Rate) | **20.83%** | **52.17%** | **+31.34%** |
| IRR_traffic (Traffic Haz.) | 4.17% | 10.00% | +5.83% |
| FRR (False Rejection Rate) | 69.35% | 82.26% | +12.91% |

| Confusion Matrix | Phase 2 | Phase 3 |
|---|---|---|
| TP (Correct Reuse) | 19 | 11 |
| FP (Cache Hazard) | 5 | **12** |
| FN (Missed Reuse) | 43 | 51 |
| TN (Correct Reject) | 53 | 46 |

> [!WARNING]
> **Phase 3's non-LLM decision step WORSENS cache hazard rate on this dataset (52.17% vs Phase 2's 20.83%).** This is driven entirely by the AMBIGUOUS-tier decision step: 32 pairs in the ambiguous band, 12 FPs, IRR_cache = 57.1%. The decision step incorrectly promotes borderline pairs to REUSE at a high hazard rate.

### 3.3 Per-Tier Breakdown

| Tier | Count | TP | FP | FN | TN | ARR | IRR_cache | CRR |
|---|---|---|---|---|---|---|---|---|
| AUTO_REUSE | 2 | 2 | 0 | 0 | 0 | 100.0% | **0.0%** | 100.0% |
| AMBIGUOUS | 32 | 9 | 12 | 1 | 10 | 65.6% | **57.1%** | 42.9% |
| BYPASS | 86 | 0 | 0 | 50 | 36 | 0.0% | 0.0% | 100.0% |

**AUTO-REUSE tier is safe (0 FPs, IRR_cache=0%).** The problem is the AMBIGUOUS-tier decision step.

### 3.4 Interpretation

The AMBIGUOUS-band traffic represents pairs where:
- Similarity is in [0.50, 0.92) — Phase 2 would have served most of these as MISS (≥ 0.85 threshold)
- Stability confidence is in [0.80, 0.90) — borderline, mostly fallback classifier

The non-LLM decision step (linear combination of similarity, confidence, and category history) systematically promotes these borderline pairs toward REUSE at a 57.1% hazard rate. This is worse than random and worse than Phase 2's fixed threshold.

**Root cause:** The linear combination weights (0.50 sim + 0.30 conf + 0.20 hist) with threshold=0.50 are too permissive for the ambiguous-similarity region. With cold-start history (0.50), and even moderate similarity/confidence, the score easily exceeds 0.50. This effectively creates a soft version of a lower-threshold Phase 2 lookup, and Phase 2's sweep shows that lower thresholds (< 0.85) have IRR_cache of 27%–40%.

---

## 4. Auto-Reuse Verbatim Proof

**Result: PASS.**

All 2 AUTO_REUSE-tier pairs:
- Produced HIT decisions (TP=2, FP=0)
- Similarity scores were identical to what Phase 2's SemanticCache.lookup() returned
- DecisionStep.decide() was never invoked for AUTO_REUSE pairs (confirmed by test)

Test: `tests/test_decision_evaluation.py::TestAutoReuseVerbatimProof::test_auto_reuse_tier_produces_hit_matching_phase2_output` — **PASSES**.

---

## 5. Test Inventory

### New tests (Phase 3)

| File | Tests | Coverage |
|---|---|---|
| `test_tier_router_unit.py` | 22 tests | TierBoundaries validation, all tier routing conditions, hermetic hot-path proof |
| `test_decision_step_unit.py` | 21 tests | CategoryHistory EWMA math, DecisionStep weight validation, judge-hook behavior |
| `test_decision_evaluation.py` | 24 tests | Schema guard (reusing Phase 2 check), DecisionMetrics structure, auto-reuse verbatim proof |
| **Total new** | **67 tests** | All pass |

### Existing tests (unchanged)

All 90 pre-Phase-3 tests continue to pass (27 Phase 1 + ~81 Phase 2 unit tests, minus 18 deselected model-integration tests).

**Full suite result: 90 + 67 = 157 tests pass (+ 18 deselected model-integration tests).**

---

## 6. What Was NOT Done

- **No LLM judge call implemented.** The non-LLM decision step was evaluated first, per the spec. Results show it is inadequate (IRR_cache=57.1% in ambiguous band). A judge call remains as a hook in `DecisionStep` but is `None` by default and was NOT invoked.
- **No modification to `src/classifier/`** — Phase 1 code untouched.
- **No modification to `src/cache/`** — Phase 2 code untouched.
- **No Phase 2 threshold adjustment** — 0.85 reference threshold was not touched.
- **No new pair dataset authored** — Option A was explicitly deferred per the dataset decision.
- **No git commit** — all changes uncommitted for manual review.
- **No post-hoc boundary tuning** — boundaries were fixed from Phase 1/2 distributions before running the evaluation.

---

## 7. Findings and Required Actions

> [!CAUTION]
> **Finding: The non-LLM decision step in Phase 3's AMBIGUOUS band makes outcomes worse, not better (on the same 120 pairs Phase 2 was evaluated on).**
>
> IRR_cache for AMBIGUOUS-band traffic: **57.1%** (12 FP / 21 hits)
> IRR_cache for Phase 3 overall: **52.17%** — worse than Phase 2's 20.83%

**What this means:**
1. The AUTO-REUSE tier is safe (0% hazard, 2 pairs) — the boundary selection for that tier works.
2. The AMBIGUOUS-tier decision step is ineffective at distinguishing safe from unsafe pairs using similarity + confidence + cold-start history alone.
3. Phase 3 in its current form should NOT be used in production — it is strictly worse than Phase 2's simple threshold at 0.85 on this same-set benchmark.

**Recommended next steps (not in scope for Phase 3):**
- Option A: Invest in a judge call (LLM-based) to improve AMBIGUOUS-band accuracy, with honest cost accounting.
- Option B: Narrow the AMBIGUOUS band further (raise `bypass_sim_ceiling`) so fewer borderline pairs reach the decision step.
- Option C: Evaluate on a fresh pair dataset (Option A dataset) before drawing conclusions about generalization.

---

## 8. Files Added/Modified

| File | Action |
|---|---|
| `src/decision/__init__.py` | Created |
| `src/decision/tier_router.py` | Created |
| `src/decision/decision_step.py` | Created |
| `src/evaluation/decision_evaluator.py` | Created |
| `src/evaluation/__init__.py` | Modified (added exports) |
| `tests/test_tier_router_unit.py` | Created |
| `tests/test_decision_step_unit.py` | Created |
| `tests/test_decision_evaluation.py` | Created |
| `docs/phase3_implementation.md` | Created |
| `docs/phase3_walkthrough.md` | Created (this file) |
