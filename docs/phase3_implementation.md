# Phase 3 — Reuse Decision Layer: Implementation Specification

**Document purpose:** This is the SPEC document. For auditors to compare stated intent against actual execution. See `docs/phase3_walkthrough.md` for what was actually done.

---

## 1. What Phase 3 Is and Isn't

Phase 3 adds a **tiered routing decision layer** above the frozen Phase 1 classifier and Phase 2 cache. It does NOT modify either Phase 1 or Phase 2.

- Phase 1 (Stability Classifier) and Phase 2 (Semantic Cache, frozen at model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) are the control groups.
- Phase 3's entire purpose is to be measured against them.

---

## 2. Step 0 — Tier Boundaries (written before any routing code)

All boundaries derive from Phase 1 and Phase 2's **already-measured** distributions. The evaluation dataset (`query_pair_reuse_benchmark.json`) was NOT consulted when choosing these values.

### 2.1 Phase 1 Confidence Distribution (existing data)

From `stability_classifier.py` and the `STABLE_CONFIDENCE_THRESHOLD` comment block:

| Cluster | Confidence | Description | Rule-layer dangerous error rate |
|---|---|---|---|
| Cluster A | ≈ 0.769 | Zero-signal queries; overridden to DYNAMIC by Sub-stage A | ~0% (override fires) |
| Cluster B | ≈ 0.881 | Queries with strong programming cues; 3/85 credibility-set queries escape as FP | ~10% on Cluster B |
| Rule layer | 0.95–1.0 | Explicit STABLE rule fired | ~0% on credibility set |

**STABLE_CONFIDENCE_THRESHOLD = 0.80** (from Phase 1) — below this, confidence never produces a valid STABLE effective_decision.

### 2.2 Phase 2 Similarity Distribution (existing sweep data)

From `docs/phase2_walkthrough.md` — **verbatim, not re-derived:**

| Threshold | Hits | IRR_cache% | Notes |
|---|---|---|---|
| 0.85 | 24 | 20.83% | Best CRR; still 5 FPs |
| 0.90 | 8 | 25.00% | |
| 0.92 | 7 | 28.57% | Fine-sweep interpolated |
| 0.97 | 2 | 50.00% | |

No threshold achieves IRR_cache < 10% while producing hits > 0.

**HIGH-similarity pairs: 66 total; 43 are UNSAFE.** Embedding similarity alone cannot distinguish safe from unsafe pairs with the stated safety bar.

### 2.3 Tier Boundary Definitions

| Parameter | Value | Justification |
|---|---|---|
| `auto_reuse_sim_floor` | **0.92** | Above Phase 2's 0.90 point (8 hits, 25% hazard); combined with confidence filter, residual hazard estimated < 5% |
| `auto_reuse_conf_floor` | **0.90** | Above Cluster B (0.881); selects rule-layer certainty where dangerous error rate ≈ 0% |
| `bypass_sim_ceiling` | **0.50** | Below this, Phase 2's sweep shows essentially no hit-rate signal |
| `bypass_conf_ceiling` | **0.80** | Matches STABLE_CONFIDENCE_THRESHOLD from Phase 1 |

**AUTO-REUSE tier:** `similarity_score >= 0.92` AND `stability_confidence >= 0.90` AND `effective_decision == STABLE`

**BYPASS tier:** `effective_decision != STABLE` OR `similarity_score < 0.50` OR `stability_confidence < 0.80`

**AMBIGUOUS tier:** Everything else (borderline on at least one signal)

### 2.4 Residual Hazard Estimate for AUTO-REUSE Tier

Using only Phase 1 and Phase 2's existing numbers:

- Phase 1 rule-layer dangerous error rate: **≈ 0%** (0 FPs on 85-query credibility set)
- Phase 2 at sim ≥ 0.92: **≈ 28.57% standalone hazard** (7 hits, 2 FPs over 120 pairs)
- Joint estimate (both filters): **< 5%** — the 2 FPs at sim ≥ 0.92 are intent-mismatch pairs that would also need a STABLE rule match to pass the confidence filter, which is unlikely by the rule engine's design

> [!IMPORTANT]
> This estimate must be verified against the 120-pair evaluation. If empirical AUTO-REUSE IRR_cache > 5%, tighten boundaries before production use.

---

## 3. Dataset Decision — Option B (not re-litigated)

**Decision already made:** Use `query_pair_reuse_benchmark.json` for Phase 3 evaluation (Option B). Option A (fresh pair dataset) was explicitly deferred.

- `query_pair_reuse_benchmark.json` is the only pair-format dataset in this repo
- Phase 2 already swept it fully
- Single-query datasets (heldout/final_test) are wrong-shaped for cache-reuse evaluation

**Required framing (not optional):** Every Phase 3 result computed against this dataset is labeled as a **SAME-SET COMPARISON**, not a generalization or held-out test.

Required statement in `docs/phase3_walkthrough.md`:

> "These results are a same-set comparison against query_pair_reuse_benchmark.json, already used to establish Phase 2's baseline. They show whether the tiered decision layer improves outcomes on known pairs, not whether it generalizes to unseen traffic. A fresh pair dataset (Option A) would be required before claiming generalization."

---

## 4. Ambiguous-Band Measurement Plan

The ambiguous-band traffic fraction is reported as the **headline DoD metric**:

```
ambiguous_fraction = (pairs routed to AMBIGUOUS) / 120
```

If `ambiguous_fraction > 0.50`, evaluation is flagged with a WARNING and the decision-step is NOT expanded. A large ambiguous band means boundaries need tightening against Phase 1/2 distributions, not more agent logic on top.

Boundaries are fixed from Phase 1/2 data before running the evaluation — no iteration against the 120-pair outcomes.

---

## 5. Decision-Step Design

### 5.1 Category Reuse History

**What counts as a category?** The 8 enterprise domains from Phase 0 (matching the `domain` field in the benchmark): `computer_science`, `science_medicine`, `mathematics`, `system_operations`, `history_geography`, `finance_economics`, `realtime_news_weather`, `legal_compliance`.

**What does "history" mean with no live traffic?** History is simulated online during evaluation: as pairs are processed in benchmark order (PAIR-001 to PAIR-120), each pair's outcome updates the running EWMA for its domain. At pair N, the history available reflects pairs 1..N-1 only. This is NOT a look-ahead.

**Cold-start:** A domain with no prior observations defaults to **0.50** (neutral, no bias).

**Staleness:** Exponentially Weighted Moving Average (EWMA), decay=0.90. With no live traffic and 120 pairs, EWMA and simple mean differ by < 0.02. EWMA is retained for correct live-traffic behavior.

### 5.2 Linear Combination Scoring

For AMBIGUOUS traffic:

```
score = 0.50 * sim_norm + 0.30 * conf_norm + 0.20 * hist_rate
```

Where:
- `sim_norm = clip((sim - bypass_sim_ceiling) / (auto_reuse_sim_floor - bypass_sim_ceiling), 0, 1)`
- `conf_norm = clip((conf - bypass_conf_ceiling) / (auto_reuse_conf_floor - bypass_conf_ceiling), 0, 1)`
- `hist_rate = domain EWMA rate` (cold-start: 0.50)

Decision threshold: `score >= 0.50` → REUSE; else BYPASS.

**Weight rationale:** Similarity gets the most weight (0.50) because it is the Phase 2 primary signal. Confidence is secondary (0.30) because Phase 1 demonstrated it distinguishes safe from borderline STABLE predictions. History is tertiary (0.20) because it is subject to cold-start and dataset skew.

### 5.3 Judge-Call Policy

"Optionally a cheap judge call" — the initial implementation builds WITHOUT a judge call. The `JudgeCallable` hook is defined in `decision_step.py` and can be wired in, but is `None` by default.

**Build order:** Evaluate the non-LLM version first. Add a judge call only if ambiguous-band accuracy is inadequate and measurably improvable. If added, its call count and latency are first-class reported metrics.

---

## 6. Files

### New files

| File | Purpose |
|---|---|
| `src/decision/__init__.py` | Package exports |
| `src/decision/tier_router.py` | Tier enum, TierBoundaries, TierRouter |
| `src/decision/decision_step.py` | CategoryHistory, DecisionStep, DecisionResult |
| `src/evaluation/decision_evaluator.py` | DecisionEvaluator, DecisionMetrics |
| `tests/test_tier_router_unit.py` | Network-free tier router tests |
| `tests/test_decision_step_unit.py` | Network-free decision step tests |
| `tests/test_decision_evaluation.py` | Schema guard + verbatim proof tests |
| `docs/phase3_implementation.md` | This document |
| `docs/phase3_walkthrough.md` | What was actually done |

### Modified files

| File | Change |
|---|---|
| `src/evaluation/__init__.py` | Added DecisionEvaluator, DecisionMetrics exports |

### Untouched files (explicitly)

- `src/classifier/` — Phase 1, not modified
- `src/cache/` — Phase 2, not modified
- All Phase 1 and Phase 2 datasets — not modified
- Phase 2 frozen threshold (0.85 reference point) — not modified or reused as Phase 3 threshold

---

## 7. DoD Requirements

| Item | Requirement |
|---|---|
| 1 | Tier boundaries defined and justified against Phase 1/2 distributions BEFORE touching evaluation dataset |
| 2 | Ambiguous-band fraction measured and reported; flagged if > 50% |
| 3 | Proof (passing test) that AUTO_REUSE never triggers decision step or judge call |
| 4 | Phase 2 vs Phase 3 same-set comparison on 120 pairs, labeled correctly |
| 5 | Both docs complete |
| 6 | All new + existing tests passing |
| 7 | No git commit |
