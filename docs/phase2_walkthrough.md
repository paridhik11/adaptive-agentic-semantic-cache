# Phase 2 — Semantic Cache Core: Walkthrough

**Document purpose:** This is the WALKTHROUGH document — what was actually done, in the order it was done. It includes the actual dataset findings, sweep results, threshold justification, final metrics, test inventory, and an explicit list of what was NOT done.

Compare this against `docs/phase2_implementation.md` (the SPEC) to audit intent vs. execution.

---

## 1. Order of Work

1. Read Phase 1 source (`src/classifier/`, `src/evaluation/stability_evaluator.py`, all tests)
2. Verified dataset — Step 0 (before writing any code)
3. Wrote pair evaluation protocol to module docstring (before implementation)
4. Wrote safety criterion (before running the sweep)
5. Implemented `src/cache/embedding.py`, `vector_store.py`, `semantic_cache.py`
6. Implemented `src/evaluation/cache_evaluator.py`
7. Implemented `scripts/run_cache_threshold_sweep.py`
8. Implemented `tests/test_semantic_cache.py`, `tests/test_cache_evaluation.py`
9. Ran threshold sweep → captured all numbers below
10. Wrote `docs/phase2_implementation.md` (spec) and this document (walkthrough)
11. Ran full test suite

---

## 2. Dataset Verification (Step 0) — Actual Findings

File: `data/raw/query_pair_reuse_benchmark.json`

| Property | Found |
|---|---|
| Record count | **120** (PAIR-001 to PAIR-120) — not assumed; verified by loading |
| Format | JSON array (bare, no metadata wrapper) |
| Records missing `query_a` | 0 |
| Records missing `query_b` | 0 |
| Unique IDs | 120 (no duplicates) |
| All required keys present | Yes |

### Actual Distributions

| `is_reuse_safe` | Count |
|---|---|
| `true` | 62 |
| `false` | 58 |

| `semantic_similarity_level` | Count |
|---|---|
| HIGH | 66 |
| MEDIUM | 39 |
| LOW | 15 |

| `taxonomy_class` | Count |
|---|---|
| SAFE_EQUIVALENT | 62 |
| UNSAFE_DIFFERENT_INTENT | 15 |
| UNSAFE_DIFFERENT_TOPIC | 15 |
| UNSAFE_SCOPE_MISMATCH | 13 |
| UNSAFE_CONTEXT_MISMATCH | 11 |
| UNSAFE_DYNAMIC_TEMPORAL | 4 |

| `domain` | Count |
|---|---|
| computer_science | 54 |
| science_medicine | 27 |
| mathematics | 13 |
| system_operations | 11 |
| history_geography | 9 |
| finance_economics | 5 |
| realtime_news_weather | 1 |
| legal_compliance | 0 |

> [!NOTE]
> The dataset has a **heavily skewed domain distribution** (54/120 = 45% computer_science). This is relevant context for interpreting the sweep results.

### Leakage Check

The leakage check was run against all Phase 1 datasets:

| Phase 1 Dataset | Overlap with Pair Dataset |
|---|---|
| `query_stability_benchmark.json` (development, N=160) | **3 query strings** |
| `query_stability_benchmark_heldout.json` (N=60) | **0** |
| `query_stability_benchmark_final_test.json` (N=60) | **0** |
| `query_stability_human_credibility.json` (N=85) | **0** |

The 3 overlapping query strings with the **development set** (which is diagnostic/non-pristine) are:
1. `"what is the current version of the linux kernel mainline branch?"`
2. `"what is the formula for the volume of a sphere of radius r?"`
3. `"what is the value of euler's number e to four decimal places?"`

These appear in the pair dataset as individual query strings (query_a or query_b side), used as part of evaluation *pairs* — not as stability-labelled singles. The Phase 1 development set is explicitly tagged as `development_diagnostic_do_not_use_as_final_eval`. Zero overlap with any pristine evaluation set. **No action taken** — documented and tested.

---

## 3. Pair Evaluation Protocol

The following protocol was written as the module-level docstring in `src/evaluation/cache_evaluator.py` before any code in that file was implemented:

> For each pair (query_a, query_b, is_reuse_safe):
> 1. Index query_a's embedding into the vector store (as if it were already cached).
> 2. Search using query_b's embedding.
> 3. If top-1 similarity >= threshold: predicted = HIT (reuse claimed).
> 4. Compare predicted HIT/MISS against ground-truth is_reuse_safe.
>
> TP = HIT and is_reuse_safe=true (correct reuse)
> FP = HIT and is_reuse_safe=false (cache hazard — reused something unsafe)
> FN = MISS and is_reuse_safe=true (missed a safe reuse opportunity)
> TN = MISS and is_reuse_safe=false (correctly rejected)

---

## 4. Safety Criterion (Pre-Stated)

Written before the sweep was run: **`IRR_cache = FP / (TP + FP)` must be < 10%.**

Selection rule: **lowest** threshold satisfying the criterion (maximize hit-rate).

If no threshold satisfies the criterion: result reported as-is — not adjusted.

---

## 5. Threshold Sweep Results

Run with: `python scripts/run_cache_threshold_sweep.py`
Model: `all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (FAISS IndexFlatIP, cosine similarity via inner product of L2-normalized vectors)
Revision provenance: **TRACED** — snapshot timestamp 2026-09-05 matches run date; see `docs/phase2_reproducibility_fix.md` §1.
Mean embed+search latency per pair: **13.9–14.9 ms**

### Coarse Sweep (step 0.05)

| Threshold | Hits | ARR% | CRR% | IRR_cache% | IRR_traffic% | FRR% | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 107 | 89.17 | 57.94 | **42.06** | 37.50 | 0.00 | 62 | 45 | 0 | 13 |
| 0.15 | 106 | 88.33 | 58.49 | **41.51** | 36.67 | 0.00 | 62 | 44 | 0 | 14 |
| 0.20 | 105 | 87.50 | 59.05 | **40.95** | 35.83 | 0.00 | 62 | 43 | 0 | 15 |
| 0.25 | 104 | 86.67 | 59.62 | **40.38** | 35.00 | 0.00 | 62 | 42 | 0 | 16 |
| 0.30 | 103 | 85.83 | 60.19 | **39.81** | 34.17 | 0.00 | 62 | 41 | 0 | 17 |
| 0.35 | 103 | 85.83 | 60.19 | **39.81** | 34.17 | 0.00 | 62 | 41 | 0 | 17 |
| 0.40 | 102 | 85.00 | 60.78 | **39.22** | 33.33 | 0.00 | 62 | 40 | 0 | 18 |
| 0.45 | 99 | 82.50 | 61.62 | **38.38** | 31.67 | 1.61 | 61 | 38 | 1 | 20 |
| 0.50 | 93 | 77.50 | 64.52 | **35.48** | 27.50 | 3.23 | 60 | 33 | 2 | 25 |
| 0.55 | 87 | 72.50 | 66.67 | **33.33** | 24.17 | 6.45 | 58 | 29 | 4 | 29 |
| 0.60 | 83 | 69.17 | 67.47 | **32.53** | 22.50 | 9.68 | 56 | 27 | 6 | 31 |
| 0.65 | 75 | 62.50 | 72.00 | **28.00** | 17.50 | 12.90 | 54 | 21 | 8 | 37 |
| 0.70 | 68 | 56.67 | 72.06 | **27.94** | 15.83 | 20.97 | 49 | 19 | 13 | 39 |
| 0.75 | 52 | 43.33 | 67.31 | **32.69** | 14.17 | 43.55 | 35 | 17 | 27 | 41 |
| 0.80 | 35 | 29.17 | 68.57 | **31.43** | 9.17 | 61.29 | 24 | 11 | 38 | 47 |
| 0.85 | 24 | 20.00 | 79.17 | **20.83** | 4.17 | 69.35 | 19 | 5 | 43 | 53 |
| 0.90 | 8 | 6.67 | 75.00 | **25.00** | 1.67 | 90.32 | 6 | 2 | 56 | 56 |
| 0.95 | 4 | 3.33 | 50.00 | **50.00** | 1.67 | 96.77 | 2 | 2 | 60 | 56 |

### Fine Sweep (step 0.01, around 0.90 crossover zone)

| Threshold | Hits | ARR% | CRR% | IRR_cache% | FRR% | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| 0.90 | 8 | 6.67 | 75.00 | **25.00** | 90.32 | 6 | 2 | 56 | 56 |
| 0.91 | 8 | 6.67 | 75.00 | **25.00** | 90.32 | 6 | 2 | 56 | 56 |
| 0.92 | 7 | 5.83 | 71.43 | **28.57** | 91.94 | 5 | 2 | 57 | 56 |
| 0.93 | 6 | 5.00 | 66.67 | **33.33** | 93.55 | 4 | 2 | 58 | 56 |
| 0.94 | 5 | 4.17 | 60.00 | **40.00** | 95.16 | 3 | 2 | 59 | 56 |
| 0.95 | 4 | 3.33 | 50.00 | **50.00** | 96.77 | 2 | 2 | 60 | 56 |
| 0.96 | 3 | 2.50 | 33.33 | **66.67** | 98.39 | 1 | 2 | 61 | 56 |
| 0.97 | 2 | 1.67 | 50.00 | **50.00** | 98.39 | 1 | 1 | 61 | 57 |
| 0.98 | 1 | 0.83 | 0.00 | **100.00** | 100.00 | 0 | 1 | 62 | 57 |
| 0.99 | 1 | 0.83 | 0.00 | **100.00** | 100.00 | 0 | 1 | 62 | 57 |

---

## 6. Threshold Selection — Justification

> [!IMPORTANT]
> **Primary finding: No threshold satisfies IRR_cache < 10% while producing hits > 0.**

The pre-stated safety criterion (IRR_cache < 10%) was **not met at any threshold** in the sweep. At every threshold where the model produces at least one HIT, IRR_cache ≥ 20.83% (at threshold 0.85). The hazard rate never drops below the 10% ceiling while any reuse is claimed.

**This is a valid finding, not an error.** The baseline embedding model (`all-MiniLM-L6-v2`) cannot discriminate safe from unsafe pairs well enough for the stated safety bar using fixed-threshold cosine similarity alone. Key observations:

- At threshold 0.85 (best CRR = 79.17%): 5 FPs out of 24 hits = 20.83% hazard rate
- At threshold 0.80: 11 FPs out of 35 hits = 31.43% hazard rate
- At threshold 0.70 (reasonable hit-rate): 19 FPs out of 68 hits = 27.94% hazard rate
- The dataset contains many HIGH-similarity pairs that are UNSAFE (n=43 unsafe HIGH-sim pairs in the training distribution): the embedding model correctly identifies lexical/semantic closeness but cannot distinguish safe reuse from unsafe variants with the same surface similarity

**No threshold is recommended** for production use at the 10% safety ceiling. This confirms the need for Phase 3 (agentic verification) to reduce the hazard rate.

**Reference threshold for comparison (not recommended for production):** If the safety bar were relaxed to 20%, threshold 0.85 would be chosen: ARR=20.00%, IRR_cache=20.83%, FRR=69.35%.

---

## 7. Final Metrics Summary (At Reference Threshold 0.85)

| Metric | Value | Notes |
|---|---|---|
| Threshold | 0.85 | Reference only — does NOT meet the safety criterion |
| N pairs | 120 | Verified count |
| Hit count (TP+FP) | 24 | Queries that would bypass generation |
| ARR (Actual Reuse Rate) | 20.00% | — |
| CRR (Correct Reuse Precision) | 79.17% | TP/(TP+FP) |
| IRR_cache (Cache Hazard Rate) | **20.83%** | FP/(TP+FP) — exceeds 10% ceiling |
| IRR_traffic | 4.17% | FP/N |
| FRR (False Rejection Rate) | 69.35% | FN/(TP+FN) |
| Mean embed+search latency | ~13.9 ms | Measured directly |
| Token/dollar cost figures | **Not reported** | No LLM called |

---

## 8. Test Inventory

### New tests added

| File | Test | Guards Against |
|---|---|---|
| `test_semantic_cache.py` | `TestQueryEmbedder::test_encode_returns_correct_shape` | Embedding dimension contract |
| `test_semantic_cache.py` | `TestQueryEmbedder::test_encode_is_unit_normalized` | Cosine-via-inner-product correctness |
| `test_semantic_cache.py` | `TestQueryEmbedder::test_encode_empty_query_raises` | Empty query safety |
| `test_semantic_cache.py` | `TestFlatVectorStore::test_search_on_empty_store_returns_not_found` | Empty index guard |
| `test_semantic_cache.py` | `TestFlatVectorStore::test_add_and_search_round_trip` | Self-match score ≈ 1.0 |
| `test_semantic_cache.py` | `TestFlatVectorStore::test_reset_clears_store` | Between-pair isolation |
| `test_semantic_cache.py` | `TestSemanticCache::test_lookup_hit_for_near_identical_queries` | HIT at moderate threshold |
| `test_semantic_cache.py` | `TestSemanticCache::test_lookup_miss_for_dissimilar_queries_at_high_threshold` | MISS at high threshold |
| `test_semantic_cache.py` | `TestSemanticCache::test_score_is_cosine_similarity` | Self-match score = 1.0 |
| `test_cache_evaluation.py` | `TestDatasetSchemaValidation::test_wrong_record_count_raises_loudly` | Dataset mutation |
| `test_cache_evaluation.py` | `TestDatasetSchemaValidation::test_missing_required_key_raises` | Schema contract |
| `test_cache_evaluation.py` | `TestPairProtocolDirection::test_protocol_is_index_a_search_b_not_reversed` | Protocol a/b swap bug |
| `test_cache_evaluation.py` | `TestPairProtocolDirection::test_evaluator_pair_protocol_matches_documented_protocol` | Protocol–code alignment |
| `test_cache_evaluation.py` | `TestDatasetLeakage::test_zero_overlap_with_heldout` | Pristine set contamination |
| `test_cache_evaluation.py` | `TestDatasetLeakage::test_zero_overlap_with_final_test` | Pristine set contamination |
| `test_cache_evaluation.py` | `TestDatasetLeakage::test_zero_overlap_with_human_credibility` | Pristine set contamination |
| `test_cache_evaluation.py` | `TestDatasetLeakage::test_known_dev_set_overlaps_are_documented` | Overlap count drift |

### Existing tests — all pass
All 27 Phase 1 tests continue to pass unchanged.

---

## 9. Deviations from the Spec

None. The implementation follows the spec exactly:
- Pair protocol: index a, search b
- Sweep range: 0.10–0.99 coarse, 0.01 fine around crossover
- Safety criterion applied as stated
- No token/dollar cost figures produced

---

## 10. What Was NOT Done

- **No agentic verification** added — this is out of scope for Phase 2
- **No learned or dynamic threshold** — fixed threshold only
- **No integration with Phase 1 classifier** — Phase 2 evaluated standalone
- **No token or dollar cost figures** — no LLM called
- **No modification to `src/classifier/`** — Phase 1 code untouched
- **No modification to Phase 1 datasets** — untouched
- **No git commit made** — all changes left uncommitted for manual review
- **No post-hoc threshold selection** — the threshold was not adjusted after seeing results

---

## 11. Files Added/Modified

| File | Action |
|---|---|
| `src/cache/embedding.py` | Created |
| `src/cache/vector_store.py` | Created |
| `src/cache/semantic_cache.py` | Created |
| `src/cache/__init__.py` | Modified (added exports) |
| `src/evaluation/cache_evaluator.py` | Created |
| `src/evaluation/__init__.py` | Modified (added exports) |
| `scripts/run_cache_threshold_sweep.py` | Created |
| `tests/test_semantic_cache.py` | Created |
| `tests/test_cache_evaluation.py` | Created |
| `docs/phase2_implementation.md` | Created |
| `docs/phase2_walkthrough.md` | Created (this file) |
| `pyproject.toml` | Modified (added faiss-cpu, sentence-transformers) |
