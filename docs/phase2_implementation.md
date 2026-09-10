# Phase 2 — Semantic Cache Core: Implementation Specification

**Document purpose:** This is the SPEC document, written for future auditors to compare stated intent against actual execution. It restates the scope boundaries, pair-evaluation protocol, safety criterion, sweep range, and cost-metric rule exactly as they were given in the implementation brief — not what was done.

---

## 1. Scope — Phase 2 Baseline Only, No Agent

Phase 2 implements a **baseline semantic cache** evaluated standalone as a control group.

**What is in scope:**
- `query → embed → vector search (top-1) → fixed similarity threshold → HIT/MISS`

**What is explicitly excluded:**
- Agentic verification
- A learned or dynamic threshold
- Integration with Phase 1 Stability Classifier output
- Any modification to `src/classifier/` or Phase 1 datasets
- Git commits (all changes left uncommitted for manual review)
- Token or dollar cost figures (no LLM is called in this baseline)

---

## 2. Dataset Verification — Step 0

Before any pipeline code was written, the dataset `data/raw/query_pair_reuse_benchmark.json` was loaded and verified. The verification report is part of the walkthrough doc (`docs/phase2_walkthrough.md`). This step was explicitly required to:

- Confirm the actual record count (not assumed to be 120)
- Verify the full schema per record
- Verify both `query_a` and `query_b` are populated for every record
- Check distributions of `is_reuse_safe`, `semantic_similarity_level`, and `taxonomy_class`
- Check for overlap with all Phase 1 datasets using the same leakage-check logic added for Phase 1

---

## 3. Pair Evaluation Protocol

The following protocol was written down before any implementation, as a docstring in `src/evaluation/cache_evaluator.py`, and implemented verbatim:

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

Each pair is evaluated independently. The vector store is reset between pairs. Only one entry (query_a) is in the store when query_b is searched.

---

## 4. Safety Criterion

**Written before any threshold sweep was run.**

The cache hazard rate (`IRR_cache = FP / (TP + FP)`) must stay **below 10%** at the chosen threshold. This is a hard ceiling, not a tradeoff weight.

**Selection rule:** The **lowest** threshold satisfying `IRR_cache < 10%` is chosen, to maximize the actual reuse rate (ARR) subject to the safety constraint. The highest threshold satisfying the criterion is not chosen.

**If the criterion cannot be met:** If no threshold achieves `IRR_cache < 10%` while producing hits > 0, the result is reported as a finding and the threshold is not adjusted post-hoc to manufacture an acceptable number.

---

## 5. Threshold Sweep Range

**Pre-stated sweep strategy:**
- Coarse sweep: 0.10 to 0.99, step 0.05 (~19 values)
- Fine sweep: 0.01-step refinement around wherever `IRR_cache` crosses the 10% safety bar (up to 10 fine steps)

Threshold selection was derived from the pre-stated safety criterion and selection rule, not picked post-hoc after seeing results.

---

## 6. Cost Metric Rule

No LLM is called in the Phase 2 baseline. The only honest cost proxies are:

- **Hit count** (TP + FP): the number of queries that would bypass generation if the cache were wired end-to-end
- **Embedding + search latency**: measured directly via `time.perf_counter()`

Dollar or token-per-query figures from assumed price-per-token rates are **not reported**. This is a hard rule, not a preference.

---

## 7. Metric Definitions

Metric formulas are taken verbatim from `docs/evaluation_metrics.md` (Phase 0 definitions). They are not redefined for Phase 2:

| Metric | Formula | Notes |
|---|---|---|
| ARR (Actual Reuse Rate) | (TP + FP) / N | Reflects traffic offloaded from LLM |
| CRR (Correct Reuse Rate) | TP / (TP + FP) | = 1.0 if no hits |
| IRR_cache (Cache Hazard Rate) | FP / (TP + FP) | Primary safety metric; = 0.0 if no hits |
| IRR_traffic | FP / N | Hazardous reuses as fraction of all traffic |
| FRR (False Rejection Rate) | FN / (TP + FN) | Missed safe reuses; = 0.0 if no safe pairs |

---

## 8. Test Requirements

Tests were required to guard:
1. `test_semantic_cache.py` — unit tests for `QueryEmbedder`, `FlatVectorStore`, `SemanticCache.lookup()`
2. `test_cache_evaluation.py`:
   - Dataset schema check that fails loudly on unexpected count or schema
   - Pair protocol direction test (swapping a/b catches accidental swap bugs)
   - Leakage check: `query_pair_reuse_benchmark.json` vs. all Phase 1 datasets
3. All 27 existing Phase 1 tests must continue to pass

---

## 9. Documentation Requirements

Two separate documents with different purposes:

1. **`docs/phase2_implementation.md`** (this document): The SPEC. For auditors to check intent vs. execution.
2. **`docs/phase2_walkthrough.md`**: What was actually done. Dataset findings, sweep table, threshold justification tied to pre-stated criterion, final metrics, test inventory, and an explicit list of what was NOT done.
