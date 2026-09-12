# Phase 2 Reproducibility Fix

**Document purpose:** This document records every change made during the Phase 2
reproducibility audit, the exact findings from each step, and the final frozen
baseline state. A future reader should be able to answer "what exact model produced
these numbers" from this document alone, without archaeology.

**Scope:** Reproducibility fix only. No Phase 3 work, no changes to the pair
evaluation protocol, safety criterion, sweep range, or metric definitions.

---

## 1. Revision Provenance (Step 1)

### Finding: TRACED — original revision confirmed

The local HuggingFace cache at:

```
~/.cache/huggingface/hub/
  models--sentence-transformers--all-MiniLM-L6-v2/
    snapshots/
      1110a243fdf4706b3f48f1d95db1a4f5529b4d41/
```

is **present** on this machine. All snapshot files carry timestamps of
**2026-09-05 14:51:xx**, matching the Phase 2 sweep run date (Phase 2
implementation started 2026-09-05T09:09:29Z; see Git history).

The snapshot folder name **IS** the commit hash that `sentence-transformers`
actually downloaded and loaded. This is not inferred or guessed — it is the
direct, unambiguous artifact of the HuggingFace Hub download mechanism.

**Revision hash (ground truth): `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**

The walkthrough numbers are therefore **provenance-confirmed**, not
provenance-unknown. They were produced by exactly this revision.

---

## 2. Pinned Revision (Step 2)

### Change: `src/cache/embedding.py`

Added a `MODEL_REVISION` constant:

```python
MODEL_REVISION: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
```

This constant is now passed to `SentenceTransformer` as:

```python
self._model = SentenceTransformer(model_name, revision=MODEL_REVISION)
```

This guarantees that any future run of the code — on any machine, at any
future date — loads exactly the same model weights that produced the published
baseline numbers, regardless of what HuggingFace Hub's "main" branch resolves
to at that time.

**Provenance label:** TRACED. The hash was obtained from the local snapshot
cache, not assumed or freshly-pinned from "current main." See §1 above.

The docstring in `embedding.py` explains the full pinning protocol, including
the requirement to re-run the sweep and document old-vs-new numbers before
any revision change is accepted.

---

## 3. Test Split Architecture (Step 3)

### Problem with original test files

Both `test_semantic_cache.py` and `test_cache_evaluation.py` instantiated
`QueryEmbedder()` directly (loading the real SentenceTransformer model) in
every test that touched cache logic. This meant:

- Every `pytest` run required network access or a local HF cache
- The CI job was silently model-dependent with no indication in the test names
- Logic tests (threshold boundary, confusion-matrix math, schema validation)
  were coupled to real embedding outputs for no reason

### New architecture

Three new test files replace the two original ones:

| File | Marker | Network? | Purpose |
|---|---|---|---|
| `test_semantic_cache_unit.py` | *(none)* | **No** | FlatVectorStore mechanics, SemanticCache decision logic, CacheLookupResult schema, module constants — all via mocked embedder |
| `test_cache_evaluation_unit.py` | *(none)* | **No** | CacheEvaluator schema validation, pair protocol mechanics (mocked), threshold sweep math, leakage check (JSON-only) |
| `test_model_integration.py` | `@pytest.mark.requires_model` | Only if HF cache absent | Real embedding similarity behaviour, tied explicitly to `MODEL_REVISION` |

The two original files (`test_semantic_cache.py`, `test_cache_evaluation.py`) have
been replaced by tombstones that document the migration map. They contain no
runnable tests.

### What is mocked in unit tests

`QueryEmbedder.encode()` is mocked via `unittest.mock.MagicMock` to return
deterministic fake unit-norm vectors using `np.random.default_rng(seed)`.
The actual FAISS IndexFlatIP math and SemanticCache decision logic runs against
these synthetic vectors — only the SentenceTransformer download path is avoided.

Key unit-test coverage that does NOT require real embeddings:
- `FlatVectorStore`: add/search/reset/dim-mismatch/metadata/top-1 selection
- `SemanticCache.lookup()`: score-at-threshold = HIT, score-below = MISS,
  empty-store = MISS, zero-threshold, invalid-threshold guards
- `CacheLookupResult.to_dict()` schema, frozen dataclass immutability
- `MODEL_REVISION` format validation (40-char hex, not "main")
- `QueryEmbedder` constructor: SentenceTransformer called with `revision=` kwarg
- `CacheEvaluator.load_and_validate_dataset()`: wrong count, missing keys,
  empty query, bad bool, duplicate ID, non-list JSON
- Confusion-matrix math: TP/FP/FN/TN counting, IRR_cache formula, FRR formula,
  zero-hits CRR default
- Threshold sweep: monotonic hit-count ordering, one point per threshold

### What requires the real model

- Actual cosine similarity values between natural-language queries
- Verification that similar queries score > 0.70 and dissimilar < 0.60
- Self-match scores ≈ 1.0 (trivially confirmed by unit tests too, but
  integration test confirms with real embeddings)
- `MODEL_REVISION` loaded in `embedder.revision` attribute

### pytest configuration

`pyproject.toml` now includes:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not requires_model'"
markers = [
    "requires_model: marks tests that load the real all-MiniLM-L6-v2 model ...",
]
```

Default `pytest` invocation → **unit tests only, 63 tests, ~8 seconds, zero network**.
Explicit `pytest -m requires_model` → **18 model integration tests**.

---

## 4. CI Cache Change (Step 4)

### Change: `.github/workflows/ci.yml`

The original CI had a single `test` job running `pytest` (which would load the
model and hit the network on every run). The new CI has two jobs:

**Job 1: `unit-tests`** (runs on both Python 3.11 and 3.12)
- Runs `pytest -v` (default, excludes `requires_model`)
- No model downloaded, no network dependency

**Job 2: `model-integration-tests`** (runs on Python 3.11 only)
- Caches `~/.cache/huggingface` via `actions/cache@v4`
- Cache key: `huggingface-all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- The key includes `MODEL_REVISION` so the cache is automatically invalidated
  if the pinned revision is ever updated (a revision change without cache
  invalidation would silently serve stale weights in CI)
- Runs `pytest -m requires_model -v`

---

## 5. Threshold Sweep Re-Run (Steps 6 & 7)

### New sweep results (with pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`)

Run: `python scripts/run_cache_threshold_sweep.py` after pinning MODEL_REVISION.

#### Coarse Sweep (step 0.05)

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

#### Fine Sweep (step 0.01, around crossover)

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

### Old vs. New comparison (Step 7)

> [!IMPORTANT]
> **The new sweep results are IDENTICAL to the published numbers, digit for digit.**
> Pinning did not change the reported baseline. The original numbers are now
> provenance-confirmed.

Every TP, FP, FN, TN count and every derived metric (ARR%, CRR%, IRR_cache%,
IRR_traffic%, FRR%) in the new run matches the values in `docs/phase2_walkthrough.md`
exactly. This is expected: the revision hash was traced from the local cache
(§1), so the same weights were already being used — we have now made that
dependency explicit and reproducible for all future runs.

**The headline finding is re-verified:** No threshold satisfies IRR_cache < 10%
while producing hits > 0. This holds under the pinned revision, not as an
assumption.

`docs/phase2_walkthrough.md` is **not modified**. The numbers are correct as
published. A note about the revision pin has been added to the model line in
the walkthrough's section 5.

---

## 6. Docstring Bug Fix (Step 8)

### Change: `src/cache/embedding.py`

**Before (incorrect):**
```
cosine similarity in [0, 1]
```

**After (correct):**
```
The mathematical range of cosine similarity for unit vectors is [-1, 1];
practical values for this model on natural-language queries are
overwhelmingly positive (>0), but the range is NOT bounded to [0, 1].
```

`vector_store.py` already had the correct `[-1, 1]` statement in its module
docstring. The `embedding.py` docstring was inconsistent; this fix aligns them.

The fix appears in both the class-level docstring and the `encode()` method
docstring's Returns section.

---

## 7. Final Frozen Baseline

The following numbers are now permanently tied to model revision
**`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`** of
`sentence-transformers/all-MiniLM-L6-v2`, as loaded via FAISS IndexFlatIP
with L2-normalized vectors (cosine similarity via inner product):

| Metric | Value | Notes |
|---|---|---|
| Model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Traced, not guessed |
| Dataset | `query_pair_reuse_benchmark.json`, N=120 | Verified; no mutations |
| Sweep range | 0.10–0.99 coarse (step 0.05) + fine 0.90–0.99 (step 0.01) | Unchanged |
| Safety criterion | IRR_cache < 10% | Pre-stated, not post-hoc |
| Safety criterion satisfied? | **No** | No threshold produces hits with IRR_cache < 10% |
| Minimum IRR_cache with hits | **20.83%** at threshold 0.85 | 5 FP / 24 hits |
| Reference threshold | 0.85 | Not recommended for production |
| ARR at 0.85 | 20.00% | — |
| CRR at 0.85 | 79.17% | — |
| IRR_traffic at 0.85 | 4.17% | — |
| FRR at 0.85 | 69.35% | — |
| IRR_cache at 0.70 | 27.94% | 19 FP / 68 hits |

Any future change to `MODEL_REVISION` in `src/cache/embedding.py` **must**:
1. Re-run `scripts/run_cache_threshold_sweep.py`
2. Compare and report old vs. new numbers in this document
3. Re-verify whether the headline finding ("no threshold satisfies IRR_cache < 10%") still holds
4. Update the CI cache key in `.github/workflows/ci.yml` to the new revision

---

## 8. Files Changed in This Fix

| File | Change |
|---|---|
| `src/cache/embedding.py` | Added `MODEL_REVISION` constant; pass `revision=` to `SentenceTransformer`; fixed cosine range docstring `[0,1]→[-1,1]` |
| `tests/test_semantic_cache_unit.py` | **NEW** — 43 network-free unit tests |
| `tests/test_cache_evaluation_unit.py` | **NEW** — 20 network-free unit tests |
| `tests/test_model_integration.py` | **NEW** — 18 model integration tests (`@pytest.mark.requires_model`) |
| `tests/test_semantic_cache.py` | **RETIRED** → tombstone with migration map |
| `tests/test_cache_evaluation.py` | **RETIRED** → tombstone with migration map |
| `pyproject.toml` | Added `addopts`, `markers` for `requires_model` |
| `.github/workflows/ci.yml` | Split into unit-tests job + model-integration-tests job with HF cache |
| `docs/phase2_reproducibility_fix.md` | **NEW** — this document |
