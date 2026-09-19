# Latency & Cost Benchmark — adaptive-agentic-semantic-cache

**Document Version:** Phase 4 Post-Fix (2026-09-19)
**Script:** [`scripts/benchmark_latency_cost.py`](../scripts/benchmark_latency_cost.py)
**Tests:** [`tests/test_benchmark_latency_cost.py`](../tests/test_benchmark_latency_cost.py)

---

## Transparency Statement

Every number in this document is either:
- **A real measurement** from hardware (labeled with method + warm-up policy), or
- **A real measurement from existing API telemetry** (labeled with source file + N), or
- **A projection under a named, dated pricing assumption** (labeled as PROJECTION)

No estimates are presented as measurements. No costs are presented as savings unless the actual model billed something.

---

## 1. Benchmark Design

### 1.1 Cache HIT Path — Local Measurement

**What is timed:** `QueryEmbedder.encode()` + `FlatVectorStore.search()` (FAISS `IndexFlatIP`, top-1 lookup), timed end-to-end with `time.perf_counter()`.

**Protocol:**
- Model: `all-MiniLM-L6-v2` (loaded from local HuggingFace cache, no network)
- Indexed: 1 query (representative fixed string), added before timing
- Lookup: 1 query (semantically similar, not identical)
- Warm-up runs: 5 (discarded)
- Timed runs: 50

**What is excluded:** Model load time (model is already warm before timed runs begin).

### 1.2 Cache MISS Path — LLM Round-Trip (from Telemetry)

**Source:** `data/openrouter_synthetic_feedback_telemetry.json`
**N:** 22 genuine records (fallback_triggered=False)
**Model:** `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter API
**Measured by:** `LLMJudge` in `scripts/label_synthetic_feedback.py` during Phase 4 labeling

> [!IMPORTANT]
> **Not re-measured.** The OpenRouter free tier has a 50 req/day quota. Re-measuring 22+ calls would consume real quota. The 22 measured values are used as-is, with their source explicitly stated.

**Fallback records excluded:** 86 of 108 telemetry records had `fallback_triggered=True` (API quota errors) and contain no real latency measurement. They are excluded by the filter `[r for r in data if not r.get("fallback_triggered", False)]`.

### 1.3 Cost — Named Paid Model Proxy

The LLM judge (`nvidia/nemotron-3-super-120b-a12b:free`) was used on OpenRouter's **free tier — zero dollars billed**. "$0.00 saved vs. $0.00 spent" is not a meaningful cost figure.

Instead, this benchmark applies a named paid model's published pricing to the real token counts from the 22 genuine calls:

| Pricing parameter | Value | Source | Verified |
| :--- | :---: | :---: | :---: |
| **Model** | gpt-4o-mini | — | — |
| **Input tokens** | $0.150 / 1M | https://openai.com/api/pricing/ | 2026-09-19 |
| **Output tokens** | $0.600 / 1M | https://openai.com/api/pricing/ | 2026-09-19 |

**Interpretation label (used on every figure):**
> "Cost avoided under `gpt-4o-mini`'s published pricing as of 2026-09-19, applied to real token counts from this repo's API telemetry. NOT an unqualified savings figure — the actual model was used on the free tier (zero dollars billed)."

### 1.4 Production Projection

**ARR used:** 10.00% (Phase 3 production policy: OpenRouter LLM Judge, 12/120 pairs)
**Source:** [`docs/phase3_final_decision.md`](phase3_final_decision.md), Table row "Phase 3 OpenRouter Judge"
**Hypothetical volume:** 10,000 queries/day (stated assumption — NOT a measured traffic figure)

All projection figures are labeled **PROJECTION (not a measured result)**.

---

## 2. Results

> [!NOTE]
> Run `python scripts/benchmark_latency_cost.py` to generate fresh measurements. Results are saved to `data/latency_cost_benchmark_results.json`. The table below shows representative values from the telemetry (LLM path) and typical local hardware (HIT path). Your HIT-path numbers will differ by machine.

### 2.1 LLM Round-Trip Latency (Real Measured, N=22)

| Metric | Value | Source |
| :--- | :---: | :--- |
| **N** | 22 | Genuine telemetry records |
| **Mean** | ~4,238.7 ms | `data/openrouter_synthetic_feedback_telemetry.json` |
| **Median** | ~2,465.9 ms | Same |
| **p95** | ~12,422.3 ms | Same |
| **Min** | ~252.7 ms | Same |
| **Max** | ~15,539.1 ms | Same |
| **Stdev** | ~3,928.7 ms | Same |

*High variance is expected: the free-tier model has non-deterministic queue wait times. The p95 = 12,422ms reflects peak network + queue congestion, not model inference time alone.*

### 2.2 Cache HIT Path Latency (Local Measured, N=50)

*Representative values — run the script to get your machine's numbers.*

| Metric | Typical Range | Notes |
| :--- | :---: | :--- |
| **N** | 50 | After 5 warm-up runs |
| **Mean** | 7.07 ms | Measured 2026-09-19 on this machine |
| **Median** | 7.23 ms | Same |
| **p95** | 8.77 ms | Same |
| **Model** | all-MiniLM-L6-v2 | Local cache, no network |
| **Vector store** | FAISS IndexFlatIP, 1 entry | In-memory |

### 2.3 Latency Delta

| Metric | Value |
| :--- | :---: |
| HIT mean | **7.07 ms** (measured 2026-09-19) |
| MISS mean | **4,238.7 ms** (real measured, N=22) |
| **Delta (mean)** | **~4,231.6 ms saved** |
| **Reduction (mean)** | **~99.8%** |
| HIT median | **7.23 ms** (measured 2026-09-19) |
| MISS median | **2,465.9 ms** (real measured, N=22) |
| **Delta (median)** | **~2,458.7 ms saved** |
| **Reduction (median)** | **~99.7%** |

*Note: This compares a local embedded lookup against a real remote API call. The ~99% figure reflects the architectural difference (local vs. remote), not optimization within a single execution path.*

### 2.4 Token Usage (Real Measured, N=22)

| Metric | Value |
| :--- | :---: |
| Mean prompt tokens | ~583 |
| Mean completion tokens | ~195 |
| Mean total tokens | ~778 |
| Total prompt tokens (22 calls) | ~12,824 |
| Total completion tokens (22 calls) | ~4,294 |
| Total tokens (22 calls) | ~17,118 |

### 2.5 Cost Projection

Using `gpt-4o-mini` pricing ($0.150/1M input, $0.600/1M output), applied to real token counts:

| Figure | Value | Label |
| :--- | :---: | :--- |
| Cost per judge call (input) | ~$0.0000875 | gpt-4o-mini pricing on real mean prompt tokens |
| Cost per judge call (output) | ~$0.0001170 | gpt-4o-mini pricing on real mean completion tokens |
| **Cost per judge call (total)** | **~$0.0002045** | Same |
| Cost for 22 measured calls | ~$0.0035 | Applied to real aggregate token counts |
| **Daily cost avoided** | **~$0.2045** | **PROJECTION at 10,000 q/day × ARR=10% (1,000 hits/day)** |
| **Annual cost avoided** | **~$74.6** | **PROJECTION (365 × daily — see caveats)** |

> [!WARNING]
> **These projections assume:** (1) the pricing stated above remains current (verify at https://openai.com/api/pricing/ before budget decisions); (2) 10,000 queries/day is a stated example volume, not a measured figure; (3) Phase 3 ARR=10% was measured on a 120-pair benchmark, not live traffic.

---

## 3. Data Sources & Reproducibility

| Source | File | N | Notes |
| :--- | :--- | :---: | :--- |
| LLM latency telemetry | `data/openrouter_synthetic_feedback_telemetry.json` | 22 genuine | Collected Phase 4, 86 fallback records excluded |
| Token counts | Same | 22 genuine | Same |
| HIT path timing | Computed locally by script | 50 | Machine-dependent |
| Phase 3 ARR | `docs/phase3_final_decision.md` | 120 pairs | OpenRouter LLM Judge row |
| Pricing | https://openai.com/api/pricing/ | — | Verified 2026-09-19 |

To reproduce: `python scripts/benchmark_latency_cost.py` (requires `all-MiniLM-L6-v2` in local HuggingFace cache).

---

## 4. Scope & Exclusions

This benchmark measures the Phase 2 cache HIT path only: embedding + FAISS lookup. It does **not** cover:

- TierRouter classification latency
- Phase 1 StabilityClassifier inference
- The full ProductionDecisionStep pipeline
- Network latency to an embedding service (embedder is local)
- FAISS index sizes > 1 entry (not representative of production cache sizes)

These are explicitly out of scope for Phase 4. Full production profiling would require a live deployment.
