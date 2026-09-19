"""Phase 4 -- Latency and Cost Benchmark for adaptive-agentic-semantic-cache.

===========================================================================
BENCHMARK DESIGN AND ASSUMPTIONS (READ BEFORE INTERPRETING RESULTS)
===========================================================================

LATENCY
-------
A. CACHE HIT PATH: embed query + FAISS top-1 lookup.
   - Measured by running HIT_PATH_RUNS (50) warm wall-clock timing iterations
     using the all-MiniLM-L6-v2 model loaded from the local cache (no network).
   - Reports mean, median, p95 across all runs.
   - 5 warm-up runs are discarded before timed measurement begins.

B. CACHE MISS -- LLM ROUND-TRIP (judge call):
   - Uses the 22 GENUINE (non-fallback) latency_ms values recorded in
     data/openrouter_synthetic_feedback_telemetry.json.
   - These are real wall-clock round-trip times to the OpenRouter API
     (nvidia/nemotron-3-super-120b-a12b:free), measured during Phase 4
     labeling. NOT re-measured here to preserve the 50-req/day free quota.
     Source is stated in every output field.

COST
----
Real token counts come from the 22 genuine telemetry records only.
The judge model (nvidia/nemotron-3-super-120b-a12b:free) was used on
OpenRouter's free tier -- zero dollars billed. Reporting "zero saved"
against zero-dollar cost is not a meaningful metric.

Instead this benchmark uses a NAMED PAID MODEL as an explicit stand-in:
  Model:   OpenAI gpt-4o-mini
  Input:   $0.150 per million tokens
  Output:  $0.600 per million tokens
  Source:  https://openai.com/api/pricing/  (verified 2026-09-19)

All dollar figures are labeled:
  "cost avoided under gpt-4o-mini published pricing as of 2026-09-19,
   applied to real token counts from this repo's API telemetry"

PRODUCTION PROJECTION
---------------------
Phase 3 adopted decision policy: ARR = 10.00% (12/120 pairs on benchmark).
Source: docs/phase3_final_decision.md, OpenRouter LLM Judge row.
Projection uses this ARR with a stated hypothetical query volume.
Results are labeled explicitly as PROJECTIONS, not measured results.

===========================================================================
SCOPE (Phase 4 discipline -- same invariants as every prior phase)
===========================================================================
- Does NOT touch TierRouter, ProductionDecisionStep, or src/classifier/
- HIT-path timing requires the real all-MiniLM-L6-v2 model (local cache).
  Run standalone:  python scripts/benchmark_latency_cost.py
  Or via pytest:   pytest -m requires_model tests/test_benchmark_latency_cost.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELEMETRY_PATH = REPO_ROOT / "data" / "openrouter_synthetic_feedback_telemetry.json"
OUTPUT_PATH = REPO_ROOT / "data" / "latency_cost_benchmark_results.json"

# GPT-4o-mini pricing -- verified 2026-09-19
# Source: https://openai.com/api/pricing/
PRICING_MODEL_NAME = "gpt-4o-mini"
PRICING_MODEL_URL = "https://openai.com/api/pricing/"
PRICING_DATE = "2026-09-19"
COST_PER_M_INPUT_TOKENS_USD = 0.150   # USD per 1,000,000 input tokens
COST_PER_M_OUTPUT_TOKENS_USD = 0.600  # USD per 1,000,000 output tokens

# Phase 3 production ARR -- OpenRouter LLM Judge, ARR=10.00%, 12 hits / 120 pairs
# Source: docs/phase3_final_decision.md
PHASE3_PRODUCTION_ARR = 0.10

# Stated hypothetical query volume for projection (NOT a measured number)
HYPOTHETICAL_QUERIES_PER_DAY = 10_000

# HIT-path benchmark parameters
HIT_PATH_RUNS = 50    # number of timed runs (spec requires >= 30)
HIT_PATH_WARMUP = 5   # warm-up runs, discarded
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_genuine_telemetry() -> List[Dict]:
    """Load and return only genuine (non-fallback) telemetry records.

    Excludes records where fallback_triggered=True. Those are API-error stubs
    that fail-closed to BYPASS and have no real latency or token measurements.
    Only records with a real request_id and non-null raw_response are evidence.
    """
    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Telemetry file not found at {TELEMETRY_PATH}. "
            "Run scripts/label_synthetic_feedback.py first."
        )
    with open(TELEMETRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    genuine = [r for r in data if not r.get("fallback_triggered", False)]
    if not genuine:
        raise ValueError(
            "No genuine (non-fallback) records found in telemetry. "
            "Cannot compute real latency or token statistics."
        )
    return genuine


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------

def compute_llm_latency_stats(genuine_records: List[Dict]) -> Dict:
    """Compute LLM round-trip latency statistics from real measured values.

    Args:
        genuine_records: Telemetry records with fallback_triggered=False.

    Returns:
        Dict with n, mean_ms, median_ms, p95_ms, min_ms, max_ms, stdev_ms, source.
    """
    latencies = [r["latency_ms"] for r in genuine_records]
    sorted_lats = sorted(latencies)
    n = len(latencies)
    p95_idx = min(int(0.95 * n), n - 1)

    return {
        "n": n,
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted_lats[p95_idx], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "stdev_ms": round(statistics.stdev(latencies) if n > 1 else 0.0, 2),
        "source": (
            "Real measured wall-clock round-trip times to OpenRouter API "
            "(nvidia/nemotron-3-super-120b-a12b:free) during Phase 4 labeling, "
            "recorded in data/openrouter_synthetic_feedback_telemetry.json. "
            "Not re-measured here -- API quota preserved."
        ),
    }


def benchmark_hit_path_latency(n_runs: int = HIT_PATH_RUNS) -> Dict:
    """Measure cache HIT path latency (embed + FAISS lookup) in wall-clock time.

    Loads the real all-MiniLM-L6-v2 model from local cache. Indexes one
    representative query, then times n_runs lookup operations after HIT_PATH_WARMUP
    warm-up runs.

    Args:
        n_runs: Number of timed iterations. Must be >= 30.

    Returns:
        Dict with n, mean_ms, median_ms, p95_ms, min_ms, max_ms, stdev_ms,
        model, vector_store, measurement_method.

    Raises:
        ValueError: If n_runs < 30 (spec requires >= 30 runs).
    """
    if n_runs < 30:
        raise ValueError(
            f"n_runs must be >= 30 per benchmark spec (got {n_runs})"
        )

    from src.cache.embedding import QueryEmbedder
    from src.cache.vector_store import FlatVectorStore

    embedder = QueryEmbedder()
    store = FlatVectorStore(dim=EMBEDDING_DIM)

    # Index one representative cached query (deterministic fixture)
    indexed_query = "How do you calculate the determinant of a 2x2 matrix?"
    store.add(embedder.encode(indexed_query), metadata={"query": indexed_query, "source": "benchmark"})

    # Lookup query -- semantically similar but not identical
    lookup_query = "What is the formula for the 2x2 matrix determinant?"

    # Warm-up (discarded)
    for _ in range(HIT_PATH_WARMUP):
        emb = embedder.encode(lookup_query)
        store.search(emb, k=1)

    # Timed measurement
    latencies_ms: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        emb = embedder.encode(lookup_query)
        store.search(emb, k=1)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    sorted_lats = sorted(latencies_ms)
    n = len(latencies_ms)
    p95_idx = min(int(0.95 * n), n - 1)

    return {
        "n": n,
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "p95_ms": round(sorted_lats[p95_idx], 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "stdev_ms": round(statistics.stdev(latencies_ms), 3),
        "model": "all-MiniLM-L6-v2",
        "vector_store": "FAISS IndexFlatIP (in-memory, 1 indexed entry)",
        "measurement_method": (
            f"Wall-clock time.perf_counter() over {n} runs "
            f"after {HIT_PATH_WARMUP} warm-up runs (discarded). "
            "Includes: QueryEmbedder.encode() + FlatVectorStore.search() (top-1 FAISS). "
            "Excludes: model load time (model already warm)."
        ),
    }


def compute_latency_delta(hit_stats: Dict, miss_stats: Dict) -> Dict:
    """Compute latency reduction of a cache hit vs. a cold LLM call.

    Args:
        hit_stats: From benchmark_hit_path_latency() -- local measured.
        miss_stats: From compute_llm_latency_stats() -- real telemetry.

    Returns:
        Dict with mean/median deltas, percentage reductions, and summary string.
    """
    delta_mean = miss_stats["mean_ms"] - hit_stats["mean_ms"]
    delta_median = miss_stats["median_ms"] - hit_stats["median_ms"]
    pct_mean = (delta_mean / miss_stats["mean_ms"]) * 100 if miss_stats["mean_ms"] > 0 else 0.0
    pct_median = (delta_median / miss_stats["median_ms"]) * 100 if miss_stats["median_ms"] > 0 else 0.0

    return {
        "hit_mean_ms": hit_stats["mean_ms"],
        "miss_mean_ms": round(miss_stats["mean_ms"], 1),
        "delta_mean_ms": round(delta_mean, 1),
        "pct_reduction_mean": round(pct_mean, 1),
        "hit_median_ms": hit_stats["median_ms"],
        "miss_median_ms": round(miss_stats["median_ms"], 1),
        "delta_median_ms": round(delta_median, 1),
        "pct_reduction_median": round(pct_median, 1),
        "summary": (
            f"A cache hit takes ~{hit_stats['mean_ms']:.1f}ms (mean) vs "
            f"~{miss_stats['mean_ms']:.0f}ms for a fresh LLM judge call -- "
            f"a {pct_mean:.0f}% or {delta_mean:.0f}ms reduction (mean). "
            f"Median: {hit_stats['median_ms']:.1f}ms vs {miss_stats['median_ms']:.0f}ms "
            f"({pct_median:.0f}% reduction). "
            "LLM latency derived from real measured telemetry. "
            "Cache hit latency measured locally on this machine."
        ),
    }


# ---------------------------------------------------------------------------
# Token and cost statistics
# ---------------------------------------------------------------------------

def compute_token_stats(genuine_records: List[Dict]) -> Dict:
    """Compute token usage statistics from real measured API calls.

    Args:
        genuine_records: Telemetry records with fallback_triggered=False.

    Returns:
        Dict with per-call and aggregate token counts, plus source description.
    """
    prompt_tokens = [r["prompt_tokens"] for r in genuine_records]
    completion_tokens = [r["completion_tokens"] for r in genuine_records]
    total_tokens = [r["total_tokens"] for r in genuine_records]
    n = len(genuine_records)

    return {
        "n_calls": n,
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 1),
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 1),
        "mean_total_tokens": round(statistics.mean(total_tokens), 1),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "total_tokens_all_calls": sum(total_tokens),
        "source": (
            f"Real measured prompt/completion token counts from {n} genuine OpenRouter "
            "API calls recorded in data/openrouter_synthetic_feedback_telemetry.json. "
            "86 fallback records (API errors) are excluded -- they have no real token counts."
        ),
    }


def compute_cost_projection(token_stats: Dict) -> Dict:
    """Project cost avoidance using named paid model pricing on real token counts.

    The judge model was used on the free tier (zero dollars billed).
    This function applies a named paid model's published pricing to the real
    token counts as an explicit proxy for what cost would have been incurred
    on a comparable paid model. All results are labeled accordingly.

    Args:
        token_stats: From compute_token_stats().

    Returns:
        Dict with per-call cost, aggregate cost, projection label, and source.
    """
    mean_prompt = token_stats["mean_prompt_tokens"]
    mean_completion = token_stats["mean_completion_tokens"]

    # Cost per single judge call (mean token usage)
    cost_per_call_input_usd = (mean_prompt / 1_000_000) * COST_PER_M_INPUT_TOKENS_USD
    cost_per_call_output_usd = (mean_completion / 1_000_000) * COST_PER_M_OUTPUT_TOKENS_USD
    cost_per_call_total_usd = cost_per_call_input_usd + cost_per_call_output_usd

    # Aggregate cost for all measured genuine calls
    total_prompt = token_stats["total_prompt_tokens"]
    total_completion = token_stats["total_completion_tokens"]
    aggregate_cost_usd = (
        (total_prompt / 1_000_000) * COST_PER_M_INPUT_TOKENS_USD
        + (total_completion / 1_000_000) * COST_PER_M_OUTPUT_TOKENS_USD
    )

    # Production projection: at Phase 3 ARR, each cache HIT avoids one judge call
    daily_hits = HYPOTHETICAL_QUERIES_PER_DAY * PHASE3_PRODUCTION_ARR
    daily_cost_avoided_usd = daily_hits * cost_per_call_total_usd

    label = (
        f"Cost avoided under {PRICING_MODEL_NAME}'s published pricing as of {PRICING_DATE} "
        f"({PRICING_MODEL_URL}), applied to real token counts from this repo's API telemetry "
        f"({token_stats['n_calls']} genuine judge calls in "
        f"data/openrouter_synthetic_feedback_telemetry.json). "
        "NOT an unqualified savings figure: the actual model "
        "(nvidia/nemotron-3-super-120b-a12b:free) was used on the free tier (zero dollars billed)."
    )
    projection_label = (
        f"PROJECTION (not a measured result): at {HYPOTHETICAL_QUERIES_PER_DAY:,} queries/day "
        f"and Phase 3 ARR={PHASE3_PRODUCTION_ARR * 100:.0f}% "
        f"(source: docs/phase3_final_decision.md), applying the {PRICING_MODEL_NAME} "
        "pricing assumption stated above. Every cache HIT avoids one judge call."
    )

    return {
        "pricing_model": PRICING_MODEL_NAME,
        "pricing_url": PRICING_MODEL_URL,
        "pricing_date": PRICING_DATE,
        "cost_per_m_input_tokens_usd": COST_PER_M_INPUT_TOKENS_USD,
        "cost_per_m_output_tokens_usd": COST_PER_M_OUTPUT_TOKENS_USD,
        "cost_per_judge_call_input_usd": round(cost_per_call_input_usd, 8),
        "cost_per_judge_call_output_usd": round(cost_per_call_output_usd, 8),
        "cost_per_judge_call_total_usd": round(cost_per_call_total_usd, 8),
        "aggregate_cost_for_measured_calls_usd": round(aggregate_cost_usd, 6),
        "label": label,
        "production_projection": {
            "hypothetical_queries_per_day": HYPOTHETICAL_QUERIES_PER_DAY,
            "phase3_arr": PHASE3_PRODUCTION_ARR,
            "arr_source": "docs/phase3_final_decision.md, OpenRouter LLM Judge, ARR=10.00%",
            "daily_cache_hits_projected": daily_hits,
            "daily_cost_avoided_usd": round(daily_cost_avoided_usd, 4),
            "label": projection_label,
        },
    }


# ---------------------------------------------------------------------------
# Top-level benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(include_hit_path: bool = True) -> Dict:
    """Run the complete latency and cost benchmark.

    Args:
        include_hit_path: If True, measure HIT-path latency (requires real model).
                          Set False to skip model loading (e.g., in unit tests that
                          only test computation logic, not wall-clock timing).

    Returns:
        Complete benchmark results dict.
    """
    print("=" * 80)
    print("LATENCY & COST BENCHMARK -- adaptive-agentic-semantic-cache")
    print("=" * 80)
    print(f"\nData source:   {TELEMETRY_PATH}")
    print(f"Pricing model: {PRICING_MODEL_NAME} ({PRICING_MODEL_URL})")
    print(f"Pricing date:  {PRICING_DATE}")

    # Load genuine telemetry
    print("\n[1/4] Loading genuine telemetry records...")
    genuine = load_genuine_telemetry()
    print(f"  Found {len(genuine)} genuine records ({108 - len(genuine)} fallback stubs excluded)")

    # LLM round-trip latency (real measured)
    print("\n[2/4] LLM round-trip latency (from real telemetry, not re-measured)...")
    llm_latency = compute_llm_latency_stats(genuine)
    print(f"  N={llm_latency['n']}  mean={llm_latency['mean_ms']:.1f}ms  "
          f"median={llm_latency['median_ms']:.1f}ms  p95={llm_latency['p95_ms']:.1f}ms")

    # Token usage (real measured)
    print("\n[3/4] Token usage statistics (from real telemetry)...")
    token_stats = compute_token_stats(genuine)
    print(f"  N={token_stats['n_calls']} calls  "
          f"mean_prompt={token_stats['mean_prompt_tokens']:.0f}  "
          f"mean_completion={token_stats['mean_completion_tokens']:.0f}  "
          f"mean_total={token_stats['mean_total_tokens']:.0f} tokens")

    # Cost projection
    cost = compute_cost_projection(token_stats)
    print(f"  Per-call cost ({PRICING_MODEL_NAME} pricing): "
          f"${cost['cost_per_judge_call_total_usd']:.6f}")
    proj = cost["production_projection"]
    print(f"  Daily cost avoided ({proj['hypothetical_queries_per_day']:,} q/day, "
          f"ARR={proj['phase3_arr']*100:.0f}%): ${proj['daily_cost_avoided_usd']:.4f} "
          f"[PROJECTION -- see label]")

    # HIT-path timing (wall-clock measurement)
    hit_latency: Optional[Dict] = None
    latency_delta: Optional[Dict] = None

    if include_hit_path:
        print(f"\n[4/4] Cache HIT path timing ({HIT_PATH_RUNS} runs after {HIT_PATH_WARMUP} warm-up)...")
        print("  Loading all-MiniLM-L6-v2 model...")
        hit_latency = benchmark_hit_path_latency(n_runs=HIT_PATH_RUNS)
        print(f"  mean={hit_latency['mean_ms']:.2f}ms  "
              f"median={hit_latency['median_ms']:.2f}ms  "
              f"p95={hit_latency['p95_ms']:.2f}ms")
        latency_delta = compute_latency_delta(hit_latency, llm_latency)
        print(f"\nLatency delta: {latency_delta['summary']}")
    else:
        print("\n[4/4] HIT-path timing skipped (include_hit_path=False).")

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": str(TELEMETRY_PATH),
        "genuine_records_used": len(genuine),
        "excluded_fallback_records": 108 - len(genuine),
        "llm_miss_path_latency": llm_latency,
        "token_usage": token_stats,
        "cost_projection": cost,
        "cache_hit_path_latency": hit_latency,
        "latency_delta": latency_delta,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {OUTPUT_PATH}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmark(include_hit_path=True)
