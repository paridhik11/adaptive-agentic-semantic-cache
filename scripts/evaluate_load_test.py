"""Phase 5 — Load Test Metrics Evaluator.

Computes empirical metrics from data/load_test_telemetry.json:
  1. Hit rate (overall and per category).
  2. Latency by path (AUTO_REUSE, AMBIGUOUS, BYPASS) with mean, median, p95, min, max, stdev.
  3. Tokens and cost avoided under gpt-4o-mini published pricing as of 2026-09-19.
  4. Incorrect-reuse rate (IRR_cache) on HIT decisions with exact Clopper-Pearson 95% CI.
  5. Ambiguous-tier judge invocation rate.

All numbers derive strictly from real run telemetry or named/dated pricing sources.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.decision.adaptive_threshold_engine import clopper_pearson_ci

TELEMETRY_PATH = REPO_ROOT / "data" / "load_test_telemetry.json"
SUMMARY_PATH = REPO_ROOT / "data" / "load_test_metrics_summary.json"

# GPT-4o-mini pricing -- verified 2026-09-19
# Source: https://openai.com/api/pricing/
PRICING_MODEL_NAME = "gpt-4o-mini"
PRICING_MODEL_URL = "https://openai.com/api/pricing/"
PRICING_DATE = "2026-09-19"
COST_PER_M_INPUT_TOKENS_USD = 0.150   # USD per 1,000,000 input tokens
COST_PER_M_OUTPUT_TOKENS_USD = 0.600  # USD per 1,000,000 output tokens


def calculate_latency_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, p95, min, max, stdev for a list of latency values."""
    if not values:
        return {
            "n": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stdev_ms": 0.0,
        }
    n = len(values)
    sorted_vals = sorted(values)
    p95_idx = min(int(0.95 * n), n - 1)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(sorted_vals[p95_idx], 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "stdev_ms": round(statistics.stdev(values) if n > 1 else 0.0, 3),
    }


def evaluate_telemetry(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive Phase 5 metrics from real telemetry records."""
    total_queries = len(records)
    if total_queries == 0:
        raise ValueError("No telemetry records to evaluate.")

    # 1. Overall and per-tier counts
    hits = [r for r in records if r["final_decision"] == "HIT"]
    misses = [r for r in records if r["final_decision"] == "MISS"]

    auto_reuse_records = [r for r in records if r["path"] == "AUTO_REUSE"]
    ambiguous_records = [r for r in records if r["path"] == "AMBIGUOUS"]
    bypass_records = [r for r in records if r["path"] == "BYPASS"]

    # 2. Per-category breakdowns
    domain_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        domain_groups.setdefault(r["domain"], []).append(r)

    category_metrics: Dict[str, Any] = {}
    for dom, d_records in sorted(domain_groups.items()):
        d_hits = [r for r in d_records if r["final_decision"] == "HIT"]
        d_misses = [r for r in d_records if r["final_decision"] == "MISS"]
        d_auto = [r for r in d_records if r["path"] == "AUTO_REUSE"]
        d_amb = [r for r in d_records if r["path"] == "AMBIGUOUS"]
        d_byp = [r for r in d_records if r["path"] == "BYPASS"]
        category_metrics[dom] = {
            "total_queries": len(d_records),
            "hits": len(d_hits),
            "misses": len(d_misses),
            "hit_rate_pct": round(len(d_hits) / len(d_records) * 100.0, 2),
            "path_counts": {
                "AUTO_REUSE": len(d_auto),
                "AMBIGUOUS": len(d_amb),
                "BYPASS": len(d_byp),
            },
        }

    # 3. Latency statistics by path
    auto_lats = [r["timings_ms"]["total_pipeline_ms"] for r in auto_reuse_records]
    amb_lats = [r["timings_ms"]["total_pipeline_ms"] for r in ambiguous_records]
    byp_lats = [r["timings_ms"]["total_pipeline_ms"] for r in bypass_records]

    auto_stats = calculate_latency_percentiles(auto_lats)
    amb_stats = calculate_latency_percentiles(amb_lats)
    byp_stats = calculate_latency_percentiles(byp_lats)

    # Component breakdown for ambiguous path
    amb_judge_lats = [r["timings_ms"]["judge_ms"] for r in ambiguous_records if r["timings_ms"]["judge_ms"] > 0]
    amb_judge_stats = calculate_latency_percentiles(amb_judge_lats)

    # Latency delta: AUTO_REUSE vs AMBIGUOUS
    delta_mean_ms = amb_stats["mean_ms"] - auto_stats["mean_ms"] if (amb_stats["mean_ms"] > 0 and auto_stats["mean_ms"] > 0) else 0.0
    pct_reduction_mean = (delta_mean_ms / amb_stats["mean_ms"] * 100.0) if amb_stats["mean_ms"] > 0 else 0.0

    # 4. Token counts and cost calculations
    judge_calls = [r for r in records if r["judge_telemetry"] is not None]
    total_prompt_tokens = sum(r["judge_telemetry"]["prompt_tokens"] for r in judge_calls)
    total_completion_tokens = sum(r["judge_telemetry"]["completion_tokens"] for r in judge_calls)
    total_tokens = sum(r["judge_telemetry"]["total_tokens"] for r in judge_calls)

    cost_avoided_usd = (
        (total_prompt_tokens / 1_000_000.0) * COST_PER_M_INPUT_TOKENS_USD
        + (total_completion_tokens / 1_000_000.0) * COST_PER_M_OUTPUT_TOKENS_USD
    )

    # 5. Safety and Hazard Metrics (IRR_cache on HITs with Clopper-Pearson CI)
    tp_count = sum(1 for r in records if r["outcome_type"] == "TP")
    fp_count = sum(1 for r in records if r["outcome_type"] == "FP")
    tn_count = sum(1 for r in records if r["outcome_type"] == "TN")
    fn_count = sum(1 for r in records if r["outcome_type"] == "FN")

    hit_n = len(hits)
    irr_cache = (fp_count / hit_n) if hit_n > 0 else 0.0
    ci_lower, ci_upper = clopper_pearson_ci(k=fp_count, n=hit_n, confidence_level=0.95) if hit_n > 0 else (0.0, 0.0)

    # Sample size power disclosure
    is_underpowered = hit_n < 36
    power_note = (
        f"Sample size n={hit_n} hits < 36. Exact Clopper-Pearson 95% CI upper bound ({ci_upper:.1%}) "
        "cannot mathematically clear <= 10.0% under finite sample theory (Phase 4 Section 8 mathematical floor: "
        "n >= 36 with k=0 required to prove CI upper <= 10.0%). Reported plainly without artificial suppression."
        if is_underpowered
        else f"Sample size n={hit_n} >= 36; statistical power adequate."
    )

    # 6. Ambiguous-Tier Invocation Rate
    amb_invocation_rate = len(ambiguous_records) / total_queries
    local_resolution_rate = (len(auto_reuse_records) + len(bypass_records)) / total_queries

    summary = {
        "metadata": {
            "total_queries": total_queries,
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_file": str(TELEMETRY_PATH),
            "disclaimer": "Metrics computed from a single synthetic load-test run, not live customer traffic.",
        },
        "hit_rate_metrics": {
            "total_queries": total_queries,
            "total_hits": len(hits),
            "total_misses": len(misses),
            "overall_hit_rate_pct": round(len(hits) / total_queries * 100.0, 2),
            "category_breakdown": category_metrics,
        },
        "path_distribution": {
            "auto_reuse_count": len(auto_reuse_records),
            "auto_reuse_pct": round(len(auto_reuse_records) / total_queries * 100.0, 2),
            "ambiguous_count": len(ambiguous_records),
            "ambiguous_pct": round(len(ambiguous_records) / total_queries * 100.0, 2),
            "bypass_count": len(bypass_records),
            "bypass_pct": round(len(bypass_records) / total_queries * 100.0, 2),
            "local_resolution_pct": round(local_resolution_rate * 100.0, 2),
        },
        "latency_metrics": {
            "auto_reuse_path": auto_stats,
            "ambiguous_judge_path": amb_stats,
            "bypass_path": byp_stats,
            "judge_call_component": amb_judge_stats,
            "delta_auto_vs_ambiguous": {
                "delta_mean_ms": round(delta_mean_ms, 3),
                "pct_reduction_mean": round(pct_reduction_mean, 2),
                "speedup_factor": round(amb_stats["mean_ms"] / auto_stats["mean_ms"], 1) if auto_stats["mean_ms"] > 0 else 0.0,
            },
        },
        "token_and_cost_metrics": {
            "judge_calls_count": len(judge_calls),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "mean_tokens_per_call": round(total_tokens / len(judge_calls), 1) if judge_calls else 0.0,
            "cost_avoided_projection_usd": round(cost_avoided_usd, 6),
            "actual_billed_usd": 0.0,
            "pricing_reference": {
                "model_name": PRICING_MODEL_NAME,
                "pricing_date": PRICING_DATE,
                "input_cost_per_m_usd": COST_PER_M_INPUT_TOKENS_USD,
                "output_cost_per_m_usd": COST_PER_M_OUTPUT_TOKENS_USD,
                "source_url": PRICING_MODEL_URL,
                "label": (
                    f"Cost avoided under {PRICING_MODEL_NAME} published pricing as of {PRICING_DATE}, "
                    "applied to real token counts from this repo's API telemetry. PROJECTION ONLY — "
                    "OpenRouter free tier billed $0.00."
                ),
            },
        },
        "safety_metrics": {
            "tp": tp_count,
            "fp": fp_count,
            "tn": tn_count,
            "fn": fn_count,
            "hit_denominator": hit_n,
            "irr_cache": round(irr_cache, 4),
            "irr_cache_pct": round(irr_cache * 100.0, 2),
            "clopper_pearson_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
            "clopper_pearson_95_ci_pct": [round(ci_lower * 100.0, 2), round(ci_upper * 100.0, 2)],
            "is_underpowered": is_underpowered,
            "power_disclosure": power_note,
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 load test metrics.")
    parser.add_argument("--telemetry", type=Path, default=TELEMETRY_PATH, help="Path to telemetry JSON")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH, help="Path to output summary JSON")

    args = parser.parse_args()

    if not args.telemetry.exists():
        print(f"Error: Telemetry file {args.telemetry} not found. Run scripts/run_load_test.py first.")
        sys.exit(1)

    with open(args.telemetry, "r", encoding="utf-8") as f:
        records = json.load(f)

    summary = evaluate_telemetry(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("PHASE 5 LOAD TEST METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Queries: {summary['hit_rate_metrics']['total_queries']}")
    print(f"Overall Hit Rate: {summary['hit_rate_metrics']['overall_hit_rate_pct']}% ({summary['hit_rate_metrics']['total_hits']} hits)")
    print(f"Path Distribution: AUTO_REUSE={summary['path_distribution']['auto_reuse_count']} ({summary['path_distribution']['auto_reuse_pct']}%), "
          f"AMBIGUOUS={summary['path_distribution']['ambiguous_count']} ({summary['path_distribution']['ambiguous_pct']}%), "
          f"BYPASS={summary['path_distribution']['bypass_count']} ({summary['path_distribution']['bypass_pct']}%)")
    print(f"Local Resolution Rate: {summary['path_distribution']['local_resolution_pct']}%")
    print("\n--- Latency Breakdown ---")
    print(f"  AUTO_REUSE Path : mean={summary['latency_metrics']['auto_reuse_path']['mean_ms']}ms, median={summary['latency_metrics']['auto_reuse_path']['median_ms']}ms, p95={summary['latency_metrics']['auto_reuse_path']['p95_ms']}ms")
    print(f"  AMBIGUOUS Path  : mean={summary['latency_metrics']['ambiguous_judge_path']['mean_ms']}ms, median={summary['latency_metrics']['ambiguous_judge_path']['median_ms']}ms, p95={summary['latency_metrics']['ambiguous_judge_path']['p95_ms']}ms")
    print(f"  BYPASS Path     : mean={summary['latency_metrics']['bypass_path']['mean_ms']}ms, median={summary['latency_metrics']['bypass_path']['median_ms']}ms, p95={summary['latency_metrics']['bypass_path']['p95_ms']}ms")
    print(f"  Speedup Factor  : {summary['latency_metrics']['delta_auto_vs_ambiguous']['speedup_factor']}x ({summary['latency_metrics']['delta_auto_vs_ambiguous']['pct_reduction_mean']}% faster)")
    print("\n--- Tokens & Cost Avoided ---")
    print(f"  Judge Calls: {summary['token_and_cost_metrics']['judge_calls_count']}")
    print(f"  Total Tokens: {summary['token_and_cost_metrics']['total_tokens']} (prompt={summary['token_and_cost_metrics']['total_prompt_tokens']}, comp={summary['token_and_cost_metrics']['total_completion_tokens']})")
    print(f"  Cost Avoided: ${summary['token_and_cost_metrics']['cost_avoided_projection_usd']:.6f} ({summary['token_and_cost_metrics']['pricing_reference']['label']})")
    print(f"  Actual Billed: ${summary['token_and_cost_metrics']['actual_billed_usd']:.6f} (OpenRouter free tier)")
    print("\n--- Safety & Cache Hazard ---")
    print(f"  TP={summary['safety_metrics']['tp']}, FP={summary['safety_metrics']['fp']}, TN={summary['safety_metrics']['tn']}, FN={summary['safety_metrics']['fn']}")
    print(f"  IRR_cache: {summary['safety_metrics']['irr_cache_pct']}% (Exact Clopper-Pearson 95% CI: {summary['safety_metrics']['clopper_pearson_95_ci_pct'][0]}% - {summary['safety_metrics']['clopper_pearson_95_ci_pct'][1]}%)")
    print(f"  Power Note: {summary['safety_metrics']['power_disclosure']}")
    print("=" * 80)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
