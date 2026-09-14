"""Run Phase 3 Gemini LLM Judge Evaluation on query_pair_reuse_benchmark.json.

Evaluates the 32 ambiguous-tier query pairs with live Gemini calls,
respecting the 15 RPM free-tier rate limit with a 4.0s inter-call delay.
Dumps full telemetry and raw response texts to artifacts for verification.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.getcwd())

from src.decision.judge_call import LLMJudge, DEFAULT_JUDGE_MODEL
from src.evaluation.decision_evaluator import DecisionEvaluator

DATASET_PATH = Path("data/raw/query_pair_reuse_benchmark.json")
OUTPUT_TELEMETRY_PATH = Path("data/gemini_judge_evaluation_telemetry.json")


def main() -> None:
    print("=" * 80)
    print("STARTING PHASE 3 GEMINI LIVE BENCHMARK EVALUATION")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {DEFAULT_JUDGE_MODEL} (Google AI Studio Free Tier)")
    print("Rate-limit pacing: 4.0s delay between calls (15 RPM limit)")
    print("=" * 80)

    # Initialize live judge with 4.0s rate limit pacing
    judge = LLMJudge(
        model=DEFAULT_JUDGE_MODEL,
        rate_limit_delay_seconds=4.0,
    )

    evaluator = DecisionEvaluator(
        judge=judge,
    )

    t_start = time.perf_counter()
    metrics = evaluator.evaluate(
        dataset_path=DATASET_PATH,
        bypass_all_ambiguous=False,
        run_judge=True,
    )
    total_duration = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Total run wall-clock duration: {total_duration:.2f} s")
    print("=" * 80 + "\n")

    # Print standard report
    evaluator.print_report(metrics)

    # Filter ambiguous-tier pairs evaluated by judge
    judge_calls_data = []
    for r in metrics.detailed_results:
        if r.get("tier") == "AMBIGUOUS" and "judge_step" in r:
            js = r["judge_step"]
            jr = js.get("judge_result", {})
            judge_calls_data.append({
                "pair_id": r["id"],
                "query_a": r["query_a"],
                "query_b": r["query_b"],
                "domain": r["domain"],
                "similarity_score": r["similarity_score"],
                "is_reuse_safe": r["is_reuse_safe"],
                "decision": js.get("decision"),
                "is_safe": jr.get("is_safe"),
                "confidence": jr.get("confidence"),
                "rationale": jr.get("rationale"),
                "prompt_tokens": jr.get("prompt_tokens"),
                "completion_tokens": jr.get("completion_tokens"),
                "total_tokens": jr.get("total_tokens"),
                "latency_ms": jr.get("latency_ms"),
                "model": jr.get("model"),
                "cost_usd": jr.get("cost_usd"),
                "fallback_triggered": jr.get("fallback_triggered"),
                "error": jr.get("error"),
                "raw_response": jr.get("raw_response"),
            })

    output_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_requested": DEFAULT_JUDGE_MODEL,
            "total_pairs": metrics.total_pairs,
            "ambiguous_count": metrics.ambiguous_count,
            "judge_calls_count": metrics.judge_calls_count,
            "judge_total_input_tokens": metrics.judge_total_input_tokens,
            "judge_total_output_tokens": metrics.judge_total_output_tokens,
            "judge_total_tokens": metrics.judge_total_tokens,
            "judge_total_latency_ms": metrics.judge_total_latency_ms,
            "judge_avg_latency_ms": metrics.judge_avg_latency_ms,
            "judge_total_cost_usd": metrics.judge_total_cost_usd,
            "total_benchmark_duration_s": round(total_duration, 2),
        },
        "metrics": {
            "arr": metrics.judge_arr,
            "crr": metrics.judge_crr,
            "irr_cache": metrics.judge_irr_cache,
            "irr_traffic": metrics.judge_irr_traffic,
            "frr": metrics.judge_frr,
            "tp": metrics.judge_tp,
            "fp": metrics.judge_fp,
            "fn": metrics.judge_fn,
            "tn": metrics.judge_tn,
            "ambiguous_tp": metrics.judge_ambiguous_tp,
            "ambiguous_fp": metrics.judge_ambiguous_fp,
            "ambiguous_fn": metrics.judge_ambiguous_fn,
            "ambiguous_tn": metrics.judge_ambiguous_tn,
        },
        "judge_calls": judge_calls_data,
    }

    OUTPUT_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TELEMETRY_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved detailed telemetry JSON to {OUTPUT_TELEMETRY_PATH}")


if __name__ == "__main__":
    main()
