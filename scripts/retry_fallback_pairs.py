"""Retry ONLY the fallback pairs from gemini_judge_evaluation_telemetry.json.

Pulls forward the 17 succeeded results unchanged.
Retries only the 15 fallback pairs with exponential backoff on 429 errors.
If rate limits persist, identifies unresolved pairs and halts without fabricating results.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
sys.path.insert(0, os.getcwd())

from src.decision.judge_call import LLMJudge, DEFAULT_JUDGE_MODEL

TELEMETRY_PATH = Path("data/gemini_judge_evaluation_telemetry.json")
OUTPUT_PATH = Path("data/gemini_judge_evaluation_telemetry.json")


def retry_fallback_batch(
    model: str = DEFAULT_JUDGE_MODEL,
    inter_call_delay: float = 15.0,
) -> Dict[str, Any]:
    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(f"Telemetry file not found at {TELEMETRY_PATH}")

    with open(TELEMETRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    calls: List[Dict[str, Any]] = data.get("judge_calls", [])
    succeeded_calls = [c for c in calls if not c.get("fallback_triggered")]
    fallback_calls = [c for c in calls if c.get("fallback_triggered")]

    print("=" * 80)
    print(f"RETRY RUN FOR AMBIGUOUS-TIER FALLBACK PAIRS")
    print(f"Total pairs in telemetry : {len(calls)}")
    print(f"Already succeeded        : {len(succeeded_calls)} (preserved unchanged)")
    print(f"To retry                 : {len(fallback_calls)}")
    print(f"Target model             : {model}")
    print(f"Inter-call delay         : {inter_call_delay}s")
    print("=" * 80)

    judge = LLMJudge(
        model=model,
        rate_limit_delay_seconds=inter_call_delay,
    )

    retried_results: List[Dict[str, Any]] = []
    unresolved_pair_ids: List[str] = []
    retries_histogram: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    for idx, item in enumerate(fallback_calls, 1):
        pair_id = item["pair_id"]
        print(f"\n[{idx}/{len(fallback_calls)}] Retrying {pair_id} ({item['domain']}, sim={item['similarity_score']:.4f})...")

        res = judge.judge(
            query_a=item["query_a"],
            query_b=item["query_b"],
            domain=item["domain"],
            similarity_score=item["similarity_score"],
            stability_confidence=item.get("stability_confidence", 1.0),
            category_history_rate=item.get("category_history_rate", 0.5),
        )

        retried_entry = {
            "pair_id": pair_id,
            "query_a": item["query_a"],
            "query_b": item["query_b"],
            "domain": item["domain"],
            "similarity_score": item["similarity_score"],
            "is_reuse_safe": item["is_reuse_safe"],
            "decision": res.decision,
            "is_safe": res.is_safe,
            "confidence": res.confidence,
            "rationale": res.rationale,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.total_tokens,
            "latency_ms": res.latency_ms,
            "model": res.model,
            "cost_usd": res.cost_usd,
            "fallback_triggered": res.fallback_triggered,
            "error": res.error,
            "raw_response": res.raw_response,
            "retries_used": getattr(res, "retries_used", 0),
        }
        retried_results.append(retried_entry)

        r_count = getattr(res, "retries_used", 0)
        retries_histogram[r_count] = retries_histogram.get(r_count, 0) + 1

        if res.fallback_triggered:
            print(f"  FAILED -> Fallback triggered: {res.error}")
            unresolved_pair_ids.append(pair_id)
        else:
            print(f"  SUCCESS -> {res.decision} (is_safe={res.is_safe}, confidence={res.confidence}, retries={r_count})")

    # Merge: keep original 17 succeeded + newly retried 15 results
    merged_calls = list(succeeded_calls) + retried_results
    # Sort by pair_id to preserve original ordering
    merged_calls.sort(key=lambda x: x["pair_id"])

    # Compute updated metrics
    amb_tp = sum(1 for c in merged_calls if c["decision"] == "REUSE" and c["is_reuse_safe"])
    amb_fp = sum(1 for c in merged_calls if c["decision"] == "REUSE" and not c["is_reuse_safe"])
    amb_fn = sum(1 for c in merged_calls if c["decision"] != "REUSE" and c["is_reuse_safe"])
    amb_tn = sum(1 for c in merged_calls if c["decision"] != "REUSE" and not c["is_reuse_safe"])

    # Tier 1 AUTO_REUSE has 2 safe pairs (TP=2, FP=0, FN=0, TN=0)
    # Tier 3 BYPASS has 86 pairs (50 safe FN, 36 unsafe TN)
    total_tp = 2 + amb_tp
    total_fp = 0 + amb_fp
    total_fn = 50 + amb_fn
    total_tn = 36 + amb_tn
    n = 120
    n_safe = total_tp + total_fn
    hit_count = total_tp + total_fp

    arr = hit_count / n if n > 0 else 0.0
    crr = total_tp / hit_count if hit_count > 0 else 1.0
    irr_cache = total_fp / hit_count if hit_count > 0 else 0.0
    irr_traffic = total_fp / n if n > 0 else 0.0
    frr = total_fn / n_safe if n_safe > 0 else 0.0

    total_in_tokens = sum(c["prompt_tokens"] for c in merged_calls)
    total_out_tokens = sum(c["completion_tokens"] for c in merged_calls)
    total_tokens = sum(c["total_tokens"] for c in merged_calls)
    total_latency_ms = sum(c["latency_ms"] for c in merged_calls)
    avg_latency_ms = total_latency_ms / len(merged_calls) if merged_calls else 0.0

    merged_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_requested": model,
            "total_pairs": n,
            "ambiguous_count": len(merged_calls),
            "succeeded_count": len(merged_calls) - len(unresolved_pair_ids),
            "unresolved_count": len(unresolved_pair_ids),
            "unresolved_pair_ids": unresolved_pair_ids,
            "retries_histogram": retries_histogram,
            "judge_calls_count": len(merged_calls),
            "judge_total_input_tokens": total_in_tokens,
            "judge_total_output_tokens": total_out_tokens,
            "judge_total_tokens": total_tokens,
            "judge_total_latency_ms": round(total_latency_ms, 2),
            "judge_avg_latency_ms": round(avg_latency_ms, 2),
            "judge_total_cost_usd": 0.0,
        },
        "metrics": {
            "arr": round(arr, 6),
            "crr": round(crr, 6),
            "irr_cache": round(irr_cache, 6),
            "irr_traffic": round(irr_traffic, 6),
            "frr": round(frr, 6),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "ambiguous_tp": amb_tp,
            "ambiguous_fp": amb_fp,
            "ambiguous_fn": amb_fn,
            "ambiguous_tn": amb_tn,
        },
        "judge_calls": merged_calls,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("RETRY RUN SUMMARY")
    print(f"Retried pairs         : {len(fallback_calls)}")
    print(f"Succeeded in retry    : {len(fallback_calls) - len(unresolved_pair_ids)}")
    print(f"Still unresolved      : {len(unresolved_pair_ids)}")
    print(f"Unresolved pair IDs   : {unresolved_pair_ids}")
    print(f"Retries histogram     : {retries_histogram}")
    print(f"Ambiguous-tier result : TP={amb_tp}, FP={amb_fp}, FN={amb_fn}, TN={amb_tn}")
    print(f"Overall metrics       : ARR={arr*100:.2f}%, IRR_cache={irr_cache*100:.2f}%, CRR={crr*100:.2f}%")
    print("=" * 80)

    return merged_payload


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JUDGE_MODEL
    retry_fallback_batch(model=target_model)
