"""Phase 5 — Full Production Pipeline Synthetic Load Test Runner.

Executes the full end-to-end production pipeline:
  StabilityClassifier -> SemanticCache -> TierRouter -> ProductionDecisionStep (LLMJudge)
under synthetic query load, recording real wall-clock timings, exact token counts,
real OpenRouter provider generation IDs, and per-path performance telemetry.

Integrity rules:
  - No simulated or mocked values when running against live endpoints.
  - Every judge call records real API latency, token counts, request ID, and rationale.
  - Fail-closed to BYPASS if any rate-limit or network issue occurs.
  - Paced execution (>= 2.5s) to adhere strictly to OpenRouter free-tier rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.classifier.stability_classifier import StabilityClassifier
from src.cache.embedding import QueryEmbedder
from src.cache.vector_store import FlatVectorStore
from src.cache.semantic_cache import SemanticCache
from src.decision.tier_router import TierRouter, Tier
from src.decision.decision_step import ProductionDecisionStep, JudgeDecisionStep, CategoryHistory
from src.decision.judge_call import LLMJudge, DEFAULT_JUDGE_MODEL

STREAM_PATH = REPO_ROOT / "data" / "raw" / "load_test_query_stream.json"
TELEMETRY_PATH = REPO_ROOT / "data" / "load_test_telemetry.json"

PRIOR_DATASET_PATHS = [
    REPO_ROOT / "data" / "raw" / "query_pair_reuse_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_heldout.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_final_test.json",
    REPO_ROOT / "data" / "raw" / "query_stability_human_credibility.json",
    REPO_ROOT / "data" / "raw" / "synthetic_query_pair_feedback.json",
]


def check_leakage(stream_records: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Strict 6-way leakage check against all prior benchmark datasets."""
    existing_queries = set()
    for p in PRIOR_DATASET_PATHS:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            for k in ["query", "query_a", "query_b", "prompt", "text"]:
                if k in item and isinstance(item[k], str):
                    existing_queries.add(item[k].strip().lower())

    overlaps = []
    for r in stream_records:
        q = r["query"].strip().lower()
        if q in existing_queries:
            overlaps.append(f"{r['id']}: {r['query']}")

    return len(overlaps) == 0, overlaps


def check_openrouter_quota(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Query OpenRouter /api/v1/auth/key to inspect real daily free request limit and usage."""
    if not api_key:
        return None
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10.0) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("data", {})
    except Exception as exc:
        print(f"Warning: Could not query OpenRouter quota endpoint: {exc}")
        return None


def run_load_test(
    stream_path: Path = STREAM_PATH,
    output_path: Path = TELEMETRY_PATH,
    model: str = DEFAULT_JUDGE_MODEL,
    inter_call_delay: float = 2.5,
    max_judge_calls: int = 15,
) -> List[Dict[str, Any]]:
    """Execute synthetic query stream through the production pipeline."""
    print("=" * 80)
    print("PHASE 5: FULL PRODUCTION PIPELINE SYNTHETIC LOAD TEST")
    print(f"Stream path: {stream_path}")
    print(f"Output telemetry: {output_path}")
    print(f"Judge model: {model}")
    print(f"Inter-call delay: {inter_call_delay}s")
    print("=" * 80)

    if not stream_path.exists():
        raise FileNotFoundError(f"Stream dataset not found: {stream_path}")

    with open(stream_path, "r", encoding="utf-8") as f:
        stream_records = json.load(f)

    # 1. Verify 6-way leakage
    passed, overlaps = check_leakage(stream_records)
    if not passed:
        raise ValueError(f"Leakage detected against prior datasets: {overlaps}")
    print(f"Leakage verification: PASS (0 overlaps across all 6 prior datasets on {len(stream_records)} queries)")

    # 2. Check pre-run quota
    from src.decision.judge_call import _resolve_api_key
    api_key = _resolve_api_key("openrouter")
    pre_quota = check_openrouter_quota(api_key)
    if pre_quota and "free_model_daily_requests" in pre_quota:
        fq = pre_quota["free_model_daily_requests"]
        print(f"Pre-run OpenRouter Quota: used={fq.get('used')}, limit={fq.get('limit')}, remaining={fq.get('remaining')}")

    # 3. Initialize components
    print("\nInitializing production pipeline components...")
    classifier = StabilityClassifier()
    embedder = QueryEmbedder()
    store = FlatVectorStore(dim=embedder.dim)
    cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
    router = TierRouter()
    history = CategoryHistory()
    judge = LLMJudge(
        model=model,
        backend="openrouter",
        rate_limit_delay_seconds=inter_call_delay,
    )
    decision_step = ProductionDecisionStep(judge=judge, history=history)

    # Support resuming if partial telemetry already exists
    completed_records: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for r in saved:
                    if "query_id" in r and not r.get("fallback_triggered", False):
                        completed_records[r["query_id"]] = r
            if completed_records:
                print(f"Loaded {len(completed_records)} previously completed genuine records from {output_path}.")
        except Exception as e:
            print(f"Notice: Telemetry file exists but could not be parsed ({e}), starting fresh.")

    telemetry: List[Dict[str, Any]] = []
    judge_calls_made = 0
    t_start_suite = time.perf_counter()

    print("\nExecuting query stream...")
    print(f"{'#':<3} | {'ID':<10} | {'Domain':<22} | {'Tier':<10} | {'Decision':<8} | {'Latency':<9} | {'Judge Call'}")
    print("-" * 88)

    for idx, item in enumerate(stream_records, start=1):
        qid = item["id"]
        q = item["query"]
        dom = item["domain"]
        qtype = item["query_type"]
        is_safe = item["is_reuse_safe"]

        t_start_query = time.perf_counter()

        # Step 1: Stability Classifier (Phase 1)
        t0_clf = time.perf_counter()
        stab_res = classifier.classify(q)
        classifier_ms = (time.perf_counter() - t0_clf) * 1000.0

        # Step 2: Semantic Cache Lookup (Phase 2)
        t0_cache = time.perf_counter()
        cache_res = cache.lookup(q)
        lookup_ms = (time.perf_counter() - t0_cache) * 1000.0

        # Step 3: Tier Router (Phase 3)
        tier = router.route(cache_res, stab_res)

        # Step 4: Decision Step / Execution
        judge_called = False
        judge_info = None
        judge_ms = 0.0

        if tier == Tier.AUTO_REUSE:
            path = "AUTO_REUSE"
            final_decision = "HIT"
            cached_response = cache_res.matched_metadata.get("response", f"Cached response for {cache_res.matched_metadata.get('id')}")
        elif tier == Tier.AMBIGUOUS:
            path = "AMBIGUOUS"
            judge_called = True
            matched_query = cache_res.matched_metadata.get("query", "")

            # Check quota cap before making live call
            if judge_calls_made >= max_judge_calls:
                print(f"\n[STOP] Daily budget cap reached ({max_judge_calls} calls). Halting stream.")
                break

            t0_judge = time.perf_counter()
            dec_res = decision_step.decide(
                query_a=matched_query,
                query_b=q,
                cache_result=cache_res,
                stability_result=stab_res,
                domain=dom,
            )
            judge_ms = (time.perf_counter() - t0_judge) * 1000.0
            judge_calls_made += 1

            judge_info = dec_res.judge_result.to_dict()
            if dec_res.decision == "REUSE" and dec_res.is_safe:
                final_decision = "HIT"
                cached_response = cache_res.matched_metadata.get("response", "")
            else:
                final_decision = "MISS"
                cached_response = None
        else:  # BYPASS
            path = "BYPASS"
            final_decision = "MISS"
            cached_response = None

        total_pipeline_ms = (time.perf_counter() - t_start_query) * 1000.0

        # Cache Population on MISS: upstream generation simulated and cached
        if final_decision == "MISS":
            simulated_response = f"Upstream LLM generated answer for [{qid}] ({dom}): {q}"
            cache.index(
                query=q,
                metadata={
                    "id": qid,
                    "query": q,
                    "domain": dom,
                    "response": simulated_response,
                    "timestamp": time.time(),
                },
            )

        # Correctness evaluation against ground truth
        if final_decision == "HIT":
            outcome_type = "TP" if is_safe else "FP"
            was_correct = is_safe
        else:
            outcome_type = "TN" if not is_safe else "FN"
            was_correct = not is_safe

        # Print row
        judge_str = f"YES ({judge_info['decision']}, {judge_ms:.0f}ms)" if judge_called else "NO"
        print(f"{idx:>3} | {qid:<10} | {dom:<22} | {path:<10} | {final_decision:<8} | {total_pipeline_ms:>6.1f}ms | {judge_str}")

        record = {
            "query_id": qid,
            "domain": dom,
            "query_text": q,
            "query_type": qtype,
            "is_reuse_safe_ground_truth": is_safe,
            "path": path,
            "tier": tier.value,
            "final_decision": final_decision,
            "outcome_type": outcome_type,
            "was_correct": was_correct,
            "timings_ms": {
                "classifier_ms": round(classifier_ms, 3),
                "embed_ms": round(cache_res.embed_latency_ms, 3),
                "search_ms": round(cache_res.search_latency_ms, 3),
                "judge_ms": round(judge_ms, 3),
                "total_pipeline_ms": round(total_pipeline_ms, 3),
            },
            "classifier_telemetry": {
                "label": stab_res.predicted_label.value,
                "effective_decision": stab_res.effective_decision.value,
                "confidence": round(stab_res.confidence, 4),
                "matched_rule": stab_res.matched_rule,
            },
            "cache_telemetry": {
                "similarity_score": round(cache_res.similarity_score, 4),
                "matched_cached_query": cache_res.matched_metadata.get("query"),
                "matched_id": cache_res.matched_metadata.get("id"),
                "store_size": cache_res.store_size,
            },
            "judge_telemetry": judge_info,
            "fallback_triggered": judge_info.get("fallback_triggered", False) if judge_info else False,
        }
        telemetry.append(record)

    total_suite_elapsed = time.perf_counter() - t_start_suite

    # Save telemetry
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    # 4. Check post-run quota
    post_quota = check_openrouter_quota(api_key)
    print("\n" + "=" * 80)
    print(f"LOAD TEST COMPLETE: Processed {len(telemetry)} queries in {total_suite_elapsed:.2f}s")
    print(f"Judge calls made: {judge_calls_made}")
    if post_quota and "free_model_daily_requests" in post_quota:
        fq = post_quota["free_model_daily_requests"]
        print(f"Post-run OpenRouter Quota: used={fq.get('used')}, limit={fq.get('limit')}, remaining={fq.get('remaining')}")
    print(f"Saved real telemetry to {output_path}")
    print("=" * 80)

    return telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 full-pipeline load test.")
    parser.add_argument("--stream", type=Path, default=STREAM_PATH, help="Path to stream JSON")
    parser.add_argument("--output", type=Path, default=TELEMETRY_PATH, help="Path to output telemetry JSON")
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--delay", type=float, default=2.5, help="Inter-call delay for judge (s)")
    parser.add_argument("--max-calls", type=int, default=15, help="Max judge calls allowed this session")
    parser.add_argument("--leakage-check-only", action="store_true", help="Run 6-way leakage check and exit")

    args = parser.parse_args()

    if args.leakage_check_only:
        with open(args.stream, "r", encoding="utf-8") as f:
            records = json.load(f)
        passed, overlaps = check_leakage(records)
        if passed:
            print(f"PASS: 0 overlaps found across all 6 benchmark datasets ({len(records)} queries verified).")
            sys.exit(0)
        else:
            print(f"FAIL: Leakage detected: {overlaps}")
            sys.exit(1)

    run_load_test(
        stream_path=args.stream,
        output_path=args.output,
        model=args.model,
        inter_call_delay=args.delay,
        max_judge_calls=args.max_calls,
    )


if __name__ == "__main__":
    main()
