"""New Dataset Load Test Runner — Phase 5 Step 1b.

Executes the full end-to-end production pipeline against data/raw/new_dataset.json:
  StabilityClassifier -> SemanticCache -> TierRouter -> ProductionDecisionStep (LLMJudge)

Dataset structure handled:
  - source="stackexchange" entries: standalone cold seeds (behavior=MISS, pair_id=own id).
    Each is expected to MISS/BYPASS. After a MISS they are indexed into cache so later
    queries may match them (replicating the original runner behavior).
  - source="hand_authored" entries: grouped into pairs by pair_id.
    - Anchor (behavior=MISS): seeds the cache on its first MISS.
    - Follow-on (behavior=HIT): a genuine paraphrase -- expected to be AUTO_REUSE or
      judged safe (HIT outcome is TP, MISS is FN).
    - Follow-on (behavior=BYPASS): an adversarial trap -- expected to be rejected by
      TierRouter or judge (MISS/BYPASS outcome is TN, HIT is FP).
  - realtime_news_weather entries: all hand_authored, all behavior=BYPASS.

Schema differences from load_test_query_stream.json:
  - "behavior" (MISS/HIT/BYPASS)  replaces  "is_reuse_safe" + "query_type"
  - "source", "pair_id", "se_score", "tags" are new fields

Invariants:
  - Does NOT modify TierRouter, ProductionDecisionStep, or src/classifier/.
  - File order is trusted for pair sequencing (anchors precede follow-ons).
  - Judge budget: live quota check before run; cap at remaining daily quota.
  - Fail-closed on any judge error (BYPASS).
  - Inter-call delay >= 2.5s to respect OpenRouter free-tier rate limits.
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

DATASET_PATH = REPO_ROOT / "data" / "raw" / "new_dataset.json"
TELEMETRY_PATH = REPO_ROOT / "data" / "new_dataset_telemetry.json"
METRICS_PATH = REPO_ROOT / "data" / "new_dataset_metrics_summary.json"

# All datasets that must not overlap with new_dataset queries
PRIOR_DATASET_PATHS = [
    REPO_ROOT / "data" / "raw" / "query_pair_reuse_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_heldout.json",
    REPO_ROOT / "data" / "raw" / "query_stability_benchmark_final_test.json",
    REPO_ROOT / "data" / "raw" / "query_stability_human_credibility.json",
    REPO_ROOT / "data" / "raw" / "synthetic_query_pair_feedback.json",
    REPO_ROOT / "data" / "raw" / "load_test_query_stream.json",  # original Phase 5 stream
]


# ---------------------------------------------------------------------------
# Schema adapter
# ---------------------------------------------------------------------------

def behavior_to_is_reuse_safe(behavior: str) -> bool:
    """Map new dataset 'behavior' field to existing correctness convention.

    HIT  -> is_reuse_safe=True  (reuse is expected and correct)
    MISS -> is_reuse_safe=False (no reuse expected; fresh generation correct)
    BYPASS -> is_reuse_safe=False (volatility/trap; bypass correct, reuse wrong)
    """
    return behavior.upper() == "HIT"


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load and minimally validate the new dataset JSON."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "IMPORTANT: Save the file in your IDE (Ctrl+S) before running this script."
        )
    with open(dataset_path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt or empty JSON in {dataset_path}: {exc}") from exc

    if not records:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    required = {"id", "query", "domain", "behavior"}
    for i, r in enumerate(records):
        missing = required - set(r.keys())
        if missing:
            raise ValueError(f"Record #{i} (id={r.get('id')}) missing fields: {missing}")

    return records


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

def check_leakage(
    dataset_records: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """7-way leakage check: new dataset vs all 6 original benchmarks + original stream."""
    existing_queries: set = set()
    for p in PRIOR_DATASET_PATHS:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        # Handle both flat arrays and metadata-wrapped formats
        if isinstance(data, dict):
            items = data.get("queries", data.get("items", data.get("data", [])))
        else:
            items = data
        for item in items:
            for k in ["query", "query_a", "query_b", "prompt", "text"]:
                v = item.get(k, "")
                if isinstance(v, str) and v.strip():
                    existing_queries.add(v.strip().lower())

    overlaps = []
    for r in dataset_records:
        q = r["query"].strip().lower()
        if q in existing_queries:
            overlaps.append(f"{r['id']}: {r['query'][:80]}")

    return len(overlaps) == 0, overlaps


# ---------------------------------------------------------------------------
# OpenRouter quota helpers
# ---------------------------------------------------------------------------

def check_openrouter_quota(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Query OpenRouter /api/v1/auth/key for live daily quota status."""
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
        print(f"Warning: Could not query OpenRouter quota: {exc}")
        return None


def get_remaining_quota(quota_data: Optional[Dict[str, Any]]) -> Optional[int]:
    """Extract remaining daily free requests from quota response."""
    if not quota_data:
        return None
    fq = quota_data.get("free_model_daily_requests", {})
    return fq.get("remaining")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_new_dataset_load_test(
    dataset_path: Path = DATASET_PATH,
    telemetry_output: Path = TELEMETRY_PATH,
    model: str = DEFAULT_JUDGE_MODEL,
    inter_call_delay: float = 2.5,
    max_judge_calls: Optional[int] = None,  # None = use live remaining quota
) -> List[Dict[str, Any]]:
    """Execute new_dataset.json through the full production pipeline."""
    print("=" * 80)
    print("NEW DATASET LOAD TEST — Full Production Pipeline")
    print(f"Dataset:   {dataset_path}")
    print(f"Telemetry: {telemetry_output}")
    print(f"Model:     {model}")
    print(f"Delay:     {inter_call_delay}s between judge calls")
    print("=" * 80)

    # 1. Load dataset
    records = load_dataset(dataset_path)
    print(f"\nLoaded {len(records)} records from dataset.")

    # Schema summary
    se_count = sum(1 for r in records if r.get("source") == "stackexchange")
    ha_count = sum(1 for r in records if r.get("source") == "hand_authored")
    hit_count = sum(1 for r in records if r.get("behavior", "").upper() == "HIT")
    miss_count = sum(1 for r in records if r.get("behavior", "").upper() == "MISS")
    bypass_count = sum(1 for r in records if r.get("behavior", "").upper() == "BYPASS")
    domains = {}
    for r in records:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    print(f"  Source breakdown: stackexchange={se_count}, hand_authored={ha_count}")
    print(f"  Behavior labels: HIT={hit_count}, MISS={miss_count}, BYPASS={bypass_count}")
    print(f"  Domain distribution: {dict(sorted(domains.items()))}")

    # 2. Leakage check (7-way)
    print("\nRunning 7-way leakage check...")
    passed, overlaps = check_leakage(records)
    if not passed:
        raise ValueError(
            f"Leakage detected: {len(overlaps)} queries overlap prior datasets.\n"
            + "\n".join(f"  {o}" for o in overlaps[:10])
        )
    print(f"Leakage check: PASS (0 overlaps across 7 prior datasets on {len(records)} queries)")

    # 3. Check and set judge call budget
    from src.decision.judge_call import _resolve_api_key
    api_key = _resolve_api_key("openrouter")
    pre_quota = check_openrouter_quota(api_key)
    remaining = get_remaining_quota(pre_quota)

    if pre_quota and "free_model_daily_requests" in pre_quota:
        fq = pre_quota["free_model_daily_requests"]
        print(f"\nPre-run OpenRouter Quota: used={fq.get('used')}, limit={fq.get('limit')}, remaining={remaining}")

    if max_judge_calls is None:
        if remaining is not None:
            max_judge_calls = remaining
            print(f"Judge call budget: {max_judge_calls} (= remaining daily quota)")
        else:
            max_judge_calls = 10
            print(f"Warning: Could not determine remaining quota. Defaulting to {max_judge_calls} judge calls.")
    else:
        print(f"Judge call budget: {max_judge_calls} (user-specified)")

    # 4. Initialize pipeline components (DO NOT modify these)
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

    # 5. Execute stream
    telemetry: List[Dict[str, Any]] = []
    judge_calls_made = 0
    t_start_suite = time.perf_counter()

    # Build pair membership index for logging
    pair_members: Dict[str, List[str]] = {}
    for r in records:
        pid = r.get("pair_id")
        if pid:
            pair_members.setdefault(pid, []).append(r["id"])

    print(f"\nExecuting {len(records)} queries...")
    header = f"{'#':>3} | {'ID':<12} | {'Source':<14} | {'Domain':<22} | {'Tier':<10} | {'Decision':<8} | {'GT':<6} | {'OK?':<4} | {'ms':>7} | Judge"
    print(header)
    print("-" * len(header))

    for idx, item in enumerate(records, start=1):
        qid = item["id"]
        q = item["query"]
        dom = item["domain"]
        source = item.get("source", "unknown")
        pair_id = item.get("pair_id")
        behavior = item.get("behavior", "MISS").upper()
        se_score = item.get("se_score")
        tags = item.get("tags", [])
        is_reuse_safe = behavior_to_is_reuse_safe(behavior)

        t_start_query = time.perf_counter()

        # Step 1: StabilityClassifier
        t0_clf = time.perf_counter()
        stab_res = classifier.classify(q)
        classifier_ms = (time.perf_counter() - t0_clf) * 1000.0

        # Step 2: SemanticCache lookup
        t0_cache = time.perf_counter()
        cache_res = cache.lookup(q)
        lookup_ms = (time.perf_counter() - t0_cache) * 1000.0

        # Step 3: TierRouter
        tier = router.route(cache_res, stab_res)

        # Step 4: Decision
        judge_called = False
        judge_info = None
        judge_ms = 0.0

        if tier == Tier.AUTO_REUSE:
            path = "AUTO_REUSE"
            final_decision = "HIT"
            cached_response = cache_res.matched_metadata.get(
                "response", f"Cached: {cache_res.matched_metadata.get('id')}"
            )
        elif tier == Tier.AMBIGUOUS:
            path = "AMBIGUOUS"
            judge_called = True
            matched_query = cache_res.matched_metadata.get("query", "")

            if judge_calls_made >= max_judge_calls:
                # Budget exhausted — fail-closed to BYPASS
                print(f"\n[BUDGET] Judge call budget exhausted ({max_judge_calls}). Routing AMBIGUOUS -> BYPASS.")
                path = "AMBIGUOUS→BYPASS"
                final_decision = "MISS"
                cached_response = None
                judge_called = False
            else:
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

        # Cache population on MISS (same as original runner)
        if final_decision == "MISS":
            simulated_response = (
                f"Upstream LLM generated answer for [{qid}] ({dom}): {q}"
            )
            cache.index(
                query=q,
                metadata={
                    "id": qid,
                    "query": q,
                    "domain": dom,
                    "source": source,
                    "pair_id": pair_id,
                    "response": simulated_response,
                    "timestamp": time.time(),
                },
            )

        # Correctness scoring (TP/TN/FP/FN)
        if final_decision == "HIT":
            outcome_type = "TP" if is_reuse_safe else "FP"
        else:
            outcome_type = "TN" if not is_reuse_safe else "FN"
        was_correct = outcome_type in ("TP", "TN")

        # Console row
        judge_str = (
            f"YES ({judge_info['decision']}, {judge_ms:.0f}ms)"
            if judge_called and judge_info
            else ("YES (budget-cap→BYPASS)" if "→" in path else "NO")
        )
        gt_str = "SAFE" if is_reuse_safe else "NOTSF"
        ok_str = "✓" if was_correct else "✗"
        print(
            f"{idx:>3} | {qid:<12} | {source:<14} | {dom:<22} | {path:<10} | "
            f"{final_decision:<8} | {gt_str:<6} | {ok_str:<4} | {total_pipeline_ms:>6.1f}ms | {judge_str}"
        )

        record = {
            "query_id": qid,
            "domain": dom,
            "query_text": q,
            "source": source,
            "pair_id": pair_id,
            "behavior_ground_truth": behavior,
            "is_reuse_safe_ground_truth": is_reuse_safe,
            "se_score": se_score,
            "tags": tags,
            # Fields matching original runner schema for evaluate_load_test compatibility
            "query_type": f"{source}_{behavior.lower()}",
            "path": path.split("→")[0],  # Use canonical path name for evaluator
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
            "fallback_triggered": (
                judge_info.get("fallback_triggered", False) if judge_info else False
            ),
        }
        telemetry.append(record)

    total_suite_elapsed = time.perf_counter() - t_start_suite

    # 6. Save telemetry
    telemetry_output.parent.mkdir(parents=True, exist_ok=True)
    with open(telemetry_output, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    # 7. Post-run quota check
    post_quota = check_openrouter_quota(api_key)
    if post_quota and "free_model_daily_requests" in post_quota:
        fq = post_quota["free_model_daily_requests"]
        post_remaining = fq.get("remaining")
    else:
        post_remaining = None

    # 8. Summary statistics
    total_q = len(telemetry)
    hits = [r for r in telemetry if r["final_decision"] == "HIT"]
    misses = [r for r in telemetry if r["final_decision"] == "MISS"]
    auto_reuse = [r for r in telemetry if r["path"] == "AUTO_REUSE"]
    ambiguous = [r for r in telemetry if r["path"] == "AMBIGUOUS"]
    bypass = [r for r in telemetry if r["path"] == "BYPASS"]
    tp = sum(1 for r in telemetry if r["outcome_type"] == "TP")
    tn = sum(1 for r in telemetry if r["outcome_type"] == "TN")
    fp = sum(1 for r in telemetry if r["outcome_type"] == "FP")
    fn = sum(1 for r in telemetry if r["outcome_type"] == "FN")
    correct = sum(1 for r in telemetry if r["was_correct"])

    # Per-source breakdown
    se_records = [r for r in telemetry if r["source"] == "stackexchange"]
    ha_records = [r for r in telemetry if r["source"] == "hand_authored"]
    ha_correct = sum(1 for r in ha_records if r["was_correct"])
    se_correct = sum(1 for r in se_records if r["was_correct"])

    print("\n" + "=" * 80)
    print("NEW DATASET LOAD TEST — COMPLETE")
    print("=" * 80)
    print(f"Total Queries Processed : {total_q}")
    print(f"Total Wall-Clock Time   : {total_suite_elapsed:.2f}s")
    print(f"Judge Calls Made        : {judge_calls_made}")
    if post_remaining is not None:
        print(f"Post-run Quota Remaining: {post_remaining}")
    print()
    print("--- Pipeline Path Distribution ---")
    print(f"  AUTO_REUSE : {len(auto_reuse):>4} ({len(auto_reuse)/total_q*100:.1f}%)")
    print(f"  AMBIGUOUS  : {len(ambiguous):>4} ({len(ambiguous)/total_q*100:.1f}%)")
    print(f"  BYPASS     : {len(bypass):>4}  ({len(bypass)/total_q*100:.1f}%)")
    print()
    print("--- Outcome Matrix ---")
    print(f"  TP (safe reuse, HIT)     : {tp}")
    print(f"  TN (unsafe/miss, BYPASS) : {tn}")
    print(f"  FP (unsafe reuse, HIT)   : {fp}  ← dangerous cache hazards")
    print(f"  FN (safe, missed)        : {fn}  ← conservative false negatives")
    print(f"  Overall Accuracy         : {correct}/{total_q} = {correct/total_q*100:.2f}%")
    print()
    print("--- By Source ---")
    print(f"  StackExchange cold seeds : {len(se_records)} queries, {se_correct} correct ({se_correct/max(len(se_records),1)*100:.1f}%)")
    print(f"  Hand-authored pairs      : {len(ha_records)} queries, {ha_correct} correct ({ha_correct/max(len(ha_records),1)*100:.1f}%)")
    print()
    print(f"Telemetry saved to: {telemetry_output}")
    print("=" * 80)

    return telemetry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run new_dataset.json through the full production pipeline."
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH,
        help="Path to new_dataset.json"
    )
    parser.add_argument(
        "--output", type=Path, default=TELEMETRY_PATH,
        help="Path to output telemetry JSON"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_JUDGE_MODEL,
        help="Judge model name (OpenRouter)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.5,
        help="Minimum inter-call delay for judge calls (seconds)"
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Max judge calls (default: use live remaining OpenRouter quota)"
    )
    parser.add_argument(
        "--leakage-check-only", action="store_true",
        help="Run 7-way leakage check and exit without executing pipeline"
    )

    args = parser.parse_args()

    if args.leakage_check_only:
        records = load_dataset(args.dataset)
        passed, overlaps = check_leakage(records)
        if passed:
            print(f"PASS: 0 overlaps found across 7 benchmark datasets ({len(records)} queries).")
            sys.exit(0)
        else:
            print(f"FAIL: {len(overlaps)} overlaps detected:")
            for o in overlaps:
                print(f"  {o}")
            sys.exit(1)

    run_new_dataset_load_test(
        dataset_path=args.dataset,
        telemetry_output=args.output,
        model=args.model,
        inter_call_delay=args.delay,
        max_judge_calls=args.max_calls,
    )


if __name__ == "__main__":
    main()
