#!/usr/bin/env python3
"""Interactive Demonstration: Adaptive Agentic Semantic Cache.

"This is a demonstration, not an evaluation. See README for evaluation results."

Demonstrates the 3-tier routing architecture on real queries drawn from the
Phase 5 synthetic load test stream (data/raw/load_test_query_stream.json) and
calibrated against recorded telemetry (data/load_test_telemetry.json).

Modes:
  1. Replay mode (default, no API key required):
     - Local stages execute live: QueryEmbedder, FlatVectorStore (FAISS),
       StabilityClassifier, and TierRouter.
     - Judge decisions and rationale are replayed from recorded telemetry.
  2. Live mode (--live, requires OPENROUTER_API_KEY):
     - Executes live calls to the production judge model.
     - Strictly respects OpenRouter free tier rate limits (50 req/day).
     - Stops cleanly on quota errors without fabricating fallback outcomes.
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

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cache.embedding import QueryEmbedder
from src.cache.semantic_cache import SemanticCache
from src.cache.vector_store import FlatVectorStore
from src.classifier.stability_classifier import StabilityClassifier
from src.decision.decision_step import CategoryHistory, ProductionDecisionStep
from src.decision.judge_call import DEFAULT_JUDGE_MODEL, LLMJudge, _resolve_api_key
from src.decision.tier_router import Tier, TierRouter

STREAM_PATH = REPO_ROOT / "data" / "raw" / "load_test_query_stream.json"
TELEMETRY_PATH = REPO_ROOT / "data" / "load_test_telemetry.json"

# Curated 12-query demonstration sequence drawn from data/raw/load_test_query_stream.json
# Covering:
#   1. AUTO_REUSE hit (exact duplicate)
#   2. AUTO_REUSE hit (paraphrase)
#   3. BYPASS via low similarity floor (<0.50)
#   4. BYPASS via StabilityClassifier gate (DYNAMIC queries with temporal anchors)
#   5. AMBIGUOUS tier judge-approved semantic equivalence hit
#   6. AMBIGUOUS tier judge-rejected near-duplicate (syntactic similarity but divergent semantics)
DEMO_QUERY_IDS = [
    "LT-CS-001",  # Cold cache seed (AVL tree complexity) -> BYPASS (sim=-inf)
    "LT-CS-002",  # Exact duplicate -> AUTO_REUSE HIT (sim=1.000)
    "LT-CS-003",  # Paraphrase -> AUTO_REUSE HIT (sim=0.957)
    "LT-CS-004",  # Novel topic (Dijkstra) -> BYPASS via low similarity (sim=0.408)
    "LT-CS-007",  # Volatile ("released today") -> BYPASS via stability gate (DYNAMIC)
    "LT-CS-010",  # Near-duplicate: Prim MST vs Dijkstra shortest paths -> AMBIGUOUS judge REJECTED
    "LT-SM-004",  # Science seed (RuBisCO fixation) -> BYPASS (low similarity)
    "LT-SM-007",  # Volatile ("pollen count today") -> BYPASS via stability gate (DYNAMIC)
    "LT-SM-009",  # Semantic equivalence: RuBisCO carboxylation -> AMBIGUOUS judge APPROVED
    "LT-MA-001",  # Math seed (Central Limit Theorem) -> BYPASS (low similarity)
    "LT-MA-007",  # Volatile ("prime rate today") -> BYPASS via stability gate (DYNAMIC)
    "LT-MA-010",  # Near-duplicate: Law of Large Numbers vs CLT -> AMBIGUOUS judge REJECTED
]


def check_openrouter_quota(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Query OpenRouter /api/v1/auth/key to inspect real daily free request limit and usage."""
    if not api_key:
        return None
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=8.0) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("data", {})
    except Exception:
        return None


def format_sim(score: float) -> str:
    """Format similarity score gracefully handling negative infinity."""
    if score == float("-inf") or score < -1.0:
        return "  N/A "
    return f"{score:6.3f}"


def run_demo(live: bool = False, judge_model: str = DEFAULT_JUDGE_MODEL) -> None:
    """Execute the end-to-end interactive demonstration."""
    print("=" * 94)
    print("This is a demonstration, not an evaluation. See README for evaluation results.")
    print("=" * 94)

    # Determine execution mode
    api_key = _resolve_api_key("openrouter")
    is_live_mode = False

    if live:
        if not api_key:
            print("\n[ERROR] --live requested but OPENROUTER_API_KEY is not set.")
            print("Falling back to REPLAY mode.\n")
            is_live_mode = False
        else:
            is_live_mode = True

    if is_live_mode:
        print("\nMODE: LIVE (Live judge inference via OpenRouter API)")
        print("WARNING: OpenRouter free tier has a strict limit of 50 requests/day.")
        quota = check_openrouter_quota(api_key)
        if quota and "free_model_daily_requests" in quota:
            fq = quota["free_model_daily_requests"]
            print(f"OpenRouter Quota Status: used={fq.get('used')}, limit={fq.get('limit')}, remaining={fq.get('remaining')}")
            if fq.get("remaining", 1) <= 0:
                print("\n[STOP] OpenRouter daily quota is exhausted. Stopping cleanly to prevent fabricated results.")
                sys.exit(0)
    else:
        print("\nMODE: REPLAY (Local embedding, FAISS search, classifier, & router live; judge replayed)")
        print("Telemetry Source: data/load_test_telemetry.json")

    print("-" * 94)

    # Load data streams
    if not STREAM_PATH.exists():
        raise FileNotFoundError(f"Missing stream dataset: {STREAM_PATH}")
    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(f"Missing recorded telemetry: {TELEMETRY_PATH}")

    with open(STREAM_PATH, "r", encoding="utf-8") as f:
        stream_data = json.load(f)
    with open(TELEMETRY_PATH, "r", encoding="utf-8") as f:
        telemetry_data = json.load(f)

    stream_by_id = {item["id"]: item for item in stream_data}
    telemetry_by_id = {item["query_id"]: item for item in telemetry_data}

    # Initialize live local components
    print("\n[1/3] Initializing local pipeline components...")
    t0_init = time.perf_counter()
    classifier = StabilityClassifier()
    embedder = QueryEmbedder()
    store = FlatVectorStore(dim=embedder.dim)
    cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
    router = TierRouter()
    history = CategoryHistory()
    t_init = (time.perf_counter() - t0_init) * 1000.0
    print(f"      Pipeline initialized in {t_init:.1f} ms (SentenceTransformer pinned revision: {embedder.revision[:7]})\n")

    # Initialize judge step if in live mode
    decision_step = None
    if is_live_mode:
        judge = LLMJudge(model=judge_model, backend="openrouter", rate_limit_delay_seconds=2.0)
        decision_step = ProductionDecisionStep(judge=judge, history=history)

    print("[2/3] Processing 12 demonstration queries through the live pipeline:")
    header = (
        f"{'#':<3} | {'Query ID':<10} | {'Tier':<10} | {'Sim':<6} | "
        f"{'Stability (Conf)':<18} | {'Decision':<8} | {'Latency':<14}"
    )
    print("=" * 94)
    print(header)
    print("-" * 94)

    auto_reuse_latencies_ms: List[float] = []
    judge_latencies_ms: List[float] = []

    for idx, qid in enumerate(DEMO_QUERY_IDS, start=1):
        item = stream_by_id[qid]
        q_text = item["query"]
        domain = item["domain"]
        telem = telemetry_by_id.get(qid, {})

        # Step 1: Stability Classifier (Live)
        t0_local = time.perf_counter()
        t0_clf = time.perf_counter()
        stab_res = classifier.classify(q_text)
        clf_ms = (time.perf_counter() - t0_clf) * 1000.0

        # Step 2: Semantic Cache Lookup (Live embedding + FAISS search)
        t0_lookup = time.perf_counter()
        cache_res = cache.lookup(q_text)
        lookup_ms = (time.perf_counter() - t0_lookup) * 1000.0

        # Step 3: TierRouter (Live)
        t0_router = time.perf_counter()
        tier = router.route(cache_res, stab_res)
        router_ms = (time.perf_counter() - t0_router) * 1000.0

        local_total_ms = (time.perf_counter() - t0_local) * 1000.0

        matched_meta = cache_res.matched_metadata or {}
        matched_id = matched_meta.get("id", "None")
        sim_score = cache_res.similarity_score

        stability_str = f"{stab_res.predicted_label.value} ({stab_res.confidence:.2f})"

        judge_label = ""
        judge_rationale = ""
        query_total_ms = local_total_ms

        if tier == Tier.AUTO_REUSE:
            final_decision = "HIT"
            auto_reuse_latencies_ms.append(local_total_ms)
            latency_str = f"{local_total_ms:>6.2f} ms"

        elif tier == Tier.AMBIGUOUS:
            matched_query = matched_meta.get("query", "")

            if is_live_mode:
                # Live judge invocation
                t0_judge = time.perf_counter()
                try:
                    dec_res = decision_step.decide(
                        query_a=matched_query,
                        query_b=q_text,
                        cache_result=cache_res,
                        stability_result=stab_res,
                        domain=domain,
                    )
                    judge_ms = (time.perf_counter() - t0_judge) * 1000.0
                    judge_latencies_ms.append(judge_ms)
                    query_total_ms = local_total_ms + judge_ms
                    judge_res = dec_res.judge_result

                    if judge_res.fallback_triggered:
                        print(f"\n[QUOTA/API ERROR] Live judge failed: {judge_res.error or 'Fallback triggered'}")
                        print("Stopping live stream cleanly without fabricating results.")
                        break

                    final_decision = "HIT" if (dec_res.decision == "REUSE" and dec_res.is_safe) else "MISS"
                    judge_label = f"judge: LIVE ({judge_ms:.0f} ms)"
                    judge_rationale = judge_res.rationale
                    latency_str = f"{query_total_ms:>6.1f} ms"
                except Exception as exc:
                    print(f"\n[ERROR] Live judge exception encountered: {exc}")
                    print("Stopping cleanly.")
                    break
            else:
                # Replay judge from recorded telemetry
                telem_judge = telem.get("judge_telemetry", {}) or {}
                recorded_judge_ms = telem.get("timings_ms", {}).get("judge_ms", 7000.0)
                judge_latencies_ms.append(recorded_judge_ms)
                query_total_ms = local_total_ms + recorded_judge_ms

                dec_str = telem_judge.get("decision", "BYPASS")
                is_safe = telem_judge.get("is_safe", False)
                final_decision = "HIT" if (dec_str == "REUSE" and is_safe) else "MISS"

                judge_label = f"judge: REPLAYED ({recorded_judge_ms:.0f} ms)"
                judge_rationale = telem_judge.get("rationale", "No rationale recorded.")
                latency_str = f"{query_total_ms:>6.1f} ms*"

        else:  # BYPASS tier
            final_decision = "MISS"
            latency_str = f"{local_total_ms:>6.2f} ms"

        # Print query summary row
        print(
            f"{idx:<3} | {qid:<10} | {tier.value:<10} | {format_sim(sim_score)} | "
            f"{stability_str:<18} | {final_decision:<8} | {latency_str:<14}"
        )
        print(f"      Q: \"{q_text}\"")

        if tier == Tier.AUTO_REUSE:
            print(f"      -> Cached match: [{matched_id}] \"{matched_meta.get('query')}\"")
        elif tier == Tier.AMBIGUOUS:
            print(f"      -> Borderline match: [{matched_id}] \"{matched_meta.get('query')}\"")
            print(f"      -> {judge_label} | Decision: {final_decision}")
            if judge_rationale:
                wrapped_rat = judge_rationale[:88] + ("..." if len(judge_rationale) > 88 else "")
                print(f"         Rationale: {wrapped_rat}")
        elif tier == Tier.BYPASS:
            if stab_res.predicted_label.value == "DYNAMIC":
                print(f"      -> Trapped by Stability Gate: volatile temporal query routed to BYPASS")
            else:
                print(f"      -> Low similarity ({format_sim(sim_score)} < 0.50): routed to BYPASS")

        # Cache population on MISS: upstream response cached for subsequent lookups
        if final_decision == "MISS":
            cache.index(
                query=q_text,
                metadata={
                    "id": qid,
                    "query": q_text,
                    "domain": domain,
                    "response": f"Upstream generated response for [{qid}]",
                    "timestamp": time.time(),
                },
            )

        print("-" * 94)

    # Summary and latency reduction computation
    print("\n[3/3] Performance & Latency Reduction Summary:")
    print("=" * 94)

    n_hits = len(auto_reuse_latencies_ms)
    n_judge = len(judge_latencies_ms)

    mean_hit_ms = sum(auto_reuse_latencies_ms) / n_hits if n_hits > 0 else 0.0
    mean_judge_ms = sum(judge_latencies_ms) / n_judge if n_judge > 0 else 0.0

    if mean_judge_ms > 0 and mean_hit_ms > 0:
        demo_reduction_pct = (1.0 - (mean_hit_ms / mean_judge_ms)) * 100.0
        demo_speedup = mean_judge_ms / mean_hit_ms
    else:
        demo_reduction_pct = 0.0
        demo_speedup = 1.0

    print(f"Demonstration Run Results:")
    print(f"  * AUTO_REUSE Mean Latency (n={n_hits}):   {mean_hit_ms:>7.2f} ms (measured live)")
    if is_live_mode:
        print(f"  * AMBIGUOUS Judge Mean Latency (n={n_judge}): {mean_judge_ms:>7.2f} ms (measured live)")
    else:
        print(f"  * AMBIGUOUS Judge Mean Latency (n={n_judge}): {mean_judge_ms:>7.2f} ms (recorded telemetry)")
    print(f"  * Measured Latency Reduction:            {demo_reduction_pct:>7.2f}% ({demo_speedup:.1f}x speedup)")

    print("\nComparison with Authoritative Benchmark Ledger (docs/FACTS.md):")
    print(f"  * FACTS.md Latency Reduction Range:     99.20% to 99.84% (row LAT-05)")
    print(f"    - Phase 5 N=70 pilot:                 99.79% (484.5x; row LAT-01)")
    print(f"    - Phase 5 N=338 scaled:               99.20% (124.6x; row LAT-02)")
    print(f"    - Phase 6 N=175 blind evaluation:     99.84% (622.7x; row LAT-03)")
    print(f"  * Note: Sample size n in this interactive demonstration is intentionally small (n={n_hits} hits).")
    print(f"    This script serves as an architectural walkthrough, not a formal evaluation.")
    print(f"    Please consult README.md and docs/FACTS.md for rigorous evaluation results and Clopper-Pearson CIs.")
    print("=" * 94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Adaptive Agentic Semantic Cache interactive demonstration.",
        epilog="Default mode is REPLAY (no API key required). Pass --live to invoke live judge calls.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live LLM judge calls via OpenRouter API (requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use in live mode (default: {DEFAULT_JUDGE_MODEL}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(live=args.live, judge_model=args.model)
