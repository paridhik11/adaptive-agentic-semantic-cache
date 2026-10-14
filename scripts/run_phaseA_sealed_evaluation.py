"""Phase A4 — Single-Run Sealed Test Stream Evaluation.

Executes the frozen Phase A verification pipeline on the pre-registered
and sealed test stream (data/raw/phaseA_sealed_test_stream.json, N=2,200).

Strict Evaluation Invariants:
  1. Cryptographic hash check: Stream SHA-256 must match data/phaseA_frozen_config.json.
  2. Single run: Executed exactly once. No post-hoc tuning.
  3. Fail-closed guards: All prospective reuses (including AUTO_REUSE >= 0.92)
     must pass rule guards (numbers, negations, entities, long prompt, generative).
  4. Local equivalence: Borderline pairs scored with fine-tuned CrossEncoder (Candidate c).
  5. Primary Endpoint: Non-exact Incorrect-Reuse Rate (IRR) with exact Clopper-Pearson 95% CI.

Outputs:
  - data/phaseA_sealed_evaluation_telemetry.json
  - data/phaseA_sealed_evaluation_summary.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import beta
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cache.normalizer import CanonicalCache, normalize_query
from src.cache.semantic_cache import SemanticCache
from src.cache.vector_store import FlatVectorStore
from src.classifier.stability_classifier import StabilityClassifier
from src.decision.local_equivalence import (
    LocalEquivalenceDecisionStep,
    LocalEquivalenceResult,
    evaluate_rule_guards,
)
from src.decision.tier_router import Tier, TierRouter

FROZEN_CONFIG_PATH = REPO_ROOT / "data" / "phaseA_frozen_config.json"
SEALED_STREAM_PATH = REPO_ROOT / "data" / "raw" / "phaseA_sealed_test_stream.json"
TELEMETRY_OUTPUT_PATH = REPO_ROOT / "data" / "phaseA_sealed_evaluation_telemetry.json"
SUMMARY_OUTPUT_PATH = REPO_ROOT / "data" / "phaseA_sealed_evaluation_summary.json"
FINETUNED_MODEL_DIR = REPO_ROOT / "models" / "phaseA_cross_encoder_finetuned"


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Clopper-Pearson 95% confidence interval via Beta distribution inversion."""
    if n == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def main() -> None:
    print("=" * 80)
    print("PHASE A4: SEALED OUT-OF-SAMPLE TEST EVALUATION (SINGLE RUN)")
    print("=" * 80)

    # 1. Verify cryptographic checksums against frozen pre-registration
    with open(FROZEN_CONFIG_PATH, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)

    with open(SEALED_STREAM_PATH, "rb") as f:
        actual_stream_hash = hashlib.sha256(f.read()).hexdigest()

    expected_stream_hash = frozen_cfg["sealed_stream"]["sha256"]
    print(f"Stream SHA-256 Checksum:  {actual_stream_hash}")
    print(f"Pre-registered Hash:      {expected_stream_hash}")

    if actual_stream_hash != expected_stream_hash:
        raise ValueError(
            f"FATAL: Stream checksum mismatch!\nActual: {actual_stream_hash}\nExpected: {expected_stream_hash}"
        )
    print("Checksum verified: Stream is IDENTICAL to pre-registered frozen test set.")

    with open(SEALED_STREAM_PATH, "r", encoding="utf-8") as f:
        stream: List[Dict[str, Any]] = json.load(f)

    print(f"\nLoaded sealed stream: {len(stream)} queries.")

    # 2. Initialize Models and Components
    print("\nInitializing Phase A Verified Pipeline components...")
    t0_init = time.time()
    embedder = SentenceTransformer(frozen_cfg["bi_encoder"]["model_name"])
    classifier = StabilityClassifier()
    vstore = FlatVectorStore(dim=frozen_cfg["bi_encoder"]["embedding_dimension"])
    cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=vstore)
    router = TierRouter(
        auto_reuse_threshold=frozen_cfg["tier_router"]["auto_reuse_threshold"],
        bypass_below_threshold=frozen_cfg["tier_router"]["bypass_below"],
    )
    cross_encoder = CrossEncoder(str(FINETUNED_MODEL_DIR))
    local_step = LocalEquivalenceDecisionStep(
        threshold=frozen_cfg["cross_encoder"]["decision_threshold"],
        fallback_to_llm=False,
        model_instance=cross_encoder,
    )
    canonical_cache = CanonicalCache()
    print(f"Pipeline initialized in {time.time() - t0_init:.2f} s")

    # 3. Execution telemetry accumulators
    telemetry: List[Dict[str, Any]] = []

    total_hits = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    non_exact_hits = 0
    non_exact_tp = 0
    non_exact_fp = 0

    exact_canonical_hits = 0

    path_counts = {"CANONICAL_EXACT": 0, "AUTO_REUSE": 0, "LOCAL_EQUIVALENCE": 0, "BYPASS": 0}
    path_latencies: Dict[str, List[float]] = {k: [] for k in path_counts}
    all_latencies_ms: List[float] = []

    source_metrics: Dict[str, Dict[str, int]] = {}

    baseline_p50 = frozen_cfg["baseline_generation_reference"]["median_latency_ms"]
    baseline_p95 = frozen_cfg["baseline_generation_reference"]["p95_latency_ms"]
    baseline_mean_tokens = frozen_cfg["baseline_generation_reference"]["mean_tokens"]

    print("\nStarting execution on sealed stream...")
    t0_eval = time.time()

    for idx, event in enumerate(stream, start=1):
        q = event["query"]
        ev_id = event["id"]
        source = event.get("source", "unknown")
        is_seed = bool(event.get("is_seed", False))
        is_safe = bool(event.get("is_reuse_safe", False))
        domain = event.get("domain", "general")

        if source not in source_metrics:
            source_metrics[source] = {"hits": 0, "tp": 0, "fp": 0, "non_exact_hits": 0, "non_exact_fp": 0}

        t0_query = time.perf_counter()

        final_decision = "BYPASS"
        path = "BYPASS"
        guard_reason = "N/A"
        similarity = 0.0
        stability_conf = 0.0
        local_score = 0.0
        matched_text = ""
        matched_id = None

        # Stage 0: Canonical Normalizer Exact Match
        canon_entry = canonical_cache.get(q)
        if canon_entry is not None:
            final_decision = "REUSE"
            path = "CANONICAL_EXACT"
            matched_text = canon_entry.get("query_text", "")
            matched_id = canon_entry.get("id")
            elapsed_ms = (time.perf_counter() - t0_query) * 1000.0
        else:
            # Stage 1: Stability Classifier
            stab = classifier.classify(q)
            stability_conf = stab.confidence

            # Stage 2: Vector Store Lookup
            lookup = cache.lookup(q)
            similarity = lookup.similarity_score

            if lookup.matched_metadata:
                matched_text = lookup.matched_metadata.get("query_text", "")
                matched_id = lookup.matched_metadata.get("id")

            # Stage 3: Tier Routing
            tier_decision = router.route(lookup, stab)

            if tier_decision == Tier.AUTO_REUSE and matched_text:
                # PHASE A INVARIANT: High-similarity candidates MUST pass rule guards
                guards_passed, guard_reason = evaluate_rule_guards(matched_text, q)
                if guards_passed:
                    final_decision = "REUSE"
                    path = "AUTO_REUSE"
                else:
                    final_decision = "BYPASS"
                    path = "BYPASS"
                elapsed_ms = (time.perf_counter() - t0_query) * 1000.0

            elif tier_decision == Tier.BYPASS or not matched_text:
                final_decision = "BYPASS"
                path = "BYPASS"
                elapsed_ms = (time.perf_counter() - t0_query) * 1000.0

            else:
                # AMBIGUOUS: Local Equivalence Decision Step (Candidate c)
                loc_res = local_step.decide(
                    query_a=matched_text,
                    query_b=q,
                    similarity_score=similarity,
                    stability_confidence=stability_conf,
                    domain=domain,
                )
                final_decision = loc_res.decision
                path = "LOCAL_EQUIVALENCE"
                guard_reason = loc_res.guard_reason
                local_score = loc_res.score
                elapsed_ms = (time.perf_counter() - t0_query) * 1000.0

        all_latencies_ms.append(elapsed_ms)
        path_counts[path] += 1
        path_latencies[path].append(elapsed_ms)

        # Evaluate outcome
        outcome = "N/A"
        is_hit = (final_decision == "REUSE")
        is_non_exact = is_hit and (path != "CANONICAL_EXACT")

        if is_seed:
            # Cold seeds are indexed to allow subsequent reuse
            outcome = "SEED_INDEXED"
            canonical_cache.put(q, {"id": ev_id, "query_text": q, "domain": domain})
            cache.index(q, metadata={"id": ev_id, "query_text": q, "domain": domain})
        else:
            # Follow-up queries test cache correctness
            if is_hit:
                total_hits += 1
                source_metrics[source]["hits"] += 1
                if is_non_exact:
                    non_exact_hits += 1
                    source_metrics[source]["non_exact_hits"] += 1

                if is_safe:
                    total_tp += 1
                    source_metrics[source]["tp"] += 1
                    if is_non_exact:
                        non_exact_tp += 1
                    outcome = "TP"
                else:
                    total_fp += 1
                    source_metrics[source]["fp"] += 1
                    if is_non_exact:
                        non_exact_fp += 1
                        source_metrics[source]["non_exact_fp"] += 1
                    outcome = "FP"
            else:
                if is_safe:
                    total_fn += 1
                    outcome = "FN"
                else:
                    total_tn += 1
                    outcome = "TN"

            if path == "CANONICAL_EXACT":
                exact_canonical_hits += 1

            # Index new query if not a hit
            if not is_hit:
                canonical_cache.put(q, {"id": ev_id, "query_text": q, "domain": domain})
                cache.index(q, metadata={"id": ev_id, "query_text": q, "domain": domain})

        telemetry.append({
            "id": ev_id,
            "query": q,
            "source": source,
            "is_seed": is_seed,
            "is_reuse_safe": is_safe,
            "final_decision": final_decision,
            "path": path,
            "outcome": outcome,
            "similarity": round(similarity, 4),
            "stability_confidence": round(stability_conf, 4),
            "local_score": round(local_score, 4),
            "guard_reason": guard_reason,
            "latency_ms": round(elapsed_ms, 3),
            "matched_text": matched_text,
            "matched_id": matched_id,
        })

        if idx % 500 == 0 or idx == len(stream):
            print(f"  Processed [{idx:04d}/{len(stream)}] queries | Path hits so far: {total_hits} (Non-exact: {non_exact_hits})")

    eval_duration_s = time.time() - t0_eval
    print(f"\nExecution completed in {eval_duration_s:.2f} s ({eval_duration_s / 60:.2f} min)")

    # 4. Endpoints & Statistical Confidence Intervals
    overall_hr = total_hits / len(stream)
    overall_hr_ci = clopper_pearson_ci(total_hits, len(stream))

    overall_irr = (total_fp / total_hits) if total_hits > 0 else 0.0
    overall_irr_ci = clopper_pearson_ci(total_fp, total_hits)

    non_exact_irr = (non_exact_fp / non_exact_hits) if non_exact_hits > 0 else 0.0
    non_exact_irr_ci = clopper_pearson_ci(non_exact_fp, non_exact_hits)

    calls_avoided_per_1k = (total_hits / len(stream)) * 1000.0
    tokens_avoided_per_1k = calls_avoided_per_1k * baseline_mean_tokens

    # Latency percentiles
    p50_decision = float(np.median(all_latencies_ms))
    p95_decision = float(np.percentile(all_latencies_ms, 95))

    # End-to-end simulated latency (hits take decision time; misses take decision + baseline generation latency)
    e2e_latencies = [
        t["latency_ms"] if t["final_decision"] == "REUSE" else (t["latency_ms"] + baseline_p50)
        for t in telemetry
    ]
    p50_e2e = float(np.median(e2e_latencies))
    p95_e2e = float(np.percentile(e2e_latencies, 95))

    # Break-even calculation:
    # Mean cache overhead on misses / Mean baseline latency saved on hits
    mean_overhead = float(np.mean(all_latencies_ms))
    mean_saved = frozen_cfg["baseline_generation_reference"]["mean_latency_ms"]
    break_even_hit_rate = mean_overhead / mean_saved if mean_saved > 0 else 0.0

    # Repeat share sweep curve: 0%, 10%, 20%, 40%, 60%
    sweep_shares = [0.0, 0.10, 0.20, 0.40, 0.60]
    repeat_curve = []
    # Observed hit rate among follow-ups
    n_followups = sum(1 for e in stream if not e["is_seed"])
    cond_hit_rate = (total_hits / n_followups) if n_followups > 0 else 0.0

    for share in sweep_shares:
        projected_hr = share * cond_hit_rate
        projected_e2e_p50 = projected_hr * p50_decision + (1.0 - projected_hr) * (p50_decision + baseline_p50)
        repeat_curve.append({
            "repeat_share": share,
            "projected_hit_rate": round(projected_hr, 4),
            "projected_p50_ms": round(projected_e2e_p50, 2),
        })

    # Gate A checks
    gate_a_passed = (non_exact_hits >= 100) and (non_exact_irr <= 0.10)

    summary = {
        "timestamp": "2026-10-12T01:30:00+05:30",
        "stream_checksum": actual_stream_hash,
        "n_queries": len(stream),
        "primary_endpoint_non_exact_irr": {
            "non_exact_hits": non_exact_hits,
            "non_exact_tp": non_exact_tp,
            "non_exact_fp": non_exact_fp,
            "point_estimate": round(non_exact_irr, 4),
            "ci_95": [round(non_exact_irr_ci[0], 4), round(non_exact_irr_ci[1], 4)],
            "target_point_estimate_ceiling": 0.10,
            "cleared_point_estimate": bool(non_exact_irr <= 0.10),
            "cleared_ci_upper_bound": bool(non_exact_irr_ci[1] <= 0.10),
        },
        "secondary_endpoints": {
            "total_hits": total_hits,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn,
            "hit_rate": round(overall_hr, 4),
            "hit_rate_ci": [round(overall_hr_ci[0], 4), round(overall_hr_ci[1], 4)],
            "overall_irr": round(overall_irr, 4),
            "overall_irr_ci": [round(overall_irr_ci[0], 4), round(overall_irr_ci[1], 4)],
            "exact_canonical_hits": exact_canonical_hits,
            "calls_avoided_per_1000": round(calls_avoided_per_1k, 1),
            "tokens_avoided_per_1000": round(tokens_avoided_per_1k, 1),
            "break_even_hit_rate": round(break_even_hit_rate, 5),
            "repeat_curve": repeat_curve,
        },
        "path_resolution": {
            "counts": path_counts,
            "latencies_ms": {
                path: {
                    "count": len(lats),
                    "mean": round(float(np.mean(lats)), 2) if lats else 0.0,
                    "p50": round(float(np.median(lats)), 2) if lats else 0.0,
                    "p95": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
                }
                for path, lats in path_latencies.items()
            },
        },
        "latency_comparison_vs_baseline": {
            "baseline_model": frozen_cfg["baseline_generation_reference"]["model"],
            "baseline_p50_ms": baseline_p50,
            "baseline_p95_ms": baseline_p95,
            "decision_p50_ms": round(p50_decision, 2),
            "decision_p95_ms": round(p95_decision, 2),
            "e2e_simulated_p50_ms": round(p50_e2e, 2),
            "e2e_simulated_p95_ms": round(p95_e2e, 2),
        },
        "source_breakdown": source_metrics,
        "gate_a": {
            "passed": gate_a_passed,
            "non_exact_hits_ge_100": bool(non_exact_hits >= 100),
            "non_exact_irr_le_10_percent": bool(non_exact_irr <= 0.10),
        },
    }

    # Persist artifacts
    with open(TELEMETRY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE A4 EVALUATION SUMMARY & GATE A VERIFICATION")
    print("=" * 80)
    print(f"Total Stream Queries:      {len(stream)}")
    print(f"Total Cache Hits:          {total_hits} (TP={total_tp}, FP={total_fp})")
    print(f"Overall Hit Rate:          {overall_hr:.2%} (95% CI: [{overall_hr_ci[0]:.2%}, {overall_hr_ci[1]:.2%}])")
    print(f"Overall IRR:               {overall_irr:.2%} (95% CI: [{overall_irr_ci[0]:.2%}, {overall_irr_ci[1]:.2%}])")
    print("-" * 80)
    print("PRIMARY ENDPOINT (NON-EXACT MATCHES):")
    print(f"  Non-Exact Hits:          {non_exact_hits} / {total_hits}")
    print(f"  Non-Exact TP:            {non_exact_tp}")
    print(f"  Non-Exact FP:            {non_exact_fp}")
    print(f"  Non-Exact IRR:           {non_exact_irr:.2%} (95% CI: [{non_exact_irr_ci[0]:.2%}, {non_exact_irr_ci[1]:.2%}])")
    print(f"  Gate A Criteria:         Point estimate <= 10.0% AND Non-exact hits >= 100")
    print(f"  Gate A Passed:           {gate_a_passed}")
    print("-" * 80)
    print(f"Calls avoided per 1,000:   {calls_avoided_per_1k:.1f} calls")
    print(f"Tokens avoided per 1,000:  {tokens_avoided_per_1k:.1f} tokens")
    print(f"Break-even Hit Rate:       {break_even_hit_rate:.4%}")
    print(f"End-to-End p50 Latency:    {p50_e2e:.2f} ms (vs No-Cache: {baseline_p50:.2f} ms)")
    print(f"End-to-End p95 Latency:    {p95_e2e:.2f} ms (vs No-Cache: {baseline_p95:.2f} ms)")
    print("=" * 80)


if __name__ == "__main__":
    main()
