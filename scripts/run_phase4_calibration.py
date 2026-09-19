"""Phase 4 — Threshold Calibration and Feedback Loop Integration.

Combines historical benchmark outcomes (N=120) with newly labeled
synthetic feedback pairs (N=108), evaluates dual sample-size gates,
runs fine-grained threshold sweeps, and computes exact Clopper-Pearson
binomial CIs and Delta-method logistic CIs.

Generates comprehensive markdown/JSON reports for phase4_walkthrough.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.cache.embedding import QueryEmbedder
from src.decision.adaptive_threshold_engine import (
    AdaptiveThresholdEngine,
    CategoryThreshold,
    DomainSweepPoint,
    DEFAULT_CALIBRATED_THRESHOLDS,
    FALLBACK_THRESHOLD,
    MINIMUM_CATEGORY_N,
    MINIMUM_MINORITY_CLASS_N,
    SAFETY_CEILING_IRR,
    calibrate_category_with_feedback,
    clopper_pearson_ci,
    wilson_score_ci,
)

BENCHMARK_PATH = REPO_ROOT / "data" / "raw" / "query_pair_reuse_benchmark.json"
FEEDBACK_TELEMETRY_PATH = REPO_ROOT / "data" / "openrouter_synthetic_feedback_telemetry.json"
RESULTS_OUTPUT_PATH = REPO_ROOT / "data" / "phase4_threshold_calibration_results.json"


def load_combined_data() -> Dict[str, List[Tuple[float, bool]]]:
    """Load and combine benchmark pairs with synthetic feedback pairs."""
    embedder = QueryEmbedder()

    # 1. Historical Benchmark (N=120)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    combined: Dict[str, List[Tuple[float, bool]]] = {}
    print(f"Loading {len(bench_data)} historical benchmark pairs...")
    for r in bench_data:
        dom = r["domain"]
        sim = float(np.dot(embedder.encode(r["query_a"]), embedder.encode(r["query_b"])))
        safe = bool(r["is_reuse_safe"])
        combined.setdefault(dom, []).append((sim, safe))

    # 2. Synthetic Feedback Pairs — genuine judge calls only
    # CRITICAL: exclude any record where fallback_triggered=True.
    # Fallback records are API-error stubs that fail-closed to BYPASS
    # (judge_decision="BYPASS", judge_is_safe=False) as a production safety
    # invariant when OpenRouter's free-tier daily quota was exhausted. They
    # carry *no* real judge signal — consuming them as "judge says unsafe"
    # labels silently contaminates the threshold sweep with 86 fake negatives.
    # Only records with a real request_id and non-null raw_response are evidence.
    if not FEEDBACK_TELEMETRY_PATH.exists():
        raise FileNotFoundError(f"Missing feedback telemetry at {FEEDBACK_TELEMETRY_PATH}")

    with open(FEEDBACK_TELEMETRY_PATH, "r", encoding="utf-8") as f:
        feedback_data = json.load(f)

    genuine_records = [r for r in feedback_data if not r.get("fallback_triggered", False)]
    excluded_count = len(feedback_data) - len(genuine_records)

    print(f"Loaded {len(feedback_data)} synthetic telemetry records from {FEEDBACK_TELEMETRY_PATH.name}.")
    print(f"  Genuine judge calls (fallback_triggered=False): {len(genuine_records)}")
    print(f"  Excluded fallback stubs (API errors, not real judge decisions): {excluded_count}")
    if excluded_count > 0:
        print(f"  WARNING: {excluded_count} stub records excluded — re-run label_synthetic_feedback.py")
        print(f"           across future quota days to obtain real labels for those pairs.")

    # Per-domain genuine counts for transparency
    domain_genuine: Dict[str, int] = {}
    domain_excluded: Dict[str, int] = {}
    for r in feedback_data:
        dom = r["domain"]
        if r.get("fallback_triggered", False):
            domain_excluded[dom] = domain_excluded.get(dom, 0) + 1
        else:
            domain_genuine[dom] = domain_genuine.get(dom, 0) + 1
    all_doms = sorted(set(list(domain_genuine) + list(domain_excluded)))
    print(f"  Per-domain genuine/excluded breakdown:")
    for dom in all_doms:
        g = domain_genuine.get(dom, 0)
        e = domain_excluded.get(dom, 0)
        print(f"    {dom:<28}: genuine={g:>3}, excluded={e:>3}")

    for r in genuine_records:
        dom = r["domain"]
        sim = float(r["similarity_score"])
        # Use the LLM judge's decision as the simulated production feedback signal.
        # Only genuine calls with real request_id and raw_response reach here.
        safe = bool(r["judge_is_safe"])
        combined.setdefault(dom, []).append((sim, safe))

    return combined


def print_sweep_table_for_domain(domain: str, points: List[DomainSweepPoint]) -> None:
    """Print formatted sweep table with Clopper-Pearson and Wilson CIs."""
    print(f"\n{'=' * 110}")
    print(f" THRESHOLD SWEEP TABLE: {domain.upper()}")
    print(f"{'=' * 110}")
    print(
        f"{'Thresh':>6} | {'Hits':>4} | {'TP':>3} | {'FP':>3} | "
        f"{'ARR%':>6} | {'CRR%':>6} | {'IRR%':>6} | "
        f"{'CP 95% CI':>17} | {'Wilson 95% CI':>17} | {'Safety Status':>15}"
    )
    print("-" * 110)

    # Subsample points for display: every 0.05 step plus points around crossover
    for pt in points:
        # Show key points (multiples of 0.05 or where clears_safety_ceiling is True)
        is_key = round(pt.threshold * 100) % 5 == 0 or pt.clears_safety_ceiling
        if not is_key and not (0.75 <= pt.threshold <= 0.90):
            continue

        cp_ci_str = f"[{pt.irr_cache_ci_lower*100:>5.1f}%, {pt.irr_cache_ci_upper*100:>5.1f}%]"
        w_ci_str = f"[{pt.wilson_ci_lower*100:>5.1f}%, {pt.wilson_ci_upper*100:>5.1f}%]"
        status = "[CLEARS <=10%]" if pt.clears_safety_ceiling else "EXCEEDS CEILING"

        print(
            f"{pt.threshold:>6.2f} | {pt.hits:>4} | {pt.tp:>3} | {pt.fp:>3} | "
            f"{pt.arr*100:>5.1f}% | {pt.crr*100:>5.1f}% | {pt.irr_cache*100:>5.1f}% | "
            f"{cp_ci_str:>17} | {w_ci_str:>17} | {status:>15}"
        )


def main() -> None:
    print("=" * 110)
    print("PHASE 4: ADAPTIVE PER-CATEGORY THRESHOLD CALIBRATION")
    print(f"Safety Ceiling: IRR_cache CI Upper Bound <= {SAFETY_CEILING_IRR*100:.0f}%")
    print(f"Dual Gates: MINIMUM_CATEGORY_N = {MINIMUM_CATEGORY_N}, MINIMUM_MINORITY_CLASS_N = {MINIMUM_MINORITY_CLASS_N}")
    print("=" * 110)

    combined_data = load_combined_data()

    new_thresholds: Dict[str, CategoryThreshold] = {}
    sweep_results: Dict[str, List[DomainSweepPoint]] = {}
    summary_rows: List[Dict[str, Any]] = []

    domains = sorted(combined_data.keys())
    for dom in domains:
        pairs = combined_data[dom]
        old_ct = DEFAULT_CALIBRATED_THRESHOLDS.get(dom)

        new_ct, points = calibrate_category_with_feedback(
            domain=dom,
            pairs=pairs,
            safety_ceiling=SAFETY_CEILING_IRR,
            minimum_category_n=MINIMUM_CATEGORY_N,
            minimum_minority_class_n=MINIMUM_MINORITY_CLASS_N,
            fallback_threshold=FALLBACK_THRESHOLD,
            confidence_level=0.95,
            sweep_start=0.50,
            sweep_stop=0.95,
            sweep_step=0.01,
        )

        new_thresholds[dom] = new_ct
        sweep_results[dom] = points

        if points:
            print_sweep_table_for_domain(dom, points)

        # Build comparison record
        row = {
            "domain": dom,
            "old_sample_size": old_ct.sample_size if old_ct else 0,
            "old_minority": old_ct.minority_count if old_ct else 0,
            "old_operating_threshold": old_ct.operating_threshold if old_ct else FALLBACK_THRESHOLD,
            "old_is_fallback": old_ct.is_fallback if old_ct else True,
            "new_sample_size": new_ct.sample_size,
            "new_minority": new_ct.minority_count,
            "new_operating_threshold": new_ct.operating_threshold,
            "new_is_fallback": new_ct.is_fallback,
            "new_fallback_reason": new_ct.fallback_reason,
            "new_selection_rule": new_ct.selection_rule,
            "irr_cache_at_operating": new_ct.irr_cache_at_operating,
            "irr_cache_ci_upper": new_ct.irr_cache_ci_upper,
            "hits_at_operating": new_ct.hits_at_operating,
            "theta_point_estimate": new_ct.threshold,
            "theta_ci_lower": new_ct.ci_lower,
            "theta_ci_upper": new_ct.ci_upper,
        }
        summary_rows.append(row)

    # Print comparative before-and-after table
    print(f"\n{'=' * 125}")
    print(" PHASE 4 BEFORE-AND-AFTER PER-CATEGORY THRESHOLD COMPARISON TABLE")
    print(f"{'=' * 125}")
    print(
        f"{'Domain':<22} | {'Old N':>5} | {'Old Min':>7} | {'Old Thresh':>10} | {'Old Status':>10} | "
        f"{'New N':>5} | {'New Min':>7} | {'New Thresh':>10} | {'IRR CI Upp':>10} | {'New Status':>18}"
    )
    print("-" * 125)

    for r in summary_rows:
        old_status = "FALLBACK" if r["old_is_fallback"] else "PER-CAT"
        if r["new_is_fallback"]:
            new_status = "FALLBACK (0.85)"
        else:
            new_status = f"ADOPTED ({r['new_operating_threshold']:.4f})"

        irr_ci_str = f"{r['irr_cache_ci_upper']*100:.1f}%" if r["irr_cache_ci_upper"] is not None else "N/A"

        print(
            f"{r['domain']:<22} | {r['old_sample_size']:>5} | {r['old_minority']:>7} | {r['old_operating_threshold']:>10.4f} | {old_status:>10} | "
            f"{r['new_sample_size']:>5} | {r['new_minority']:>7} | {r['new_operating_threshold']:>10.4f} | {irr_ci_str:>10} | {new_status:>18}"
        )

    # Save complete results
    output_payload = {
        "summary": summary_rows,
        "calibrated_thresholds": {dom: ct.to_dict() for dom, ct in new_thresholds.items()},
        "sweep_points": {
            dom: [pt.to_dict() for pt in pts] for dom, pts in sweep_results.items()
        },
    }

    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved calibration results and sweep telemetry to:\n  {RESULTS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
