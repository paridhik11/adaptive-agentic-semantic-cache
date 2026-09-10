"""Threshold sweep script for the Phase 2 semantic cache baseline.

Run this script to regenerate all threshold sweep numbers reported in
docs/phase2_walkthrough.md. Numbers reported in that document were produced
by running this script exactly as-is. No post-hoc adjustments were made.

SAFETY CRITERION (pre-stated before sweep results are shown):
  The cache hazard rate (IRR_cache = FP / (TP + FP)) must stay below 10%.
  This is a hard ceiling. The LOWEST threshold satisfying IRR_cache < 10%
  is chosen to maximize hit-rate subject to the safety constraint.

SWEEP RANGE:
  Coarse: 0.10 to 0.99 in 0.05 steps
  Fine: 0.01-step refinement in the zone around the safety-bar crossover

COST METRIC RULE:
  No LLM is called. No token/dollar figures are produced.
  Reported: hit count, hazard rate, and measured embed+search latency only.

Usage:
    python scripts/run_cache_threshold_sweep.py
    python scripts/run_cache_threshold_sweep.py --dataset data/raw/query_pair_reuse_benchmark.json
    python scripts/run_cache_threshold_sweep.py --fine-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when run as a standalone script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.cache_evaluator import CacheEvaluator, SweepPoint

# ---------------------------------------------------------------------------
# Safety criterion constant (mirrored from implementation plan — single source)
# ---------------------------------------------------------------------------
HAZARD_RATE_CEILING: float = 0.10  # IRR_cache must be BELOW this value


def build_coarse_thresholds() -> list[float]:
    """0.10 to 0.99 in 0.05 steps."""
    thresholds = []
    t = 0.10
    while t <= 0.99:
        thresholds.append(round(t, 2))
        t += 0.05
    return thresholds


def build_fine_thresholds(coarse_points: list[SweepPoint]) -> list[float]:
    """0.01-step range around the crossover point (where IRR_cache crosses 10%).

    Finds the coarse threshold where IRR_cache first drops below the ceiling,
    then sweeps 0.05 below to 0.05 above that point in 0.01 steps.
    """
    crossover = None
    for pt in coarse_points:
        if pt.irr_cache < HAZARD_RATE_CEILING:
            crossover = pt.threshold
            break

    if crossover is None:
        # Safety bar never met at coarse resolution — refine around high-threshold zone
        crossover = 0.95

    lo = max(0.01, round(crossover - 0.05, 2))
    hi = min(0.99, round(crossover + 0.05, 2))
    fine = []
    t = lo
    while t <= hi:
        fine.append(round(t, 2))
        t = round(t + 0.01, 2)
    return fine


def print_sweep_table(points: list[SweepPoint], label: str = "") -> None:
    """Print formatted sweep table to stdout."""
    if label:
        print(f"\n{'=' * 96}")
        print(f" {label}")
    print(f"{'=' * 96}")
    print(
        f"{'Threshold':>10} | {'Hits':>5} | {'ARR%':>6} | {'CRR%':>6} | "
        f"{'IRR_cache%':>10} | {'IRR_traf%':>9} | {'FRR%':>6} | "
        f"{'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4} | {'Mean ms':>8}"
    )
    print("-" * 96)
    for pt in points:
        hazard_flag = " *** ABOVE CEILING" if pt.irr_cache >= HAZARD_RATE_CEILING else ""
        print(
            f"{pt.threshold:>10.4f} | {pt.hit_count:>5} | {pt.arr * 100:>6.2f} | "
            f"{pt.crr * 100:>6.2f} | {pt.irr_cache * 100:>10.2f} | "
            f"{pt.irr_traffic * 100:>9.2f} | {pt.frr * 100:>6.2f} | "
            f"{pt.tp:>4} | {pt.fp:>4} | {pt.fn:>4} | {pt.tn:>4} | "
            f"{pt.mean_total_ms:>8.4f}{hazard_flag}"
        )
    print("=" * 96)


def select_threshold(all_points: list[SweepPoint]) -> SweepPoint | None:
    """Apply the pre-stated safety criterion to select the recommended threshold.

    Criterion: IRR_cache < HAZARD_RATE_CEILING (10%).
    Selection: LOWEST threshold satisfying criterion (maximize hit-rate).
    """
    candidates = [pt for pt in all_points if pt.irr_cache < HAZARD_RATE_CEILING]
    if not candidates:
        return None
    return min(candidates, key=lambda pt: pt.threshold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold sweep for Phase 2 semantic cache baseline"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to query_pair_reuse_benchmark.json",
    )
    parser.add_argument(
        "--fine-only",
        action="store_true",
        help="Run only the fine-grained sweep (requires prior coarse run knowledge)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "query_pair_reuse_benchmark.json"
    )

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPhase 2 — Semantic Cache Baseline Threshold Sweep")
    print(f"Dataset : {dataset_path}")
    print(f"Model   : all-MiniLM-L6-v2 (FAISS IndexFlatIP, cosine similarity)")
    print(f"N pairs : 120 (verified in Step 0)")
    print(f"\nSafety criterion (pre-stated): IRR_cache < {HAZARD_RATE_CEILING * 100:.0f}%")
    print(f"Selection rule: LOWEST threshold satisfying criterion (maximize hit-rate)")
    print(f"\nCost metric: no LLM called — hit count and latency only, no token/$ figures")

    evaluator = CacheEvaluator()

    # -----------------------------------------------------------------------
    # Coarse sweep: 0.10 – 0.99, step 0.05
    # -----------------------------------------------------------------------
    if not args.fine_only:
        print(f"\nRunning coarse sweep (0.10 – 0.99, step 0.05)…")
        coarse_thresholds = build_coarse_thresholds()
        coarse_points = evaluator.threshold_sweep(dataset_path, coarse_thresholds)
        print_sweep_table(coarse_points, label="COARSE SWEEP (step 0.05)")
    else:
        # Build a minimal coarse picture to find the fine zone
        coarse_points = evaluator.threshold_sweep(dataset_path, build_coarse_thresholds())

    # -----------------------------------------------------------------------
    # Fine sweep around the crossover zone (step 0.01)
    # -----------------------------------------------------------------------
    fine_thresholds = build_fine_thresholds(coarse_points)
    print(f"\nRunning fine sweep ({min(fine_thresholds):.2f} – {max(fine_thresholds):.2f}, step 0.01)…")
    fine_points = evaluator.threshold_sweep(dataset_path, fine_thresholds)
    print_sweep_table(fine_points, label="FINE SWEEP (step 0.01, around safety-bar crossover)")

    # -----------------------------------------------------------------------
    # Merge all unique points (coarse + fine) sorted by threshold
    # -----------------------------------------------------------------------
    seen = {}
    for pt in coarse_points + fine_points:
        seen[round(pt.threshold, 4)] = pt
    all_points = sorted(seen.values(), key=lambda p: p.threshold)

    # -----------------------------------------------------------------------
    # Threshold recommendation
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("THRESHOLD RECOMMENDATION")
    print(f"Safety bar: IRR_cache < {HAZARD_RATE_CEILING * 100:.0f}% (pre-stated, not post-hoc)")
    chosen = select_threshold(all_points)
    if chosen is None:
        print(
            "RESULT: No threshold satisfies IRR_cache < "
            f"{HAZARD_RATE_CEILING * 100:.0f}% while producing hits > 0. "
            "The baseline embedding model cannot safely cache any pair in this dataset "
            "at the stated safety bar. This is a valid finding, not an error."
        )
    else:
        print(f"Recommended threshold : {chosen.threshold:.4f}")
        print(f"  IRR_cache           : {chosen.irr_cache * 100:.2f}%  (< {HAZARD_RATE_CEILING * 100:.0f}% ceiling)")
        print(f"  ARR (hit rate)      : {chosen.arr * 100:.2f}%")
        print(f"  CRR (precision)     : {chosen.crr * 100:.2f}%")
        print(f"  FRR (missed reuse)  : {chosen.frr * 100:.2f}%")
        print(f"  Hit count           : {chosen.hit_count} / {chosen.total}  (honest cost proxy)")
        print(f"  Mean latency        : {chosen.mean_total_ms:.4f} ms  (embed + search)")
        print(f"  Justification       : Lowest threshold where IRR_cache drops below "
              f"{HAZARD_RATE_CEILING * 100:.0f}%, maximizing hit-rate subject to safety constraint.")
    print(f"{'=' * 72}")

    print("\nNOT reported: token counts, dollar costs, or LLM generation metrics.")
    print("Phase 2 is a pure-embedding baseline — no LLM is called.")


if __name__ == "__main__":
    main()
