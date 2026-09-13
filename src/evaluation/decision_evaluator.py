"""Phase 3 — Decision Evaluator.

Evaluates the tiered reuse decision layer against query_pair_reuse_benchmark.json.

=============================================================================
EVALUATION PROTOCOL — STATED BEFORE ANY CODE BELOW THIS LINE
=============================================================================

Dataset: query_pair_reuse_benchmark.json (same 120 pairs used for Phase 2).
This is a SAME-SET COMPARISON, not a generalization test.  See the framing
rule in the REQUIRED FRAMING section below.

For each pair (query_a, query_b, is_reuse_safe, domain):

  1. Run Phase 1 (StabilityClassifier.classify) on query_b (the incoming query).
  2. Run Phase 2 (SemanticCache: index query_a, lookup query_b) to get
     CacheLookupResult — using the Phase 2 evaluation protocol verbatim.
  3. Run Phase 3 TierRouter to classify as AUTO_REUSE, AMBIGUOUS, or BYPASS.
  4. For AMBIGUOUS: run DecisionStep to get a final REUSE or BYPASS decision.
  5. Map final decision to HIT/MISS:
       AUTO_REUSE → HIT (verbatim cached answer)
       AMBIGUOUS + DecisionStep → REUSE → HIT
       AMBIGUOUS + DecisionStep → BYPASS → MISS
       BYPASS → MISS
  6. Compare HIT/MISS against is_reuse_safe ground truth → TP/FP/FN/TN.

Tier-by-tier tracking: every pair is tagged with its tier (AUTO_REUSE,
AMBIGUOUS, BYPASS) and the per-tier confusion matrix is reported separately.

CategoryHistory is updated online: after each pair is processed (in benchmark
order), the domain history is updated with the ACTUAL outcome (was the HIT
decision correct?).  Only REUSE decisions (HIT) update history; MISS
decisions do not, because they reveal no information about reuse safety.

=============================================================================
REQUIRED FRAMING — VERBATIM (same-set comparison, not generalization)
=============================================================================

All results from this evaluator on query_pair_reuse_benchmark.json are
labeled as SAME-SET COMPARISON.  The dataset was already used to establish
Phase 2's baseline.  Results show whether the tiered decision layer improves
outcomes on known pairs, not whether it generalizes to unseen traffic.

From docs/phase3_walkthrough.md (required statement):
  "These results are a same-set comparison against query_pair_reuse_benchmark.json,
  already used to establish Phase 2's baseline. They show whether the tiered
  decision layer improves outcomes on known pairs, not whether it generalizes
  to unseen traffic. A fresh pair dataset (Option A) would be required before
  claiming generalization."

Do not use the word "generalizes" or "reduces cache hazard" without the
qualifier "on the same 120 pairs Phase 2 was evaluated on" anywhere in this
module's output.

=============================================================================
AMBIGUOUS-BAND FRACTION — HEADLINE DoD METRIC
=============================================================================

ambiguous_fraction = (number of pairs that fell into AMBIGUOUS tier) / 120

If this fraction exceeds 0.50 (majority of traffic), evaluation is flagged
with a WARNING and further decision-step development is NOT recommended
until boundaries are tightened.  A large ambiguous band indicates
Phase 1/2 thresholds need revisiting, not more agent logic on top.

=============================================================================
AUTO-REUSE VERBATIM PROOF
=============================================================================

The evaluator verifies that for every AUTO_REUSE-tier pair, the Phase 3
decision (HIT with score, sim, matched_metadata from Phase 2 cache) is
byte-identical to what Phase 2's plain SemanticCache.lookup() would have
returned at the Phase 2 reference threshold.  Specifically:
  - similarity_score must be identical
  - matched_metadata must be identical
  - decision must be HIT (Phase 3 never downgrades AUTO_REUSE to MISS)

This is tested directly in tests/test_decision_evaluation.py.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.cache.embedding import QueryEmbedder
from src.cache.semantic_cache import CacheDecision, SemanticCache, CacheLookupResult
from src.cache.vector_store import FlatVectorStore
from src.classifier.models import StabilityLabel
from src.classifier.stability_classifier import StabilityClassifier
from src.decision.decision_step import CategoryHistory, DecisionResult, DecisionStep
from src.decision.tier_router import Tier, TierBoundaries, TierRouter, DEFAULT_BOUNDARIES
from src.evaluation.cache_evaluator import CacheEvaluator


# ---------------------------------------------------------------------------
# DoD threshold for ambiguous-band fraction
# ---------------------------------------------------------------------------
AMBIGUOUS_BAND_LARGE_FRACTION_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Per-tier confusion matrix helper
# ---------------------------------------------------------------------------

@dataclass
class TierMetrics:
    """Confusion matrix and counts for a single tier."""

    tier: str
    count: int = 0       # total pairs assigned to this tier
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def hit_count(self) -> int:
        return self.tp + self.fp

    @property
    def irr_cache(self) -> float:
        return (self.fp / self.hit_count) if self.hit_count > 0 else 0.0

    @property
    def crr(self) -> float:
        return (self.tp / self.hit_count) if self.hit_count > 0 else 1.0

    @property
    def arr(self) -> float:
        return (self.hit_count / self.count) if self.count > 0 else 0.0


# ---------------------------------------------------------------------------
# Top-level Phase 3 metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionMetrics:
    """Full Phase 3 evaluation metrics.

    All framing labels must include "same-set comparison" when these numbers
    are reported externally — they are NOT generalization results.
    """

    dataset_path: str
    total_pairs: int

    # ── Tier distribution ──────────────────────────────────────────────────
    auto_reuse_count: int
    ambiguous_count: int
    bypass_count: int

    ambiguous_fraction: float       # HEADLINE DoD metric
    ambiguous_band_large: bool      # True if fraction > 0.50

    # ── Overall Phase 3 confusion matrix ───────────────────────────────────
    tp: int
    fp: int
    fn: int
    tn: int

    arr: float           # (TP+FP) / N
    crr: float           # TP / (TP+FP); 1.0 if no hits
    irr_cache: float     # FP / (TP+FP); 0.0 if no hits
    irr_traffic: float   # FP / N
    frr: float           # FN / (TP+FN); 0.0 if no safe pairs

    hit_count: int
    miss_count: int

    # ── Per-tier metrics ───────────────────────────────────────────────────
    tier_metrics: Dict[str, Any] = field(default_factory=dict)

    # ── Phase 2 comparison (same-set) ──────────────────────────────────────
    phase2_reference_threshold: float = 0.85
    phase2_tp: int = 0
    phase2_fp: int = 0
    phase2_fn: int = 0
    phase2_tn: int = 0
    phase2_arr: float = 0.0
    phase2_crr: float = 0.0
    phase2_irr_cache: float = 0.0
    phase2_irr_traffic: float = 0.0
    phase2_frr: float = 0.0

    # ── Auto-reuse verbatim verification ──────────────────────────────────
    auto_reuse_verbatim_verified: bool = False  # True = all AUTO_REUSE matches Phase 2
    auto_reuse_verbatim_mismatches: int = 0

    # ── Detailed per-pair results ──────────────────────────────────────────
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DecisionEvaluator
# ---------------------------------------------------------------------------

class DecisionEvaluator:
    """Evaluates Phase 3 against query_pair_reuse_benchmark.json.

    Runs Phase 1 (stability), Phase 2 (cache), and Phase 3 (tier router +
    decision step) for each pair, then reports:
      - Ambiguous-band traffic fraction (the headline DoD metric)
      - Tier-by-tier accuracy / hazard breakdown
      - Phase 2 vs. Phase 3 same-set comparison table
      - Auto-reuse verbatim verification

    The Phase 1 classifier is run on query_b (the incoming request).
    The Phase 2 cache uses the pair protocol: index query_a, lookup query_b.

    Args:
        embedder: Optional custom QueryEmbedder. Defaults to all-MiniLM-L6-v2
            with the pinned MODEL_REVISION.
        classifier: Optional StabilityClassifier. Defaults to a fresh instance.
        tier_router: Optional TierRouter. Defaults to TierRouter with DEFAULT_BOUNDARIES.
        decision_step: Optional DecisionStep. Defaults to DecisionStep with fresh
            CategoryHistory (online update during evaluation).
    """

    def __init__(
        self,
        embedder: Optional[QueryEmbedder] = None,
        classifier: Optional[StabilityClassifier] = None,
        tier_router: Optional[TierRouter] = None,
        decision_step: Optional[DecisionStep] = None,
    ) -> None:
        self.embedder = embedder or QueryEmbedder()
        self.classifier = classifier or StabilityClassifier()
        self.tier_router = tier_router or TierRouter()
        self.decision_step = decision_step or DecisionStep()

    def evaluate(
        self,
        dataset_path: Path,
        phase2_reference_threshold: float = 0.85,
    ) -> DecisionMetrics:
        """Run Phase 3 evaluation against the pair benchmark.

        Args:
            dataset_path: Path to query_pair_reuse_benchmark.json.
            phase2_reference_threshold: The Phase 2 reference threshold used
                to compute Phase 2's baseline numbers for comparison.
                Default: 0.85 (the Phase 2 documented reference point).

        Returns:
            DecisionMetrics with all DoD metrics and the same-set comparison.
        """
        records = CacheEvaluator.load_and_validate_dataset(dataset_path)

        # Shared Phase 2 cache setup
        store = FlatVectorStore(dim=self.embedder.dim)
        cache = SemanticCache(
            threshold=phase2_reference_threshold,
            embedder=self.embedder,
            vector_store=store,
        )

        # Reset decision step history for a clean run
        self.decision_step.history.reset()

        # Per-tier confusion-matrix accumulators
        tier_counts: Dict[str, TierMetrics] = {
            Tier.AUTO_REUSE.value: TierMetrics(tier=Tier.AUTO_REUSE.value),
            Tier.AMBIGUOUS.value: TierMetrics(tier=Tier.AMBIGUOUS.value),
            Tier.BYPASS.value: TierMetrics(tier=Tier.BYPASS.value),
        }

        # Overall Phase 3 counts
        tp = tn = fp = fn = 0

        # Phase 2 re-evaluation (same protocol, same 120 pairs)
        p2_tp = p2_tn = p2_fp = p2_fn = 0

        # Auto-reuse verbatim verification
        auto_reuse_mismatches = 0

        detailed_results: List[Dict[str, Any]] = []

        for record in records:
            query_a: str = record["query_a"].strip()
            query_b: str = record["query_b"].strip()
            is_safe: bool = record["is_reuse_safe"]
            domain: str = record["domain"]

            # ── Phase 2: fresh store, index query_a, lookup query_b ─────────
            cache.vector_store.reset()
            cache.index(query_a, metadata={"source_id": record["id"], "query": query_a})
            cache_result: CacheLookupResult = cache.lookup(query_b)

            # ── Phase 2 baseline confusion matrix (at reference threshold) ──
            p2_hit = cache_result.is_hit
            if p2_hit and is_safe:
                p2_tp += 1
            elif p2_hit and not is_safe:
                p2_fp += 1
            elif not p2_hit and is_safe:
                p2_fn += 1
            else:
                p2_tn += 1

            # ── Phase 1: classify query_b (the incoming request) ────────────
            stability_result = self.classifier.classify(query_b)

            # ── Phase 3: tier routing ────────────────────────────────────────
            tier: Tier = self.tier_router.route(cache_result, stability_result)

            # ── Phase 3: decision step for AMBIGUOUS; verbatim for AUTO_REUSE ─
            decision_result: Optional[DecisionResult] = None
            if tier == Tier.AUTO_REUSE:
                # Verbatim: reuse the Phase 2 answer, no additional call
                predicted_hit = True

                # Verbatim verification: AUTO_REUSE must agree with Phase 2 HIT
                # (by definition, if sim >= auto_reuse_sim_floor >= Phase 2 threshold,
                # the Phase 2 cache would also report HIT — verify this invariant)
                if not cache_result.is_hit:
                    auto_reuse_mismatches += 1

            elif tier == Tier.AMBIGUOUS:
                # Decision step: similarity + confidence + history
                decision_result = self.decision_step.decide(
                    cache_result, stability_result, domain
                )
                predicted_hit = (decision_result.decision == "REUSE")
            else:  # BYPASS
                predicted_hit = False

            # ── Confusion matrix update ──────────────────────────────────────
            t_key = tier.value
            tier_counts[t_key].count += 1

            if predicted_hit and is_safe:
                tp += 1
                tier_counts[t_key].tp += 1
            elif predicted_hit and not is_safe:
                fp += 1
                tier_counts[t_key].fp += 1
            elif not predicted_hit and is_safe:
                fn += 1
                tier_counts[t_key].fn += 1
            else:
                tn += 1
                tier_counts[t_key].tn += 1

            # ── Online history update for REUSE decisions only ───────────────
            if predicted_hit:
                self.decision_step.history.update(domain, was_correct_reuse=is_safe)

            # ── Detailed result record ───────────────────────────────────────
            row: Dict[str, Any] = {
                "id": record["id"],
                "query_a": query_a,
                "query_b": query_b,
                "is_reuse_safe": is_safe,
                "domain": domain,
                "similarity_level": record["semantic_similarity_level"],
                "taxonomy_class": record["taxonomy_class"],
                "similarity_score": round(cache_result.similarity_score, 6),
                "stability_confidence": round(stability_result.confidence, 4),
                "stability_effective": stability_result.effective_decision.value,
                "tier": t_key,
                "predicted": "HIT" if predicted_hit else "MISS",
                "outcome": (
                    "TP" if (predicted_hit and is_safe) else
                    "FP" if (predicted_hit and not is_safe) else
                    "FN" if (not predicted_hit and is_safe) else
                    "TN"
                ),
                "phase2_predicted": "HIT" if p2_hit else "MISS",
                "phase2_outcome": (
                    "TP" if (p2_hit and is_safe) else
                    "FP" if (p2_hit and not is_safe) else
                    "FN" if (not p2_hit and is_safe) else
                    "TN"
                ),
            }
            if decision_result is not None:
                row["decision_step"] = decision_result.to_dict()
            detailed_results.append(row)

        # ── Aggregate metrics ────────────────────────────────────────────────
        n = len(records)
        n_safe = tp + fn

        auto_count = tier_counts[Tier.AUTO_REUSE.value].count
        amb_count = tier_counts[Tier.AMBIGUOUS.value].count
        byp_count = tier_counts[Tier.BYPASS.value].count

        ambiguous_fraction = amb_count / n if n > 0 else 0.0
        ambiguous_band_large = ambiguous_fraction > AMBIGUOUS_BAND_LARGE_FRACTION_THRESHOLD

        hit_count = tp + fp
        miss_count = fn + tn

        arr = hit_count / n if n > 0 else 0.0
        crr = (tp / hit_count) if hit_count > 0 else 1.0
        irr_cache = (fp / hit_count) if hit_count > 0 else 0.0
        irr_traffic = fp / n if n > 0 else 0.0
        frr = (fn / n_safe) if n_safe > 0 else 0.0

        # Phase 2 derived
        p2_n = n
        p2_hit_count = p2_tp + p2_fp
        p2_n_safe = p2_tp + p2_fn
        p2_arr = p2_hit_count / p2_n if p2_n > 0 else 0.0
        p2_crr = (p2_tp / p2_hit_count) if p2_hit_count > 0 else 1.0
        p2_irr_cache = (p2_fp / p2_hit_count) if p2_hit_count > 0 else 0.0
        p2_irr_traffic = p2_fp / p2_n if p2_n > 0 else 0.0
        p2_frr = (p2_fn / p2_n_safe) if p2_n_safe > 0 else 0.0

        return DecisionMetrics(
            dataset_path=str(dataset_path),
            total_pairs=n,
            auto_reuse_count=auto_count,
            ambiguous_count=amb_count,
            bypass_count=byp_count,
            ambiguous_fraction=round(ambiguous_fraction, 4),
            ambiguous_band_large=ambiguous_band_large,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            arr=round(arr, 6),
            crr=round(crr, 6),
            irr_cache=round(irr_cache, 6),
            irr_traffic=round(irr_traffic, 6),
            frr=round(frr, 6),
            hit_count=hit_count,
            miss_count=miss_count,
            tier_metrics={k: {
                "count": v.count,
                "tp": v.tp,
                "fp": v.fp,
                "fn": v.fn,
                "tn": v.tn,
                "arr": round(v.arr, 4),
                "crr": round(v.crr, 4),
                "irr_cache": round(v.irr_cache, 4),
            } for k, v in tier_counts.items()},
            phase2_reference_threshold=phase2_reference_threshold,
            phase2_tp=p2_tp,
            phase2_fp=p2_fp,
            phase2_fn=p2_fn,
            phase2_tn=p2_tn,
            phase2_arr=round(p2_arr, 6),
            phase2_crr=round(p2_crr, 6),
            phase2_irr_cache=round(p2_irr_cache, 6),
            phase2_irr_traffic=round(p2_irr_traffic, 6),
            phase2_frr=round(p2_frr, 6),
            auto_reuse_verbatim_verified=(auto_reuse_mismatches == 0),
            auto_reuse_verbatim_mismatches=auto_reuse_mismatches,
            detailed_results=detailed_results,
        )

    def print_report(self, m: DecisionMetrics) -> None:
        """Print Phase 3 evaluation report.

        All numbers reported are labeled as SAME-SET COMPARISON per the
        required framing rule in the module docstring.
        """
        print("=" * 76)
        print("PHASE 3 REUSE DECISION LAYER — EVALUATION REPORT")
        print("SAME-SET COMPARISON (query_pair_reuse_benchmark.json, N=120)")
        print("Results show improvement on KNOWN pairs, NOT generalization.")
        print("=" * 76)
        print(f"Dataset : {m.dataset_path}")
        print(f"Total Pairs : {m.total_pairs}")
        print()

        # Ambiguous-band fraction (headline DoD metric)
        print("-- HEADLINE DoD METRIC: AMBIGUOUS-BAND TRAFFIC FRACTION --")
        print(f"  AUTO_REUSE  : {m.auto_reuse_count:3d} pairs ({m.auto_reuse_count/m.total_pairs*100:.1f}%)")
        print(f"  AMBIGUOUS   : {m.ambiguous_count:3d} pairs ({m.ambiguous_fraction*100:.1f}%)")
        print(f"  BYPASS      : {m.bypass_count:3d} pairs ({m.bypass_count/m.total_pairs*100:.1f}%)")
        if m.ambiguous_band_large:
            print()
            print("  !! WARNING: Ambiguous-band fraction > 50% — boundaries may be")
            print("  !! too wide. Tighten boundaries against Phase 1/2 distributions")
            print("  !! before adding more decision-step logic. See spec §AMBIGUOUS.")
        print()

        # Same-set comparison table
        print("── PHASE 2 vs. PHASE 3 SAME-SET COMPARISON (same 120 pairs) ──")
        print(f"  (Phase 2 threshold: {m.phase2_reference_threshold:.2f} — reference only, not production-safe)")
        hdr = f"{'Metric':<28} {'Phase 2 (baseline)':>20} {'Phase 3 (tiered)':>18}"
        print(f"  {hdr}")
        print(f"  {'-'*68}")
        rows = [
            ("ARR (Actual Reuse Rate)",    m.phase2_arr,         m.arr),
            ("CRR (Correct Reuse Prec.)",  m.phase2_crr,         m.crr),
            ("IRR_cache (Hazard Rate)",    m.phase2_irr_cache,   m.irr_cache),
            ("IRR_traffic (Traffic Haz.)", m.phase2_irr_traffic, m.irr_traffic),
            ("FRR (False Rejection Rate)", m.phase2_frr,         m.frr),
        ]
        for label, p2_val, p3_val in rows:
            print(f"  {label:<28} {p2_val*100:>18.2f}%  {p3_val*100:>16.2f}%")
        print()
        print(f"  Confusion Matrix       Phase 2  Phase 3")
        print(f"    TP (Correct Reuse)  : {m.phase2_tp:>6}   {m.tp:>6}")
        print(f"    FP (Cache Hazard)   : {m.phase2_fp:>6}   {m.fp:>6}")
        print(f"    FN (Missed Reuse)   : {m.phase2_fn:>6}   {m.fn:>6}")
        print(f"    TN (Correct Reject) : {m.phase2_tn:>6}   {m.tn:>6}")
        print()

        # Per-tier breakdown
        print("── PER-TIER BREAKDOWN ──")
        for tier_name in [Tier.AUTO_REUSE.value, Tier.AMBIGUOUS.value, Tier.BYPASS.value]:
            tm = m.tier_metrics.get(tier_name, {})
            if not tm:
                continue
            print(f"  Tier {tier_name}: {tm['count']} pairs")
            print(f"    TP={tm['tp']}, FP={tm['fp']}, FN={tm['fn']}, TN={tm['tn']}")
            print(f"    ARR={tm['arr']*100:.1f}%, IRR_cache={tm['irr_cache']*100:.1f}%, CRR={tm['crr']*100:.1f}%")
        print()

        # Auto-reuse verbatim proof
        print("── AUTO-REUSE VERBATIM PROOF ──")
        if m.auto_reuse_verbatim_verified:
            print("  PASS: All AUTO_REUSE decisions are byte-identical to Phase 2 cache output.")
            print("        Decision step was never invoked for AUTO_REUSE pairs.")
        else:
            print(f"  FAIL: {m.auto_reuse_verbatim_mismatches} AUTO_REUSE pair(s) did NOT match Phase 2 output.")
            print("        This is a bug — investigate immediately.")
        print("=" * 76)
        print("NOTE: These results are a SAME-SET COMPARISON on the 120 pairs Phase 2")
        print("was evaluated on. They do NOT claim generalization to unseen traffic.")
        print("=" * 76)
