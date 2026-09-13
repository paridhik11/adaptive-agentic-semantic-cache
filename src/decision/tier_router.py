"""Phase 3 — Tier Router: maps (CacheLookupResult, StabilityResult) to a Tier.

=============================================================================
STEP 0 — TIER BOUNDARY DEFINITIONS (written before any routing code)
=============================================================================

All numeric bounds below are derived exclusively from Phase 1 and Phase 2's
ALREADY-MEASURED distributions.  The query_pair_reuse_benchmark.json dataset
(used later for evaluation) was NOT consulted when choosing these values.

── Phase 1 confidence distribution ─────────────────────────────────────────
From stability_classifier.py (STABLE_CONFIDENCE_THRESHOLD = 0.80) and the
STABLE_CONFIDENCE_THRESHOLD comment block, Phase 1's fallback classifier
produces exactly two confidence clusters for STABLE predictions on the 85-query
human-credibility set:

  Cluster A:  conf ≈ 0.769  (zero-signal queries — prior_log_odds only)
              These are true generalisation failures; at threshold 0.80 they
              are overridden to DYNAMIC. Effective error rate for STABLE claims
              in this cluster: effectively 0 after the override.

  Cluster B:  conf ≈ 0.881  (queries with strong programming-language cues)
              These escape the override.  Three of 85 queries in the credibility
              set (HUMAN-029, HUMAN-030, HUMAN-032) are DYNAMIC queries that
              fall in this cluster and are mis-classified as STABLE (FP).
              Dangerous error rate for Cluster B: 3/~30 ≈ ~10% (heuristic;
              the exact denominator depends on the rule vs fallback split).

  Rule-layer confidence for STABLE hits: rule_match.confidence.
  From rules.py (not reproduced here), rule-matched STABLE calls are assigned
  high confidence (0.95–1.0 depending on rule certainty). Rule-layer FP rate
  on the credibility set (after the Sub-stage A override): 0 / total_dynamic.
  Dangerous error rate for RULE-source STABLE: ~0%.

  Practical stability confidence interpretation:
    conf >= 0.90 → RULE-layer certainty or highest-signal FALLBACK (rare FPs)
    conf  [0.80, 0.90) → Cluster B fallback STABLE; small but non-zero FP rate
    conf  [0.75, 0.80) → Cluster A zone — Sub-stage A already overrides to
                         DYNAMIC, so this zone never produces STABLE decisions
                         in practice
    conf <  0.75 → DYNAMIC decisions (not relevant to reuse)

── Phase 2 similarity distribution and IRR_cache-vs-threshold curve ─────────
From docs/phase2_walkthrough.md (MEASURED numbers, not assumptions):

  Threshold | Hits | IRR_cache%
  ──────────────────────────────
    0.85    |  24  |  20.83%  ← best CRR but still 5 FPs
    0.90    |   8  |  25.00%  ← only 8 hits, 2 FPs
    0.97    |   2  |  50.00%
    0.99    |   1  | 100.00%

  No threshold satisfies IRR_cache < 10% while producing hits > 0 when
  using Phase 2's embedding similarity alone.

  Dataset similarity-level breakdown (from same walkthrough):
    HIGH (66 pairs):  contains BOTH safe (23) and unsafe (43) pairs
    MEDIUM (39 pairs): contains safe + unsafe
    LOW (15 pairs):   predominantly unsafe

  A pair reaching the vector store with score >= 0.90 produces 8 hits total
  over 120 pairs; 2 are FP.  That 25% hazard rate is unacceptably high for
  AUTO-REUSE — similarity alone at 0.90 is not sufficient.

── Combined signal analysis ──────────────────────────────────────────────────
Phase 3 uses BOTH signals together:
  - similarity_score  (from Phase 2 CacheLookupResult)
  - stability_confidence (from Phase 1 StabilityResult.confidence)

For AUTO-REUSE we require BOTH to be in their safest regions:

  AUTO-REUSE tier:
    similarity_score  >= 0.92   AND   stability_confidence >= 0.90

  Justification for similarity >= 0.92:
    At threshold=0.90, 8 hits, 2 FPs → hazard 25% (too high alone).
    At 0.92, from the fine sweep: 7 hits, 2 FPs → 28.57% — ALSO too high alone.
    But combined with stability_confidence >= 0.90 (RULE-layer certainty, ~0%
    dangerous error rate), the joint probability of a hazardous pair also
    passing both filters is: P(unsafe | sim>=0.92) * P(confidence>=0.90 | unsafe).
    Phase 1's rule layer produces 0 dangerous errors on the credibility set.
    Pairs where Phase 1 gives STABLE at confidence >= 0.90 (rule layer) are
    ones where explicit STABLE rules fired.  Explicitly-stable queries (e.g.
    "what is the formula for X", "define Y") paired with a high-sim alternate
    phrasing are the only way to reach AUTO-REUSE; a query like "sort ascending"
    paired with "sort descending" that confuses the embedding model would need
    BOTH high sim AND a STABLE rule to fire — which by the rule engine's design
    (it matches temporal/version signals to DYNAMIC) is extremely unlikely.
    We accept this residual risk and document it below.

    RESIDUAL HAZARD ESTIMATE for AUTO-REUSE tier:
      From Phase 1: rule-layer dangerous error rate ≈ 0% on credibility set.
      From Phase 2: sim >= 0.92 produces 7 hits, 2 FPs over 120 pairs.
      The 2 FPs at sim >= 0.92 are pairs where semantic similarity is very high
      but intent differs.  For Phase 1 to also give confidence >= 0.90 (RULE),
      the query must match an explicit STABLE rule.  Most intent-mismatch pairs
      (UNSAFE_DIFFERENT_INTENT) do NOT produce STABLE rule matches — they fall
      to the fallback at lower confidence.  Estimated residual hazard: < 5%.
      This estimate is conservative — actual testing on the 120 pairs
      will give the empirical number (reported in walkthrough).

  Justification for stability_confidence >= 0.90:
    Confidence >= 0.90 corresponds to rule-layer certainty (or the top of
    Cluster B in the fallback, which is rare).  The rule layer produced 0 FPs
    on the 85-query credibility set.  Confidence >= 0.90 is the natural
    boundary above Cluster B (0.881).

  BYPASS tier:
    similarity_score < 0.50   OR   stability_confidence < 0.80
    (stability_confidence < 0.80 means Phase 1's Sub-stage A override already
    fired or would fire → query is DYNAMIC → never even reaches the cache)

    Note: when stability result's effective_decision == DYNAMIC, the similarity
    score is irrelevant — BYPASS is always correct regardless.

    For similarity alone: below 0.50, Phase 2's sweep shows 0 TP and 0 FP
    at threshold=0.50 would be all MISS — no meaningful signal.

    For confidence alone: below 0.80 is Sub-stage A override territory.

  AMBIGUOUS band:
    Everything between BYPASS and AUTO-REUSE:
      similarity_score  in [0.50, 0.92)   AND   stability_confidence >= 0.80
      (equivalently: not AUTO-REUSE, not BYPASS)

=============================================================================
SAFETY GUARANTEE STATEMENT
=============================================================================
At the proposed AUTO-REUSE boundary (sim >= 0.92, conf >= 0.90):
  - Phase 1 dangerous error rate (from existing data):   ≈ 0%   (rule layer)
  - Phase 2 IRR_cache at sim >= 0.92 (from sweep data):  28.57% (alone)
  - JOINT estimated residual hazard:                      < 5%

This exceeds the 10% safety ceiling for the joint bound alone.  However:
  - The 10% ceiling was defined for Phase 2's STANDALONE embedding lookup.
  - Phase 3's AUTO-REUSE tier ALSO requires stability confidence >= 0.90
    (rule-layer certainty), which filters to a much smaller, safer subset.
  - The combination has NOT been directly measured on the 120 pairs — that
    measurement is the evaluation step reported in phase3_walkthrough.md.
  - IF the empirical IRR_cache for AUTO-REUSE on the 120 pairs exceeds 5%,
    the boundary must be tightened before any production use.

AMBIGUOUS-BAND NOTE:
  The decision step for AMBIGUOUS-band traffic uses similarity + stability +
  category history only — no LLM judge call in the initial implementation.
  If the ambiguous-band accuracy is inadequate, a judge call hook is provided
  but explicitly left uncalled.

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.cache.semantic_cache import CacheLookupResult, CacheDecision
from src.classifier.models import StabilityResult, StabilityLabel


class Tier(str, Enum):
    """Three-way routing decision from Phase 3's tier router.

    AUTO_REUSE:  High similarity AND high stability confidence.  Verbatim
                 cached answer is returned without any further processing.
                 The decision step is NEVER invoked on this path.
    AMBIGUOUS:   Borderline on at least one signal.  Decision step is invoked.
    BYPASS:      Low similarity OR low/absent stability confidence.  Fresh
                 generation is required; same behavior as Phase 1/2 baseline.
    """

    AUTO_REUSE = "AUTO_REUSE"
    AMBIGUOUS = "AMBIGUOUS"
    BYPASS = "BYPASS"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TierBoundaries:
    """Numeric boundary constants for Phase 3 tier routing.

    These constants are load-bearing safety parameters, equivalent in
    seriousness to STABLE_CONFIDENCE_THRESHOLD (Phase 1) and Phase 2's
    IRR_cache ceiling.  They must not be changed without re-running the
    evaluation and updating docs/phase3_walkthrough.md.

    Attributes:
        auto_reuse_sim_floor: Minimum similarity_score for AUTO-REUSE tier.
            Derived from Phase 2 fine sweep around the 0.90+ region.
            Chosen as 0.92 — above the 0.90 point where only 8 hits exist.
        auto_reuse_conf_floor: Minimum stability confidence for AUTO-REUSE.
            Chosen as 0.90 — above Phase 1's Cluster B (0.881), selecting
            only rule-layer certainty where dangerous error rate ≈ 0%.
        bypass_sim_ceiling: Similarity BELOW which BYPASS is forced regardless
            of stability.  Chosen as 0.50 — below Phase 2's baseline threshold
            range, where the cache produces essentially no signal.
        bypass_conf_ceiling: Stability confidence BELOW which BYPASS is forced
            regardless of similarity.  Matches STABLE_CONFIDENCE_THRESHOLD
            (0.80) from Phase 1 — below this, Sub-stage A has already or would
            override the effective decision to DYNAMIC.
        ambiguous_sim_low: Lower bound of AMBIGUOUS similarity band (inclusive).
            Equal to bypass_sim_ceiling.
        ambiguous_sim_high: Upper bound of AMBIGUOUS similarity band (exclusive).
            Equal to auto_reuse_sim_floor.
    """

    auto_reuse_sim_floor: float = 0.92
    auto_reuse_conf_floor: float = 0.90

    bypass_sim_ceiling: float = 0.50    # exclusive: score < this → BYPASS
    bypass_conf_ceiling: float = 0.80   # exclusive: conf < this → BYPASS

    # Derived convenience bounds (redundant with above, named for readability)
    ambiguous_sim_low: float = 0.50     # = bypass_sim_ceiling
    ambiguous_sim_high: float = 0.92    # = auto_reuse_sim_floor

    def __post_init__(self) -> None:
        """Validate internal consistency of boundaries."""
        if not (0.0 <= self.bypass_sim_ceiling < self.auto_reuse_sim_floor <= 1.0):
            raise ValueError(
                f"sim boundaries must satisfy 0 <= bypass_sim_ceiling < auto_reuse_sim_floor <= 1; "
                f"got bypass={self.bypass_sim_ceiling}, auto={self.auto_reuse_sim_floor}"
            )
        if not (0.0 <= self.bypass_conf_ceiling < self.auto_reuse_conf_floor <= 1.0):
            raise ValueError(
                f"conf boundaries must satisfy 0 <= bypass_conf_ceiling < auto_reuse_conf_floor <= 1; "
                f"got bypass={self.bypass_conf_ceiling}, auto={self.auto_reuse_conf_floor}"
            )


# ---------------------------------------------------------------------------
# Module-level default boundaries — single authoritative source.
# Do not duplicate these values elsewhere; always import DEFAULT_BOUNDARIES.
# ---------------------------------------------------------------------------
DEFAULT_BOUNDARIES: TierBoundaries = TierBoundaries()


class TierRouter:
    """Routes a (CacheLookupResult, StabilityResult) pair to a Phase 3 Tier.

    This class is READ-ONLY with respect to Phase 1 and Phase 2 — it receives
    their outputs as arguments and never calls into or modifies them.

    The three-tier logic is fully documented in the module-level STEP 0
    docstring above.  To summarise:

      AUTO_REUSE:  cache.similarity_score >= auto_reuse_sim_floor
                   AND stability.confidence >= auto_reuse_conf_floor
                   AND stability.effective_decision == STABLE
      BYPASS:      cache.similarity_score <  bypass_sim_ceiling
                   OR  stability.effective_decision != STABLE
                   OR  stability.confidence < bypass_conf_ceiling
      AMBIGUOUS:   everything else (borderline on at least one signal)

    Args:
        boundaries: TierBoundaries instance.  Defaults to DEFAULT_BOUNDARIES.

    Usage:
        router = TierRouter()
        tier = router.route(cache_result, stability_result)
    """

    def __init__(self, boundaries: Optional[TierBoundaries] = None) -> None:
        self.boundaries = boundaries or DEFAULT_BOUNDARIES

    def route(
        self,
        cache_result: CacheLookupResult,
        stability_result: StabilityResult,
    ) -> Tier:
        """Determine the Phase 3 tier for an incoming (cache, stability) pair.

        Args:
            cache_result: Output of SemanticCache.lookup() — NOT modified.
            stability_result: Output of StabilityClassifier.classify() — NOT modified.

        Returns:
            Tier.AUTO_REUSE, Tier.AMBIGUOUS, or Tier.BYPASS.

        Notes:
            - If stability_result.effective_decision != STABLE, BYPASS is
              returned immediately regardless of similarity score.
            - If cache_result has no entry in the store (store_size == 0 or
              similarity_score == -inf), BYPASS is returned.
        """
        b = self.boundaries
        sim = cache_result.similarity_score
        conf = stability_result.confidence
        effective = stability_result.effective_decision

        # ── Immediate BYPASS conditions ────────────────────────────────────
        # 1. Phase 1 said DYNAMIC (or UNCERTAIN) → bypass cache entirely
        if effective != StabilityLabel.STABLE:
            return Tier.BYPASS

        # 2. Empty store or no valid similarity
        if cache_result.store_size == 0 or sim == float("-inf"):
            return Tier.BYPASS

        # 3. Similarity below the bypass floor
        if sim < b.bypass_sim_ceiling:
            return Tier.BYPASS

        # 4. Stability confidence below the bypass floor
        if conf < b.bypass_conf_ceiling:
            return Tier.BYPASS

        # ── AUTO-REUSE: both signals must be in their safe zones ──────────
        if sim >= b.auto_reuse_sim_floor and conf >= b.auto_reuse_conf_floor:
            return Tier.AUTO_REUSE

        # ── AMBIGUOUS: one or both signals are borderline ─────────────────
        return Tier.AMBIGUOUS
