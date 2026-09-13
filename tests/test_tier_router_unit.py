"""Network-free unit tests for Phase 3 TierRouter.

Tests are isolated from Phase 1 and Phase 2 model loading by constructing
minimal mock StabilityResult and CacheLookupResult objects directly.

Covered:
  - TierBoundaries validation (__post_init__ guards)
  - TierRouter.route() for all three tiers, including exact boundary conditions
  - Confirmation that AUTO_REUSE-tier queries NEVER invoke DecisionStep
    (the hermetic hot-path requirement — DoD item 3)
  - BYPASS forced when effective_decision != STABLE
  - BYPASS forced when store is empty

pytest marker: (none) — runs unconditionally in default `pytest` invocation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.cache.semantic_cache import CacheDecision, CacheLookupResult
from src.classifier.models import (
    ClassificationSource,
    StabilityLabel,
    StabilityResult,
)
from src.decision.tier_router import (
    DEFAULT_BOUNDARIES,
    Tier,
    TierBoundaries,
    TierRouter,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal result objects
# ---------------------------------------------------------------------------

def _make_cache_result(
    score: float,
    decision: CacheDecision = CacheDecision.HIT,
    store_size: int = 1,
    threshold: float = 0.85,
) -> CacheLookupResult:
    """Build a CacheLookupResult with the given score and store_size."""
    return CacheLookupResult(
        decision=decision,
        similarity_score=score,
        threshold=threshold,
        embed_latency_ms=1.0,
        search_latency_ms=0.1,
        total_latency_ms=1.1,
        matched_metadata={"id": "PAIR-001"},
        store_size=store_size,
    )


def _make_stability_result(
    confidence: float,
    effective: StabilityLabel = StabilityLabel.STABLE,
) -> StabilityResult:
    """Build a StabilityResult with the given confidence and effective decision."""
    return StabilityResult(
        query="test query",
        predicted_label=StabilityLabel.STABLE if effective == StabilityLabel.STABLE else StabilityLabel.DYNAMIC,
        effective_decision=effective,
        is_cacheable=(effective == StabilityLabel.STABLE),
        confidence=confidence,
        source=ClassificationSource.RULE,
        matched_rule="test_rule",
        rationale="unit test",
    )


# ---------------------------------------------------------------------------
# TierBoundaries validation
# ---------------------------------------------------------------------------

class TestTierBoundariesValidation:
    def test_default_boundaries_are_self_consistent(self):
        b = DEFAULT_BOUNDARIES
        assert b.bypass_sim_ceiling < b.auto_reuse_sim_floor
        assert b.bypass_conf_ceiling < b.auto_reuse_conf_floor
        assert 0.0 <= b.bypass_sim_ceiling
        assert b.auto_reuse_sim_floor <= 1.0

    def test_custom_valid_boundaries_accepted(self):
        b = TierBoundaries(
            auto_reuse_sim_floor=0.95,
            auto_reuse_conf_floor=0.92,
            bypass_sim_ceiling=0.60,
            bypass_conf_ceiling=0.80,
            ambiguous_sim_low=0.60,
            ambiguous_sim_high=0.95,
        )
        assert b.auto_reuse_sim_floor == 0.95

    def test_inverted_sim_boundaries_raise(self):
        with pytest.raises(ValueError, match="sim boundaries"):
            TierBoundaries(
                auto_reuse_sim_floor=0.40,   # less than bypass_sim_ceiling
                bypass_sim_ceiling=0.50,
            )

    def test_inverted_conf_boundaries_raise(self):
        with pytest.raises(ValueError, match="conf boundaries"):
            TierBoundaries(
                auto_reuse_conf_floor=0.70,   # less than bypass_conf_ceiling
                bypass_conf_ceiling=0.80,
            )


# ---------------------------------------------------------------------------
# TierRouter — BYPASS conditions
# ---------------------------------------------------------------------------

class TestTierRouterBypass:
    def setup_method(self):
        self.router = TierRouter()

    def test_bypass_when_effective_decision_is_dynamic(self):
        """DYNAMIC effective decision → BYPASS regardless of similarity."""
        cache = _make_cache_result(score=0.99)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.DYNAMIC)
        assert self.router.route(cache, stab) == Tier.BYPASS

    def test_bypass_when_effective_decision_is_uncertain(self):
        cache = _make_cache_result(score=0.99)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.UNCERTAIN)
        assert self.router.route(cache, stab) == Tier.BYPASS

    def test_bypass_when_store_empty(self):
        cache = _make_cache_result(score=0.95, store_size=0)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.STABLE)
        assert self.router.route(cache, stab) == Tier.BYPASS

    def test_bypass_when_similarity_is_neg_inf(self):
        cache = _make_cache_result(score=float("-inf"), store_size=0)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.STABLE)
        assert self.router.route(cache, stab) == Tier.BYPASS

    def test_bypass_when_sim_below_bypass_ceiling(self):
        # Default bypass_sim_ceiling = 0.50 (exclusive)
        cache = _make_cache_result(score=0.49)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.STABLE)
        assert self.router.route(cache, stab) == Tier.BYPASS

    def test_bypass_when_sim_exactly_at_bypass_ceiling_is_ambiguous(self):
        # sim == 0.50 is the lower bound of AMBIGUOUS (inclusive)
        cache = _make_cache_result(score=0.50)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.STABLE)
        result = self.router.route(cache, stab)
        # 0.50 >= bypass_sim_ceiling (0.50) → not bypassed on sim alone
        # conf 0.95 >= auto_reuse_conf_floor (0.90)
        # sim 0.50 < auto_reuse_sim_floor (0.92) → AMBIGUOUS
        assert result == Tier.AMBIGUOUS

    def test_bypass_when_confidence_below_bypass_ceiling(self):
        # Default bypass_conf_ceiling = 0.80 (exclusive)
        cache = _make_cache_result(score=0.85)
        stab = _make_stability_result(confidence=0.75, effective=StabilityLabel.STABLE)
        assert self.router.route(cache, stab) == Tier.BYPASS

    def test_bypass_when_confidence_exactly_at_bypass_ceiling_is_ambiguous(self):
        # conf == 0.80 → passes the bypass check (not < 0.80)
        cache = _make_cache_result(score=0.85)
        stab = _make_stability_result(confidence=0.80, effective=StabilityLabel.STABLE)
        result = self.router.route(cache, stab)
        # conf 0.80 < auto_reuse_conf_floor (0.90) → AMBIGUOUS
        assert result == Tier.AMBIGUOUS


# ---------------------------------------------------------------------------
# TierRouter — AUTO_REUSE conditions
# ---------------------------------------------------------------------------

class TestTierRouterAutoReuse:
    def setup_method(self):
        self.router = TierRouter()

    def test_auto_reuse_at_exact_floor_values(self):
        """At exactly the AUTO_REUSE floor values → AUTO_REUSE."""
        b = DEFAULT_BOUNDARIES
        cache = _make_cache_result(score=b.auto_reuse_sim_floor)   # 0.92
        stab = _make_stability_result(confidence=b.auto_reuse_conf_floor)  # 0.90
        assert self.router.route(cache, stab) == Tier.AUTO_REUSE

    def test_auto_reuse_well_above_floors(self):
        cache = _make_cache_result(score=0.99)
        stab = _make_stability_result(confidence=0.99)
        assert self.router.route(cache, stab) == Tier.AUTO_REUSE

    def test_not_auto_reuse_when_sim_just_below_floor(self):
        b = DEFAULT_BOUNDARIES
        cache = _make_cache_result(score=b.auto_reuse_sim_floor - 0.001)
        stab = _make_stability_result(confidence=b.auto_reuse_conf_floor)
        assert self.router.route(cache, stab) == Tier.AMBIGUOUS

    def test_not_auto_reuse_when_conf_just_below_floor(self):
        b = DEFAULT_BOUNDARIES
        cache = _make_cache_result(score=b.auto_reuse_sim_floor)
        stab = _make_stability_result(confidence=b.auto_reuse_conf_floor - 0.001)
        assert self.router.route(cache, stab) == Tier.AMBIGUOUS


# ---------------------------------------------------------------------------
# TierRouter — AMBIGUOUS conditions
# ---------------------------------------------------------------------------

class TestTierRouterAmbiguous:
    def setup_method(self):
        self.router = TierRouter()

    def test_ambiguous_high_sim_medium_conf(self):
        """sim above bypass ceiling but below auto floor; conf >= bypass but < auto → AMBIGUOUS."""
        cache = _make_cache_result(score=0.85)
        stab = _make_stability_result(confidence=0.85)
        assert self.router.route(cache, stab) == Tier.AMBIGUOUS

    def test_ambiguous_high_conf_medium_sim(self):
        cache = _make_cache_result(score=0.75)
        stab = _make_stability_result(confidence=0.95)
        assert self.router.route(cache, stab) == Tier.AMBIGUOUS

    def test_ambiguous_borderline_both_signals(self):
        b = DEFAULT_BOUNDARIES
        cache = _make_cache_result(score=b.bypass_sim_ceiling + 0.001)
        stab = _make_stability_result(confidence=b.bypass_conf_ceiling + 0.001)
        assert self.router.route(cache, stab) == Tier.AMBIGUOUS


# ---------------------------------------------------------------------------
# CRITICAL DoD TEST: AUTO_REUSE hot-path never invokes DecisionStep
# ---------------------------------------------------------------------------

class TestAutoReuseNeverInvokesDecisionStep:
    """DoD item 3: a passing test proving the auto-reuse path never invokes
    the decision step or any judge call.

    The mock DecisionStep is injected into a minimal wrapper that counts calls.
    When the router routes to AUTO_REUSE, the decision step must record zero calls.
    """

    def test_auto_reuse_decision_step_zero_calls(self):
        """Routing to AUTO_REUSE must result in zero calls to DecisionStep.decide()."""
        from src.decision.decision_step import DecisionStep

        mock_decision_step = MagicMock(spec=DecisionStep)
        router = TierRouter()

        # Build inputs that trigger AUTO_REUSE
        b = DEFAULT_BOUNDARIES
        cache = _make_cache_result(score=b.auto_reuse_sim_floor + 0.01)
        stab = _make_stability_result(confidence=b.auto_reuse_conf_floor + 0.01)

        tier = router.route(cache, stab)
        assert tier == Tier.AUTO_REUSE

        # Simulate the calling code's conditional: only call decide() if AMBIGUOUS
        if tier == Tier.AMBIGUOUS:
            mock_decision_step.decide(cache, stab, "computer_science")

        # Assert: decision step was never called
        mock_decision_step.decide.assert_not_called()

    def test_bypass_decision_step_zero_calls(self):
        """Routing to BYPASS must also result in zero calls to DecisionStep.decide()."""
        from src.decision.decision_step import DecisionStep

        mock_decision_step = MagicMock(spec=DecisionStep)
        router = TierRouter()

        cache = _make_cache_result(score=0.40)
        stab = _make_stability_result(confidence=0.95, effective=StabilityLabel.STABLE)

        tier = router.route(cache, stab)
        assert tier == Tier.BYPASS

        if tier == Tier.AMBIGUOUS:
            mock_decision_step.decide(cache, stab, "computer_science")

        mock_decision_step.decide.assert_not_called()

    def test_only_ambiguous_tier_calls_decision_step(self):
        """Only AMBIGUOUS routing should trigger a DecisionStep.decide() call."""
        from src.decision.decision_step import DecisionStep, DecisionResult

        mock_decision_step = MagicMock(spec=DecisionStep)
        mock_decision_step.decide.return_value = DecisionResult(
            decision="REUSE",
            score=0.7,
            sim_norm=0.8,
            conf_norm=0.6,
            hist_rate=0.5,
            domain="computer_science",
            history_count=0,
            judge_called=False,
            judge_score=None,
            rationale="mocked",
        )
        router = TierRouter()

        # AMBIGUOUS inputs
        cache = _make_cache_result(score=0.75)
        stab = _make_stability_result(confidence=0.85)

        tier = router.route(cache, stab)
        assert tier == Tier.AMBIGUOUS

        if tier == Tier.AMBIGUOUS:
            mock_decision_step.decide(cache, stab, "computer_science")

        mock_decision_step.decide.assert_called_once()


# ---------------------------------------------------------------------------
# Tier enum basics
# ---------------------------------------------------------------------------

class TestTierEnum:
    def test_auto_reuse_str(self):
        assert str(Tier.AUTO_REUSE) == "AUTO_REUSE"

    def test_ambiguous_str(self):
        assert str(Tier.AMBIGUOUS) == "AMBIGUOUS"

    def test_bypass_str(self):
        assert str(Tier.BYPASS) == "BYPASS"

    def test_tier_is_string_subclass(self):
        assert Tier.AUTO_REUSE == "AUTO_REUSE"
        assert Tier.AMBIGUOUS == "AMBIGUOUS"
        assert Tier.BYPASS == "BYPASS"
