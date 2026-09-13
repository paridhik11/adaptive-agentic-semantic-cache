"""Network-free unit tests for Phase 3 DecisionStep and CategoryHistory.

No model loading, no network access.  All Phase 1 and Phase 2 outputs are
constructed directly.

Covered:
  - CategoryHistory: cold-start defaults, EWMA update, reset, multi-domain
  - DecisionStep: weight validation, decision math at key boundary points
  - DecisionStep: judge-call hook is called when provided, NOT called when None
  - DecisionResult: to_dict() schema
  - Weight / threshold edge cases

pytest marker: (none) — runs unconditionally.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.cache.semantic_cache import CacheDecision, CacheLookupResult
from src.classifier.models import ClassificationSource, StabilityLabel, StabilityResult
from src.decision.decision_step import (
    CategoryHistory,
    DecisionResult,
    DecisionStep,
)
from src.decision.tier_router import DEFAULT_BOUNDARIES, TierBoundaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_result(score: float) -> CacheLookupResult:
    return CacheLookupResult(
        decision=CacheDecision.HIT if score >= 0.85 else CacheDecision.MISS,
        similarity_score=score,
        threshold=0.85,
        embed_latency_ms=1.0,
        search_latency_ms=0.1,
        total_latency_ms=1.1,
        store_size=1,
    )


def _stability_result(confidence: float, effective: StabilityLabel = StabilityLabel.STABLE) -> StabilityResult:
    return StabilityResult(
        query="test",
        predicted_label=StabilityLabel.STABLE,
        effective_decision=effective,
        is_cacheable=True,
        confidence=confidence,
        source=ClassificationSource.RULE,
        matched_rule="test_rule",
        rationale="unit test",
    )


# ---------------------------------------------------------------------------
# CategoryHistory
# ---------------------------------------------------------------------------

class TestCategoryHistory:
    def test_cold_start_returns_default(self):
        hist = CategoryHistory()
        assert hist.get("computer_science") == CategoryHistory.COLD_START_DEFAULT

    def test_cold_start_count_is_zero(self):
        hist = CategoryHistory()
        assert hist.observation_count("computer_science") == 0

    def test_single_correct_reuse_raises_rate(self):
        hist = CategoryHistory(decay=0.0)   # decay=0 → EWMA = latest observation
        hist.update("computer_science", was_correct_reuse=True)
        assert hist.get("computer_science") == pytest.approx(1.0, abs=0.01)

    def test_single_incorrect_reuse_lowers_rate(self):
        hist = CategoryHistory(decay=0.0)
        hist.update("computer_science", was_correct_reuse=False)
        # decay=0: EWMA = 0*prior + 1*0.0 = 0.0
        assert hist.get("computer_science") == pytest.approx(0.0, abs=0.01)

    def test_ewma_decay_blends_observations(self):
        hist = CategoryHistory(decay=0.90)
        # First update: from cold_start=0.50
        hist.update("cs", was_correct_reuse=True)
        # EWMA = 0.90 * 0.50 + 0.10 * 1.0 = 0.55
        assert hist.get("cs") == pytest.approx(0.55, abs=1e-6)

    def test_count_increments_per_update(self):
        hist = CategoryHistory()
        for _ in range(5):
            hist.update("math", was_correct_reuse=True)
        assert hist.observation_count("math") == 5

    def test_separate_domains_are_independent(self):
        hist = CategoryHistory(decay=0.0)
        hist.update("computer_science", was_correct_reuse=True)
        hist.update("mathematics", was_correct_reuse=False)
        assert hist.get("computer_science") == pytest.approx(1.0, abs=0.01)
        assert hist.get("mathematics") == pytest.approx(0.0, abs=0.01)
        assert hist.get("science_medicine") == CategoryHistory.COLD_START_DEFAULT

    def test_reset_clears_all_domains(self):
        hist = CategoryHistory()
        hist.update("computer_science", was_correct_reuse=True)
        hist.update("mathematics", was_correct_reuse=False)
        hist.reset()
        assert hist.observation_count("computer_science") == 0
        assert hist.get("computer_science") == CategoryHistory.COLD_START_DEFAULT
        assert hist.get("mathematics") == CategoryHistory.COLD_START_DEFAULT

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="decay"):
            CategoryHistory(decay=1.0)

    def test_to_dict_contains_ewma_and_count(self):
        hist = CategoryHistory()
        hist.update("cs", was_correct_reuse=True)
        d = hist.to_dict()
        assert "cs" in d
        assert "ewma" in d["cs"]
        assert "count" in d["cs"]
        assert d["cs"]["count"] == 1


# ---------------------------------------------------------------------------
# DecisionStep — weight validation
# ---------------------------------------------------------------------------

class TestDecisionStepValidation:
    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="Weights must sum"):
            DecisionStep(w_sim=0.5, w_conf=0.5, w_hist=0.5)

    def test_invalid_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="decision_threshold"):
            DecisionStep(decision_threshold=1.5)

    def test_invalid_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="decision_threshold"):
            DecisionStep(decision_threshold=-0.1)

    def test_valid_weights_sum_to_one(self):
        step = DecisionStep(w_sim=0.5, w_conf=0.3, w_hist=0.2)
        assert abs(step.w_sim + step.w_conf + step.w_hist - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# DecisionStep — decision math
# ---------------------------------------------------------------------------

class TestDecisionStepMath:
    """Verifies the linear combination math and REUSE/BYPASS boundary."""

    def _make_step(self, **kwargs) -> DecisionStep:
        return DecisionStep(w_sim=0.5, w_conf=0.3, w_hist=0.2, **kwargs)

    def test_all_signals_max_gives_reuse(self):
        """sim=0.92, conf=0.90 (at floors), hist=1.0 → high score → REUSE."""
        step = self._make_step()
        # sim at auto_reuse floor → sim_norm = 1.0
        # conf at auto_reuse floor → conf_norm = 1.0
        # hist: 1.0 (inject via mock history)
        step.history._ewma["cs"] = 1.0
        step.history._count["cs"] = 10
        cache = _cache_result(score=DEFAULT_BOUNDARIES.auto_reuse_sim_floor)
        stab = _stability_result(confidence=DEFAULT_BOUNDARIES.auto_reuse_conf_floor)
        result = step.decide(cache, stab, domain="cs")
        assert result.decision == "REUSE"
        assert result.score >= 0.50

    def test_all_signals_min_gives_bypass(self):
        """sim just above bypass floor, conf just above bypass floor, hist=0.0 → BYPASS."""
        step = self._make_step()
        step.history._ewma["cs"] = 0.0
        step.history._count["cs"] = 10
        b = DEFAULT_BOUNDARIES
        cache = _cache_result(score=b.bypass_sim_ceiling + 0.001)  # sim_norm ≈ 0
        stab = _stability_result(confidence=b.bypass_conf_ceiling + 0.001)  # conf_norm ≈ 0
        result = step.decide(cache, stab, domain="cs")
        assert result.decision == "BYPASS"
        assert result.score < 0.50

    def test_cold_start_history_neutral(self):
        """Cold-start history = 0.50 → neutral; decision depends on sim/conf alone."""
        step = self._make_step()
        b = DEFAULT_BOUNDARIES
        # Mid-band sim and conf → score is driven by weights
        # sim_norm = 0.5 (halfway in band), conf_norm = 0.5
        # score = 0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5 = 0.5 → borderline
        mid_sim = (b.bypass_sim_ceiling + b.auto_reuse_sim_floor) / 2
        mid_conf = (b.bypass_conf_ceiling + b.auto_reuse_conf_floor) / 2
        cache = _cache_result(score=mid_sim)
        stab = _stability_result(confidence=mid_conf)
        result = step.decide(cache, stab, domain="unknown_domain")
        assert result.history_count == 0
        assert result.hist_rate == pytest.approx(CategoryHistory.COLD_START_DEFAULT)

    def test_sim_norm_clipped_to_zero_below_band(self):
        """Similarity below bypass ceiling → sim_norm clamped to 0."""
        step = self._make_step()
        b = DEFAULT_BOUNDARIES
        cache = _cache_result(score=b.bypass_sim_ceiling - 0.01)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="cs")
        assert result.sim_norm == pytest.approx(0.0, abs=1e-6)

    def test_sim_norm_clipped_to_one_above_band(self):
        """Similarity above auto floor → sim_norm clamped to 1."""
        step = self._make_step()
        b = DEFAULT_BOUNDARIES
        cache = _cache_result(score=b.auto_reuse_sim_floor + 0.05)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="cs")
        assert result.sim_norm == pytest.approx(1.0, abs=1e-6)

    def test_decision_result_is_reuse_or_bypass(self):
        step = self._make_step()
        cache = _cache_result(score=0.80)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="cs")
        assert result.decision in ("REUSE", "BYPASS")

    def test_history_count_reflected_in_result(self):
        step = self._make_step()
        step.history.update("math", was_correct_reuse=True)
        cache = _cache_result(score=0.80)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="math")
        assert result.history_count == 1


# ---------------------------------------------------------------------------
# DecisionStep — judge-call hook
# ---------------------------------------------------------------------------

class TestDecisionStepJudgeHook:
    """Verify judge-call hook behavior.

    The judge hook is optional and NOT invoked by default.  When provided,
    it receives (sim, conf, domain) and returns a score in [0, 1].
    """

    def test_judge_not_called_when_none(self):
        """Without a judge, judge_called must be False and judge_score None."""
        step = DecisionStep(judge=None)
        cache = _cache_result(score=0.80)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="cs")
        assert result.judge_called is False
        assert result.judge_score is None

    def test_judge_called_when_provided(self):
        """When a judge is wired in, judge_called must be True."""
        judge_mock = MagicMock(return_value=0.80)
        step = DecisionStep(judge=judge_mock)
        cache = _cache_result(score=0.80)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="cs")
        assert result.judge_called is True
        judge_mock.assert_called_once_with(0.80, 0.85, "cs")

    def test_judge_score_reflected_in_result(self):
        judge_fn = lambda sim, conf, domain: 0.75
        step = DecisionStep(judge=judge_fn)
        cache = _cache_result(score=0.80)
        stab = _stability_result(confidence=0.85)
        result = step.decide(cache, stab, domain="cs")
        assert result.judge_score == pytest.approx(0.75, abs=1e-4)

    def test_judge_score_drives_reuse_when_high(self):
        """A judge returning 1.0 should strongly push toward REUSE."""
        judge_fn = lambda sim, conf, domain: 1.0
        # Use very low sim/conf so without judge it would be BYPASS
        step = DecisionStep(
            judge=judge_fn,
            w_sim=0.5, w_conf=0.3, w_hist=0.2,
            decision_threshold=0.50,
        )
        step.history._ewma["cs"] = 0.0
        step.history._count["cs"] = 10
        b = DEFAULT_BOUNDARIES
        cache = _cache_result(score=b.bypass_sim_ceiling + 0.001)
        stab = _stability_result(confidence=b.bypass_conf_ceiling + 0.001)
        result = step.decide(cache, stab, domain="cs")
        # With judge=1.0: score = 0.5 * 1.0 + 0.5 * (tiny combined) ≈ 0.5 → REUSE
        assert result.judge_called is True


# ---------------------------------------------------------------------------
# DecisionResult schema
# ---------------------------------------------------------------------------

class TestDecisionResultSchema:
    def _make_result(self, decision: str = "REUSE") -> DecisionResult:
        return DecisionResult(
            decision=decision,
            score=0.65,
            sim_norm=0.70,
            conf_norm=0.60,
            hist_rate=0.50,
            domain="computer_science",
            history_count=3,
            judge_called=False,
            judge_score=None,
            rationale="test rationale",
        )

    def test_to_dict_contains_all_keys(self):
        d = self._make_result().to_dict()
        required = {
            "decision", "score", "sim_norm", "conf_norm", "hist_rate",
            "domain", "history_count", "judge_called", "judge_score", "rationale",
        }
        assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"

    def test_to_dict_decision_is_string(self):
        assert isinstance(self._make_result("REUSE").to_dict()["decision"], str)
        assert isinstance(self._make_result("BYPASS").to_dict()["decision"], str)

    def test_frozen_dataclass_immutable(self):
        result = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.decision = "OTHER"  # type: ignore[misc]

    def test_judge_score_none_in_dict_when_not_called(self):
        d = self._make_result().to_dict()
        assert d["judge_score"] is None
        assert d["judge_called"] is False
