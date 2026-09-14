"""Network-free unit tests for Phase 3 AdaptiveThresholdEngine.

No network access, no live model loading. Validates:
  - Dual sample-size gates (MINIMUM_CATEGORY_N=20 AND MINIMUM_MINORITY_CLASS_N=10)
  - Operating threshold set to ci_upper (upper bound of 95% CI) for conservative bias
  - Fallback to 0.85 for domains failing either gate or unseen domains
  - Point estimate vs operating threshold serialization and reporting
"""

from __future__ import annotations

import pytest

from src.decision.adaptive_threshold_engine import (
    AdaptiveThresholdEngine,
    CategoryThreshold,
    ThresholdDecisionResult,
    MINIMUM_CATEGORY_N,
    MINIMUM_MINORITY_CLASS_N,
    FALLBACK_THRESHOLD,
    fit_logistic_threshold,
)


class TestPrecalibratedEngine:
    """Test default pre-calibrated engine behavior with dual gates and ci_upper operating thresholds."""

    @pytest.fixture
    def engine(self) -> AdaptiveThresholdEngine:
        return AdaptiveThresholdEngine()

    def test_populated_category_computer_science_passes_dual_gates(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("computer_science")
        assert ct.is_fallback is False
        assert ct.sample_size == 54
        assert ct.minority_count == 26  # 26 safe, 28 unsafe -> min=26 >= 10
        assert ct.threshold == pytest.approx(0.6855, abs=1e-3)  # point estimate theta
        assert ct.operating_threshold == pytest.approx(0.7924, abs=1e-3)  # ci_upper
        assert ct.standard_error is not None and ct.standard_error > 0.0
        assert ct.ci_lower is not None and ct.ci_upper is not None
        assert ct.ci_lower < ct.threshold < ct.ci_upper
        assert ct.ci_lower == pytest.approx(0.5785, abs=1e-3)
        assert ct.ci_upper == pytest.approx(0.7924, abs=1e-3)
        assert ct.b1 is not None and ct.b1 > 0.0

    def test_populated_category_science_medicine_passes_dual_gates(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("science_medicine")
        assert ct.is_fallback is False
        assert ct.sample_size == 27
        assert ct.minority_count == 10  # 17 safe, 10 unsafe -> min=10 >= 10 (meets gate exactly)
        assert ct.threshold == pytest.approx(0.6171, abs=1e-3)  # point estimate theta
        assert ct.operating_threshold == pytest.approx(0.7245, abs=1e-3)  # ci_upper
        assert ct.standard_error is not None and ct.standard_error > 0.0
        assert ct.ci_lower is not None and ct.ci_upper is not None
        assert ct.ci_lower < ct.threshold < ct.ci_upper
        assert ct.ci_lower == pytest.approx(0.5097, abs=1e-3)
        assert ct.ci_upper == pytest.approx(0.7245, abs=1e-3)
        assert ct.b1 is not None and ct.b1 > 0.0

    def test_sparse_category_realtime_news_weather_falls_back(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("realtime_news_weather")
        assert ct.is_fallback is True
        assert ct.sample_size == 1
        assert ct.minority_count == 0
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.ci_lower is None
        assert ct.ci_upper is None
        assert ct.fallback_reason is not None

    def test_sparse_category_finance_economics_falls_back(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("finance_economics")
        assert ct.is_fallback is True
        assert ct.sample_size == 5
        assert ct.minority_count == 1  # 1 safe, 4 unsafe -> min=1 < 10
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.ci_lower is None

    def test_sparse_category_mathematics_falls_back(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("mathematics")
        assert ct.is_fallback is True
        assert ct.sample_size == 13
        assert ct.minority_count == 2  # 11 safe, 2 unsafe -> min=2 < 10
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)

    def test_sparse_category_system_operations_falls_back(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("system_operations")
        assert ct.is_fallback is True
        assert ct.sample_size == 11
        assert ct.minority_count == 2  # 2 safe, 9 unsafe -> min=2 < 10
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)

    def test_sparse_category_history_geography_falls_back(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("history_geography")
        assert ct.is_fallback is True
        assert ct.sample_size == 9
        assert ct.minority_count == 4  # 5 safe, 4 unsafe -> min=4 < 10
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)

    def test_absent_category_legal_compliance_falls_back(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("legal_compliance")
        assert ct.is_fallback is True
        assert ct.sample_size == 0
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)

    def test_unseen_domain_string_defaults_to_fallback(
        self, engine: AdaptiveThresholdEngine
    ):
        ct = engine.get_threshold("astronomy_astrophysics")
        assert ct.is_fallback is True
        assert ct.sample_size == 0
        assert ct.threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert ct.operating_threshold == pytest.approx(FALLBACK_THRESHOLD)
        assert "unknown" in ct.fallback_reason.lower()


class TestDualSampleGates:
    """Test both MINIMUM_CATEGORY_N (20) and MINIMUM_MINORITY_CLASS_N (10) gates."""

    def test_constants_defined(self):
        assert MINIMUM_CATEGORY_N == 20
        assert MINIMUM_MINORITY_CLASS_N == 10
        assert FALLBACK_THRESHOLD == pytest.approx(0.85)

    def test_domain_passing_total_n_but_failing_minority_gate_falls_back(self):
        """A domain with N=25 total pairs but only 2 unsafe pairs (minority=2 < 10)
        must FAIL the minority-class gate and fall back to 0.85."""
        sims = [0.70 + 0.01 * i for i in range(23)] + [0.40, 0.45]
        safe = [True] * 23 + [False] * 2  # N=25 >= 20, but minority=2 < 10
        data = [("imbalanced_domain", s, sf) for s, sf in zip(sims, safe)]

        engine = AdaptiveThresholdEngine.fit(
            data, minimum_category_n=20, minimum_minority_class_n=10
        )
        ct = engine.get_threshold("imbalanced_domain")
        assert ct.is_fallback is True
        assert ct.sample_size == 25
        assert ct.minority_count == 2
        assert ct.operating_threshold == pytest.approx(0.85)
        assert "minority" in ct.fallback_reason.lower()

    def test_domain_failing_total_n_gate_falls_back(self):
        """A domain with N=19 total pairs (< 20) fails total N gate and falls back."""
        sims = [0.70 + 0.01 * i for i in range(10)] + [0.40 + 0.01 * i for i in range(9)]
        safe = [True] * 10 + [False] * 9  # N=19 < 20
        data = [("sparse_domain_19", s, sf) for s, sf in zip(sims, safe)]

        engine = AdaptiveThresholdEngine.fit(
            data, minimum_category_n=20, minimum_minority_class_n=10
        )
        ct = engine.get_threshold("sparse_domain_19")
        assert ct.is_fallback is True
        assert ct.sample_size == 19
        assert ct.operating_threshold == pytest.approx(0.85)
        assert "minimum_n" in ct.fallback_reason.lower()

    def test_domain_passing_both_gates_fits_per_category_with_ci_upper(self):
        """A domain with N=20 total pairs and 10 in minority class passes both gates.
        operating_threshold must equal ci_upper (not point estimate theta)."""
        sims = [0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90] + \
               [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
        safe = [True] * 10 + [False] * 10  # N=20 >= 20, minority=10 >= 10
        data = [("balanced_domain_20", s, sf) for s, sf in zip(sims, safe)]

        engine = AdaptiveThresholdEngine.fit(
            data, minimum_category_n=20, minimum_minority_class_n=10
        )
        ct = engine.get_threshold("balanced_domain_20")
        assert ct.is_fallback is False
        assert ct.sample_size == 20
        assert ct.minority_count == 10
        assert ct.operating_threshold == pytest.approx(ct.ci_upper)
        assert ct.operating_threshold > ct.threshold  # upper CI bound > point estimate


class TestDecideRoutingOperatingThreshold:
    """Test decide() uses operating_threshold (ci_upper for per-category, 0.85 for fallback)."""

    @pytest.fixture
    def engine(self) -> AdaptiveThresholdEngine:
        return AdaptiveThresholdEngine()

    def test_computer_science_between_theta_and_ci_upper_bypasses(
        self, engine: AdaptiveThresholdEngine
    ):
        # CS: theta=0.6855, ci_upper=0.7924
        # Similarity of 0.7500 is above point estimate (0.6855) but below ci_upper (0.7924)
        # Under the refined engine, this MUST BYPASS (conservative safety margin)
        res = engine.decide(similarity_score=0.7500, domain="computer_science")
        assert res.decision == "BYPASS"
        assert res.is_fallback is False
        assert res.operating_threshold == pytest.approx(0.7924, abs=1e-3)
        assert res.point_estimate == pytest.approx(0.6855, abs=1e-3)

    def test_computer_science_above_ci_upper_reuses(
        self, engine: AdaptiveThresholdEngine
    ):
        # Similarity of 0.8000 is above ci_upper (0.7924) -> REUSE
        res = engine.decide(similarity_score=0.8000, domain="computer_science")
        assert res.decision == "REUSE"
        assert res.is_fallback is False
        assert res.operating_threshold == pytest.approx(0.7924, abs=1e-3)

    def test_science_medicine_between_theta_and_ci_upper_bypasses(
        self, engine: AdaptiveThresholdEngine
    ):
        # Science: theta=0.6171, ci_upper=0.7245
        # Similarity of 0.6500 is above theta but below ci_upper -> BYPASS
        res = engine.decide(similarity_score=0.6500, domain="science_medicine")
        assert res.decision == "BYPASS"
        assert res.is_fallback is False
        assert res.operating_threshold == pytest.approx(0.7245, abs=1e-3)

    def test_science_medicine_above_ci_upper_reuses(
        self, engine: AdaptiveThresholdEngine
    ):
        # Similarity of 0.7500 is above ci_upper (0.7245) -> REUSE
        res = engine.decide(similarity_score=0.7500, domain="science_medicine")
        assert res.decision == "REUSE"
        assert res.is_fallback is False
        assert res.operating_threshold == pytest.approx(0.7245, abs=1e-3)

    def test_fallback_domain_uses_0_85_operating_threshold(
        self, engine: AdaptiveThresholdEngine
    ):
        # finance_economics is fallback (threshold=0.85)
        res_bypass = engine.decide(similarity_score=0.8200, domain="finance_economics")
        assert res_bypass.decision == "BYPASS"
        assert res_bypass.operating_threshold == pytest.approx(0.85)
        assert res_bypass.is_fallback is True

        res_reuse = engine.decide(similarity_score=0.8800, domain="finance_economics")
        assert res_reuse.decision == "REUSE"
        assert res_reuse.operating_threshold == pytest.approx(0.85)
        assert res_reuse.is_fallback is True


class TestSerializationAndDataclasses:
    """Test serialization includes operating_threshold and minority_count."""

    def test_category_threshold_to_dict(self):
        ct = CategoryThreshold(
            domain="computer_science",
            sample_size=54,
            minority_count=26,
            threshold=0.6855,
            operating_threshold=0.7924,
            is_fallback=False,
            ci_lower=0.5785,
            ci_upper=0.7924,
            standard_error=0.0546,
            b0=-3.6696,
            b1=5.3535,
        )
        d = ct.to_dict()
        assert d["domain"] == "computer_science"
        assert d["sample_size"] == 54
        assert d["minority_count"] == 26
        assert d["threshold"] == 0.6855
        assert d["operating_threshold"] == 0.7924
        assert d["is_fallback"] is False
        assert d["ci_lower"] == 0.5785
        assert d["ci_upper"] == 0.7924

    def test_threshold_decision_result_to_dict(self):
        engine = AdaptiveThresholdEngine()
        res = engine.decide(similarity_score=0.80, domain="computer_science")
        d = res.to_dict()
        assert d["decision"] == "REUSE"
        assert d["similarity_score"] == 0.80
        assert d["threshold"] == 0.7924
        assert d["operating_threshold"] == 0.7924
        assert d["point_estimate"] == 0.6855
        assert d["domain"] == "computer_science"
        assert d["is_fallback"] is False
