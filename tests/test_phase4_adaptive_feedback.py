"""Unit tests for Phase 4 — Adaptive Policy / Feedback Loop.

Covers:
1. Exact Clopper-Pearson binomial confidence intervals (scipy.stats.beta).
2. Wilson score confidence intervals (scipy.stats.norm.ppf).
3. 5-way dataset leakage verification and synthetic dataset schema integrity.
4. Threshold sweep mechanics, binomial CI bounding, and safety ceiling enforcement.
5. Dual-gate fallback behavior (MINIMUM_CATEGORY_N, MINIMUM_MINORITY_CLASS_N).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
from scipy import stats

from src.decision.adaptive_threshold_engine import (
    AdaptiveThresholdEngine,
    CategoryThreshold,
    DomainSweepPoint,
    FALLBACK_THRESHOLD,
    MINIMUM_CATEGORY_N,
    MINIMUM_MINORITY_CLASS_N,
    SAFETY_CEILING_IRR,
    calibrate_category_with_feedback,
    clopper_pearson_ci,
    sweep_domain_thresholds,
    wilson_score_ci,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_query_pair_feedback.json"


# ===========================================================================
# 1. Clopper-Pearson Exact Binomial CI Tests
# ===========================================================================

class TestClopperPearsonCI:
    """Test exact Clopper-Pearson binomial confidence interval calculations."""

    def test_n_zero_returns_zero_zero(self):
        low, high = clopper_pearson_ci(0, 0)
        assert low == 0.0
        assert high == 0.0

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError):
            clopper_pearson_ci(-1, 10)
        with pytest.raises(ValueError):
            clopper_pearson_ci(11, 10)
        with pytest.raises(ValueError):
            clopper_pearson_ci(5, -1)

    def test_zero_successes_exact_formula(self):
        """When k=0, lower bound is 0.0 and upper bound is 1 - (alpha/2)^(1/n)."""
        n = 10
        alpha = 0.05
        expected_upper = 1.0 - (alpha / 2.0) ** (1.0 / n)
        low, high = clopper_pearson_ci(0, n, confidence_level=0.95)
        assert low == 0.0
        assert pytest.approx(high, rel=1e-5) == expected_upper
        assert pytest.approx(high, abs=1e-4) == 0.3085

    def test_zero_successes_reaches_ten_percent_ceiling_at_n36(self):
        """At 95% confidence, k=0 clears 10% ceiling (upper < 0.10) precisely when n >= 36."""
        # At n=35, upper > 0.0999 (rounds to 0.1000)
        _, high_35 = clopper_pearson_ci(0, 35, confidence_level=0.95)
        assert high_35 > 0.0999

        # At n=36, upper = 0.0974 <= 0.10
        _, high_36 = clopper_pearson_ci(0, 36, confidence_level=0.95)
        assert high_36 <= 0.10
        assert pytest.approx(high_36, abs=1e-4) == 0.0974

    def test_all_successes_exact_formula(self):
        """When k=n, upper bound is 1.0 and lower bound is (alpha/2)^(1/n)."""
        n = 10
        alpha = 0.05
        expected_lower = (alpha / 2.0) ** (1.0 / n)
        low, high = clopper_pearson_ci(n, n, confidence_level=0.95)
        assert high == 1.0
        assert pytest.approx(low, rel=1e-5) == expected_lower

    def test_symmetric_half_split(self):
        """k=5, n=10 is symmetric around 0.50."""
        low, high = clopper_pearson_ci(5, 10, confidence_level=0.95)
        assert pytest.approx(0.50 - low, abs=1e-5) == pytest.approx(high - 0.50, abs=1e-5)
        assert 0.18 < low < 0.20
        assert 0.80 < high < 0.82

    def test_confidence_level_monotonicity(self):
        """Higher confidence level produces wider confidence intervals."""
        low_90, high_90 = clopper_pearson_ci(3, 20, confidence_level=0.90)
        low_95, high_95 = clopper_pearson_ci(3, 20, confidence_level=0.95)
        low_99, high_99 = clopper_pearson_ci(3, 20, confidence_level=0.99)

        assert low_99 < low_95 < low_90
        assert high_90 < high_95 < high_99


# ===========================================================================
# 2. Wilson Score CI Tests
# ===========================================================================

class TestWilsonScoreCI:
    """Test Wilson score binomial proportion interval calculations."""

    def test_n_zero_returns_zero_zero(self):
        low, high = wilson_score_ci(0, 0)
        assert low == 0.0
        assert high == 0.0

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError):
            wilson_score_ci(-1, 10)
        with pytest.raises(ValueError):
            wilson_score_ci(11, 10)

    def test_zero_successes_wilson_upper_bound(self):
        """For k=0, Wilson upper bound is z^2 / (n + z^2)."""
        n = 20
        z = stats.norm.ppf(0.975)
        expected_upper = (z ** 2) / (n + z ** 2)
        low, high = wilson_score_ci(0, n, confidence_level=0.95)
        assert low == 0.0
        assert pytest.approx(high, rel=1e-5) == expected_upper
        assert pytest.approx(high, abs=1e-4) == 0.1611

    def test_half_split_symmetric_around_half(self):
        low, high = wilson_score_ci(10, 20, confidence_level=0.95)
        assert pytest.approx((low + high) / 2.0, abs=1e-6) == 0.50


# ===========================================================================
# 3. Synthetic Dataset Leakage & Schema Validation
# ===========================================================================

class TestSyntheticDatasetLeakage:
    """Verify synthetic query pairs exhibit 0 leakage against all 5 datasets."""

    EXISTING_DATASETS = {
        "pair_benchmark": DATA_DIR / "query_pair_reuse_benchmark.json",
        "stability_dev": DATA_DIR / "query_stability_benchmark.json",
        "stability_heldout": DATA_DIR / "query_stability_benchmark_heldout.json",
        "stability_final_test": DATA_DIR / "query_stability_benchmark_final_test.json",
        "stability_human_cred": DATA_DIR / "query_stability_human_credibility.json",
    }

    @classmethod
    def setup_class(cls):
        assert SYNTHETIC_DATASET_PATH.exists(), f"Missing {SYNTHETIC_DATASET_PATH}"
        with open(SYNTHETIC_DATASET_PATH, "r", encoding="utf-8") as f:
            cls.synthetic_pairs = json.load(f)

    def test_record_count_is_108(self):
        assert len(self.synthetic_pairs) == 108

    def test_required_keys_present_in_every_record(self):
        required = {"id", "query_a", "query_b", "domain", "semantic_similarity_level", "is_reuse_safe", "taxonomy_class", "rejection_reason", "rationale"}
        for r in self.synthetic_pairs:
            missing = required - set(r.keys())
            assert not missing, f"Record {r.get('id')} missing keys: {missing}"
            assert isinstance(r["is_reuse_safe"], bool)
            assert r["domain"] in {
                "computer_science", "mathematics", "system_operations",
                "history_geography", "finance_economics", "science_medicine",
                "realtime_news_weather"
            }

    def test_internal_uniqueness(self):
        queries: Set[str] = set()
        ids: Set[str] = set()
        for r in self.synthetic_pairs:
            assert r["id"] not in ids, f"Duplicate ID: {r['id']}"
            ids.add(r["id"])

            qa = r["query_a"].strip().lower()
            qb = r["query_b"].strip().lower()
            assert qa != qb, f"Self-identity in {r['id']}"
            assert qa not in queries, f"Duplicate query string in synthetic set: {qa}"
            assert qb not in queries, f"Duplicate query string in synthetic set: {qb}"
            queries.add(qa)
            queries.add(qb)

    def test_zero_leakage_against_all_existing_benchmarks(self):
        synth_queries = set()
        for r in self.synthetic_pairs:
            synth_queries.add(r["query_a"].strip().lower())
            synth_queries.add(r["query_b"].strip().lower())

        for name, path in self.EXISTING_DATASETS.items():
            assert path.exists(), f"Missing dataset: {path}"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("queries", [])
            ext_qs = set()
            for it in items:
                if "query" in it:
                    ext_qs.add(it["query"].strip().lower())
                if "query_a" in it:
                    ext_qs.add(it["query_a"].strip().lower())
                if "query_b" in it:
                    ext_qs.add(it["query_b"].strip().lower())

            overlap = synth_queries & ext_qs
            assert not overlap, (
                f"LEAKAGE DETECTED against {name}: {len(overlap)} overlapping strings! "
                f"Examples: {sorted(overlap)[:3]}"
            )


# ===========================================================================
# 4. Threshold Sweep & Calibration Mechanics
# ===========================================================================

class TestThresholdSweepMechanics:
    """Test sweep_domain_thresholds and calibrate_category_with_feedback."""

    def test_monotonic_hit_count_behavior(self):
        # Create 40 synthetic points with similarities from 0.50 to 0.90
        sims = [0.50 + i * 0.01 for i in range(40)]
        safe = [True] * 40

        points = sweep_domain_thresholds(sims, safe, start=0.50, stop=0.90, step=0.05)
        assert len(points) > 0

        # Hits should be monotonically non-increasing with higher threshold
        for i in range(len(points) - 1):
            assert points[i].hits >= points[i + 1].hits

    def test_clears_safety_ceiling_requires_ci_upper_under_ten_percent(self):
        # 36 safe items at high similarity (0.85), 0 unsafe
        sims = [0.85] * 36
        safe = [True] * 36

        points = sweep_domain_thresholds(sims, safe, safety_ceiling=0.10, start=0.80, stop=0.90, step=0.05)
        pt_80 = next(p for p in points if abs(p.threshold - 0.80) < 1e-4)
        assert pt_80.hits == 36
        assert pt_80.fp == 0
        assert pt_80.irr_cache_ci_upper <= 0.10
        assert pt_80.clears_safety_ceiling is True

    def test_fails_safety_ceiling_if_hits_insufficient_even_with_zero_fp(self):
        # 20 safe items at similarity 0.85, 0 unsafe.
        # k=0, n=20 -> Clopper-Pearson upper bound = 16.8% > 10%
        sims = [0.85] * 20
        safe = [True] * 20

        points = sweep_domain_thresholds(sims, safe, safety_ceiling=0.10, start=0.80, stop=0.90, step=0.05)
        pt_80 = next(p for p in points if abs(p.threshold - 0.80) < 1e-4)
        assert pt_80.hits == 20
        assert pt_80.fp == 0
        assert pt_80.irr_cache_ci_upper > 0.10
        assert pt_80.clears_safety_ceiling is False


# ===========================================================================
# 5. Dual Gate Calibration Tests
# ===========================================================================

class TestDualGateCalibration:
    """Verify dual gates (N >= 20, min >= 10) strictly trigger fallback."""

    def test_fails_gate1_when_n_less_than_twenty(self):
        # 15 pairs (10 safe, 5 unsafe) -> N=15 < 20
        pairs = [(0.80, True)] * 10 + [(0.80, False)] * 5
        ct, sweep = calibrate_category_with_feedback("test_dom", pairs)

        assert ct.is_fallback is True
        assert ct.operating_threshold == FALLBACK_THRESHOLD
        assert "minimum_n" in ct.fallback_reason
        assert len(sweep) == 0

    def test_fails_gate2_when_minority_less_than_ten(self):
        # 25 pairs (23 safe, 2 unsafe) -> N=25 >= 20, but minority=2 < 10
        pairs = [(0.80, True)] * 23 + [(0.80, False)] * 2
        ct, sweep = calibrate_category_with_feedback("test_dom", pairs)

        assert ct.is_fallback is True
        assert ct.operating_threshold == FALLBACK_THRESHOLD
        assert "minimum_minority_n" in ct.fallback_reason
        assert len(sweep) == 0

    def test_adopts_lowest_qualifying_threshold_when_gates_pass(self):
        # 50 pairs: 38 safe at 0.80+, 12 unsafe at 0.60
        # N=50 >= 20, minority=12 >= 10 -> Passes dual gates
        # At threshold 0.75: all 38 safe are hits, 0 unsafe are hits
        # hits=38, fp=0 -> CI upper bound = 9.25% <= 10%
        pairs = [(0.82, True)] * 38 + [(0.60, False)] * 12
        ct, sweep = calibrate_category_with_feedback(
            "test_dom", pairs, sweep_start=0.70, sweep_stop=0.85, sweep_step=0.01
        )

        assert ct.is_fallback is False
        assert ct.operating_threshold < FALLBACK_THRESHOLD
        assert ct.irr_cache_ci_upper <= 0.10
        assert ct.hits_at_operating == 38
        assert "LOWEST_SWEEP_THRESHOLD_CLEARED_SAFETY_CEILING" in ct.selection_rule
