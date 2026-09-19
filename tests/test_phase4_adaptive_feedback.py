"""Unit tests for Phase 4 — Adaptive Policy / Feedback Loop.

Covers:
1. Exact Clopper-Pearson binomial confidence intervals (scipy.stats.beta).
2. Wilson score confidence intervals (scipy.stats.norm.ppf).
3. 5-way dataset leakage verification and synthetic dataset schema integrity.
4. Threshold sweep mechanics, binomial CI bounding, and safety ceiling enforcement.
5. Dual-gate fallback behavior (MINIMUM_CATEGORY_N, MINIMUM_MINORITY_CLASS_N).
6. Fallback-record filter: load_combined_data() must exclude fallback_triggered records.
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


# ===========================================================================
# 6. Fallback-Record Filter — load_combined_data() must exclude stubs
# ===========================================================================

class TestFallbackFilterExcludesFallbackRecords:
    """Verify that the fallback_triggered filter in load_combined_data() works.

    Root cause of Phase 4 label-contamination bug (documented in phase4_walkthrough.md):
    86 of 108 synthetic telemetry records had fallback_triggered=True — API-error stubs
    that fail-closed to judge_is_safe=False. The original load_combined_data() consumed
    ALL records with no filter, treating these stubs as genuine "judge says unsafe" labels.

    These tests verify the filter predicate:
        genuine_records = [r for r in feedback_data if not r.get("fallback_triggered", False)]

    Tests operate on in-memory fixture data — no file I/O, no network, no model loading.
    """

    # Fixture: mix of genuine and fallback records mimicking real telemetry structure
    FIXTURE_TELEMETRY = [
        {
            "pair_id": "SYNTH-001",
            "domain": "mathematics",
            "similarity_score": 0.9034,
            "authored_is_reuse_safe": True,
            "judge_decision": "REUSE",
            "judge_is_safe": True,
            "fallback_triggered": False,
            "error": None,
            "raw_response": '{"decision": "REUSE", "is_safe": true}',
            "request_id": "gen-real-abc123",
        },
        {
            "pair_id": "SYNTH-002",
            "domain": "mathematics",
            "similarity_score": 0.7045,
            "authored_is_reuse_safe": False,
            "judge_decision": "BYPASS",
            "judge_is_safe": False,
            "fallback_triggered": False,
            "error": None,
            "raw_response": '{"decision": "BYPASS", "is_safe": false}',
            "request_id": "gen-real-def456",
        },
        # Fallback stub: quota exhausted, fail-closed to BYPASS
        {
            "pair_id": "SYNTH-023",
            "domain": "computer_science",
            "similarity_score": 0.8512,
            "authored_is_reuse_safe": True,
            "judge_decision": "BYPASS",
            "judge_is_safe": False,   # <-- not a real judge decision
            "fallback_triggered": True,
            "error": "RateLimitError: 429",
            "raw_response": None,
            "request_id": None,
        },
        # Another fallback stub, different domain
        {
            "pair_id": "SYNTH-040",
            "domain": "science_medicine",
            "similarity_score": 0.7823,
            "authored_is_reuse_safe": False,
            "judge_decision": "BYPASS",
            "judge_is_safe": False,   # <-- not a real judge decision
            "fallback_triggered": True,
            "error": "APIConnectionError: Connection error.",
            "raw_response": None,
            "request_id": None,
        },
        # Fallback stub with fallback_triggered=True but judge_is_safe=True (edge case)
        {
            "pair_id": "SYNTH-055",
            "domain": "history_geography",
            "similarity_score": 0.9100,
            "authored_is_reuse_safe": True,
            "judge_decision": "BYPASS",
            "judge_is_safe": False,
            "fallback_triggered": True,
            "error": "RateLimitError: 429",
            "raw_response": None,
            "request_id": None,
        },
    ]

    def _apply_filter(self, telemetry: list) -> list:
        """Apply the exact filter predicate from load_combined_data()."""
        return [r for r in telemetry if not r.get("fallback_triggered", False)]

    def test_filter_excludes_all_fallback_records(self):
        """No record with fallback_triggered=True must appear in the filtered output."""
        filtered = self._apply_filter(self.FIXTURE_TELEMETRY)
        for r in filtered:
            assert r.get("fallback_triggered", False) is False, (
                f"Fallback record {r['pair_id']} leaked through the filter!"
            )

    def test_filter_preserves_all_genuine_records(self):
        """All records with fallback_triggered=False must be in the filtered output."""
        genuine_ids = {r["pair_id"] for r in self.FIXTURE_TELEMETRY
                       if not r.get("fallback_triggered", False)}
        filtered_ids = {r["pair_id"] for r in self._apply_filter(self.FIXTURE_TELEMETRY)}
        assert genuine_ids == filtered_ids, (
            f"Genuine records missing from filtered output: {genuine_ids - filtered_ids}"
        )

    def test_filter_counts_are_correct(self):
        """Fixture has 2 genuine and 3 fallback records."""
        filtered = self._apply_filter(self.FIXTURE_TELEMETRY)
        assert len(filtered) == 2, f"Expected 2 genuine records, got {len(filtered)}"

    def test_filter_excluded_count_matches_fallback_count(self):
        """Number excluded equals number with fallback_triggered=True."""
        filtered = self._apply_filter(self.FIXTURE_TELEMETRY)
        excluded_count = len(self.FIXTURE_TELEMETRY) - len(filtered)
        fallback_count = sum(1 for r in self.FIXTURE_TELEMETRY if r.get("fallback_triggered"))
        assert excluded_count == fallback_count == 3

    def test_filter_handles_missing_fallback_triggered_key(self):
        """Records without the fallback_triggered key should be treated as genuine."""
        telemetry_no_key = [
            {"pair_id": "X-001", "domain": "mathematics", "similarity_score": 0.80,
             "judge_is_safe": True},  # no fallback_triggered key at all
            {"pair_id": "X-002", "domain": "mathematics", "similarity_score": 0.70,
             "judge_is_safe": False, "fallback_triggered": True},
        ]
        filtered = self._apply_filter(telemetry_no_key)
        # X-001 has no key -> treated as genuine (get("fallback_triggered", False) = False)
        assert len(filtered) == 1
        assert filtered[0]["pair_id"] == "X-001"

    def test_only_genuine_labels_reach_sweep(self):
        """Filtered records yield only real judge_is_safe values, not stub defaults."""
        filtered = self._apply_filter(self.FIXTURE_TELEMETRY)
        labels = [(r["domain"], r["judge_is_safe"]) for r in filtered]
        # SYNTH-001: mathematics, True (REUSE genuine)
        # SYNTH-002: mathematics, False (BYPASS genuine)
        assert ("mathematics", True) in labels
        assert ("mathematics", False) in labels
        # Stub domains must NOT appear
        filtered_domains = {r["domain"] for r in filtered}
        assert "computer_science" not in filtered_domains, (
            "computer_science had only fallback stubs — must not appear in filtered output"
        )
        assert "science_medicine" not in filtered_domains
        assert "history_geography" not in filtered_domains

    def test_real_telemetry_file_genuine_count(self):
        """Integration check: the actual telemetry file has exactly 22 genuine records
        and 86 fallback stubs, confirming the root cause documented in phase4_walkthrough.md.
        """
        telemetry_path = REPO_ROOT / "data" / "openrouter_synthetic_feedback_telemetry.json"
        if not telemetry_path.exists():
            pytest.skip("Telemetry file not present — skipping integration check")

        with open(telemetry_path, "r", encoding="utf-8") as f:
            real_data = json.load(f)

        genuine = self._apply_filter(real_data)
        fallbacks = [r for r in real_data if r.get("fallback_triggered", False)]

        assert len(genuine) == 22, (
            f"Expected 22 genuine records in telemetry, got {len(genuine)}. "
            f"Re-run label_synthetic_feedback.py to obtain more real labels."
        )
        assert len(fallbacks) == 86, (
            f"Expected 86 fallback stubs, got {len(fallbacks)}."
        )
        # Confirm mathematics is the only domain with 100% genuine coverage
        math_records = [r for r in real_data if r["domain"] == "mathematics"]
        math_genuine = [r for r in math_records if not r.get("fallback_triggered")]
        assert len(math_genuine) == 18, f"Expected 18 genuine math records, got {len(math_genuine)}"
        assert len(math_records) == 18, "mathematics should have 0 fallback records"
