"""Unit and integration tests for Phase 5 — Load Test, Metrics, and Dashboard.

Validates:
  - 6-way leakage check against all prior benchmark datasets (asserts 0 overlaps).
  - Dataset schema, category distribution (10 queries per domain), and query taxonomy.
  - Metrics evaluation logic (hit rate, latency percentiles, cost calculation, Clopper-Pearson CI).
  - Sample power disclosure flag (underpowered flag when hits < 36).
  - End-to-end pipeline runner on miniature stream with mock judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluate_load_test import (
    calculate_latency_percentiles,
    evaluate_telemetry,
    COST_PER_M_INPUT_TOKENS_USD,
    COST_PER_M_OUTPUT_TOKENS_USD,
)
from scripts.run_load_test import check_leakage, run_load_test
from src.decision.adaptive_threshold_engine import clopper_pearson_ci

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAM_PATH = REPO_ROOT / "data" / "raw" / "load_test_query_stream.json"


# ---------------------------------------------------------------------------
# 1. Dataset Integrity & 6-Way Leakage Verification Tests
# ---------------------------------------------------------------------------

class TestLoadTestDatasetIntegrity:
    def test_stream_dataset_file_exists(self):
        assert STREAM_PATH.exists(), f"Stream dataset missing at {STREAM_PATH}"

    def test_stream_dataset_schema_and_size(self):
        with open(STREAM_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) == 70, f"Expected exactly 70 queries, got {len(records)}"

        expected_domains = {
            "computer_science",
            "science_medicine",
            "mathematics",
            "system_operations",
            "history_geography",
            "finance_economics",
            "realtime_news_weather",
        }

        domain_counts = {}
        required_keys = {"id", "domain", "query", "query_type", "is_reuse_safe", "expected_tier"}

        for r in records:
            for k in required_keys:
                assert k in r, f"Record {r.get('id')} missing key '{k}'"
            dom = r["domain"]
            assert dom in expected_domains, f"Unexpected domain '{dom}'"
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

        for dom in expected_domains:
            assert domain_counts.get(dom) == 10, f"Domain {dom} expected 10 queries, got {domain_counts.get(dom)}"

    def test_strict_six_way_leakage_verification(self):
        """Verify 0 query string overlaps against all 6 prior benchmark files."""
        with open(STREAM_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        passed, overlaps = check_leakage(records)
        assert passed is True, f"Leakage detected in {len(overlaps)} queries: {overlaps}"
        assert len(overlaps) == 0


# ---------------------------------------------------------------------------
# 2. Metrics Evaluation Unit Tests
# ---------------------------------------------------------------------------

class TestMetricsCalculationUnit:
    def test_calculate_latency_percentiles_empty(self):
        stats = calculate_latency_percentiles([])
        assert stats["n"] == 0
        assert stats["mean_ms"] == 0.0
        assert stats["median_ms"] == 0.0
        assert stats["p95_ms"] == 0.0

    def test_calculate_latency_percentiles_populated(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = calculate_latency_percentiles(vals)
        assert stats["n"] == 10
        assert stats["mean_ms"] == pytest.approx(55.0, abs=1e-2)
        assert stats["median_ms"] == pytest.approx(55.0, abs=1e-2)
        assert stats["p95_ms"] == pytest.approx(100.0, abs=1e-2)
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 100.0

    def test_evaluate_telemetry_synthetic_fixture(self):
        records = [
            # AUTO_REUSE Hit (TP)
            {
                "query_id": "T1",
                "domain": "computer_science",
                "path": "AUTO_REUSE",
                "final_decision": "HIT",
                "outcome_type": "TP",
                "was_correct": True,
                "timings_ms": {"total_pipeline_ms": 7.5, "judge_ms": 0.0},
                "judge_telemetry": None,
            },
            # AMBIGUOUS Hit (TP, safe reuse decided by judge)
            {
                "query_id": "T2",
                "domain": "computer_science",
                "path": "AMBIGUOUS",
                "final_decision": "HIT",
                "outcome_type": "TP",
                "was_correct": True,
                "timings_ms": {"total_pipeline_ms": 3200.0, "judge_ms": 3190.0},
                "judge_telemetry": {
                    "decision": "REUSE",
                    "is_safe": True,
                    "confidence": 0.95,
                    "prompt_tokens": 200,
                    "completion_tokens": 40,
                    "total_tokens": 240,
                    "latency_ms": 3190.0,
                },
            },
            # BYPASS Miss (TN)
            {
                "query_id": "T3",
                "domain": "mathematics",
                "path": "BYPASS",
                "final_decision": "MISS",
                "outcome_type": "TN",
                "was_correct": True,
                "timings_ms": {"total_pipeline_ms": 4.5, "judge_ms": 0.0},
                "judge_telemetry": None,
            },
            # BYPASS Miss (TN)
            {
                "query_id": "T4",
                "domain": "mathematics",
                "path": "BYPASS",
                "final_decision": "MISS",
                "outcome_type": "TN",
                "was_correct": True,
                "timings_ms": {"total_pipeline_ms": 4.8, "judge_ms": 0.0},
                "judge_telemetry": None,
            },
        ]

        summary = evaluate_telemetry(records)

        # Hit rate: 2 hits / 4 queries = 50.0%
        assert summary["hit_rate_metrics"]["total_queries"] == 4
        assert summary["hit_rate_metrics"]["total_hits"] == 2
        assert summary["hit_rate_metrics"]["overall_hit_rate_pct"] == 50.0

        # Path distribution
        assert summary["path_distribution"]["auto_reuse_count"] == 1
        assert summary["path_distribution"]["ambiguous_count"] == 1
        assert summary["path_distribution"]["bypass_count"] == 2
        assert summary["path_distribution"]["local_resolution_pct"] == 75.0

        # Cost avoided calculation
        expected_cost = (200 / 1e6) * COST_PER_M_INPUT_TOKENS_USD + (40 / 1e6) * COST_PER_M_OUTPUT_TOKENS_USD
        assert summary["token_and_cost_metrics"]["cost_avoided_projection_usd"] == pytest.approx(expected_cost, abs=1e-8)

        # Safety metrics (TP=2, FP=0)
        assert summary["safety_metrics"]["irr_cache"] == 0.0
        assert summary["safety_metrics"]["is_underpowered"] is True
        assert "cannot mathematically clear" in summary["safety_metrics"]["power_disclosure"]


# ---------------------------------------------------------------------------
# 3. Pipeline Runner Mocked Integration Test
# ---------------------------------------------------------------------------

class TestPipelineRunnerMockedIntegration:
    def test_run_load_test_mini_stream_with_mock_judge(self, tmp_path):
        mini_stream = [
            {
                "id": "MINI-001",
                "domain": "mathematics",
                "query": "State the definition and conditions of the central limit theorem in probability theory.",
                "query_type": "UNIQUE_SEED",
                "reference_seed_id": None,
                "is_reuse_safe": False,
                "expected_tier": "BYPASS",
            },
            {
                "id": "MINI-002",
                "domain": "mathematics",
                "query": "State the definition and conditions of the central limit theorem in probability theory.",
                "query_type": "EXACT_REPEAT",
                "reference_seed_id": "MINI-001",
                "is_reuse_safe": True,
                "expected_tier": "AUTO_REUSE",
            },
        ]

        stream_file = tmp_path / "mini_stream.json"
        telemetry_file = tmp_path / "mini_telemetry.json"
        with open(stream_file, "w", encoding="utf-8") as f:
            json.dump(mini_stream, f)

        # Run with mock quota
        with patch("scripts.run_load_test.check_openrouter_quota", return_value={"free_model_daily_requests": {"used": 0, "limit": 50, "remaining": 50}}):
            telemetry = run_load_test(
                stream_path=stream_file,
                output_path=telemetry_file,
                inter_call_delay=0.0,
            )

        assert len(telemetry) == 2
        assert telemetry[0]["path"] == "BYPASS"
        assert telemetry[0]["final_decision"] == "MISS"
        assert telemetry[1]["path"] == "AUTO_REUSE"
        assert telemetry[1]["final_decision"] == "HIT"
        assert telemetry_file.exists()
