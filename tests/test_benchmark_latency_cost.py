"""Tests for scripts/benchmark_latency_cost.py — computation logic only.

These tests cover the mathematical computation functions in the benchmark script
(token cost projection, latency delta, statistics) WITHOUT running wall-clock
timing loops or loading the real SentenceTransformer model.

NOT covered here (by design):
- Actual wall-clock latency values (non-deterministic, machine-dependent)
- Loading the all-MiniLM-L6-v2 model
- Making any network requests

pytest marker: (none) — these run unconditionally in the default invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Make the scripts/ directory importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_latency_cost import (
    COST_PER_M_INPUT_TOKENS_USD,
    COST_PER_M_OUTPUT_TOKENS_USD,
    HIT_PATH_RUNS,
    HYPOTHETICAL_QUERIES_PER_DAY,
    PHASE3_PRODUCTION_ARR,
    PRICING_DATE,
    PRICING_MODEL_NAME,
    PRICING_MODEL_URL,
    compute_cost_projection,
    compute_latency_delta,
    compute_llm_latency_stats,
    compute_token_stats,
    load_genuine_telemetry,
)


# ---------------------------------------------------------------------------
# Fixtures — deterministic fake telemetry records
# ---------------------------------------------------------------------------

def _make_genuine_records(n: int = 5) -> List[Dict]:
    """Create N fake genuine telemetry records with known values."""
    return [
        {
            "pair_id": f"SYNTH-{i:03d}",
            "domain": "mathematics",
            "similarity_score": 0.80,
            "judge_is_safe": True,
            "fallback_triggered": False,
            "latency_ms": float(1000 * (i + 1)),   # 1000, 2000, ..., N*1000 ms
            "prompt_tokens": 580 + i,
            "completion_tokens": 195 + i,
            "total_tokens": 775 + 2 * i,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: load_genuine_telemetry() filter predicate
# ---------------------------------------------------------------------------

class TestLoadGenuineTelemetry:
    """The telemetry file must exist and return >= 1 genuine record."""

    def test_real_telemetry_file_returns_genuine(self):
        """Integration check: the real telemetry file yields genuine records."""
        telemetry_path = REPO_ROOT / "data" / "openrouter_synthetic_feedback_telemetry.json"
        if not telemetry_path.exists():
            pytest.skip("Telemetry file not present -- skipping")

        genuine = load_genuine_telemetry()
        assert len(genuine) == 69, (
            f"Expected 69 genuine records, got {len(genuine)}. "
            "Run label_synthetic_feedback.py to obtain more real labels."
        )

    def test_filter_excludes_fallback_records(self):
        """Verify load_genuine_telemetry() only returns non-fallback records.
        Tested indirectly via the actual telemetry file if it exists."""
        telemetry_path = REPO_ROOT / "data" / "openrouter_synthetic_feedback_telemetry.json"
        if not telemetry_path.exists():
            pytest.skip("Telemetry file not present -- skipping")

        genuine = load_genuine_telemetry()
        for r in genuine:
            assert not r.get("fallback_triggered", False), (
                f"load_genuine_telemetry() returned a fallback record: {r.get('pair_id')}"
            )

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """If the telemetry file doesn't exist, FileNotFoundError is raised."""
        import scripts.benchmark_latency_cost as bm
        monkeypatch.setattr(bm, "TELEMETRY_PATH", tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError, match="Telemetry"):
            bm.load_genuine_telemetry()


# ---------------------------------------------------------------------------
# Tests: compute_llm_latency_stats()
# ---------------------------------------------------------------------------

class TestComputeLLMLatencyStats:
    """LLM latency statistics from real measured records."""

    def test_known_latencies_mean(self):
        """Mean of [1000, 2000, 3000, 4000, 5000] = 3000.0"""
        records = _make_genuine_records(n=5)
        stats = compute_llm_latency_stats(records)
        assert stats["mean_ms"] == pytest.approx(3000.0, rel=1e-3)

    def test_known_latencies_median(self):
        """Median of [1000, 2000, 3000, 4000, 5000] = 3000.0"""
        records = _make_genuine_records(n=5)
        stats = compute_llm_latency_stats(records)
        assert stats["median_ms"] == pytest.approx(3000.0, rel=1e-3)

    def test_n_equals_record_count(self):
        records = _make_genuine_records(n=7)
        stats = compute_llm_latency_stats(records)
        assert stats["n"] == 7

    def test_p95_is_at_or_near_max_for_small_n(self):
        """For N=5, p95 index = int(0.95*5)=4 which is the last element."""
        records = _make_genuine_records(n=5)
        stats = compute_llm_latency_stats(records)
        # latencies are 1000, 2000, 3000, 4000, 5000 -- sorted, p95_idx=4 -> 5000.0
        assert stats["p95_ms"] == pytest.approx(5000.0, rel=1e-3)

    def test_min_max_correct(self):
        records = _make_genuine_records(n=5)
        stats = compute_llm_latency_stats(records)
        assert stats["min_ms"] == pytest.approx(1000.0, rel=1e-3)
        assert stats["max_ms"] == pytest.approx(5000.0, rel=1e-3)

    def test_source_field_present(self):
        records = _make_genuine_records(n=3)
        stats = compute_llm_latency_stats(records)
        assert "source" in stats
        assert "telemetry" in stats["source"].lower()

    def test_single_record_stdev_is_zero(self):
        records = _make_genuine_records(n=1)
        stats = compute_llm_latency_stats(records)
        assert stats["stdev_ms"] == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_token_stats()
# ---------------------------------------------------------------------------

class TestComputeTokenStats:
    """Token usage statistics from real measured records."""

    def test_mean_prompt_tokens(self):
        """With i in [0..4], prompt_tokens = [580, 581, 582, 583, 584] -> mean = 582.0"""
        records = _make_genuine_records(n=5)
        stats = compute_token_stats(records)
        assert stats["mean_prompt_tokens"] == pytest.approx(582.0, abs=0.1)

    def test_total_prompt_tokens(self):
        records = _make_genuine_records(n=5)
        stats = compute_token_stats(records)
        # 580+581+582+583+584 = 2910
        assert stats["total_prompt_tokens"] == 2910

    def test_n_calls_equals_record_count(self):
        records = _make_genuine_records(n=6)
        stats = compute_token_stats(records)
        assert stats["n_calls"] == 6

    def test_source_field_present_and_mentions_fallback(self):
        records = _make_genuine_records(n=3)
        stats = compute_token_stats(records)
        assert "source" in stats
        assert "fallback" in stats["source"].lower()


# ---------------------------------------------------------------------------
# Tests: compute_cost_projection()
# ---------------------------------------------------------------------------

class TestComputeCostProjection:
    """Cost projection uses real token stats + named pricing assumption."""

    def _make_token_stats(
        self,
        mean_prompt: float = 583.0,
        mean_completion: float = 195.0,
        n_calls: int = 22,
        total_prompt: int = 12824,
        total_completion: int = 4294,
    ) -> Dict:
        """Return a token_stats dict with known values."""
        return {
            "n_calls": n_calls,
            "mean_prompt_tokens": mean_prompt,
            "mean_completion_tokens": mean_completion,
            "mean_total_tokens": mean_prompt + mean_completion,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens_all_calls": total_prompt + total_completion,
            "source": "test fixture",
        }

    def test_cost_per_call_input_matches_formula(self):
        """cost_per_call_input = mean_prompt / 1e6 * COST_PER_M_INPUT"""
        ts = self._make_token_stats(mean_prompt=600.0)
        cost = compute_cost_projection(ts)
        expected = (600.0 / 1_000_000) * COST_PER_M_INPUT_TOKENS_USD
        assert cost["cost_per_judge_call_input_usd"] == pytest.approx(expected, rel=1e-6)

    def test_cost_per_call_output_matches_formula(self):
        """cost_per_call_output = mean_completion / 1e6 * COST_PER_M_OUTPUT"""
        ts = self._make_token_stats(mean_completion=200.0)
        cost = compute_cost_projection(ts)
        expected = (200.0 / 1_000_000) * COST_PER_M_OUTPUT_TOKENS_USD
        assert cost["cost_per_judge_call_output_usd"] == pytest.approx(expected, rel=1e-6)

    def test_cost_per_call_total_is_sum_of_input_and_output(self):
        ts = self._make_token_stats(mean_prompt=583.0, mean_completion=195.0)
        cost = compute_cost_projection(ts)
        expected_total = (
            cost["cost_per_judge_call_input_usd"]
            + cost["cost_per_judge_call_output_usd"]
        )
        assert cost["cost_per_judge_call_total_usd"] == pytest.approx(expected_total, rel=1e-6)

    def test_daily_projection_formula(self):
        """daily_cost_avoided = (queries/day * ARR) * cost_per_call

        Note: daily_cost_avoided_usd is stored rounded to 4 decimal places in the
        script output. The tolerance here uses abs=0.001 (sub-cent precision) which
        is appropriate for daily dollar projection figures.
        """
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        daily_hits = HYPOTHETICAL_QUERIES_PER_DAY * PHASE3_PRODUCTION_ARR
        # Recompute expected from the same formula (not from the rounded intermediate)
        expected_daily = daily_hits * (
            (ts["mean_prompt_tokens"] / 1_000_000) * COST_PER_M_INPUT_TOKENS_USD
            + (ts["mean_completion_tokens"] / 1_000_000) * COST_PER_M_OUTPUT_TOKENS_USD
        )
        assert cost["production_projection"]["daily_cost_avoided_usd"] == pytest.approx(
            expected_daily, abs=0.001  # within $0.001 (sub-cent) -- sufficient for projections
        )

    def test_pricing_model_name_is_stated(self):
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        assert cost["pricing_model"] == PRICING_MODEL_NAME
        assert PRICING_MODEL_NAME in cost["label"]

    def test_pricing_date_is_stated(self):
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        assert PRICING_DATE in cost["label"]

    def test_projection_label_says_projection(self):
        """The projection must be explicitly labeled as a projection, not a measurement."""
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        proj_label = cost["production_projection"]["label"]
        assert "PROJECTION" in proj_label.upper(), (
            "Projection label must contain 'PROJECTION' to clearly indicate it's not measured"
        )

    def test_label_mentions_free_tier_not_savings(self):
        """Label must disclaim that the actual model was free-tier (not presented as savings)."""
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        label = cost["label"]
        assert "free tier" in label.lower() or "free-tier" in label.lower(), (
            "Label must state that the actual model used the free tier (zero dollars billed)"
        )

    def test_pricing_url_is_present(self):
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        assert "pricing_url" in cost
        assert cost["pricing_url"].startswith("https://")

    def test_arr_source_is_documented(self):
        ts = self._make_token_stats()
        cost = compute_cost_projection(ts)
        arr_source = cost["production_projection"].get("arr_source", "")
        assert len(arr_source) > 10, (
            "arr_source must document where the ARR figure comes from"
        )

    def test_zero_tokens_yields_zero_cost(self):
        """Edge case: zero tokens should produce zero cost."""
        ts = self._make_token_stats(mean_prompt=0.0, mean_completion=0.0, total_prompt=0, total_completion=0)
        cost = compute_cost_projection(ts)
        assert cost["cost_per_judge_call_total_usd"] == pytest.approx(0.0, abs=1e-10)
        assert cost["production_projection"]["daily_cost_avoided_usd"] == pytest.approx(0.0, abs=1e-10)

    def test_real_telemetry_token_counts_produce_realistic_cost(self):
        """Using real measured token counts (N=22), cost per call should be in a sane range.
        Mean prompt=583, completion=195 -> cost ~ $0.0000874 + $0.0001170 = ~$0.000205
        """
        ts = self._make_token_stats(
            mean_prompt=583.0, mean_completion=195.0, n_calls=22,
            total_prompt=12824, total_completion=4294
        )
        cost = compute_cost_projection(ts)
        # Should be in the range $0.000100 to $0.000500 per call
        assert 0.0001 < cost["cost_per_judge_call_total_usd"] < 0.0005, (
            f"Cost per call out of expected range: {cost['cost_per_judge_call_total_usd']}"
        )


# ---------------------------------------------------------------------------
# Tests: compute_latency_delta()
# ---------------------------------------------------------------------------

class TestComputeLatencyDelta:
    """Latency delta computations."""

    def _make_hit_stats(self, mean_ms: float = 25.0, median_ms: float = 23.0) -> Dict:
        return {
            "n": 50, "mean_ms": mean_ms, "median_ms": median_ms,
            "p95_ms": 35.0, "min_ms": 18.0, "max_ms": 45.0, "stdev_ms": 5.0,
        }

    def _make_miss_stats(self, mean_ms: float = 4238.7, median_ms: float = 2465.9) -> Dict:
        return {
            "n": 22, "mean_ms": mean_ms, "median_ms": median_ms,
            "p95_ms": 12422.3, "min_ms": 252.7, "max_ms": 15539.1, "stdev_ms": 3928.7,
        }

    def test_delta_mean_is_miss_minus_hit(self):
        hit = self._make_hit_stats(mean_ms=25.0)
        miss = self._make_miss_stats(mean_ms=4238.7)
        delta = compute_latency_delta(hit, miss)
        assert delta["delta_mean_ms"] == pytest.approx(4238.7 - 25.0, abs=0.5)

    def test_delta_median_is_miss_minus_hit(self):
        hit = self._make_hit_stats(median_ms=23.0)
        miss = self._make_miss_stats(median_ms=2465.9)
        delta = compute_latency_delta(hit, miss)
        assert delta["delta_median_ms"] == pytest.approx(2465.9 - 23.0, abs=0.5)

    def test_pct_reduction_mean_formula(self):
        """pct_reduction = (miss - hit) / miss * 100"""
        hit = self._make_hit_stats(mean_ms=25.0)
        miss = self._make_miss_stats(mean_ms=4238.7)
        delta = compute_latency_delta(hit, miss)
        expected_pct = (4238.7 - 25.0) / 4238.7 * 100
        assert delta["pct_reduction_mean"] == pytest.approx(expected_pct, abs=0.5)

    def test_zero_miss_latency_gives_zero_reduction(self):
        """Edge case: if miss latency = 0, reduction = 0 (no division by zero)."""
        hit = self._make_hit_stats(mean_ms=25.0, median_ms=23.0)
        miss = self._make_miss_stats(mean_ms=0.0, median_ms=0.0)
        delta = compute_latency_delta(hit, miss)
        assert delta["pct_reduction_mean"] == 0.0
        assert delta["pct_reduction_median"] == 0.0

    def test_summary_string_is_present(self):
        hit = self._make_hit_stats()
        miss = self._make_miss_stats()
        delta = compute_latency_delta(hit, miss)
        assert "summary" in delta
        assert len(delta["summary"]) > 20

    def test_reduction_is_positive_when_miss_greater_than_hit(self):
        hit = self._make_hit_stats(mean_ms=25.0)
        miss = self._make_miss_stats(mean_ms=4000.0)
        delta = compute_latency_delta(hit, miss)
        assert delta["delta_mean_ms"] > 0
        assert delta["pct_reduction_mean"] > 0


# ---------------------------------------------------------------------------
# Tests: benchmark constants
# ---------------------------------------------------------------------------

class TestBenchmarkConstants:
    """Verify benchmark constants are set to reasonable values."""

    def test_hit_path_runs_at_least_30(self):
        assert HIT_PATH_RUNS >= 30, (
            f"HIT_PATH_RUNS must be >= 30 per spec, got {HIT_PATH_RUNS}"
        )

    def test_phase3_arr_is_ten_percent(self):
        assert PHASE3_PRODUCTION_ARR == pytest.approx(0.10, abs=1e-6), (
            "PHASE3_PRODUCTION_ARR must be 0.10 (Phase 3 production policy ARR)"
        )

    def test_pricing_date_is_present(self):
        assert PRICING_DATE and len(PRICING_DATE) == 10  # YYYY-MM-DD

    def test_pricing_model_url_is_openai(self):
        assert "openai.com" in PRICING_MODEL_URL

    def test_pricing_model_name_is_gpt4o_mini(self):
        assert PRICING_MODEL_NAME == "gpt-4o-mini"

    def test_input_pricing_is_0_15_per_million(self):
        assert COST_PER_M_INPUT_TOKENS_USD == pytest.approx(0.150, abs=1e-6)

    def test_output_pricing_is_0_60_per_million(self):
        assert COST_PER_M_OUTPUT_TOKENS_USD == pytest.approx(0.600, abs=1e-6)
