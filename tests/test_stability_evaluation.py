"""Integration and benchmark evaluation tests for Phase 1 Stability Classifier."""

from pathlib import Path
import pytest

from src.classifier.models import ConditionalRoutingPolicy
from src.evaluation.stability_evaluator import StabilityEvaluator

DEV_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "query_stability_benchmark.json"
)
HELDOUT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "query_stability_benchmark_heldout.json"
)
FINAL_TEST_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "query_stability_benchmark_final_test.json"
)


def test_stability_evaluator_dev_benchmark_run():
    """Verify evaluation runs over the 160-query development benchmark."""
    assert DEV_BENCHMARK_PATH.exists(), f"Dev benchmark file not found at {DEV_BENCHMARK_PATH}"

    evaluator = StabilityEvaluator()
    metrics = evaluator.evaluate_benchmark(DEV_BENCHMARK_PATH)

    # 1. Verification of query counts
    assert metrics.total_queries == 160, f"Expected 160 queries, got {metrics.total_queries}"

    # 2. Observed safety metric: dangerous errors on dev benchmark
    assert metrics.dangerous_error_count == 0, (
        f"Hazard observed on dev benchmark: {metrics.dangerous_error_count} dynamic queries were predicted as STABLE"
    )

    # 3. Overall Accuracy on development benchmark
    assert metrics.accuracy >= 0.90, f"Dev accuracy {metrics.accuracy * 100:.2f}% below 90% threshold"

    # 4. Routing resolution breakdown
    assert metrics.rule_resolved_pct >= 50.0

    # 5. Latency measurement exists and is positive
    assert metrics.latency_mean_ms > 0.0


def test_stability_evaluator_heldout_challenge_run():
    """Verify evaluation runs over the challenge / validation benchmark."""
    assert HELDOUT_BENCHMARK_PATH.exists(), f"Heldout benchmark file not found at {HELDOUT_BENCHMARK_PATH}"

    evaluator = StabilityEvaluator()
    metrics = evaluator.evaluate_benchmark(HELDOUT_BENCHMARK_PATH)

    # 1. Verification of query counts
    assert metrics.total_queries >= 50, f"Expected at least 50 challenge queries, got {metrics.total_queries}"

    # 2. Observed accuracy on challenge data
    assert metrics.accuracy >= 0.85, f"Challenge accuracy {metrics.accuracy * 100:.2f}% below 85% threshold"

    # 3. Empirical latency is measured
    assert metrics.latency_p50_ms > 0.0
    assert metrics.latency_mean_ms < 10.0


def test_stability_evaluator_final_test_run():
    """Verify evaluation runs over the pristine unseen final test benchmark."""
    assert FINAL_TEST_BENCHMARK_PATH.exists(), f"Final test file not found at {FINAL_TEST_BENCHMARK_PATH}"

    evaluator = StabilityEvaluator()
    metrics = evaluator.evaluate_benchmark(FINAL_TEST_BENCHMARK_PATH)

    # 1. Verification of query counts
    assert metrics.total_queries >= 50, f"Expected at least 50 final test queries, got {metrics.total_queries}"

    # 2. Observed accuracy on final test data
    assert metrics.accuracy >= 0.85, f"Final test accuracy {metrics.accuracy * 100:.2f}% below 85% threshold"

    # 3. Empirical latency is measured
    assert metrics.latency_p50_ms > 0.0
    assert metrics.latency_mean_ms < 10.0


def test_stability_evaluator_conditional_policies_comparison():
    """Verify metrics reflect different CONDITIONALLY_STABLE policies."""
    evaluator = StabilityEvaluator()

    metrics_fwd = evaluator.evaluate_benchmark(
        DEV_BENCHMARK_PATH,
        conditional_policy=ConditionalRoutingPolicy.FORWARD_TO_CACHE_CANDIDATE,
    )
    metrics_byp = evaluator.evaluate_benchmark(
        DEV_BENCHMARK_PATH,
        conditional_policy=ConditionalRoutingPolicy.CONSERVATIVE_BYPASS,
    )

    assert metrics_fwd.total_queries == metrics_byp.total_queries
    assert metrics_byp.tn >= metrics_fwd.tn


def test_stability_evaluator_human_credibility_run():
    """Verify evaluation runs seamlessly on metadata-wrapped human credibility dataset."""
    human_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "query_stability_human_credibility.json"
    assert human_path.exists(), f"Missing file: {human_path}"

    evaluator = StabilityEvaluator()
    metrics = evaluator.evaluate_benchmark(human_path)

    assert metrics.total_queries == 85
    assert "DEV/DIAGNOSTIC" in metrics.dataset_role or "Human Credibility" in metrics.dataset_role
    assert metrics.latency_p50_ms > 0.0


def test_stability_evaluator_dataset_loader_error_handling(tmp_path):
    """Verify loader raises informative ValueError on invalid dataset formats."""
    evaluator = StabilityEvaluator()

    # 1. Nonexistent file
    with pytest.raises(FileNotFoundError):
        evaluator.load_benchmark_dataset(tmp_path / "nonexistent.json")

    # 2. Corrupt JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ unquoted_key: 123 ", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted or invalid JSON"):
        evaluator.load_benchmark_dataset(bad_json)

    # 3. Dict without recognized query list key
    no_key_json = tmp_path / "no_key.json"
    no_key_json.write_text('{"foo": "bar", "count": 10}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object must contain a list of queries"):
        evaluator.load_benchmark_dataset(no_key_json)

    # 4. Item missing 'query' field
    missing_query_json = tmp_path / "missing_query.json"
    missing_query_json.write_text('[{"id": "TEST-1", "label": "STABLE"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or invalid 'query'"):
        evaluator.load_benchmark_dataset(missing_query_json)

