"""Step 0: Run trained StabilityClassifier against final test benchmark."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from src.evaluation.stability_evaluator import StabilityEvaluator

FINAL_TEST_PATH = Path("data/raw/query_stability_benchmark_final_test.json")

evaluator = StabilityEvaluator()
m = evaluator.evaluate_benchmark(FINAL_TEST_PATH)

print("=== STEP 0: StabilityClassifier Final Test Evaluation ===")
print(f"Dataset:          {m.benchmark_name}")
print(f"Dataset Role:     {m.dataset_role}")
print(f"Total Queries:    {m.total_queries}")
print(f"Correct:          {m.correct_predictions}")
print(f"Overall Accuracy: {m.accuracy * 100:.2f}%")
print()
print("--- Confusion Matrix (Positive=STABLE, Negative=DYNAMIC) ---")
print(f"  TP (True STABLE  -> pred STABLE):  {m.tp}")
print(f"  TN (True DYNAMIC -> pred DYNAMIC): {m.tn}")
print(f"  FP (True DYNAMIC -> pred STABLE):  {m.fp}  <- dangerous (stale cache hazard)")
print(f"  FN (True STABLE  -> pred DYNAMIC): {m.fn}  <- conservative (unnecessary bypass)")
print()
print("--- Per-Class Metrics ---")
print(f"  STABLE  Precision={m.stable_precision*100:.2f}%  Recall={m.stable_recall*100:.2f}%  F1={m.stable_f1*100:.2f}%")
print(f"  DYNAMIC Precision={m.dynamic_precision*100:.2f}%  Recall={m.dynamic_recall*100:.2f}%  F1={m.dynamic_f1*100:.2f}%")
print()
print(f"  Dangerous Error Rate (FP/Total Dynamic): {m.dangerous_error_rate*100:.2f}%  ({m.dangerous_error_count} errors)")
print(f"  Conservative Error Rate (FN/Total Stable): {m.conservative_error_rate*100:.2f}%  ({m.conservative_error_count} errors)")
print()
print(f"  Rule-resolved:     {m.rule_resolved_count} ({m.rule_resolved_pct:.1f}%)")
print(f"  Fallback-resolved: {m.fallback_resolved_count} ({m.fallback_resolved_pct:.1f}%)")
print()
print(f"  Latency Mean={m.latency_mean_ms:.4f}ms  P50={m.latency_p50_ms:.4f}ms  P95={m.latency_p95_ms:.4f}ms")
print()
passes = m.accuracy >= 0.85
status = "PASS" if passes else "FAIL"
print(f"pytest assertion (accuracy >= 85%): {status}  (measured: {m.accuracy*100:.2f}%)")
print()
result = {
    "total_queries": m.total_queries,
    "correct": m.correct_predictions,
    "accuracy": round(m.accuracy, 6),
    "tp": m.tp, "tn": m.tn, "fp": m.fp, "fn": m.fn,
    "stable_precision": round(m.stable_precision, 6),
    "stable_recall": round(m.stable_recall, 6),
    "stable_f1": round(m.stable_f1, 6),
    "dynamic_precision": round(m.dynamic_precision, 6),
    "dynamic_recall": round(m.dynamic_recall, 6),
    "dynamic_f1": round(m.dynamic_f1, 6),
    "dangerous_error_rate": round(m.dangerous_error_rate, 6),
    "dangerous_error_count": m.dangerous_error_count,
    "conservative_error_rate": round(m.conservative_error_rate, 6),
    "conservative_error_count": m.conservative_error_count,
    "rule_resolved_count": m.rule_resolved_count,
    "rule_resolved_pct": round(m.rule_resolved_pct, 2),
    "fallback_resolved_count": m.fallback_resolved_count,
    "fallback_resolved_pct": round(m.fallback_resolved_pct, 2),
    "latency_mean_ms": round(m.latency_mean_ms, 4),
    "latency_p50_ms": round(m.latency_p50_ms, 4),
    "latency_p95_ms": round(m.latency_p95_ms, 4),
    "pytest_85pct_pass": passes,
}
print("JSON:")
print(json.dumps(result, indent=2))
