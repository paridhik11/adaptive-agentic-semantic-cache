"""Formal benchmark evaluation runner for the Stability Classifier."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional

from src.classifier.models import ClassificationSource, ConditionalRoutingPolicy, StabilityLabel
from src.classifier.stability_classifier import StabilityClassifier


@dataclass
class StabilityMetrics:
    """Calculated metrics for stability classifier benchmark."""

    benchmark_name: str
    dataset_role: str  # "Development", "Challenge/Validation", "Final Test", "Human Credibility Audit", etc.
    total_queries: int
    correct_predictions: int
    accuracy: float

    # Confusion matrix counts (Positive = STABLE, Negative = DYNAMIC)
    # TP: True STABLE predicted as STABLE (safe cache candidate)
    # TN: True DYNAMIC predicted as DYNAMIC (safe bypass)
    # FP: True DYNAMIC predicted as STABLE (OBSERVED DANGEROUS ERROR: stale cache hit hazard)
    # FN: True STABLE predicted as DYNAMIC (OBSERVED CONSERVATIVE ERROR: unnecessary LLM regeneration)
    tp: int
    tn: int
    fp: int
    fn: int

    # Class-level metrics
    stable_precision: float
    stable_recall: float
    stable_f1: float

    dynamic_precision: float
    dynamic_recall: float
    dynamic_f1: float

    # Critical safety error rates
    dangerous_error_count: int
    dangerous_error_rate: float  # FP / Total Dynamic

    conservative_error_count: int
    conservative_error_rate: float  # FN / Total Stable

    # Pipeline routing resolution counts
    rule_resolved_count: int
    rule_resolved_pct: float
    fallback_resolved_count: int
    fallback_resolved_pct: float

    # Measured empirical latencies (ms)
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float

    conditional_policy: str = "FORWARD_TO_CACHE_CANDIDATE"
    domain_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)


class StabilityEvaluator:
    """Evaluates stability classification performance against benchmark datasets."""

    def __init__(self, classifier: Optional[StabilityClassifier] = None) -> None:
        """Initialize evaluator with stability classifier."""
        self.classifier = classifier or StabilityClassifier()

    @staticmethod
    def load_benchmark_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
        """Load and normalize a benchmark dataset from JSON.

        Supports both:
        1. Direct JSON array: [ { "query": "...", "stability_label": "..." }, ... ]
        2. Metadata-wrapped JSON object: { "metadata": {...}, "queries": [ { "query": "...", ... }, ... ] }
           (also supports keys 'items', 'data', 'samples')

        Normalizes field names:
        - 'query' (required non-empty string)
        - 'id' (defaults to STAB-### if absent)
        - 'stability_label' (resolves from 'stability_label', 'expected_label', 'label')
        - 'domain' (resolves from 'domain', 'category', defaults to 'general')
        - 'rationale' (resolves from 'rationale', 'reason', defaults to '')

        Raises:
            FileNotFoundError: If dataset_path does not exist.
            ValueError: If JSON structure is invalid or query items lack required fields.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupted or invalid JSON in {dataset_path}: {e}") from e

        # Surface dataset_role from JSON metadata when present (Phase 1.1 Sub-stage B).
        # Note: load_benchmark_dataset is a @staticmethod so dataset_role is extracted
        # separately in evaluate_benchmark via a pre-peek of the raw JSON top level.
        if isinstance(raw_data, list):
            raw_items = raw_data
        elif isinstance(raw_data, dict):
            candidates = ["queries", "items", "data", "samples"]
            found_key = next((k for k in candidates if k in raw_data and isinstance(raw_data[k], list)), None)
            if found_key is not None:
                raw_items = raw_data[found_key]
            else:
                raise ValueError(
                    f"Invalid dataset structure in {dataset_path}: JSON object must contain a list of queries under "
                    f"one of the keys: {candidates}. Found keys: {list(raw_data.keys())}"
                )
        else:
            raise ValueError(
                f"Invalid dataset format in {dataset_path}: Expected JSON array or object with queries list, "
                f"got {type(raw_data).__name__}"
            )


        if not raw_items:
            raise ValueError(f"Empty dataset in {dataset_path}: Contains 0 queries.")

        normalized_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid item format at index {idx} in {dataset_path}: Expected object/dict, got {type(item).__name__}"
                )

            query = item.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                item_id = item.get("id", f"index-{idx}")
                raise ValueError(f"Missing or invalid 'query' string in item {item_id} of {dataset_path}")

            item_id = item.get("id", f"STAB-{idx:03d}")
            raw_label = (
                item.get("stability_label")
                or item.get("expected_label")
                or item.get("label")
                or "STABLE"
            )
            raw_label_upper = str(raw_label).strip().upper()
            if raw_label_upper not in ("STABLE", "DYNAMIC", "CONDITIONALLY_STABLE"):
                raw_label_upper = "DYNAMIC" if "DYN" in raw_label_upper else "STABLE"

            domain = item.get("domain") or item.get("category") or "general"
            rationale = item.get("rationale") or item.get("reason") or ""

            normalized_items.append({
                "id": str(item_id),
                "query": query.strip(),
                "stability_label": raw_label_upper,
                "domain": str(domain).strip(),
                "rationale": str(rationale).strip(),
                "raw_item": item,
            })

        return normalized_items

    def evaluate_benchmark(
        self,
        dataset_path: Optional[Path] = None,
        conditional_policy: ConditionalRoutingPolicy = ConditionalRoutingPolicy.FORWARD_TO_CACHE_CANDIDATE,
        dataset_role: Optional[str] = None,
    ) -> StabilityMetrics:
        """Run evaluation over a stability benchmark JSON dataset.

        Args:
            dataset_path: Path to benchmark JSON file. Defaults to data/raw/query_stability_benchmark.json.
            conditional_policy: Routing policy for CONDITIONALLY_STABLE queries.
            dataset_role: Optional descriptive role ("Development", "Challenge/Validation", "Final Test", "Human Credibility Audit").

        Returns:
            StabilityMetrics object containing complete performance evaluation.
        """
        if dataset_path is None:
            dataset_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "raw"
                / "query_stability_benchmark.json"
            )

        items = self.load_benchmark_dataset(dataset_path)
        benchmark_name = dataset_path.name

        # Extract dataset_role from JSON top-level, if present (Phase 1.1 Sub-stage B).
        # This requires a cheap second JSON load of just the top-level key; the full
        # normalised items were already loaded above — no re-parsing of query content.
        json_role: Optional[str] = None
        try:
            with open(dataset_path, "r", encoding="utf-8") as _f:
                _raw = json.load(_f)
            if isinstance(_raw, dict):
                json_role = _raw.get("dataset_role") or None
        except Exception:
            pass  # Non-critical: fall back to filename heuristic

        # Auto-populate dataset_role: caller-supplied > JSON metadata > filename heuristic
        if dataset_role is None:
            if json_role == "development_diagnostic_do_not_use_as_final_eval":
                # Preserve the canonical machine-readable role while adding a human label
                if "human" in benchmark_name or "credibility" in benchmark_name:
                    dataset_role = "DEV/DIAGNOSTIC — Human Credibility Audit (do not use as final eval)"
                else:
                    dataset_role = "DEV/DIAGNOSTIC — Development Benchmark (do not use as final eval)"
            elif json_role is not None:
                dataset_role = json_role
            elif "human" in benchmark_name or "credibility" in benchmark_name:
                dataset_role = "Human Credibility Audit"
            elif "final_test" in benchmark_name:
                dataset_role = "Final Test (Pristine — operationally untouched)"
            elif "heldout" in benchmark_name:
                dataset_role = "Challenge / Validation (operationally untouched)"
            else:
                dataset_role = "Development"

        self.classifier.conditional_policy = conditional_policy

        detailed_results: List[Dict[str, Any]] = []
        domain_stats = defaultdict(lambda: {"total": 0, "correct": 0, "dangerous": 0, "conservative": 0})

        tp = tn = fp = fn = 0
        rule_count = 0
        fallback_count = 0
        latencies_ms: List[float] = []

        for item in items:
            query = item["query"]
            raw_ground_truth = item["stability_label"]
            domain = item["domain"]

            # Map ground truth to binary safe-cache evaluation
            if raw_ground_truth == "DYNAMIC":
                ground_truth = StabilityLabel.DYNAMIC
            elif raw_ground_truth == "CONDITIONALLY_STABLE":
                if conditional_policy == ConditionalRoutingPolicy.CONSERVATIVE_BYPASS:
                    ground_truth = StabilityLabel.DYNAMIC
                else:
                    ground_truth = StabilityLabel.STABLE
            else:
                ground_truth = StabilityLabel.STABLE

            t0 = time.perf_counter()
            res = self.classifier.classify(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            pred = res.effective_decision
            is_correct = (pred == ground_truth)

            if res.source == ClassificationSource.RULE:
                rule_count += 1
            else:
                fallback_count += 1

            # Update confusion matrix
            if ground_truth == StabilityLabel.STABLE and pred == StabilityLabel.STABLE:
                tp += 1
            elif ground_truth == StabilityLabel.DYNAMIC and pred == StabilityLabel.DYNAMIC:
                tn += 1
            elif ground_truth == StabilityLabel.DYNAMIC and pred == StabilityLabel.STABLE:
                fp += 1  # OBSERVED DANGEROUS ERROR
            elif ground_truth == StabilityLabel.STABLE and pred == StabilityLabel.DYNAMIC:
                fn += 1  # OBSERVED CONSERVATIVE ERROR

            # Update domain stats
            domain_stats[domain]["total"] += 1
            if is_correct:
                domain_stats[domain]["correct"] += 1
            if ground_truth == StabilityLabel.DYNAMIC and pred == StabilityLabel.STABLE:
                domain_stats[domain]["dangerous"] += 1
            if ground_truth == StabilityLabel.STABLE and pred == StabilityLabel.DYNAMIC:
                domain_stats[domain]["conservative"] += 1

            detailed_results.append({
                "id": item["id"],
                "query": query,
                "domain": domain,
                "raw_ground_truth": raw_ground_truth,
                "evaluated_ground_truth": ground_truth.value,
                "predicted_label": res.predicted_label.value,
                "effective_decision": res.effective_decision.value,
                "is_correct": is_correct,
                "source": res.source.value,
                "matched_rule": res.matched_rule,
                "confidence": res.confidence,
                "latency_ms": round(elapsed_ms, 4),
                "rationale": res.rationale,
            })

        total = len(items)
        correct = tp + tn
        accuracy = (correct / total) if total > 0 else 0.0

        total_dynamic = tn + fp
        total_stable = tp + fn

        # Stable metrics
        stable_prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        stable_rec = (tp / total_stable) if total_stable > 0 else 0.0
        stable_f1 = (
            (2 * stable_prec * stable_rec) / (stable_prec + stable_rec)
            if (stable_prec + stable_rec) > 0
            else 0.0
        )

        # Dynamic metrics
        dynamic_prec = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        dynamic_rec = (tn / total_dynamic) if total_dynamic > 0 else 0.0
        dynamic_f1 = (
            (2 * dynamic_prec * dynamic_rec) / (dynamic_prec + dynamic_rec)
            if (dynamic_prec + dynamic_rec) > 0
            else 0.0
        )

        dangerous_error_rate = (fp / total_dynamic) if total_dynamic > 0 else 0.0
        conservative_error_rate = (fn / total_stable) if total_stable > 0 else 0.0

        rule_pct = (rule_count / total * 100.0) if total > 0 else 0.0
        fallback_pct = (fallback_count / total * 100.0) if total > 0 else 0.0

        # Latency metrics
        lat_mean = statistics.mean(latencies_ms) if latencies_ms else 0.0
        lat_p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
        lat_p95 = (
            statistics.quantiles(latencies_ms, n=20)[18]
            if len(latencies_ms) >= 20
            else max(latencies_ms)
        )

        # Domain breakdown
        domain_breakdown: Dict[str, Dict[str, Any]] = {}
        for dom, stats in domain_stats.items():
            dom_tot = stats["total"]
            dom_acc = (stats["correct"] / dom_tot) if dom_tot > 0 else 0.0
            domain_breakdown[dom] = {
                "total": dom_tot,
                "correct": stats["correct"],
                "accuracy": round(dom_acc, 4),
                "dangerous_errors": stats["dangerous"],
                "conservative_errors": stats["conservative"],
            }

        return StabilityMetrics(
            benchmark_name=benchmark_name,
            dataset_role=dataset_role,
            total_queries=total,
            correct_predictions=correct,
            accuracy=accuracy,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            stable_precision=stable_prec,
            stable_recall=stable_rec,
            stable_f1=stable_f1,
            dynamic_precision=dynamic_prec,
            dynamic_recall=dynamic_rec,
            dynamic_f1=dynamic_f1,
            dangerous_error_count=fp,
            dangerous_error_rate=dangerous_error_rate,
            conservative_error_count=fn,
            conservative_error_rate=conservative_error_rate,
            rule_resolved_count=rule_count,
            rule_resolved_pct=rule_pct,
            fallback_resolved_count=fallback_count,
            fallback_resolved_pct=fallback_pct,
            latency_mean_ms=round(lat_mean, 4),
            latency_p50_ms=round(lat_p50, 4),
            latency_p95_ms=round(lat_p95, 4),
            conditional_policy=conditional_policy.value,
            domain_breakdown=domain_breakdown,
            detailed_results=detailed_results,
        )

    def print_report(self, metrics: StabilityMetrics) -> None:
        """Print a structured evaluation report to console."""
        print("=" * 74)
        print(f"STABILITY CLASSIFIER BENCHMARK REPORT: {metrics.benchmark_name.upper()}")
        print(f"Dataset Role : {metrics.dataset_role}")
        # Phase 1.1 Sub-stage B: warn prominently when this is a dev/diagnostic run
        if "diagnostic" in metrics.dataset_role.lower() or "do not use" in metrics.dataset_role.lower():
            print("-" * 74)
            print("! DEV/DIAGNOSTIC DATASET — numbers below are NOT final evaluation results.")
            print("! This dataset was used during classifier development and/or rule design.")
            print("! Do not cite these figures as held-out accuracy. Use a pristine eval set.")
        elif "pristine" in metrics.dataset_role.lower() or "untouched" in metrics.dataset_role.lower():
            print("-" * 74)
            print("  OPERATIONALLY UNTOUCHED dataset — treat results with appropriate care")
            print("  (independence is not cryptographically proven; see B.0 findings).")
        print(f"Policy       : CONDITIONALLY_STABLE -> {metrics.conditional_policy}")
        print("=" * 74)
        print(f"Total Benchmark Queries : {metrics.total_queries}")
        print(f"Overall Accuracy        : {metrics.accuracy * 100:.2f}% ({metrics.correct_predictions}/{metrics.total_queries})")
        print("-" * 74)
        print("CONFUSION MATRIX (STABLE=Positive, DYNAMIC=Negative)")
        print(f"  True Positives  (TP, Correct STABLE)   : {metrics.tp:3d}")
        print(f"  True Negatives  (TN, Correct DYNAMIC)  : {metrics.tn:3d}")
        print(f"  False Positives (FP, DANGEROUS ERROR)  : {metrics.fp:3d}  (Dynamic predicted as Stable)")
        print(f"  False Negatives (FN, CONSERVATIVE ERR) : {metrics.fn:3d}  (Stable predicted as Dynamic)")
        print("-" * 74)
        print("CLASS-LEVEL PERFORMANCE")
        print(f"  STABLE  -> Precision: {metrics.stable_precision * 100:.2f}%, Recall: {metrics.stable_recall * 100:.2f}%, F1: {metrics.stable_f1 * 100:.2f}%")
        print(f"  DYNAMIC -> Precision: {metrics.dynamic_precision * 100:.2f}%, Recall: {metrics.dynamic_recall * 100:.2f}%, F1: {metrics.dynamic_f1 * 100:.2f}%")
        print("-" * 74)
        print("SAFETY & CACHE HAZARD METRICS (OBSERVED EMPIRICAL COUNTS)")
        print(f"  Dangerous Error Rate (FP / Dynamic)    : {metrics.dangerous_error_rate * 100:.2f}% ({metrics.dangerous_error_count} / {metrics.tn + metrics.fp} dynamic queries)")
        print(f"  Conservative Error Rate (FN / Stable)  : {metrics.conservative_error_rate * 100:.2f}% ({metrics.conservative_error_count} / {metrics.tp + metrics.fn} stable queries)")
        print("-" * 74)
        print("PIPELINE ROUTING RESOLUTION")
        print(f"  Stage 1 (Rule-Based) Resolved          : {metrics.rule_resolved_count:3d} ({metrics.rule_resolved_pct:.1f}%)")
        print(f"  Stage 2 (Fallback) Resolved            : {metrics.fallback_resolved_count:3d} ({metrics.fallback_resolved_pct:.1f}%)")
        print("-" * 74)
        print("MEASURED LATENCY (EMPIRICAL PER QUERY)")
        print(f"  Mean Latency : {metrics.latency_mean_ms:.4f} ms")
        print(f"  P50 (Median) : {metrics.latency_p50_ms:.4f} ms")
        print(f"  P95 Latency  : {metrics.latency_p95_ms:.4f} ms")
        print("-" * 74)
        print("DOMAIN BREAKDOWN")
        for dom, stats in sorted(metrics.domain_breakdown.items()):
            print(f"  {dom:<25} : Acc={stats['accuracy'] * 100:5.1f}% | Total={stats['total']:2d} | Dangerous={stats['dangerous_errors']} | Conservative={stats['conservative_errors']}")
        print("=" * 74)

    def print_tri_comparison(self, m_dev: StabilityMetrics, m_chal: StabilityMetrics, m_test: StabilityMetrics) -> None:
        """Print consolidated comparison table across Development, Challenge, and Final Test datasets."""
        print("=" * 90)
        print("TRI-DATASET STABILITY CLASSIFIER BENCHMARK COMPARISON REPORT")
        print("=" * 90)
        print(f"{'Metric':<36} | {'Development (N=160)':<16} | {'Challenge (N=60)':<16} | {'Final Test (N=60)':<16}")
        print("-" * 90)
        print(f"{'Overall Accuracy':<36} | {m_dev.accuracy * 100:6.2f}% ({m_dev.correct_predictions}/{m_dev.total_queries}) | {m_chal.accuracy * 100:6.2f}% ({m_chal.correct_predictions}/{m_chal.total_queries}) | {m_test.accuracy * 100:6.2f}% ({m_test.correct_predictions}/{m_test.total_queries})")
        print(f"{'Dangerous Errors (FP: Dyn -> Stable)':<36} | {m_dev.dangerous_error_count:2d} ({m_dev.dangerous_error_rate*100:4.1f}%)       | {m_chal.dangerous_error_count:2d} ({m_chal.dangerous_error_rate*100:4.1f}%)       | {m_test.dangerous_error_count:2d} ({m_test.dangerous_error_rate*100:4.1f}%)")
        print(f"{'Conservative Errors (FN: Stab -> Dyn)':<36} | {m_dev.conservative_error_count:2d} ({m_dev.conservative_error_rate*100:4.1f}%)       | {m_chal.conservative_error_count:2d} ({m_chal.conservative_error_rate*100:4.1f}%)       | {m_test.conservative_error_count:2d} ({m_test.conservative_error_rate*100:4.1f}%)")
        print(f"{'STABLE Precision / Recall':<36} | {m_dev.stable_precision*100:5.1f}% / {m_dev.stable_recall*100:5.1f}% | {m_chal.stable_precision*100:5.1f}% / {m_chal.stable_recall*100:5.1f}% | {m_test.stable_precision*100:5.1f}% / {m_test.stable_recall*100:5.1f}%")
        print(f"{'DYNAMIC Precision / Recall':<36} | {m_dev.dynamic_precision*100:5.1f}% / {m_dev.dynamic_recall*100:5.1f}% | {m_chal.dynamic_precision*100:5.1f}% / {m_chal.dynamic_recall*100:5.1f}% | {m_test.dynamic_precision*100:5.1f}% / {m_test.dynamic_recall*100:5.1f}%")
        print(f"{'Stage 1 (Rule-Layer) Coverage':<36} | {m_dev.rule_resolved_pct:5.1f}% ({m_dev.rule_resolved_count}/{m_dev.total_queries})   | {m_chal.rule_resolved_pct:5.1f}% ({m_chal.rule_resolved_count}/{m_chal.total_queries})   | {m_test.rule_resolved_pct:5.1f}% ({m_test.rule_resolved_count}/{m_test.total_queries})")
        print(f"{'Measured Median Latency (P50)':<36} | {m_dev.latency_p50_ms:6.4f} ms      | {m_chal.latency_p50_ms:6.4f} ms      | {m_test.latency_p50_ms:6.4f} ms")
        print(f"{'Measured P95 Latency':<36} | {m_dev.latency_p95_ms:6.4f} ms      | {m_chal.latency_p95_ms:6.4f} ms      | {m_test.latency_p95_ms:6.4f} ms")
        print("=" * 90)


def main() -> None:
    """CLI entry point for running stability evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Query Stability Classifier on Benchmark")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to benchmark JSON file")
    parser.add_argument("--all", action="store_true", help="Run tri-dataset benchmark comparison")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["FORWARD_TO_CACHE_CANDIDATE", "CONSERVATIVE_BYPASS"],
        default="FORWARD_TO_CACHE_CANDIDATE",
        help="Routing policy for CONDITIONALLY_STABLE queries",
    )
    args = parser.parse_args()

    policy = ConditionalRoutingPolicy(args.policy)
    evaluator = StabilityEvaluator()
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

    if args.all:
        dev_path = data_dir / "query_stability_benchmark.json"
        chal_path = data_dir / "query_stability_benchmark_heldout.json"
        test_path = data_dir / "query_stability_benchmark_final_test.json"

        m_dev = evaluator.evaluate_benchmark(dev_path, conditional_policy=policy, dataset_role="Development")
        m_chal = evaluator.evaluate_benchmark(chal_path, conditional_policy=policy, dataset_role="Challenge / Validation")
        m_test = evaluator.evaluate_benchmark(test_path, conditional_policy=policy, dataset_role="Final Test (Pristine)")

        evaluator.print_tri_comparison(m_dev, m_chal, m_test)
    else:
        metrics = evaluator.evaluate_benchmark(args.dataset, conditional_policy=policy)
        evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
