"""Semantic cache benchmark evaluator for Phase 2.

=============================================================================
PAIR EVALUATION PROTOCOL — STATED EXPLICITLY BEFORE ANY CODE BELOW THIS LINE
=============================================================================

For each pair (query_a, query_b, is_reuse_safe) in the benchmark:

  1. Index query_a's embedding into the vector store (as if it were already
     cached — i.e., a prior user asked query_a and the result was stored).
  2. Search using query_b's embedding (query_b is the incoming request).
  3. If top-1 similarity >= threshold: predicted = HIT (reuse claimed).
  4. Compare predicted HIT/MISS against ground-truth is_reuse_safe.

Confusion matrix:
  TP = HIT  and is_reuse_safe=True   (correct reuse)
  FP = HIT  and is_reuse_safe=False  (cache hazard — reused something unsafe)
  FN = MISS and is_reuse_safe=True   (missed a safe reuse opportunity)
  TN = MISS and is_reuse_safe=False  (correctly rejected)

Each pair is evaluated independently. The vector store is reset between pairs
(one fresh index add per pair, one search per pair).

=============================================================================
METRIC DEFINITIONS — VERBATIM FROM docs/evaluation_metrics.md
=============================================================================

  ARR (Actual Reuse Rate):
    ARR = (TP + FP) / N
    Proportion of total traffic where the system decides to reuse.

  CRR (Correct Reuse Rate / Precision):
    CRR = TP / (TP + FP)
    Proportion of reused answers that were genuinely safe.
    Edge case: if TP + FP == 0, CRR := 1.0 (no unsafe answers served).

  IRR_cache (Cache Hazard Rate):
    IRR_cache = FP / (TP + FP) = 1 - CRR
    Proportion of reused answers that were incorrect (the critical safety metric).
    Edge case: if TP + FP == 0, IRR_cache := 0.0.

  IRR_traffic (Traffic-level Hazard Rate):
    IRR_traffic = FP / N
    Hazardous reuses as a fraction of total traffic.

  FRR (False Rejection Rate):
    FRR = FN / (TP + FN)
    Proportion of safely reusable opportunities unnecessarily regenerated.
    Edge case: if TP + FN == 0, FRR := 0.0.

No token/dollar figures are produced. No LLM is called. The only cost proxies
reported are hit count and measured embedding + search latency.

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.cache.embedding import QueryEmbedder
from src.cache.semantic_cache import CacheDecision, SemanticCache
from src.cache.vector_store import FlatVectorStore


# ---------------------------------------------------------------------------
# Required schema for query_pair_reuse_benchmark.json records
# ---------------------------------------------------------------------------
REQUIRED_PAIR_KEYS: frozenset[str] = frozenset({
    "id",
    "query_a",
    "query_b",
    "domain",
    "semantic_similarity_level",
    "is_reuse_safe",
    "taxonomy_class",
    "rejection_reason",
    "rationale",
})

EXPECTED_RECORD_COUNT: int = 120  # verified empirically in Step 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """Metrics at a single threshold value."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    total: int

    # Derived metrics
    arr: float       # Actual Reuse Rate = (TP+FP) / N
    crr: float       # Correct Reuse Rate = TP / (TP+FP), 1.0 if no hits
    irr_cache: float # Cache Hazard Rate = FP / (TP+FP), 0.0 if no hits
    irr_traffic: float  # Traffic-level hazard = FP / N
    frr: float       # False Rejection Rate = FN / (TP+FN), 0.0 if no safe pairs

    hit_count: int   # TP + FP (honest cost proxy: queries bypassing generation)
    miss_count: int  # FN + TN

    mean_embed_ms: float
    mean_search_ms: float
    mean_total_ms: float


@dataclass
class CacheMetrics:
    """Full evaluation metrics at a given threshold."""

    dataset_path: str
    threshold: float
    total_pairs: int
    tp: int
    fp: int
    fn: int
    tn: int

    arr: float
    crr: float
    irr_cache: float
    irr_traffic: float
    frr: float

    hit_count: int
    miss_count: int

    latency_mean_embed_ms: float
    latency_mean_search_ms: float
    latency_mean_total_ms: float
    latency_p50_total_ms: float
    latency_p95_total_ms: float

    detailed_results: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class CacheEvaluator:
    """Evaluates the semantic cache baseline against the pair-reuse benchmark.

    The pair evaluation protocol (index query_a, search query_b, compare to
    is_reuse_safe ground truth) is implemented exactly as documented in the
    module-level docstring. No deviations.

    Args:
        embedder: Optional custom QueryEmbedder. Defaults to all-MiniLM-L6-v2.
    """

    def __init__(self, embedder: Optional[QueryEmbedder] = None) -> None:
        self.embedder = embedder or QueryEmbedder()

    @staticmethod
    def load_and_validate_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
        """Load query_pair_reuse_benchmark.json with strict schema validation.

        Args:
            dataset_path: Path to the JSON file.

        Returns:
            List of validated record dicts.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If the record count, format, or schema is unexpected.
                This is intentionally loud — unexpected structure must be caught
                before any evaluation is run.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {dataset_path}: {e}") from e

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array, got {type(data).__name__} in {dataset_path}. "
                f"The pair benchmark must be a bare JSON array of record objects."
            )

        actual_count = len(data)
        if actual_count != EXPECTED_RECORD_COUNT:
            raise ValueError(
                f"Record count mismatch in {dataset_path}: "
                f"expected {EXPECTED_RECORD_COUNT}, found {actual_count}. "
                f"If the dataset was intentionally extended, update EXPECTED_RECORD_COUNT "
                f"in cache_evaluator.py after explicit review."
            )

        seen_ids: set[str] = set()
        for idx, record in enumerate(data, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {idx} is not a dict: {type(record).__name__}")

            missing = REQUIRED_PAIR_KEYS - set(record.keys())
            if missing:
                raise ValueError(
                    f"Record {record.get('id', f'index-{idx}')} is missing keys: {sorted(missing)}"
                )

            record_id = record["id"]
            if record_id in seen_ids:
                raise ValueError(f"Duplicate ID: {record_id}")
            seen_ids.add(record_id)

            if not record.get("query_a") or not str(record["query_a"]).strip():
                raise ValueError(f"Empty query_a in record {record_id}")
            if not record.get("query_b") or not str(record["query_b"]).strip():
                raise ValueError(f"Empty query_b in record {record_id}")
            if not isinstance(record["is_reuse_safe"], bool):
                raise ValueError(f"is_reuse_safe must be bool in {record_id}")

        return data

    def evaluate(
        self,
        dataset_path: Path,
        threshold: float,
    ) -> CacheMetrics:
        """Run the pair evaluation protocol at a given similarity threshold.

        For each pair:
          1. Reset the store (fresh context per pair).
          2. Index query_a.
          3. Search query_b.
          4. Decision = HIT if score >= threshold, else MISS.
          5. Compare to is_reuse_safe.

        Args:
            dataset_path: Path to query_pair_reuse_benchmark.json.
            threshold: Cosine similarity threshold in [0, 1].

        Returns:
            CacheMetrics with full confusion matrix, derived metrics, and latencies.
        """
        records = self.load_and_validate_dataset(dataset_path)

        # Build a shared embedder + store; threshold passed per-lookup.
        store = FlatVectorStore(dim=self.embedder.dim)
        cache = SemanticCache(
            threshold=threshold,
            embedder=self.embedder,
            vector_store=store,
        )

        tp = tn = fp = fn = 0
        embed_latencies: List[float] = []
        search_latencies: List[float] = []
        total_latencies: List[float] = []
        detailed_results: List[Dict[str, Any]] = []

        for record in records:
            query_a: str = record["query_a"].strip()
            query_b: str = record["query_b"].strip()
            is_safe: bool = record["is_reuse_safe"]

            # --- Step 1: fresh store (one entry per pair) ---
            cache.vector_store.reset()

            # --- Step 2: index query_a (simulates a cached prior response) ---
            cache.index(query_a, metadata={"source_id": record["id"], "query": query_a})

            # --- Step 3: lookup query_b (the incoming request) ---
            result = cache.lookup(query_b)

            embed_latencies.append(result.embed_latency_ms)
            search_latencies.append(result.search_latency_ms)
            total_latencies.append(result.total_latency_ms)

            # --- Step 4: update confusion matrix ---
            predicted_hit = result.decision == CacheDecision.HIT

            if predicted_hit and is_safe:
                tp += 1
            elif predicted_hit and not is_safe:
                fp += 1
            elif not predicted_hit and is_safe:
                fn += 1
            else:  # not predicted_hit and not is_safe
                tn += 1

            detailed_results.append({
                "id": record["id"],
                "query_a": query_a,
                "query_b": query_b,
                "is_reuse_safe": is_safe,
                "domain": record["domain"],
                "similarity_level": record["semantic_similarity_level"],
                "taxonomy_class": record["taxonomy_class"],
                "predicted": "HIT" if predicted_hit else "MISS",
                "score": round(result.similarity_score, 6),
                "threshold": threshold,
                "outcome": (
                    "TP" if (predicted_hit and is_safe) else
                    "FP" if (predicted_hit and not is_safe) else
                    "FN" if (not predicted_hit and is_safe) else
                    "TN"
                ),
                "embed_ms": result.embed_latency_ms,
                "search_ms": result.search_latency_ms,
                "total_ms": result.total_latency_ms,
            })

        n = len(records)
        hit_count = tp + fp
        miss_count = fn + tn
        n_safe = tp + fn

        arr = hit_count / n if n > 0 else 0.0
        crr = (tp / hit_count) if hit_count > 0 else 1.0
        irr_cache = (fp / hit_count) if hit_count > 0 else 0.0
        irr_traffic = fp / n if n > 0 else 0.0
        frr = (fn / n_safe) if n_safe > 0 else 0.0

        mean_embed = statistics.mean(embed_latencies) if embed_latencies else 0.0
        mean_search = statistics.mean(search_latencies) if search_latencies else 0.0
        mean_total = statistics.mean(total_latencies) if total_latencies else 0.0
        p50_total = statistics.median(total_latencies) if total_latencies else 0.0
        p95_total = (
            statistics.quantiles(total_latencies, n=20)[18]
            if len(total_latencies) >= 20
            else max(total_latencies, default=0.0)
        )

        return CacheMetrics(
            dataset_path=str(dataset_path),
            threshold=threshold,
            total_pairs=n,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            arr=arr,
            crr=crr,
            irr_cache=irr_cache,
            irr_traffic=irr_traffic,
            frr=frr,
            hit_count=hit_count,
            miss_count=miss_count,
            latency_mean_embed_ms=round(mean_embed, 4),
            latency_mean_search_ms=round(mean_search, 4),
            latency_mean_total_ms=round(mean_total, 4),
            latency_p50_total_ms=round(p50_total, 4),
            latency_p95_total_ms=round(p95_total, 4),
            detailed_results=detailed_results,
        )

    def threshold_sweep(
        self,
        dataset_path: Path,
        thresholds: Sequence[float],
    ) -> List[SweepPoint]:
        """Evaluate across a sequence of thresholds efficiently.

        Pre-computes all embeddings ONCE, then sweeps thresholds by comparing
        pre-computed similarity scores — avoiding redundant re-embedding.
        This is mathematically equivalent to calling evaluate() per threshold:
        the similarity score between (query_a, query_b) does not change with
        the threshold; only the HIT/MISS decision boundary changes.

        Latency reported: actual per-pair embed + search time from a single
        evaluate() call at the first threshold (used for all sweep points).

        Args:
            dataset_path: Path to query_pair_reuse_benchmark.json.
            thresholds: Sequence of float values in [0, 1].

        Returns:
            List of SweepPoint, one per threshold, in input order.
        """
        if not thresholds:
            return []

        records = self.load_and_validate_dataset(dataset_path)
        n = len(records)
        n_safe = sum(1 for r in records if r["is_reuse_safe"])

        # Pre-compute all similarity scores once (single pass through the data)
        scores: List[float] = []
        ground_truths: List[bool] = []
        embed_latencies: List[float] = []
        search_latencies: List[float] = []
        total_latencies: List[float] = []

        store = FlatVectorStore(dim=self.embedder.dim)

        for record in records:
            query_a: str = record["query_a"].strip()
            query_b: str = record["query_b"].strip()
            is_safe: bool = record["is_reuse_safe"]

            store.reset()

            # Time: embed query_a
            t0 = time.perf_counter()
            vec_a = self.embedder.encode(query_a)
            t1 = time.perf_counter()

            # Index query_a
            store.add(vec_a, metadata={"id": record["id"]})

            # Time: embed query_b + search
            t2 = time.perf_counter()
            vec_b = self.embedder.encode(query_b)
            t3 = time.perf_counter()
            result = store.search(vec_b, k=1)
            t4 = time.perf_counter()

            embed_ms = ((t1 - t0) + (t3 - t2)) * 1000.0
            search_ms = (t4 - t3) * 1000.0
            total_ms = embed_ms + search_ms

            scores.append(result.score if result.found else float("-inf"))
            ground_truths.append(is_safe)
            embed_latencies.append(embed_ms)
            search_latencies.append(search_ms)
            total_latencies.append(total_ms)

        # Pre-compute latency stats once (same for all thresholds)
        mean_embed = statistics.mean(embed_latencies) if embed_latencies else 0.0
        mean_search = statistics.mean(search_latencies) if search_latencies else 0.0
        mean_total = statistics.mean(total_latencies) if total_latencies else 0.0

        # Now sweep thresholds using pre-computed scores
        points: List[SweepPoint] = []
        for thr in thresholds:
            tp = tn = fp = fn = 0
            for score, is_safe in zip(scores, ground_truths):
                predicted_hit = score >= thr
                if predicted_hit and is_safe:
                    tp += 1
                elif predicted_hit and not is_safe:
                    fp += 1
                elif not predicted_hit and is_safe:
                    fn += 1
                else:
                    tn += 1

            hit_count = tp + fp
            arr = hit_count / n if n > 0 else 0.0
            crr = (tp / hit_count) if hit_count > 0 else 1.0
            irr_cache = (fp / hit_count) if hit_count > 0 else 0.0
            irr_traffic = fp / n if n > 0 else 0.0
            frr = (fn / n_safe) if n_safe > 0 else 0.0

            points.append(SweepPoint(
                threshold=thr,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                total=n,
                arr=arr,
                crr=crr,
                irr_cache=irr_cache,
                irr_traffic=irr_traffic,
                frr=frr,
                hit_count=hit_count,
                miss_count=fn + tn,
                mean_embed_ms=round(mean_embed, 4),
                mean_search_ms=round(mean_search, 4),
                mean_total_ms=round(mean_total, 4),
            ))
        return points


    def print_metrics(self, m: CacheMetrics) -> None:
        """Print a structured evaluation report to stdout."""
        print("=" * 72)
        print(f"SEMANTIC CACHE EVALUATION REPORT")
        print(f"Dataset : {m.dataset_path}")
        print(f"Threshold: {m.threshold:.4f}")
        print("=" * 72)
        print(f"Total Pairs : {m.total_pairs}")
        print(f"Hit Count   : {m.hit_count}  (TP+FP — queries that would bypass generation)")
        print(f"Miss Count  : {m.miss_count}  (FN+TN — fresh generation required)")
        print("-" * 72)
        print("CONFUSION MATRIX (HIT=Positive, SAFE=Positive-class)")
        print(f"  TP (Correct Reuse)      : {m.tp:3d}")
        print(f"  FP (Cache Hazard)       : {m.fp:3d}")
        print(f"  FN (Missed Reuse)       : {m.fn:3d}")
        print(f"  TN (Correct Rejection)  : {m.tn:3d}")
        print("-" * 72)
        print("METRICS (verbatim from docs/evaluation_metrics.md)")
        print(f"  ARR (Actual Reuse Rate)        : {m.arr * 100:.2f}%")
        print(f"  CRR (Correct Reuse Precision)  : {m.crr * 100:.2f}%")
        print(f"  IRR_cache (Cache Hazard Rate)  : {m.irr_cache * 100:.2f}%")
        print(f"  IRR_traffic (Traffic Hazard)   : {m.irr_traffic * 100:.2f}%")
        print(f"  FRR (False Rejection Rate)     : {m.frr * 100:.2f}%")
        print("-" * 72)
        print("MEASURED LATENCY (embed + search; no LLM call)")
        print(f"  Mean Embed   : {m.latency_mean_embed_ms:.4f} ms")
        print(f"  Mean Search  : {m.latency_mean_search_ms:.4f} ms")
        print(f"  Mean Total   : {m.latency_mean_total_ms:.4f} ms")
        print(f"  P50 Total    : {m.latency_p50_total_ms:.4f} ms")
        print(f"  P95 Total    : {m.latency_p95_total_ms:.4f} ms")
        print("=" * 72)


def main() -> None:
    """CLI entry point for single-threshold cache evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate semantic cache baseline on pair-reuse benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to query_pair_reuse_benchmark.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold (default: 0.85)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "query_pair_reuse_benchmark.json"
    )

    evaluator = CacheEvaluator()
    metrics = evaluator.evaluate(dataset_path, threshold=args.threshold)
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    main()
