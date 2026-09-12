"""Network-free unit tests for Phase 2 cache evaluation components.

These tests mock QueryEmbedder.encode() to return deterministic fake vectors.
No SentenceTransformer is loaded; no network access is required or permitted.

Covered:
  - CacheEvaluator.load_and_validate_dataset() — schema validation
    (wrong count, missing keys, empty queries, bad bool, duplicate IDs, non-list JSON)
  - CacheEvaluator.evaluate() — pair protocol mechanics via mocked embedder
  - CacheEvaluator.threshold_sweep() — metric math (TP/FP/FN/TN, IRR_cache, FRR, etc.)
  - Pair leakage check — dataset file reads only (no embedding)

NOT covered here (see test_cache_evaluation_integration.py, requires_model marker):
  - Real embedding similarity scores from the pinned model
  - Actual pair evaluations against the benchmark with real embeddings

pytest marker: (none) — these run unconditionally in the default `pytest` invocation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cache.embedding import EMBEDDING_DIM
from src.evaluation.cache_evaluator import (
    CacheEvaluator,
    EXPECTED_RECORD_COUNT,
    REQUIRED_PAIR_KEYS,
    SweepPoint,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PAIR_DATASET_PATH = DATA_DIR / "query_pair_reuse_benchmark.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_record(pair_id: str = "PAIR-001") -> Dict[str, Any]:
    """Build a minimal valid pair record."""
    return {
        "id": pair_id,
        "query_a": "How do I reverse a string in Python?",
        "query_b": "Python string reversal using slicing syntax",
        "domain": "computer_science",
        "semantic_similarity_level": "HIGH",
        "is_reuse_safe": True,
        "taxonomy_class": "SAFE_EQUIVALENT",
        "rejection_reason": None,
        "rationale": "Identical programming intent and technique.",
    }


def write_temp_dataset(records: List[Dict[str, Any]]) -> Path:
    """Write records to a temp JSON file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump(records, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _unit_vec(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_mock_evaluator(encode_map: dict | None = None) -> CacheEvaluator:
    """Return a CacheEvaluator whose embedder.encode() returns deterministic fake vectors.

    encode_map: optional dict[str, np.ndarray] to control per-query responses.
                If not provided, all queries return _unit_vec(42).
    """
    mock_embedder = MagicMock()
    mock_embedder.dim = EMBEDDING_DIM

    def _encode(query: str) -> np.ndarray:
        if encode_map and query.strip() in encode_map:
            return encode_map[query.strip()]
        return _unit_vec(42)

    mock_embedder.encode.side_effect = _encode
    evaluator = CacheEvaluator(embedder=mock_embedder)
    return evaluator


# ---------------------------------------------------------------------------
# 1. Schema validation tests (no embedder needed)
# ---------------------------------------------------------------------------

class TestDatasetSchemaValidation:
    def test_real_dataset_loads_without_error(self):
        """The real benchmark dataset must load cleanly with no validation errors."""
        records = CacheEvaluator.load_and_validate_dataset(PAIR_DATASET_PATH)
        assert len(records) == EXPECTED_RECORD_COUNT

    def test_wrong_record_count_raises_loudly(self):
        """A dataset with the wrong number of records must raise ValueError."""
        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, 6)]
        path = write_temp_dataset(records)
        try:
            with pytest.raises(ValueError, match="Record count mismatch"):
                CacheEvaluator.load_and_validate_dataset(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_required_key_raises(self):
        """A record missing a required key must raise ValueError naming the missing key."""
        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        del records[0]["is_reuse_safe"]
        path = write_temp_dataset(records)
        try:
            with pytest.raises(ValueError, match="missing keys"):
                CacheEvaluator.load_and_validate_dataset(path)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_query_a_raises(self):
        """A record with an empty query_a must raise ValueError."""
        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        records[0]["query_a"] = "   "
        path = write_temp_dataset(records)
        try:
            with pytest.raises(ValueError, match="Empty query_a"):
                CacheEvaluator.load_and_validate_dataset(path)
        finally:
            path.unlink(missing_ok=True)

    def test_non_bool_is_reuse_safe_raises(self):
        """is_reuse_safe must be a boolean, not a string."""
        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        records[0]["is_reuse_safe"] = "true"
        path = write_temp_dataset(records)
        try:
            with pytest.raises(ValueError, match="is_reuse_safe must be bool"):
                CacheEvaluator.load_and_validate_dataset(path)
        finally:
            path.unlink(missing_ok=True)

    def test_duplicate_id_raises(self):
        """Duplicate IDs must raise ValueError."""
        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        records[1]["id"] = records[0]["id"]
        path = write_temp_dataset(records)
        try:
            with pytest.raises(ValueError, match="Duplicate ID"):
                CacheEvaluator.load_and_validate_dataset(path)
        finally:
            path.unlink(missing_ok=True)

    def test_non_list_json_raises(self):
        """A JSON object (not array) must raise ValueError."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        json.dump({"queries": []}, tmp)
        tmp.close()
        path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="JSON array"):
                CacheEvaluator.load_and_validate_dataset(path)
        finally:
            path.unlink(missing_ok=True)

    def test_all_required_keys_are_present_in_real_dataset(self):
        """Every record in the real dataset must have all required keys."""
        records = CacheEvaluator.load_and_validate_dataset(PAIR_DATASET_PATH)
        for rec in records:
            missing = REQUIRED_PAIR_KEYS - set(rec.keys())
            assert not missing, f"Record {rec['id']} missing keys: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 2. Pair protocol mechanics (mocked embedder, controlled confusion matrix)
# ---------------------------------------------------------------------------

class TestPairProtocolMechanics:
    """Verifies CacheEvaluator.evaluate() builds the correct confusion matrix.

    Uses a synthetic 4-pair dataset with fully controlled cosine scores
    (via mocked embedder returning known unit vectors) to verify TP/FP/FN/TN
    assignment and derived metric math independently of any real model.
    """

    def _build_4pair_dataset(self) -> tuple[Path, dict]:
        """
        Create 4 synthetic pairs to exercise all 4 confusion-matrix cells:
          Pair 1: safe=True,  we'll make score HIGH → TP  (at threshold ~0.80)
          Pair 2: safe=False, we'll make score HIGH → FP
          Pair 3: safe=True,  we'll make score LOW  → FN
          Pair 4: safe=False, we'll make score LOW  → TN
        """
        records = [
            {**make_valid_record("PAIR-001"), "is_reuse_safe": True,
             "query_a": "qTP_a", "query_b": "qTP_b"},
            {**make_valid_record("PAIR-002"), "is_reuse_safe": False,
             "query_a": "qFP_a", "query_b": "qFP_b"},
            {**make_valid_record("PAIR-003"), "is_reuse_safe": True,
             "query_a": "qFN_a", "query_b": "qFN_b"},
            {**make_valid_record("PAIR-004"), "is_reuse_safe": False,
             "query_a": "qTN_a", "query_b": "qTN_b"},
        ]
        # Pad to EXPECTED_RECORD_COUNT with safe=True, low-score pairs
        for i in range(5, EXPECTED_RECORD_COUNT + 1):
            records.append({
                **make_valid_record(f"PAIR-{i:03d}"),
                "is_reuse_safe": True,
                "query_a": f"qPad{i}_a",
                "query_b": f"qPad{i}_b",
            })
        path = write_temp_dataset(records)

        # Build encode_map:
        # HIGH-score pairs: both A and B map to the SAME unit vector → score ≈ 1.0
        high_vec = _unit_vec(100)
        # LOW-score pairs: A maps to one vec, B maps to an orthogonal vec → score ≈ 0
        low_vec_a = _unit_vec(200)
        low_vec_b_raw = np.random.default_rng(201).standard_normal(EMBEDDING_DIM).astype(np.float32)
        low_vec_b_raw -= low_vec_b_raw.dot(low_vec_a) * low_vec_a
        low_vec_b = (low_vec_b_raw / np.linalg.norm(low_vec_b_raw)).astype(np.float32)
        # Pad pairs: A and B map to same low vec → score ≈ 1.0, but is_reuse_safe=True → TP
        # We need pad pairs to be TPs (safe + hit) or TNs; easiest: make them low score
        # so they're FN (safe + miss) — doesn't matter for our 4-cell check of first 4 pairs.
        pad_vec = _unit_vec(300)

        encode_map = {
            "qTP_a": high_vec, "qTP_b": high_vec,
            "qFP_a": high_vec, "qFP_b": high_vec,
            "qFN_a": low_vec_a, "qFN_b": low_vec_b,
            "qTN_a": low_vec_a, "qTN_b": low_vec_b,
        }
        # All pad queries get pad_vec (score≈1.0 vs themselves → HIT; all safe → TP)
        for i in range(5, EXPECTED_RECORD_COUNT + 1):
            encode_map[f"qPad{i}_a"] = pad_vec
            encode_map[f"qPad{i}_b"] = pad_vec

        return path, encode_map

    def test_confusion_matrix_all_four_cells(self):
        """With a controlled 4-pair fixture, verify TP/FP/FN/TN are all counted correctly."""
        path, encode_map = self._build_4pair_dataset()
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        try:
            # threshold=0.5: HIGH pairs (score≈1.0) → HIT; LOW pairs (score≈0.0) → MISS
            metrics = evaluator.evaluate(path, threshold=0.5)
            # First 4 pairs: TP=1, FP=1, FN=1, TN=1
            # Pad pairs (116): all safe + score≈1.0 → all TP
            assert metrics.tp >= 1, f"Expected at least 1 TP; got {metrics.tp}"
            assert metrics.fp >= 1, f"Expected at least 1 FP; got {metrics.fp}"
            assert metrics.fn >= 1, f"Expected at least 1 FN; got {metrics.fn}"
            assert metrics.tn >= 1, f"Expected at least 1 TN; got {metrics.tn}"
            # Total must equal EXPECTED_RECORD_COUNT
            assert metrics.tp + metrics.fp + metrics.fn + metrics.tn == EXPECTED_RECORD_COUNT
        finally:
            path.unlink(missing_ok=True)

    def test_irr_cache_formula(self):
        """IRR_cache = FP / (TP + FP) must be computed correctly."""
        path, encode_map = self._build_4pair_dataset()
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        try:
            metrics = evaluator.evaluate(path, threshold=0.5)
            if metrics.hit_count > 0:
                expected_irr = metrics.fp / metrics.hit_count
                assert abs(metrics.irr_cache - expected_irr) < 1e-9
        finally:
            path.unlink(missing_ok=True)

    def test_frr_formula(self):
        """FRR = FN / (TP + FN) must be computed correctly."""
        path, encode_map = self._build_4pair_dataset()
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        try:
            metrics = evaluator.evaluate(path, threshold=0.5)
            n_safe = metrics.tp + metrics.fn
            if n_safe > 0:
                expected_frr = metrics.fn / n_safe
                assert abs(metrics.frr - expected_frr) < 1e-9
        finally:
            path.unlink(missing_ok=True)

    def test_zero_hits_crr_defaults_to_one(self):
        """With no hits (threshold=1.1 equivalent), CRR must default to 1.0."""
        # Use all-orthogonal pairs: every pair scores near 0 → all MISS at threshold=0.99
        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        for rec in records:
            rec["is_reuse_safe"] = True  # all safe
        path = write_temp_dataset(records)
        # Encode_map: a-vectors and b-vectors are nearly orthogonal
        encode_map = {}
        for i in range(1, EXPECTED_RECORD_COUNT + 1):
            pid = f"PAIR-{i:03d}"
            rec = records[i - 1]
            encode_map[rec["query_a"]] = _unit_vec(i * 2)
            # orthogonal b: subtract projection
            va = encode_map[rec["query_a"]]
            vb_raw = np.random.default_rng(i * 2 + 1).standard_normal(EMBEDDING_DIM).astype(np.float32)
            vb_raw -= vb_raw.dot(va) * va
            encode_map[rec["query_b"]] = (vb_raw / np.linalg.norm(vb_raw)).astype(np.float32)
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        try:
            metrics = evaluator.evaluate(path, threshold=0.99)
            assert metrics.hit_count == 0, f"Expected 0 hits, got {metrics.hit_count}"
            assert metrics.crr == 1.0, f"CRR must default to 1.0 when no hits; got {metrics.crr}"
            assert metrics.irr_cache == 0.0
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Threshold sweep metric math
# ---------------------------------------------------------------------------

class TestThresholdSweepMath:
    """Verifies threshold_sweep() correctly varies the decision boundary."""

    def test_higher_threshold_produces_fewer_hits(self):
        """At high thresholds, fewer pairs should be HITs than at low thresholds."""
        path, encode_map = TestPairProtocolMechanics()._build_4pair_dataset()
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        try:
            points = evaluator.threshold_sweep(path, thresholds=[0.1, 0.5, 0.95])
            hits = [p.hit_count for p in points]
            assert hits[0] >= hits[1] >= hits[2], (
                f"Hit counts should be non-increasing with threshold; got {hits}"
            )
        finally:
            path.unlink(missing_ok=True)

    def test_sweep_returns_one_point_per_threshold(self):
        path, encode_map = TestPairProtocolMechanics()._build_4pair_dataset()
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        thresholds = [0.1, 0.5, 0.9]
        try:
            points = evaluator.threshold_sweep(path, thresholds=thresholds)
            assert len(points) == len(thresholds)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_thresholds_returns_empty_list(self):
        evaluator = _make_mock_evaluator()
        points = evaluator.threshold_sweep(PAIR_DATASET_PATH, thresholds=[])
        assert points == []

    def test_sweep_total_matches_dataset_count(self):
        path, encode_map = TestPairProtocolMechanics()._build_4pair_dataset()
        evaluator = _make_mock_evaluator(encode_map=encode_map)
        try:
            points = evaluator.threshold_sweep(path, thresholds=[0.5])
            assert points[0].total == EXPECTED_RECORD_COUNT
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. Leakage check (reads JSON files, no embedding)
# ---------------------------------------------------------------------------

class TestDatasetLeakage:
    """Verify query_pair_reuse_benchmark.json does not leak into Phase 1 eval datasets.

    These tests only open and parse JSON files — no model loading.
    """

    PHASE1_PATHS = {
        "development": DATA_DIR / "query_stability_benchmark.json",
        "heldout": DATA_DIR / "query_stability_benchmark_heldout.json",
        "final_test": DATA_DIR / "query_stability_benchmark_final_test.json",
        "human_credibility": DATA_DIR / "query_stability_human_credibility.json",
    }

    @staticmethod
    def _extract_queries(path: Path) -> set[str]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("queries", [])
        else:
            return set()
        return {item["query"].strip().lower() for item in items}

    def _pair_queries(self) -> set[str]:
        with open(PAIR_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = set()
        for r in data:
            queries.add(r["query_a"].strip().lower())
            queries.add(r["query_b"].strip().lower())
        return queries

    def test_zero_overlap_with_heldout(self):
        pair_qs = self._pair_queries()
        p1_qs = self._extract_queries(self.PHASE1_PATHS["heldout"])
        overlap = pair_qs & p1_qs
        assert not overlap, (
            f"LEAKAGE: {len(overlap)} query string(s) from pair benchmark overlap "
            f"with Phase 1 heldout set: {sorted(overlap)[:5]}"
        )

    def test_zero_overlap_with_final_test(self):
        pair_qs = self._pair_queries()
        p1_qs = self._extract_queries(self.PHASE1_PATHS["final_test"])
        overlap = pair_qs & p1_qs
        assert not overlap, (
            f"LEAKAGE: {len(overlap)} query string(s) from pair benchmark overlap "
            f"with Phase 1 final-test set: {sorted(overlap)[:5]}"
        )

    def test_zero_overlap_with_human_credibility(self):
        pair_qs = self._pair_queries()
        p1_qs = self._extract_queries(self.PHASE1_PATHS["human_credibility"])
        overlap = pair_qs & p1_qs
        assert not overlap, (
            f"LEAKAGE: {len(overlap)} query string(s) from pair benchmark overlap "
            f"with Phase 1 human-credibility set: {sorted(overlap)[:5]}"
        )

    def test_known_dev_set_overlaps_are_documented(self):
        """Verify the 3 known development-set overlaps exist and haven't grown."""
        pair_qs = self._pair_queries()
        p1_dev_qs = self._extract_queries(self.PHASE1_PATHS["development"])
        overlap = pair_qs & p1_dev_qs

        known_overlapping_queries = {
            "what is the current version of the linux kernel mainline branch?",
            "what is the formula for the volume of a sphere of radius r?",
            "what is the value of euler's number e to four decimal places?",
        }

        assert overlap == known_overlapping_queries, (
            f"Development-set overlap has changed from the documented 3 queries.\n"
            f"Expected: {sorted(known_overlapping_queries)}\n"
            f"Found   : {sorted(overlap)}\n"
            f"If the dataset was intentionally updated, update this test AND "
            f"docs/phase2_walkthrough.md with the new overlap count and rationale."
        )

    def test_pair_dataset_record_count_matches_expectation(self):
        """The pair dataset must have exactly EXPECTED_RECORD_COUNT records."""
        with open(PAIR_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == EXPECTED_RECORD_COUNT, (
            f"Pair dataset count changed: expected {EXPECTED_RECORD_COUNT}, "
            f"found {len(data)}. If intentional, update EXPECTED_RECORD_COUNT in "
            f"cache_evaluator.py and this test."
        )
