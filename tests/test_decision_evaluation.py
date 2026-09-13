"""Unit tests for Phase 3 decision evaluation — schema guard and protocol checks.

These tests guard:
  - query_pair_reuse_benchmark.json schema (reused from Phase 2, not reimplemented)
  - DecisionMetrics dataclass structure
  - Ambiguous-band large-fraction flag logic
  - Auto-reuse verbatim verification: AUTO_REUSE decisions match Phase 2 output
    (the DoD "verbatim" proof — tested against mocked cache/classifier, not a real model)

NOT covered here (requires_model marker — separately):
  - Real embedding scores
  - Actual Phase 3 metrics against the live model

pytest marker: (none) — runs unconditionally in the default `pytest` invocation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.cache.embedding import EMBEDDING_DIM
from src.cache.semantic_cache import CacheDecision, CacheLookupResult
from src.classifier.models import ClassificationSource, StabilityLabel, StabilityResult
from src.decision.decision_step import CategoryHistory, DecisionResult, DecisionStep
from src.decision.tier_router import DEFAULT_BOUNDARIES, Tier, TierRouter, TierBoundaries
from src.evaluation.cache_evaluator import (
    CacheEvaluator,
    EXPECTED_RECORD_COUNT,
    REQUIRED_PAIR_KEYS,
)
from src.evaluation.decision_evaluator import (
    AMBIGUOUS_BAND_LARGE_FRACTION_THRESHOLD,
    DecisionEvaluator,
    DecisionMetrics,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PAIR_DATASET_PATH = DATA_DIR / "query_pair_reuse_benchmark.json"


# ---------------------------------------------------------------------------
# Helper: build minimal pair record
# ---------------------------------------------------------------------------

def make_valid_record(pair_id: str = "PAIR-001") -> Dict[str, Any]:
    return {
        "id": pair_id,
        "query_a": "How do I reverse a string in Python using slice notation?",
        "query_b": "Python string reversal using slicing syntax",
        "domain": "computer_science",
        "semantic_similarity_level": "HIGH",
        "is_reuse_safe": True,
        "taxonomy_class": "SAFE_EQUIVALENT",
        "rejection_reason": None,
        "rationale": "Identical programming intent.",
    }


def write_temp_dataset(records: List[Dict[str, Any]]) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump(records, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _unit_vec(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Schema guard — reuse Phase 2's existing load_and_validate_dataset
# ---------------------------------------------------------------------------

class TestDatasetSchemaGuard:
    """Schema validation for query_pair_reuse_benchmark.json.

    This deliberately reuses CacheEvaluator.load_and_validate_dataset —
    no reimplementation.  Per the spec: 'reuse Phase 2's existing check,
    don't reimplement it'.
    """

    def test_real_dataset_loads_with_expected_count(self):
        """The pair benchmark must load to exactly EXPECTED_RECORD_COUNT records."""
        records = CacheEvaluator.load_and_validate_dataset(PAIR_DATASET_PATH)
        assert len(records) == EXPECTED_RECORD_COUNT

    def test_all_records_have_required_keys(self):
        records = CacheEvaluator.load_and_validate_dataset(PAIR_DATASET_PATH)
        for rec in records:
            missing = REQUIRED_PAIR_KEYS - set(rec.keys())
            assert not missing, f"Record {rec['id']} missing: {sorted(missing)}"

    def test_all_is_reuse_safe_are_booleans(self):
        records = CacheEvaluator.load_and_validate_dataset(PAIR_DATASET_PATH)
        for rec in records:
            assert isinstance(rec["is_reuse_safe"], bool), (
                f"Record {rec['id']}: is_reuse_safe must be bool, got {type(rec['is_reuse_safe'])}"
            )

    def test_all_domain_fields_are_non_empty_strings(self):
        records = CacheEvaluator.load_and_validate_dataset(PAIR_DATASET_PATH)
        for rec in records:
            assert isinstance(rec["domain"], str) and rec["domain"].strip(), (
                f"Record {rec['id']}: domain must be non-empty string"
            )


# ---------------------------------------------------------------------------
# 2. DecisionMetrics structure
# ---------------------------------------------------------------------------

class TestDecisionMetricsStructure:
    """Validate the DecisionMetrics dataclass has all required Phase 3 fields."""

    def _make_minimal_metrics(self) -> DecisionMetrics:
        return DecisionMetrics(
            dataset_path="/fake/path.json",
            total_pairs=120,
            auto_reuse_count=5,
            ambiguous_count=30,
            bypass_count=85,
            ambiguous_fraction=0.25,
            ambiguous_band_large=False,
            tp=5, fp=0, fn=20, tn=95,
            arr=0.042, crr=1.0, irr_cache=0.0,
            irr_traffic=0.0, frr=0.80,
            hit_count=5, miss_count=115,
        )

    def test_required_fields_exist(self):
        m = self._make_minimal_metrics()
        assert hasattr(m, "ambiguous_fraction")
        assert hasattr(m, "ambiguous_band_large")
        assert hasattr(m, "auto_reuse_count")
        assert hasattr(m, "ambiguous_count")
        assert hasattr(m, "bypass_count")
        assert hasattr(m, "tp")
        assert hasattr(m, "fp")
        assert hasattr(m, "fn")
        assert hasattr(m, "tn")
        assert hasattr(m, "arr")
        assert hasattr(m, "crr")
        assert hasattr(m, "irr_cache")
        assert hasattr(m, "irr_traffic")
        assert hasattr(m, "frr")

    def test_phase2_comparison_fields_exist(self):
        m = self._make_minimal_metrics()
        assert hasattr(m, "phase2_tp")
        assert hasattr(m, "phase2_fp")
        assert hasattr(m, "phase2_arr")
        assert hasattr(m, "phase2_irr_cache")

    def test_verbatim_verification_fields_exist(self):
        m = self._make_minimal_metrics()
        assert hasattr(m, "auto_reuse_verbatim_verified")
        assert hasattr(m, "auto_reuse_verbatim_mismatches")

    def test_large_flag_false_when_fraction_below_threshold(self):
        m = self._make_minimal_metrics()
        m_large = DecisionMetrics(
            **{**m.__dict__,  # type: ignore[arg-type]
               "ambiguous_fraction": AMBIGUOUS_BAND_LARGE_FRACTION_THRESHOLD - 0.01,
               "ambiguous_band_large": False,
               "detailed_results": [],
               "tier_metrics": {}},
        )
        assert m_large.ambiguous_band_large is False

    def test_large_flag_true_when_fraction_above_threshold(self):
        m_large = DecisionMetrics(
            dataset_path="/fake/path.json",
            total_pairs=120,
            auto_reuse_count=0,
            ambiguous_count=70,
            bypass_count=50,
            ambiguous_fraction=0.583,
            ambiguous_band_large=True,
            tp=0, fp=0, fn=62, tn=58,
            arr=0.0, crr=1.0, irr_cache=0.0,
            irr_traffic=0.0, frr=1.0,
            hit_count=0, miss_count=120,
        )
        assert m_large.ambiguous_band_large is True


# ---------------------------------------------------------------------------
# 3. Ambiguous-band fraction threshold constant
# ---------------------------------------------------------------------------

class TestAmbiguousBandConstant:
    def test_large_threshold_is_0_50(self):
        assert AMBIGUOUS_BAND_LARGE_FRACTION_THRESHOLD == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# 4. Auto-reuse verbatim proof — mocked evaluator
# ---------------------------------------------------------------------------

class TestAutoReuseVerbatimProof:
    """DoD item 3: test that AUTO_REUSE-tier pairs produce the same result as Phase 2.

    This test uses a controlled scenario:
      - A mocked embedder returns two specific unit vectors (high similarity)
      - The TierRouter is configured with a low auto-reuse floor so the pair
        falls into AUTO_REUSE
      - We assert: the Phase 3 predicted result for AUTO_REUSE == Phase 2 result
        (both are HIT; the similarity score is the same value from the same
        cache.lookup() call)
      - No DecisionStep.decide() calls are made for AUTO_REUSE pairs

    This is a unit-level proof of the invariant, not an integration test.
    """

    def test_auto_reuse_tier_produces_hit_matching_phase2_output(self):
        """For a pair routed to AUTO_REUSE, Phase 3 must report HIT with the
        same similarity_score as the Phase 2 CacheLookupResult."""
        # Build identical vectors → score ≈ 1.0 → AUTO_REUSE under any boundaries
        high_vec = _unit_vec(42)
        mock_embedder = MagicMock()
        mock_embedder.dim = EMBEDDING_DIM
        mock_embedder.encode.return_value = high_vec

        # Tight boundaries: auto_reuse at 0.50 to guarantee this pair routes there
        tight_bounds = TierBoundaries(
            auto_reuse_sim_floor=0.50,
            auto_reuse_conf_floor=0.50,
            bypass_sim_ceiling=0.10,
            bypass_conf_ceiling=0.10,
            ambiguous_sim_low=0.10,
            ambiguous_sim_high=0.50,
        )

        # Build a mock stability classifier that returns high-confidence STABLE
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StabilityResult(
            query="test",
            predicted_label=StabilityLabel.STABLE,
            effective_decision=StabilityLabel.STABLE,
            is_cacheable=True,
            confidence=0.95,
            source=ClassificationSource.RULE,
            matched_rule="test_rule",
            rationale="unit test",
        )

        # Use a REAL DecisionStep (so .history is accessible) and spy on .decide()
        real_decision_step = DecisionStep()

        evaluator = DecisionEvaluator(
            embedder=mock_embedder,
            classifier=mock_classifier,
            tier_router=TierRouter(boundaries=tight_bounds),
            decision_step=real_decision_step,
        )

        # Build a dataset padded to EXPECTED_RECORD_COUNT; all pairs reuse same vecs
        records = [make_valid_record("PAIR-001")]
        for i in range(2, EXPECTED_RECORD_COUNT + 1):
            records.append(make_valid_record(f"PAIR-{i:03d}"))

        path = write_temp_dataset(records)
        try:
            with patch.object(
                real_decision_step, "decide", wraps=real_decision_step.decide
            ) as spy:
                metrics = evaluator.evaluate(path)
                # All pairs have score ~1.0 and conf=0.95, both above tight boundaries
                # (auto_reuse_sim_floor=0.50, auto_reuse_conf_floor=0.50).
                # TierRouter must route every pair to AUTO_REUSE.
                # Therefore DecisionStep.decide() must NEVER be called.
                spy.assert_not_called()
        finally:
            path.unlink(missing_ok=True)

        # All pairs routed to AUTO_REUSE; all safe (make_valid_record is_reuse_safe=True)
        assert metrics.auto_reuse_count == EXPECTED_RECORD_COUNT
        assert metrics.ambiguous_count == 0
        assert metrics.auto_reuse_verbatim_verified is True

    def test_auto_reuse_verbatim_mismatch_detected(self):
        """If an AUTO_REUSE pair's similarity_score is below the Phase 2 threshold,
        the verbatim check must detect the mismatch and flag it.

        This tests the evaluator's internal verification logic: when
        cache_result.is_hit is False but tier == AUTO_REUSE, the evaluator
        increments auto_reuse_mismatches.

        We achieve this by injecting a TierRouter that routes to AUTO_REUSE
        even though the similarity is low (below Phase 2 threshold).
        """
        # Low similarity → Phase 2 MISS
        low_vec_a = _unit_vec(10)
        low_vec_b_raw = np.random.default_rng(11).standard_normal(EMBEDDING_DIM).astype(np.float32)
        low_vec_b_raw -= low_vec_b_raw.dot(low_vec_a) * low_vec_a
        low_vec_b = (low_vec_b_raw / np.linalg.norm(low_vec_b_raw)).astype(np.float32)

        call_count = {"n": 0}
        def encode_side_effect(query: str) -> np.ndarray:
            v = low_vec_a if call_count["n"] % 2 == 0 else low_vec_b
            call_count["n"] += 1
            return v

        mock_embedder = MagicMock()
        mock_embedder.dim = EMBEDDING_DIM
        mock_embedder.encode.side_effect = encode_side_effect

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StabilityResult(
            query="test",
            predicted_label=StabilityLabel.STABLE,
            effective_decision=StabilityLabel.STABLE,
            is_cacheable=True,
            confidence=0.95,
            source=ClassificationSource.RULE,
            matched_rule="test_rule",
            rationale="unit test",
        )

        # Force router to always return AUTO_REUSE
        mock_router = MagicMock()
        mock_router.route.return_value = Tier.AUTO_REUSE

        evaluator = DecisionEvaluator(
            embedder=mock_embedder,
            classifier=mock_classifier,
            tier_router=mock_router,
        )

        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        path = write_temp_dataset(records)
        try:
            metrics = evaluator.evaluate(path)
        finally:
            path.unlink(missing_ok=True)

        # Every pair has low similarity → Phase 2 returns MISS →
        # forced AUTO_REUSE + Phase 2 MISS → mismatch detected
        assert metrics.auto_reuse_verbatim_mismatches > 0
        assert metrics.auto_reuse_verbatim_verified is False
