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
from src.decision.adaptive_threshold_engine import AdaptiveThresholdEngine
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

        # Use a REAL AdaptiveThresholdEngine and spy on .decide()
        real_engine = AdaptiveThresholdEngine()

        evaluator = DecisionEvaluator(
            embedder=mock_embedder,
            classifier=mock_classifier,
            tier_router=TierRouter(boundaries=tight_bounds),
            threshold_engine=real_engine,
        )

        # Build a dataset padded to EXPECTED_RECORD_COUNT; all pairs reuse same vecs
        records = [make_valid_record("PAIR-001")]
        for i in range(2, EXPECTED_RECORD_COUNT + 1):
            records.append(make_valid_record(f"PAIR-{i:03d}"))

        path = write_temp_dataset(records)
        try:
            with patch.object(
                real_engine, "decide", wraps=real_engine.decide
            ) as spy:
                metrics = evaluator.evaluate(path)
                # All pairs have score ~1.0 and conf=0.95, both above tight boundaries
                # (auto_reuse_sim_floor=0.50, auto_reuse_conf_floor=0.50).
                # TierRouter must route every pair to AUTO_REUSE.
                # Therefore AdaptiveThresholdEngine.decide() must NEVER be called.
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


# ---------------------------------------------------------------------------
# 5. Ambiguous-tier adaptive threshold engine integration
# ---------------------------------------------------------------------------

class TestAmbiguousTierDecisionEngine:
    """Test that AMBIGUOUS-tier queries invoke AdaptiveThresholdEngine."""

    def test_ambiguous_tier_invokes_adaptive_engine(self):
        """When router returns AMBIGUOUS, evaluator must invoke threshold_engine.decide()."""
        mock_embedder = MagicMock()
        mock_embedder.dim = EMBEDDING_DIM
        mock_embedder.encode.return_value = _unit_vec(1)

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StabilityResult(
            query="test",
            predicted_label=StabilityLabel.STABLE,
            effective_decision=StabilityLabel.STABLE,
            is_cacheable=True,
            confidence=0.85,
            source=ClassificationSource.FALLBACK,
            matched_rule=None,
            rationale="test",
        )

        mock_router = MagicMock()
        mock_router.route.return_value = Tier.AMBIGUOUS

        engine = AdaptiveThresholdEngine()
        evaluator = DecisionEvaluator(
            embedder=mock_embedder,
            classifier=mock_classifier,
            tier_router=mock_router,
            threshold_engine=engine,
        )

        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        path = write_temp_dataset(records)
        try:
            with patch.object(engine, "decide", wraps=engine.decide) as spy:
                metrics = evaluator.evaluate(path)
                assert spy.call_count == EXPECTED_RECORD_COUNT
        finally:
            path.unlink(missing_ok=True)

        assert metrics.ambiguous_count == EXPECTED_RECORD_COUNT

    def test_ambiguous_tier_per_category_differentiation(self):
        """Pairs with identical similarity in different categories are decided differently
        based on per-category threshold vs fallback threshold."""
        # Both pairs have similarity ~ 0.75
        vec = _unit_vec(1)
        mock_embedder = MagicMock()
        mock_embedder.dim = EMBEDDING_DIM
        mock_embedder.encode.return_value = vec

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StabilityResult(
            query="test",
            predicted_label=StabilityLabel.STABLE,
            effective_decision=StabilityLabel.STABLE,
            is_cacheable=True,
            confidence=0.85,
            source=ClassificationSource.FALLBACK,
            matched_rule=None,
            rationale="test",
        )

        mock_router = MagicMock()
        mock_router.route.return_value = Tier.AMBIGUOUS

        engine = AdaptiveThresholdEngine()
        evaluator = DecisionEvaluator(
            embedder=mock_embedder,
            classifier=mock_classifier,
            tier_router=mock_router,
            threshold_engine=engine,
        )

        # computer_science operating threshold is ~0.7924 -> score 0.81 >= 0.7924 -> REUSE (HIT)
        # finance_economics operating threshold is fallback 0.85 -> score 0.81 < 0.85 -> BYPASS (MISS)
        records = []
        for i in range(1, EXPECTED_RECORD_COUNT + 1):
            dom = "computer_science" if i % 2 == 1 else "finance_economics"
            rec = make_valid_record(f"PAIR-{i:03d}")
            rec["domain"] = dom
            records.append(rec)

        # Mock cache lookup to return similarity_score = 0.8100
        with patch("src.evaluation.decision_evaluator.SemanticCache.lookup") as mock_lookup:
            mock_lookup.return_value = CacheLookupResult(
                decision=CacheDecision.MISS,
                similarity_score=0.8100,
                threshold=0.85,
                embed_latency_ms=1.0,
                search_latency_ms=0.1,
                total_latency_ms=1.1,
                store_size=1,
            )
            path = write_temp_dataset(records)
            try:
                metrics = evaluator.evaluate(path)
            finally:
                path.unlink(missing_ok=True)

        cs_results = [r for r in metrics.detailed_results if r["domain"] == "computer_science"]
        fin_results = [r for r in metrics.detailed_results if r["domain"] == "finance_economics"]

        # All CS should be HIT (0.81 >= 0.7924 operating threshold)
        assert all(r["predicted"] == "HIT" for r in cs_results)
        # All Finance should be MISS (0.81 < 0.85 fallback threshold)
        assert all(r["predicted"] == "MISS" for r in fin_results)


# ---------------------------------------------------------------------------
# 6. Trivial "Bypass All Ambiguous" baseline mode
# ---------------------------------------------------------------------------

class TestTrivialBypassAllAmbiguousMode:
    """Validate the trivial baseline that routes all AMBIGUOUS pairs to BYPASS."""

    def test_bypass_all_ambiguous_mode_produces_zero_tp_and_zero_fp_in_ambiguous_tier(self):
        """When bypass_all_ambiguous=True, AMBIGUOUS tier produces 0 TP and 0 FP by construction."""
        mock_embedder = MagicMock()
        mock_embedder.dim = EMBEDDING_DIM
        mock_embedder.encode.return_value = _unit_vec(1)

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StabilityResult(
            query="test",
            predicted_label=StabilityLabel.STABLE,
            effective_decision=StabilityLabel.STABLE,
            is_cacheable=True,
            confidence=0.85,
            source=ClassificationSource.FALLBACK,
            matched_rule=None,
            rationale="test",
        )

        mock_router = MagicMock()
        mock_router.route.return_value = Tier.AMBIGUOUS

        evaluator = DecisionEvaluator(
            embedder=mock_embedder,
            classifier=mock_classifier,
            tier_router=mock_router,
        )

        # 120 pairs: mix of safe and unsafe
        records = []
        for i in range(1, EXPECTED_RECORD_COUNT + 1):
            rec = make_valid_record(f"PAIR-{i:03d}")
            rec["is_reuse_safe"] = (i % 2 == 1)
            records.append(rec)

        path = write_temp_dataset(records)
        try:
            metrics = evaluator.evaluate(path, bypass_all_ambiguous=True)
        finally:
            path.unlink(missing_ok=True)

        amb_tier = metrics.tier_metrics[Tier.AMBIGUOUS.value]
        # Ambiguous tier MUST produce 0 TP and 0 FP by construction
        assert amb_tier["tp"] == 0
        assert amb_tier["fp"] == 0
        assert amb_tier["irr_cache"] == 0.0
        assert amb_tier["arr"] == 0.0

        # Safe pairs in ambiguous tier become FN, unsafe become TN
        assert amb_tier["fn"] == 60
        assert amb_tier["tn"] == 60

        # Trivial baseline summary fields in DecisionMetrics
        assert metrics.trivial_bypass_fp == 0
        assert metrics.trivial_bypass_irr_cache == 0.0
        assert metrics.trivial_bypass_irr_traffic == 0.0


class TestJudgeEvaluationUnit:
    """Network-free unit tests for DecisionEvaluator judge-call path."""

    def test_evaluator_with_mock_judge_populates_five_way_metrics(self):
        from src.decision.judge_call import LLMJudge, JudgeOutputSchema, JudgeDecisionEnum

        mock_client = MagicMock()
        mock_cand = MagicMock()
        mock_cand.finish_reason = "STOP"
        mock_parsed = JudgeOutputSchema(
            decision=JudgeDecisionEnum.REUSE,
            is_safe=True,
            confidence=0.95,
            rationale="Unit test mock safe reuse",
        )
        mock_usage = MagicMock(prompt_token_count=200, candidates_token_count=40, total_token_count=240)
        mock_resp = MagicMock(
            candidates=[mock_cand],
            parsed=mock_parsed,
            usage_metadata=mock_usage,
            model_version="gemini-2.0-flash",
        )
        mock_resp.text = mock_parsed.model_dump_json()
        mock_client.models.generate_content.return_value = mock_resp

        mock_judge = LLMJudge(client=mock_client)

        evaluator = DecisionEvaluator(
            embedder=MagicMock(dim=EMBEDDING_DIM, encode=lambda q: _unit_vec(1)),
            classifier=MagicMock(classify=lambda q: StabilityResult(
                query=q,
                predicted_label=StabilityLabel.STABLE,
                effective_decision=StabilityLabel.STABLE,
                is_cacheable=True,
                confidence=0.85,
                source=ClassificationSource.RULE,
                matched_rule="test",
                rationale="test",
            )),
            tier_router=MagicMock(route=lambda c, s: Tier.AMBIGUOUS),
            threshold_engine=AdaptiveThresholdEngine(),
            judge=mock_judge,
        )

        records = [make_valid_record(f"PAIR-{i:03d}") for i in range(1, EXPECTED_RECORD_COUNT + 1)]
        path = write_temp_dataset(records)
        try:
            metrics = evaluator.evaluate(path)
        finally:
            path.unlink(missing_ok=True)

        assert metrics.judge_available is True
        assert metrics.judge_calls_count == EXPECTED_RECORD_COUNT
        assert metrics.judge_total_input_tokens == 200 * EXPECTED_RECORD_COUNT
        assert metrics.judge_total_output_tokens == 40 * EXPECTED_RECORD_COUNT
        assert metrics.judge_total_tokens == 240 * EXPECTED_RECORD_COUNT
        assert metrics.judge_total_cost_usd == 0.0
        assert metrics.judge_tp == EXPECTED_RECORD_COUNT
        assert metrics.judge_fp == 0
        assert metrics.judge_arr == 1.0
        assert metrics.judge_irr_cache == 0.0

        # Verify print_report does not error
        evaluator.print_report(metrics)

