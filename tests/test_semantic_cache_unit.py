"""Unit tests for Phase 2 semantic cache components — NO NETWORK, NO REAL MODEL.

These tests mock QueryEmbedder.encode() to return deterministic fake vectors.
No SentenceTransformer is loaded; no network access is required or permitted.

Covered:
  - FlatVectorStore mechanics (add/search/reset/dim-mismatch)
  - SemanticCache.lookup() HIT/MISS decision logic at various threshold boundaries
  - CacheLookupResult schema and to_dict() contract
  - CacheDecision enum basics
  - QueryEmbedder public API surface (constants, dim property, empty-query guard)
    tested via mocking — does NOT exercise real model weights

NOT covered here (see test_model_integration.py, requires_model marker):
  - Real embedding similarity between natural-language queries
  - Actual cosine scores from the all-MiniLM-L6-v2 model

pytest marker: (none) — these run unconditionally in the default `pytest` invocation.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.cache.embedding import DEFAULT_MODEL_NAME, EMBEDDING_DIM, MODEL_REVISION
from src.cache.vector_store import FlatVectorStore, SearchResult
from src.cache.semantic_cache import SemanticCache, CacheDecision, CacheLookupResult


# ---------------------------------------------------------------------------
# Helper: make a deterministic unit-norm fake vector for a given seed
# ---------------------------------------------------------------------------

def _unit_vec(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Return a deterministic L2-normalized float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_mock_embedder(encode_fn=None):
    """Return a mock QueryEmbedder whose encode() is fully deterministic."""
    mock = MagicMock()
    mock.dim = EMBEDDING_DIM
    mock.model_name = DEFAULT_MODEL_NAME
    mock.revision = MODEL_REVISION
    if encode_fn is not None:
        mock.encode.side_effect = encode_fn
    else:
        mock.encode.return_value = _unit_vec(seed=42)
    return mock


# ---------------------------------------------------------------------------
# MODULE CONSTANTS — verified without loading any model
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_default_model_name(self):
        assert DEFAULT_MODEL_NAME == "all-MiniLM-L6-v2"

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 384

    def test_model_revision_is_pinned(self):
        """MODEL_REVISION must be a 40-character hex SHA-1 hash — not empty, not 'main'."""
        assert isinstance(MODEL_REVISION, str), "MODEL_REVISION must be a str"
        assert len(MODEL_REVISION) == 40, (
            f"MODEL_REVISION must be a 40-char SHA-1 hash; got len={len(MODEL_REVISION)!r}: {MODEL_REVISION!r}"
        )
        assert MODEL_REVISION != "main", "MODEL_REVISION must not be the unpinned 'main' branch"
        # Validate it looks like a hex string
        int(MODEL_REVISION, 16)  # raises ValueError if not valid hex


# ---------------------------------------------------------------------------
# FlatVectorStore — pure mechanics, zero network dependency
# ---------------------------------------------------------------------------

class TestFlatVectorStore:
    def test_empty_store_size(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        assert store.size == 0

    def test_add_increases_size(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        store.add(_unit_vec(1), metadata={"key": "value"})
        assert store.size == 1

    def test_add_returns_position_zero_for_first_entry(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        pos = store.add(_unit_vec(1))
        assert pos == 0

    def test_add_returns_sequential_positions(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        for i in range(5):
            pos = store.add(_unit_vec(i))
            assert pos == i

    def test_search_on_empty_store_returns_not_found(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        result = store.search(_unit_vec(99), k=1)
        assert result.found is False
        assert result.score == float("-inf")
        assert result.index_position == -1
        assert result.metadata == {}

    def test_exact_self_match_scores_near_one(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        vec = _unit_vec(42)
        store.add(vec, metadata={"id": "self"})
        result = store.search(vec, k=1)
        assert result.found is True
        assert result.score == pytest.approx(1.0, abs=1e-5), (
            f"Exact self-match of a unit vector must score ~1.0, got {result.score}"
        )
        assert result.metadata["id"] == "self"

    def test_reset_clears_all_vectors_and_metadata(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        store.add(_unit_vec(1), metadata={"x": 1})
        store.add(_unit_vec(2), metadata={"x": 2})
        assert store.size == 2
        store.reset()
        assert store.size == 0
        result = store.search(_unit_vec(1))
        assert result.found is False

    def test_dimension_mismatch_on_add_raises(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        wrong_vec = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            store.add(wrong_vec)

    def test_dimension_mismatch_on_search_raises(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        store.add(_unit_vec(1))
        wrong_vec = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            store.search(wrong_vec)

    def test_metadata_stored_per_entry(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        store.add(_unit_vec(1), metadata={"source": "test", "index": 42})
        result = store.search(_unit_vec(1))
        assert result.metadata["source"] == "test"
        assert result.metadata["index"] == 42

    def test_missing_metadata_defaults_to_empty_dict(self):
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        store.add(_unit_vec(1))
        result = store.search(_unit_vec(1))
        assert result.metadata == {}

    def test_top1_retrieves_nearest_not_farthest(self):
        """With two entries, top-1 must return the more similar one."""
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        target = _unit_vec(0)
        # near_vec: nearly identical to target (perturb slightly)
        near_vec = (target + np.full(EMBEDDING_DIM, 0.001, dtype=np.float32))
        near_vec = (near_vec / np.linalg.norm(near_vec)).astype(np.float32)
        # far_vec: orthogonal-ish (different seed)
        far_vec = _unit_vec(999)

        store.add(near_vec, metadata={"name": "near"})
        store.add(far_vec, metadata={"name": "far"})

        result = store.search(target, k=1)
        assert result.found is True
        assert result.metadata["name"] == "near", (
            f"Top-1 should be the near vector; got {result.metadata['name']} "
            f"with score {result.score:.4f}"
        )


# ---------------------------------------------------------------------------
# SemanticCache — HIT/MISS decision logic with mocked embedder
# ---------------------------------------------------------------------------

class TestSemanticCacheDecisionLogic:
    """Tests SemanticCache.lookup() decision logic without any real embeddings.

    The embedder is mocked to return controlled vectors so that threshold
    boundary conditions can be tested precisely. The actual cosine similarity
    math is exercised via FlatVectorStore (which is real code, not mocked).
    """

    def _make_cache(self, threshold: float, vec_a: np.ndarray, vec_b: np.ndarray) -> SemanticCache:
        """Build a SemanticCache with a mock embedder returning vec_a then vec_b."""
        calls = [vec_a, vec_b]
        idx = {"i": 0}

        def encode_side_effect(query: str) -> np.ndarray:
            v = calls[idx["i"] % len(calls)]
            idx["i"] += 1
            return v

        mock_embedder = _make_mock_embedder(encode_fn=encode_side_effect)
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        return SemanticCache(threshold=threshold, embedder=mock_embedder, vector_store=store)

    def test_exact_match_above_zero_threshold_is_hit(self):
        """Identical vectors: score=1.0, any threshold <= 1.0 → HIT."""
        vec = _unit_vec(1)
        cache = self._make_cache(threshold=0.85, vec_a=vec, vec_b=vec)
        cache.index("indexed query")
        result = cache.lookup("lookup query")
        assert result.decision == CacheDecision.HIT
        assert result.is_hit
        assert result.similarity_score == pytest.approx(1.0, abs=1e-5)

    def test_score_exactly_at_threshold_is_hit(self):
        """When score >= threshold, must be HIT (>= is the documented comparison).

        FIX NOTE (Part B — test-construction issue):
        The original test set threshold=actual_score where actual_score was computed
        via an independent float64 np.dot(vec_a, vec_b). Relying on bit-exact equality
        between external float64 math and FAISS's internal float32 IndexFlatIP inner
        product is fragile across BLAS backends/platforms (e.g. 1 ULP delta on numpy 2.5.3).

        Rather than weakening the assertion by subtracting an arbitrary epsilon (which would
        stop verifying exact equality at the boundary), we derive the threshold directly
        from the system itself:
        1. Run a probe lookup with a permissive threshold (0.0) to capture the exact
           similarity score reported by FAISS/system for this vector pair.
        2. Set threshold = system_score (in the exact same float32 arithmetic domain used
           by production lookup).
        3. Assert that a lookup at this threshold returns HIT and satisfies score == threshold.
        """
        vec_a = _unit_vec(1)
        # Create vec_b with a dot product to vec_a close to 0.70.
        perp_raw = np.random.default_rng(77).standard_normal(EMBEDDING_DIM).astype(np.float32)
        perp_raw -= perp_raw.dot(vec_a) * vec_a
        perp = (perp_raw / np.linalg.norm(perp_raw)).astype(np.float32)
        cos_theta = 0.70
        sin_theta = float(np.sqrt(1 - cos_theta**2))
        vec_b = (cos_theta * vec_a + sin_theta * perp).astype(np.float32)
        vec_b = (vec_b / np.linalg.norm(vec_b)).astype(np.float32)

        # Probe lookup: query the system with threshold=0.0 to read back the exact
        # similarity score produced by FAISS/vector store for this vector pair.
        probe_cache = self._make_cache(threshold=0.0, vec_a=vec_a, vec_b=vec_b)
        probe_cache.index("indexed query")
        probe_result = probe_cache.lookup("lookup query")
        assert probe_result.similarity_score is not None
        system_score = probe_result.similarity_score

        # Verify construction is approximately around the 0.70 target
        assert abs(system_score - 0.70) < 1e-4, (
            f"Construction error: expected ~0.70, got {system_score:.6f}"
        )

        # Exact boundary test: threshold is set exactly to system_score (true score == threshold)
        cache = self._make_cache(threshold=system_score, vec_a=vec_a, vec_b=vec_b)
        cache.index("indexed query")
        result = cache.lookup("lookup query")
        assert result.decision == CacheDecision.HIT, (
            f"score ({result.similarity_score:.8f}) >= threshold ({system_score:.8f}) must be HIT; "
            f"got decision={result.decision}"
        )
        assert result.similarity_score == system_score
        assert result.threshold == system_score

    def test_score_just_below_threshold_is_miss(self):
        """When score < threshold, must be MISS."""
        vec_a = _unit_vec(1)
        perp_raw = np.random.default_rng(88).standard_normal(EMBEDDING_DIM).astype(np.float32)
        perp_raw -= perp_raw.dot(vec_a) * vec_a
        perp = (perp_raw / np.linalg.norm(perp_raw)).astype(np.float32)
        cos_theta = 0.699  # just below 0.70
        sin_theta = float(np.sqrt(max(0.0, 1 - cos_theta**2)))
        vec_b = (cos_theta * vec_a + sin_theta * perp).astype(np.float32)
        vec_b = (vec_b / np.linalg.norm(vec_b)).astype(np.float32)

        cache = self._make_cache(threshold=0.70, vec_a=vec_a, vec_b=vec_b)
        cache.index("indexed query")
        result = cache.lookup("lookup query")
        assert result.decision == CacheDecision.MISS, (
            f"score < threshold must be MISS; score={result.similarity_score:.6f}"
        )
        assert not result.is_hit

    def test_miss_on_empty_store(self):
        """Without any indexed entries, lookup must always return MISS."""
        mock_embedder = _make_mock_embedder()
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        cache = SemanticCache(threshold=0.0, embedder=mock_embedder, vector_store=store)
        result = cache.lookup("any query")
        assert result.decision == CacheDecision.MISS
        assert result.store_size == 0

    def test_zero_threshold_always_hits_with_entry(self):
        """At threshold=0.0, any stored entry causes a HIT (score >= 0.0 for any unit pair)."""
        vec_a = _unit_vec(1)
        vec_b = _unit_vec(999)  # very different
        # Ensure dot product is >= 0 by checking; if not, negate one
        if np.dot(vec_a, vec_b) < 0:
            vec_b = -vec_b
        cache = self._make_cache(threshold=0.0, vec_a=vec_a, vec_b=vec_b)
        cache.index("indexed query")
        result = cache.lookup("lookup query")
        # At threshold=0.0, positive score → HIT
        if result.similarity_score >= 0.0:
            assert result.decision == CacheDecision.HIT

    def test_invalid_threshold_high_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=1.5)

    def test_invalid_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            SemanticCache(threshold=-0.1)

    def test_store_size_reported_correctly(self):
        mock_embedder = _make_mock_embedder()
        mock_embedder.encode.return_value = _unit_vec(1)
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        cache = SemanticCache(threshold=0.85, embedder=mock_embedder, vector_store=store)
        for i in range(4):
            # Each call to index() uses encode(), then lookup() uses encode() again
            cache.index(f"q{i}")
        result = cache.lookup("lookup")
        assert result.store_size == 4

    def test_index_returns_sequential_int_positions(self):
        mock_embedder = _make_mock_embedder()
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        cache = SemanticCache(threshold=0.85, embedder=mock_embedder, vector_store=store)
        for expected_pos in range(3):
            pos = cache.index(f"query {expected_pos}")
            assert isinstance(pos, int)
            assert pos == expected_pos


# ---------------------------------------------------------------------------
# CacheLookupResult — schema and to_dict() contract
# ---------------------------------------------------------------------------

class TestCacheLookupResultSchema:
    """Validates CacheLookupResult structure without requiring real embeddings."""

    def _make_result(self, decision=CacheDecision.HIT, score=0.90, threshold=0.85):
        return CacheLookupResult(
            decision=decision,
            similarity_score=score,
            threshold=threshold,
            embed_latency_ms=5.1234,
            search_latency_ms=0.0234,
            total_latency_ms=5.1468,
            matched_metadata={"id": "PAIR-001"},
            store_size=1,
        )

    def test_is_hit_true_for_hit_decision(self):
        result = self._make_result(decision=CacheDecision.HIT)
        assert result.is_hit is True

    def test_is_hit_false_for_miss_decision(self):
        result = self._make_result(decision=CacheDecision.MISS, score=0.50, threshold=0.85)
        assert result.is_hit is False

    def test_to_dict_contains_all_required_keys(self):
        result = self._make_result()
        d = result.to_dict()
        required_keys = {
            "decision", "similarity_score", "threshold",
            "embed_latency_ms", "search_latency_ms", "total_latency_ms",
            "matched_metadata", "store_size",
        }
        assert required_keys.issubset(d.keys()), (
            f"to_dict() missing keys: {required_keys - d.keys()}"
        )

    def test_to_dict_decision_is_string(self):
        """to_dict() must serialize decision as a plain string, not an enum."""
        result = self._make_result(decision=CacheDecision.HIT)
        d = result.to_dict()
        assert isinstance(d["decision"], str)
        assert d["decision"] == "HIT"

    def test_to_dict_scores_are_rounded(self):
        result = self._make_result(score=0.9876543210)
        d = result.to_dict()
        # similarity_score rounded to 6 decimal places
        assert d["similarity_score"] == round(0.9876543210, 6)

    def test_frozen_dataclass_immutable(self):
        result = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.similarity_score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CacheDecision enum
# ---------------------------------------------------------------------------

class TestCacheDecisionEnum:
    def test_str_hit(self):
        assert str(CacheDecision.HIT) == "HIT"

    def test_str_miss(self):
        assert str(CacheDecision.MISS) == "MISS"

    def test_hit_is_string_subclass(self):
        assert CacheDecision.HIT == "HIT"

    def test_miss_is_string_subclass(self):
        assert CacheDecision.MISS == "MISS"


# ---------------------------------------------------------------------------
# QueryEmbedder public API surface — tested via mocking, no real model
# ---------------------------------------------------------------------------

class TestQueryEmbedderAPISurface:
    """Tests the public API of QueryEmbedder without loading a real model.

    The SentenceTransformer constructor is patched so no network call or
    local model load occurs. The encode() logic itself is exercised.
    """

    def test_empty_query_raises_value_error(self):
        """The empty-query guard runs before the model is invoked."""
        with patch("src.cache.embedding.SentenceTransformer"):
            from src.cache.embedding import QueryEmbedder
            embedder = QueryEmbedder()
            with pytest.raises(ValueError, match="non-empty"):
                embedder.encode("")

    def test_whitespace_only_query_raises_value_error(self):
        with patch("src.cache.embedding.SentenceTransformer"):
            from src.cache.embedding import QueryEmbedder
            embedder = QueryEmbedder()
            with pytest.raises(ValueError, match="non-empty"):
                embedder.encode("   ")

    def test_dim_property_returns_embedding_dim(self):
        with patch("src.cache.embedding.SentenceTransformer"):
            from src.cache.embedding import QueryEmbedder
            embedder = QueryEmbedder()
            assert embedder.dim == EMBEDDING_DIM

    def test_model_name_stored(self):
        with patch("src.cache.embedding.SentenceTransformer"):
            from src.cache.embedding import QueryEmbedder
            embedder = QueryEmbedder()
            assert embedder.model_name == DEFAULT_MODEL_NAME

    def test_revision_stored(self):
        with patch("src.cache.embedding.SentenceTransformer"):
            from src.cache.embedding import QueryEmbedder
            embedder = QueryEmbedder()
            assert embedder.revision == MODEL_REVISION

    def test_sentencetransformer_called_with_revision(self):
        """SentenceTransformer must be called with model_name AND revision= kwarg."""
        with patch("src.cache.embedding.SentenceTransformer") as mock_st:
            from src.cache.embedding import QueryEmbedder
            QueryEmbedder()
            mock_st.assert_called_once_with(DEFAULT_MODEL_NAME, revision=MODEL_REVISION)

    def test_encode_returns_float32(self):
        """encode() must cast to float32 regardless of model output dtype."""
        with patch("src.cache.embedding.SentenceTransformer") as mock_st:
            from src.cache.embedding import QueryEmbedder
            # Make the mock model return float64 — encode() should still cast to float32
            fake_vec = np.ones(EMBEDDING_DIM, dtype=np.float64)
            mock_st.return_value.encode.return_value = fake_vec
            embedder = QueryEmbedder()
            result = embedder.encode("hello")
            assert result.dtype == np.float32

    def test_encode_passes_strip_query_to_model(self):
        """encode() strips whitespace before passing to the underlying model."""
        with patch("src.cache.embedding.SentenceTransformer") as mock_st:
            from src.cache.embedding import QueryEmbedder
            fake_vec = np.ones(EMBEDDING_DIM, dtype=np.float32)
            mock_st.return_value.encode.return_value = fake_vec
            embedder = QueryEmbedder()
            embedder.encode("  hello world  ")
            call_args = mock_st.return_value.encode.call_args
            assert call_args[0][0] == "hello world"
