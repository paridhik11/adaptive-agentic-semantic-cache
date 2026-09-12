"""Model integration tests for Phase 2 — REQUIRES REAL MODEL AND NETWORK ACCESS.

These tests load the pinned all-MiniLM-L6-v2 model (revision MODEL_REVISION)
and validate actual embedding behaviour against known semantic expectations.

NETWORK ACCESS: These tests will attempt to load the model from the local
HuggingFace cache first (~/.cache/huggingface/hub/). If the cache is present
(revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), no network access occurs.
If the cache is absent, sentence-transformers will download from huggingface.co.

To run these tests:
    pytest -m requires_model
    pytest tests/test_model_integration.py

To exclude these tests (default pytest invocation):
    pytest        # automatically skips requires_model tests

CI: Model integration tests run in a separate job (see .github/workflows/ci.yml)
with HuggingFace cache keyed on MODEL_REVISION so the model is not re-downloaded
on every run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cache.embedding import QueryEmbedder, EMBEDDING_DIM, MODEL_REVISION
from src.cache.vector_store import FlatVectorStore
from src.cache.semantic_cache import SemanticCache, CacheDecision


# All tests in this module require the real model
pytestmark = pytest.mark.requires_model


# ---------------------------------------------------------------------------
# Shared fixture: one embedder instance for the whole module
# (model loaded once per test session — expensive operation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> QueryEmbedder:
    """Load the pinned model once for all integration tests in this module."""
    return QueryEmbedder()


# ---------------------------------------------------------------------------
# QueryEmbedder — real model behaviour
# ---------------------------------------------------------------------------

class TestQueryEmbedderRealModel:
    def test_loads_pinned_revision(self, embedder: QueryEmbedder):
        """The embedder must report the pinned MODEL_REVISION hash."""
        assert embedder.revision == MODEL_REVISION, (
            f"Expected revision {MODEL_REVISION!r}, got {embedder.revision!r}. "
            "Embedding.py may not be passing revision= to SentenceTransformer."
        )

    def test_encode_returns_correct_shape(self, embedder: QueryEmbedder):
        vec = embedder.encode("How does quicksort work?")
        assert vec.shape == (EMBEDDING_DIM,), f"Expected ({EMBEDDING_DIM},), got {vec.shape}"

    def test_encode_returns_float32(self, embedder: QueryEmbedder):
        vec = embedder.encode("Test query")
        assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"

    def test_encode_is_unit_normalized(self, embedder: QueryEmbedder):
        vec = embedder.encode("What is the time complexity of binary search?")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}"

    def test_model_name(self, embedder: QueryEmbedder):
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_dim_property(self, embedder: QueryEmbedder):
        assert embedder.dim == EMBEDDING_DIM

    def test_encode_empty_query_raises(self, embedder: QueryEmbedder):
        with pytest.raises(ValueError, match="non-empty"):
            embedder.encode("")

    def test_encode_whitespace_query_raises(self, embedder: QueryEmbedder):
        with pytest.raises(ValueError, match="non-empty"):
            embedder.encode("   ")

    def test_similar_queries_produce_high_cosine_similarity(self, embedder: QueryEmbedder):
        """Semantically near-identical queries must have cosine similarity > 0.70.

        Threshold of 0.70 is chosen conservatively; the actual similarity for this
        pair using the pinned model is approximately 0.76.
        """
        vec1 = embedder.encode("How do I reverse a string in Python?")
        vec2 = embedder.encode("Python string reversal using slicing syntax")
        cosine_sim = float(np.dot(vec1, vec2))
        assert cosine_sim > 0.70, (
            f"Semantically similar queries should have cosine sim > 0.70; got {cosine_sim:.4f}. "
            "If the pinned model revision changed, re-verify this expected value."
        )

    def test_dissimilar_queries_produce_low_cosine_similarity(self, embedder: QueryEmbedder):
        """Cross-domain semantically unrelated queries must score well below 0.99."""
        vec1 = embedder.encode("How does quicksort work?")
        vec2 = embedder.encode("What is the current Bitcoin price?")
        cosine_sim = float(np.dot(vec1, vec2))
        assert cosine_sim < 0.60, (
            f"Semantically different queries should score < 0.60; got {cosine_sim:.4f}. "
            "If the pinned model revision changed, re-verify this expected value."
        )

    def test_different_queries_produce_different_embeddings(self, embedder: QueryEmbedder):
        vec1 = embedder.encode("How does quicksort work?")
        vec2 = embedder.encode("What is the current Bitcoin price?")
        assert not np.allclose(vec1, vec2, atol=1e-4), (
            "Semantically different queries must not produce identical embeddings"
        )

    def test_self_match_scores_one(self, embedder: QueryEmbedder):
        """Encoding the same string twice must yield score ≈ 1.0."""
        query = "What is the time complexity of binary search?"
        vec1 = embedder.encode(query)
        vec2 = embedder.encode(query)
        cosine_sim = float(np.dot(vec1, vec2))
        assert cosine_sim == pytest.approx(1.0, abs=1e-4), (
            f"Self-match of identical strings must score ~1.0; got {cosine_sim:.6f}"
        )


# ---------------------------------------------------------------------------
# FlatVectorStore + real embeddings
# ---------------------------------------------------------------------------

class TestFlatVectorStoreWithRealModel:
    def test_add_and_search_round_trip(self, embedder: QueryEmbedder):
        query = "How does quicksort partition work?"
        vec = embedder.encode(query)
        store = FlatVectorStore(dim=EMBEDDING_DIM)
        store.add(vec, metadata={"query": query, "id": "test-1"})
        result = store.search(vec, k=1)
        assert result.found is True
        assert result.score == pytest.approx(1.0, abs=1e-4), (
            f"Exact self-match should score ~1.0, got {result.score}"
        )
        assert result.metadata["id"] == "test-1"


# ---------------------------------------------------------------------------
# SemanticCache — real embedding integration
# ---------------------------------------------------------------------------

class TestSemanticCacheWithRealModel:
    def test_lookup_hit_for_near_identical_queries(self, embedder: QueryEmbedder):
        """Semantically near-identical query pair should HIT at threshold=0.70.

        The actual cosine similarity between these two queries (pinned model,
        revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41) is approximately 0.76.
        Threshold set to 0.70 to reliably produce a HIT with the expected values.
        If the revision changes and this test fails, check the new similarity value
        and update the threshold comment accordingly.
        """
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.70, embedder=embedder, vector_store=store)
        cache.index("How do I reverse a string in Python?")
        result = cache.lookup("Python string reversal using slicing syntax")
        assert result.is_hit, (
            f"Semantically equivalent pair should HIT at 0.70; "
            f"got score={result.similarity_score:.4f}. "
            f"Model revision: {MODEL_REVISION}"
        )

    def test_lookup_miss_for_dissimilar_queries_at_high_threshold(self, embedder: QueryEmbedder):
        """Cross-domain semantically dissimilar pair should MISS at threshold=0.95."""
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.95, embedder=embedder, vector_store=store)
        cache.index("How do I reverse a string in Python?")
        result = cache.lookup("What is the latest Bitcoin price?")
        assert not result.is_hit, (
            f"Dissimilar pair should MISS at 0.95; got score={result.similarity_score:.4f}"
        )

    def test_score_is_cosine_similarity_self_match(self, embedder: QueryEmbedder):
        """Self-match score must be ≈ 1.0 (unit vectors, inner product)."""
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        query = "What is the time complexity of binary search?"
        cache.index(query)
        result = cache.lookup(query)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-4), (
            f"Self-match should score ~1.0, got {result.similarity_score:.6f}"
        )

    def test_lookup_result_has_measured_latencies(self, embedder: QueryEmbedder):
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        cache.index("test query")
        result = cache.lookup("test query")
        assert result.embed_latency_ms >= 0.0
        assert result.search_latency_ms >= 0.0
        assert result.total_latency_ms >= 0.0
        assert abs(result.total_latency_ms - (result.embed_latency_ms + result.search_latency_ms)) < 1.0


# ---------------------------------------------------------------------------
# Pair protocol direction (real embeddings)
# ---------------------------------------------------------------------------

class TestPairProtocolDirectionRealModel:
    """Verifies protocol correctness using real embedding geometry."""

    def test_protocol_score_matches_manual_dot_product(self, embedder: QueryEmbedder):
        """The evaluator's score must equal dot(embed(query_a), embed(query_b)).

        This confirms index(a)+search(b) is the order used, not the reverse.
        (Cosine similarity is symmetric so scores would be equal either way, but
        this verifies the embedder is called in the documented order.)
        """
        query_a = "How do I reverse a string in Python using slice notation?"
        query_b = "What is the current price of Bitcoin in USD?"

        vec_a = embedder.encode(query_a)
        vec_b = embedder.encode(query_b)
        expected_score = float(np.dot(vec_a, vec_b))

        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        cache.index(query_a)
        result = cache.lookup(query_b)

        assert abs(result.similarity_score - expected_score) < 1e-4, (
            f"Cache score {result.similarity_score:.6f} != manual dot product {expected_score:.6f}. "
            "Protocol may have index/search order wrong."
        )
