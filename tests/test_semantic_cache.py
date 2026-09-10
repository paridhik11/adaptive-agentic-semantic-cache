"""Unit tests for Phase 2 semantic cache components.

Covers:
  - QueryEmbedder: encoding, normalization, empty-query guard
  - FlatVectorStore: add/search/reset, empty-store guard, dim mismatch
  - SemanticCache: lookup HIT/MISS, threshold boundaries, index+lookup round-trip
  - CacheDecision enum and CacheLookupResult structure
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cache.embedding import QueryEmbedder, DEFAULT_MODEL_NAME, EMBEDDING_DIM
from src.cache.vector_store import FlatVectorStore, SearchResult
from src.cache.semantic_cache import SemanticCache, CacheDecision, CacheLookupResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> QueryEmbedder:
    """Shared embedder instance (model loaded once per module)."""
    return QueryEmbedder()


@pytest.fixture
def fresh_store() -> FlatVectorStore:
    """Empty FlatVectorStore with standard dimension."""
    return FlatVectorStore(dim=EMBEDDING_DIM)


@pytest.fixture
def cache_high_threshold(embedder: QueryEmbedder) -> SemanticCache:
    """SemanticCache at threshold=0.99 — almost nothing should hit."""
    store = FlatVectorStore(dim=embedder.dim)
    return SemanticCache(threshold=0.99, embedder=embedder, vector_store=store)


@pytest.fixture
def cache_zero_threshold(embedder: QueryEmbedder) -> SemanticCache:
    """SemanticCache at threshold=0.0 — everything with any entry hits."""
    store = FlatVectorStore(dim=embedder.dim)
    return SemanticCache(threshold=0.0, embedder=embedder, vector_store=store)


# ---------------------------------------------------------------------------
# QueryEmbedder tests
# ---------------------------------------------------------------------------

class TestQueryEmbedder:
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

    def test_encode_empty_query_raises(self, embedder: QueryEmbedder):
        with pytest.raises(ValueError, match="non-empty"):
            embedder.encode("")

    def test_encode_whitespace_query_raises(self, embedder: QueryEmbedder):
        with pytest.raises(ValueError, match="non-empty"):
            embedder.encode("   ")

    def test_dim_property(self, embedder: QueryEmbedder):
        assert embedder.dim == EMBEDDING_DIM

    def test_model_name(self, embedder: QueryEmbedder):
        assert embedder.model_name == DEFAULT_MODEL_NAME

    def test_different_queries_produce_different_embeddings(self, embedder: QueryEmbedder):
        vec1 = embedder.encode("How does quicksort work?")
        vec2 = embedder.encode("What is the current Bitcoin price?")
        # Inner product should not be ~1.0 for semantically different queries
        cosine_sim = float(np.dot(vec1, vec2))
        assert cosine_sim < 0.99, (
            f"Semantically different queries should have dissimilar embeddings; got cos={cosine_sim:.4f}"
        )

    def test_similar_queries_produce_similar_embeddings(self, embedder: QueryEmbedder):
        vec1 = embedder.encode("How do I reverse a string in Python?")
        vec2 = embedder.encode("Python string reversal using slicing syntax")
        cosine_sim = float(np.dot(vec1, vec2))
        assert cosine_sim > 0.70, (
            f"Semantically similar queries should have high cosine similarity; got {cosine_sim:.4f}"
        )


# ---------------------------------------------------------------------------
# FlatVectorStore tests
# ---------------------------------------------------------------------------

class TestFlatVectorStore:
    def test_empty_store_size(self, fresh_store: FlatVectorStore):
        assert fresh_store.size == 0

    def test_add_increases_size(self, fresh_store: FlatVectorStore, embedder: QueryEmbedder):
        vec = embedder.encode("test query")
        fresh_store.add(vec, metadata={"key": "value"})
        assert fresh_store.size == 1

    def test_search_on_empty_store_returns_not_found(self, fresh_store: FlatVectorStore, embedder: QueryEmbedder):
        vec = embedder.encode("test query")
        result = fresh_store.search(vec, k=1)
        assert result.found is False
        assert result.score == float("-inf")
        assert result.index_position == -1

    def test_add_and_search_round_trip(self, fresh_store: FlatVectorStore, embedder: QueryEmbedder):
        query = "How does quicksort partition work?"
        vec = embedder.encode(query)
        fresh_store.add(vec, metadata={"query": query, "id": "test-1"})

        # Searching with the exact same vector must return score ≈ 1.0
        result = fresh_store.search(vec, k=1)
        assert result.found is True
        assert result.score == pytest.approx(1.0, abs=1e-4), f"Exact match should score ~1.0, got {result.score}"
        assert result.metadata["id"] == "test-1"

    def test_reset_clears_store(self, fresh_store: FlatVectorStore, embedder: QueryEmbedder):
        vec = embedder.encode("Something")
        fresh_store.add(vec)
        assert fresh_store.size == 1
        fresh_store.reset()
        assert fresh_store.size == 0
        # Search on reset store should return not-found
        result = fresh_store.search(vec)
        assert result.found is False

    def test_dimension_mismatch_raises(self, fresh_store: FlatVectorStore):
        wrong_dim_vec = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            fresh_store.add(wrong_dim_vec)

    def test_metadata_stored_per_entry(self, fresh_store: FlatVectorStore, embedder: QueryEmbedder):
        vec = embedder.encode("Query with metadata")
        fresh_store.add(vec, metadata={"source": "test", "index": 42})
        result = fresh_store.search(vec)
        assert result.metadata["source"] == "test"
        assert result.metadata["index"] == 42


# ---------------------------------------------------------------------------
# SemanticCache tests
# ---------------------------------------------------------------------------

class TestSemanticCache:
    def test_lookup_returns_miss_on_empty_store(self, cache_high_threshold: SemanticCache):
        result = cache_high_threshold.lookup("Any query")
        assert result.decision == CacheDecision.MISS
        assert not result.is_hit
        assert result.store_size == 0

    def test_lookup_returns_hit_at_zero_threshold(self, cache_zero_threshold: SemanticCache):
        """At threshold=0.0, any entry in the store causes a HIT."""
        cache_zero_threshold.index("Some cached query", metadata={"id": "q0"})
        result = cache_zero_threshold.lookup("Completely different query about potatoes")
        assert result.decision == CacheDecision.HIT
        assert result.is_hit

    def test_lookup_hit_for_near_identical_queries(self, embedder: QueryEmbedder):
        """Semantically near-identical query pair should HIT at a moderate threshold.

        The actual cosine similarity between these two queries (all-MiniLM-L6-v2)
        is approximately 0.76. Threshold is set to 0.70 to reliably produce a HIT.
        The sweep confirmed the model operates in this similarity range for equivalent pairs.
        """
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.70, embedder=embedder, vector_store=store)
        cache.index("How do I reverse a string in Python?")
        result = cache.lookup("Python string reversal using slicing syntax")
        assert result.is_hit, (
            f"Semantically equivalent pair should HIT at 0.70; got score={result.similarity_score:.4f}"
        )

    def test_lookup_miss_for_dissimilar_queries_at_high_threshold(self, embedder: QueryEmbedder):
        """Semantically dissimilar pair should MISS at a high threshold."""
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.95, embedder=embedder, vector_store=store)
        cache.index("How do I reverse a string in Python?")
        result = cache.lookup("What is the latest Bitcoin price?")
        assert not result.is_hit, (
            f"Dissimilar pair should MISS at 0.95; got score={result.similarity_score:.4f}"
        )

    def test_lookup_result_has_measured_latencies(self, embedder: QueryEmbedder):
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        cache.index("test query")
        result = cache.lookup("test query")
        assert result.embed_latency_ms >= 0.0
        assert result.search_latency_ms >= 0.0
        assert result.total_latency_ms >= 0.0
        # total should be approximately embed + search
        assert abs(result.total_latency_ms - (result.embed_latency_ms + result.search_latency_ms)) < 1.0

    def test_invalid_threshold_raises(self, embedder: QueryEmbedder):
        with pytest.raises(ValueError):
            SemanticCache(threshold=1.5)
        with pytest.raises(ValueError):
            SemanticCache(threshold=-0.1)

    def test_index_returns_int_position(self, embedder: QueryEmbedder):
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        pos = cache.index("First query")
        assert isinstance(pos, int)
        assert pos == 0

    def test_store_size_tracks_indexed_entries(self, embedder: QueryEmbedder):
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        for i in range(3):
            cache.index(f"Query number {i}")
        result = cache.lookup("Some lookup query")
        assert result.store_size == 3

    def test_lookup_result_to_dict(self, embedder: QueryEmbedder):
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        cache.index("test")
        result = cache.lookup("test")
        d = result.to_dict()
        required_keys = {
            "decision", "similarity_score", "threshold",
            "embed_latency_ms", "search_latency_ms", "total_latency_ms",
            "matched_metadata", "store_size"
        }
        assert required_keys.issubset(d.keys()), f"Missing keys: {required_keys - d.keys()}"

    def test_score_is_cosine_similarity(self, embedder: QueryEmbedder):
        """Score from an exact self-match should be ≈ 1.0 (unit vectors, inner product)."""
        store = FlatVectorStore(dim=embedder.dim)
        cache = SemanticCache(threshold=0.85, embedder=embedder, vector_store=store)
        query = "What is the time complexity of binary search?"
        cache.index(query)
        result = cache.lookup(query)
        assert result.similarity_score == pytest.approx(1.0, abs=1e-4), (
            f"Self-match should score ~1.0, got {result.similarity_score:.6f}"
        )
