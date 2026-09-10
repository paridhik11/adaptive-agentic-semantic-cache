"""Semantic cache: query → embed → vector search → HIT/MISS.

This is the Phase 2 baseline implementation.

SCOPE (Phase 2 Baseline — DO NOT expand without explicit scope change):
  - query → embed → vector search (top-1) → fixed similarity threshold
    → above threshold: HIT (reuse) / below: MISS (fresh query)
  - No agentic verification
  - No learned/dynamic threshold
  - No classifier integration (Phase 1 output is NOT consulted)
  - No token/dollar cost figures are produced

The SemanticCache is stateful: you add entries via index() and look them up
via lookup(). For pair-wise benchmark evaluation, the evaluator clears the
store between each pair via cache.vector_store.reset().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from src.cache.embedding import QueryEmbedder, DEFAULT_MODEL_NAME
from src.cache.vector_store import FlatVectorStore, SearchResult


class CacheDecision(str, Enum):
    """Outcome of a semantic cache lookup."""

    HIT = "HIT"    # Top-1 similarity >= threshold: reuse is claimed
    MISS = "MISS"  # Top-1 similarity < threshold OR no entries: fresh query required

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CacheLookupResult:
    """Complete structured output from SemanticCache.lookup().

    Attributes:
        decision: HIT or MISS.
        similarity_score: Raw cosine similarity score from FAISS (inner product
            on L2-normalized vectors). In [-1, 1]; -inf if the store was empty.
        threshold: The similarity threshold applied for this lookup.
        embed_latency_ms: Time to produce the query embedding (ms).
        search_latency_ms: Time to run the FAISS top-1 search (ms).
        total_latency_ms: embed + search time (ms). Excludes indexing time.
        matched_metadata: Metadata dict of the top-1 match if HIT; empty if MISS.
        store_size: Number of entries in the store at lookup time.
    """

    decision: CacheDecision
    similarity_score: float
    threshold: float
    embed_latency_ms: float
    search_latency_ms: float
    total_latency_ms: float
    matched_metadata: Dict[str, Any] = field(default_factory=dict)
    store_size: int = 0

    @property
    def is_hit(self) -> bool:
        return self.decision == CacheDecision.HIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "similarity_score": round(self.similarity_score, 6),
            "threshold": self.threshold,
            "embed_latency_ms": round(self.embed_latency_ms, 4),
            "search_latency_ms": round(self.search_latency_ms, 4),
            "total_latency_ms": round(self.total_latency_ms, 4),
            "matched_metadata": self.matched_metadata,
            "store_size": self.store_size,
        }


class SemanticCache:
    """Embedding-based semantic cache with a fixed similarity threshold.

    Args:
        threshold: Cosine similarity threshold in [0, 1]. A lookup returns HIT
            iff the top-1 similarity >= threshold. Defaults to 0.85.
        embedder: Optional custom QueryEmbedder. Defaults to all-MiniLM-L6-v2.
        vector_store: Optional custom FlatVectorStore. Defaults to a new
            FlatVectorStore(dim=384).

    Usage:
        cache = SemanticCache(threshold=0.85)
        cache.index("How do I reverse a string in Python?", metadata={"id": "q1"})
        result = cache.lookup("Python string reversal with slicing")
        # result.decision == CacheDecision.HIT if similarity >= 0.85
    """

    def __init__(
        self,
        threshold: float = 0.85,
        embedder: Optional[QueryEmbedder] = None,
        vector_store: Optional[FlatVectorStore] = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.embedder: QueryEmbedder = embedder or QueryEmbedder()
        self.vector_store: FlatVectorStore = vector_store or FlatVectorStore(dim=self.embedder.dim)

    def index(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Embed query and add it to the vector store.

        Args:
            query: Non-empty query string to cache.
            metadata: Optional dict stored alongside the embedding (e.g. {"id": "PAIR-001"}).

        Returns:
            Integer index position of the stored entry.
        """
        embedding = self.embedder.encode(query)
        return self.vector_store.add(embedding, metadata=metadata)

    def lookup(self, query: str) -> CacheLookupResult:
        """Look up a query in the cache.

        Phase 2 protocol:
          1. Embed the incoming query.
          2. Search the vector store for top-1 by cosine similarity.
          3. If similarity >= self.threshold: HIT; else: MISS.

        Args:
            query: Non-empty incoming query string.

        Returns:
            CacheLookupResult with decision, score, and measured latencies.
        """
        t0 = time.perf_counter()
        embedding = self.embedder.encode(query)
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        result: SearchResult = self.vector_store.search(embedding, k=1)
        t3 = time.perf_counter()
        search_ms = (t3 - t2) * 1000.0

        if result.found and result.score >= self.threshold:
            decision = CacheDecision.HIT
        else:
            decision = CacheDecision.MISS

        return CacheLookupResult(
            decision=decision,
            similarity_score=result.score,
            threshold=self.threshold,
            embed_latency_ms=round(embed_ms, 4),
            search_latency_ms=round(search_ms, 4),
            total_latency_ms=round(embed_ms + search_ms, 4),
            matched_metadata=result.metadata if result.found else {},
            store_size=self.vector_store.size,
        )
