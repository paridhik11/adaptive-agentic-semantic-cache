"""FAISS-backed flat cosine-similarity vector store for the semantic cache.

Design:
  - Index type: IndexFlatIP (inner product on L2-normalized vectors == cosine sim)
  - All vectors must be L2-normalized before add/search (enforced at this layer).
  - Metadata (arbitrary dict) is stored in a parallel Python list, keyed by the
    FAISS integer index position (0-based). FAISS only stores the vectors.
  - Thread safety: NOT guaranteed. This is a single-threaded evaluation tool.

Cosine similarity via inner product:
  If u and v are both L2-normalized (||u||=||v||=1), then:
      u · v  =  ||u|| * ||v|| * cos(θ)  =  cos(θ)  ∈ [−1, 1]
  FAISS IndexFlatIP returns the inner product directly as the similarity score.
  Scores closer to 1.0 indicate more similar vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


@dataclass
class SearchResult:
    """Result of a top-1 vector search."""

    found: bool                  # True if at least one vector exists in the index
    score: float                 # Cosine similarity in [-1, 1]; -inf if not found
    index_position: int          # 0-based FAISS index position; -1 if not found
    metadata: Dict[str, Any]     # Caller-supplied metadata for the stored entry


class FlatVectorStore:
    """In-memory FAISS flat cosine-similarity vector store.

    Args:
        dim: Embedding dimensionality. Must match the embedder's output dim.

    Usage:
        store = FlatVectorStore(dim=384)
        store.add(embedding, metadata={"query": "How does quicksort work?"})
        result = store.search(query_embedding, k=1)
        if result.found and result.score >= threshold:
            # cache hit
            ...
        store.reset()   # empty the index (used between pair evaluations)
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a single L2-normalized embedding to the store.

        Args:
            embedding: numpy.ndarray of shape (dim,). Must be L2-normalized.
            metadata: Optional arbitrary dict stored alongside the vector.

        Returns:
            Integer index position of the newly added entry (0-based).

        Raises:
            ValueError: If embedding has wrong shape.
        """
        embedding = self._validate_and_reshape(embedding)
        self._index.add(embedding)
        self._metadata.append(metadata or {})
        return len(self._metadata) - 1

    def search(self, embedding: np.ndarray, k: int = 1) -> SearchResult:
        """Search for the k nearest vectors by cosine similarity.

        Args:
            embedding: L2-normalized query vector of shape (dim,).
            k: Number of nearest neighbours to retrieve. Phase 2 always uses k=1.

        Returns:
            SearchResult for the top-1 match (even if k > 1, only top-1 is returned
            in the result struct). Returns found=False if the index is empty.
        """
        if self._index.ntotal == 0:
            return SearchResult(found=False, score=float("-inf"), index_position=-1, metadata={})

        embedding = self._validate_and_reshape(embedding)
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(embedding, actual_k)

        top_score = float(scores[0][0])
        top_idx = int(indices[0][0])

        return SearchResult(
            found=True,
            score=top_score,
            index_position=top_idx,
            metadata=self._metadata[top_idx] if 0 <= top_idx < len(self._metadata) else {},
        )

    def reset(self) -> None:
        """Clear all vectors and metadata. Used between pair evaluations."""
        self._index.reset()
        self._metadata.clear()

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_reshape(self, embedding: np.ndarray) -> np.ndarray:
        """Validate shape and return a contiguous float32 (1, dim) array for FAISS."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {embedding.shape[1]}"
            )
        return np.ascontiguousarray(embedding, dtype=np.float32)
