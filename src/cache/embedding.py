"""Query embedding module for the semantic cache.

Wraps sentence-transformers to produce L2-normalized dense embeddings
for cosine-similarity-via-inner-product in FAISS IndexFlatIP.

Model: all-MiniLM-L6-v2
  - Output dimension: 384
  - Licence: Apache-2.0
  - Size: ~22 MB
  - Chosen as the project's lightweight, sentence-level embedding standard.

This module is intentionally thin — it owns only the encode + normalize step.
No caching layer, no batching policy, no model-selection logic beyond the
single default model name. Those concerns belong upstream (in SemanticCache).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Public constant: the single authoritative default model name for Phase 2.
# Any change here affects all components that call QueryEmbedder() with no args.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384


class QueryEmbedder:
    """Produces L2-normalized query embeddings for cosine similarity lookup.

    Embeddings are normalized to unit length so that inner-product search in
    FAISS IndexFlatIP is equivalent to cosine similarity in [0, 1].

    Args:
        model_name: sentence-transformers model to load. Defaults to
            DEFAULT_MODEL_NAME ("all-MiniLM-L6-v2").

    Usage:
        embedder = QueryEmbedder()
        vec = embedder.encode("How does quicksort work?")
        # vec.shape == (384,); np.linalg.norm(vec) ≈ 1.0
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)

    def encode(self, query: str) -> np.ndarray:
        """Encode a single query string into an L2-normalized embedding.

        Args:
            query: Non-empty query string.

        Returns:
            numpy.ndarray of shape (EMBEDDING_DIM,) with unit L2 norm.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        # sentence_transformers returns a numpy array of shape (dim,) for a
        # single string. normalize_embeddings=True applies L2 normalization.
        embedding: np.ndarray = self._model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    @property
    def dim(self) -> int:
        """Dimensionality of produced embeddings."""
        return EMBEDDING_DIM
