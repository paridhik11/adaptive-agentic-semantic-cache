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

REVISION PINNING — READ BEFORE CHANGING MODEL_REVISION
-------------------------------------------------------
MODEL_REVISION is a load-bearing constant for Phase 2 reproducibility, treated
with the same seriousness as STABLE_CONFIDENCE_THRESHOLD in Phase 1.

Provenance: TRACED. The hash below was obtained from the local HuggingFace
snapshot cache at:
    ~/.cache/huggingface/hub/
        models--sentence-transformers--all-MiniLM-L6-v2/
        snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/

The snapshot directory timestamps (2026-09-05 14:51:xx) match the date of the
Phase 2 sweep run, confirming this is the exact model weights that produced
the published baseline numbers (IRR_cache=20.83% at threshold=0.85,
IRR_cache=27.94% at threshold=0.70, "no threshold satisfies IRR_cache < 10%"
finding). See docs/phase2_walkthrough.md and docs/phase2_reproducibility_fix.md.

DO NOT update MODEL_REVISION without:
  1. Re-running scripts/run_cache_threshold_sweep.py with the new revision.
  2. Diffing and documenting old vs. new numbers in phase2_walkthrough.md.
  3. Adding a dated note explaining why the revision changed.

If you are pinning a new fresh revision (not traced to an existing run), label
it explicitly as "fresh-pin" in the comment — do not pretend provenance exists
when it doesn't.
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

# ---------------------------------------------------------------------------
# LOAD-BEARING REPRODUCIBILITY CONSTANT — see module docstring before changing.
#
# Provenance: TRACED to the Phase 2 original sweep run (2026-09-05).
# The snapshot directory name in ~/.cache/huggingface/hub/
#   models--sentence-transformers--all-MiniLM-L6-v2/snapshots/
#   is literally this commit hash — it is the ground truth, not a guess.
# ---------------------------------------------------------------------------
MODEL_REVISION: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class QueryEmbedder:
    """Produces L2-normalized query embeddings for cosine similarity lookup.

    Embeddings are normalized to unit length so that inner-product search in
    FAISS IndexFlatIP is equivalent to cosine similarity. The mathematical range
    of cosine similarity for unit vectors is [-1, 1]; practical values for this
    model on natural-language queries are overwhelmingly positive (>0), but the
    range is NOT bounded to [0, 1] — do not assume non-negativity in downstream
    threshold logic. See vector_store.py's docstring for the authoritative
    mathematical statement.

    Args:
        model_name: sentence-transformers model to load. Defaults to
            DEFAULT_MODEL_NAME ("all-MiniLM-L6-v2").
        revision: Exact HuggingFace commit hash to load. Defaults to
            MODEL_REVISION. Override only with an explicit traced or
            fresh-pin hash; see module docstring for the pinning protocol.

    Usage:
        embedder = QueryEmbedder()
        vec = embedder.encode("How does quicksort work?")
        # vec.shape == (384,); np.linalg.norm(vec) ≈ 1.0
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = MODEL_REVISION,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self._model: SentenceTransformer = SentenceTransformer(
            model_name,
            revision=revision,
        )

    def encode(self, query: str) -> np.ndarray:
        """Encode a single query string into an L2-normalized embedding.

        Args:
            query: Non-empty query string.

        Returns:
            numpy.ndarray of shape (EMBEDDING_DIM,) with unit L2 norm.
            Values are in [-1, 1] (cosine similarity range for unit vectors),
            though in practice overwhelmingly positive for this model.

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
