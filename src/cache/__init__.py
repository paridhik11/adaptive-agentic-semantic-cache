"""Semantic cache and vector store package — Phase 2 baseline."""

from src.cache.embedding import QueryEmbedder, DEFAULT_MODEL_NAME, EMBEDDING_DIM
from src.cache.vector_store import FlatVectorStore, SearchResult
from src.cache.semantic_cache import SemanticCache, CacheDecision, CacheLookupResult

__all__ = [
    "QueryEmbedder",
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIM",
    "FlatVectorStore",
    "SearchResult",
    "SemanticCache",
    "CacheDecision",
    "CacheLookupResult",
]
