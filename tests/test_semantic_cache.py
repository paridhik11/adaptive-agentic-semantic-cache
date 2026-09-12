"""RETIRED — replaced by test_semantic_cache_unit.py and test_model_integration.py.

This file is kept as a tombstone to record the original test inventory from Phase 2.
It has been emptied to prevent double-execution of tests that now live in the
split test files.

Migration map:
  TestQueryEmbedder       → test_model_integration.py::TestQueryEmbedderRealModel
                            (requires_model) + test_semantic_cache_unit.py::TestQueryEmbedderAPISurface
  TestFlatVectorStore     → test_semantic_cache_unit.py::TestFlatVectorStore (unit, no model)
                            + test_model_integration.py::TestFlatVectorStoreWithRealModel
  TestSemanticCache       → test_semantic_cache_unit.py::TestSemanticCacheDecisionLogic (unit)
                            + test_model_integration.py::TestSemanticCacheWithRealModel (integration)

See docs/phase2_reproducibility_fix.md for the full split rationale.
"""
