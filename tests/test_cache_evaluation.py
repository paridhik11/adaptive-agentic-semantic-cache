"""RETIRED — replaced by test_cache_evaluation_unit.py and test_model_integration.py.

This file is kept as a tombstone to record the original test inventory from Phase 2.
It has been emptied to prevent double-execution of tests that now live in the
split test files.

Migration map:
  TestDatasetSchemaValidation → test_cache_evaluation_unit.py::TestDatasetSchemaValidation
                                (no embedder needed; all schema tests are network-free)
  TestPairProtocolDirection   → test_cache_evaluation_unit.py::TestPairProtocolMechanics
                                (mocked embedder for mechanics)
                                + test_model_integration.py::TestPairProtocolDirectionRealModel
                                (real embedder for score verification)
  TestDatasetLeakage          → test_cache_evaluation_unit.py::TestDatasetLeakage
                                (JSON-only reads, no embedder at all)

See docs/phase2_reproducibility_fix.md for the full split rationale.
"""
