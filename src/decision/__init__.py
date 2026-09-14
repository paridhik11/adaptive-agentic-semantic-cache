"""Phase 3 — Reuse Decision Layer.

This package adds a tiered routing layer ABOVE the frozen Phase 1 stability
classifier and Phase 2 semantic cache.  It does not modify either component.

Public API:
    Tier              — enum: AUTO_REUSE | AMBIGUOUS | BYPASS
    TierBoundaries    — frozen dataclass holding all six numeric bounds
    TierRouter        — routes a (CacheLookupResult, StabilityResult) pair to a Tier
    CategoryHistory   — in-memory per-domain reuse-history tracker
    DecisionStep      — ambiguous-band decision logic (no LLM)
    DecisionResult    — output dataclass from DecisionStep
"""

from src.decision.tier_router import Tier, TierBoundaries, TierRouter
from src.decision.decision_step import (
    CategoryHistory,
    DecisionResult,
    DecisionStep,
    JudgeDecisionResult,
    JudgeDecisionStep,
    ProductionDecisionStep,
)
from src.decision.adaptive_threshold_engine import (
    AdaptiveThresholdEngine,
    CategoryThreshold,
    ThresholdDecisionResult,
    MINIMUM_CATEGORY_N,
    MINIMUM_MINORITY_CLASS_N,
    FALLBACK_THRESHOLD,
)
from src.decision.judge_call import (
    LLMJudge,
    JudgeResult,
    JudgeOutputSchema,
    JudgeDecisionEnum,
)

__all__ = [
    "Tier",
    "TierBoundaries",
    "TierRouter",
    "CategoryHistory",
    "DecisionResult",
    "DecisionStep",
    "JudgeDecisionResult",
    "JudgeDecisionStep",
    "ProductionDecisionStep",
    "AdaptiveThresholdEngine",
    "CategoryThreshold",
    "ThresholdDecisionResult",
    "MINIMUM_CATEGORY_N",
    "MINIMUM_MINORITY_CLASS_N",
    "FALLBACK_THRESHOLD",
    "LLMJudge",
    "JudgeResult",
    "JudgeOutputSchema",
    "JudgeDecisionEnum",
]

