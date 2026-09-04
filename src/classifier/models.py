"""Data models, enums, and schemas for the query stability classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class StabilityLabel(str, Enum):
    """Ground-truth and predicted stability classification labels."""

    STABLE = "STABLE"
    DYNAMIC = "DYNAMIC"
    CONDITIONALLY_STABLE = "CONDITIONALLY_STABLE"
    UNCERTAIN = "UNCERTAIN"

    def __str__(self) -> str:
        return self.value


class ClassificationSource(str, Enum):
    """Source that produced the stability decision."""

    RULE = "RULE"
    FALLBACK = "FALLBACK"

    def __str__(self) -> str:
        return self.value


class ConditionalRoutingPolicy(str, Enum):
    """Policy for routing CONDITIONALLY_STABLE queries in the cache pipeline.

    - FORWARD_TO_CACHE_CANDIDATE: Treats conditionally stable queries as STABLE candidates
      for vector similarity search; compatibility (e.g. OS/version constraints) is verified
      at the downstream decision/context layer (Phase 3/4).
    - CONSERVATIVE_BYPASS: Treats conditionally stable queries as DYNAMIC at the gateway,
      bypassing cache to strictly eliminate stale context risk without checking candidates.
    """

    FORWARD_TO_CACHE_CANDIDATE = "FORWARD_TO_CACHE_CANDIDATE"
    CONSERVATIVE_BYPASS = "CONSERVATIVE_BYPASS"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RuleMatch:
    """Outcome of a single or aggregate rule evaluation."""

    matched: bool
    label: Optional[StabilityLabel] = None
    rule_name: Optional[str] = None
    confidence: float = 0.0  # Heuristic rule confidence score [0.0 - 1.0]
    rationale: Optional[str] = None


@dataclass
class StabilityResult:
    """Complete structured output from the stability classifier.

    Note on confidence: The `confidence` attribute is a heuristic score representing
    rule certainty or fallback score divergence, NOT a statistically calibrated posterior probability.
    """

    query: str
    predicted_label: StabilityLabel
    effective_decision: StabilityLabel  # Binary routing: STABLE (cache candidate) vs DYNAMIC (bypass)
    is_cacheable: bool
    confidence: float  # Heuristic score [0.0 - 1.0]
    source: ClassificationSource
    matched_rule: Optional[str] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "query": self.query,
            "predicted_label": self.predicted_label.value,
            "effective_decision": self.effective_decision.value,
            "is_cacheable": self.is_cacheable,
            "confidence": round(self.confidence, 4),
            "source": self.source.value,
            "matched_rule": self.matched_rule,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
