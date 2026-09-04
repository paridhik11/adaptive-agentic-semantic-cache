"""Two-stage query stability classifier orchestrating rule pass and fallback model.

Sub-stage A (Phase 1.1): uncertainty-default safety fix.
When the fallback classifier is uncertain (confidence < STABLE_CONFIDENCE_THRESHOLD),
the effective decision is forced to DYNAMIC, not STABLE. This implements the
conservative-bias principle: a stale cache hit is worse than a wasted LLM call.
"""

from __future__ import annotations

import time
from typing import Optional

from src.classifier.fallback import BaseFallbackClassifier, LightweightHeuristicClassifier
from src.classifier.models import (
    ClassificationSource,
    ConditionalRoutingPolicy,
    StabilityLabel,
    StabilityResult,
)
from src.classifier.rules import StabilityRuleEngine

# ---------------------------------------------------------------------------
# Safety threshold — Sub-stage A (Phase 1.1)
# ---------------------------------------------------------------------------
# When the fallback classifier predicts STABLE, its confidence must meet or
# exceed this threshold for the effective decision to remain STABLE.
# Any fallback STABLE prediction below this value is overridden to DYNAMIC
# (bypass cache) to implement the conservative-bias / uncertainty-default
# principle of the project.
#
# Value rationale: the LightweightHeuristicClassifier reports
#   confidence ≈ 1 − dynamic_score for STABLE predictions.
# With zero active signals (prior_log_odds = −1.2), dynamic_score ≈ 0.232,
# so confidence ≈ 0.768 — intentionally below 0.80, meaning zero-signal
# queries are flipped to DYNAMIC by this guard.  Only queries where the
# heuristic scores at least +0.2 log-odds net toward STABLE (i.e., the
# stable cues clearly outweigh dynamic cues) will clear the threshold.
#
# This constant must remain the single authoritative source for this
# boundary; do not duplicate the check elsewhere.
STABLE_CONFIDENCE_THRESHOLD: float = 0.80


class StabilityClassifier:
    """Conservative, explainable two-stage stability classifier."""

    def __init__(
        self,
        rule_engine: Optional[StabilityRuleEngine] = None,
        fallback_classifier: Optional[BaseFallbackClassifier] = None,
        conditional_policy: ConditionalRoutingPolicy = ConditionalRoutingPolicy.FORWARD_TO_CACHE_CANDIDATE,
    ) -> None:
        """Initialize stability classifier.

        Args:
            rule_engine: Optional custom rule engine. Defaults to StabilityRuleEngine.
            fallback_classifier: Optional custom fallback classifier. Defaults to LightweightHeuristicClassifier.
            conditional_policy: Routing policy for CONDITIONALLY_STABLE queries (FORWARD_TO_CACHE_CANDIDATE or CONSERVATIVE_BYPASS).
        """
        self.rule_engine = rule_engine or StabilityRuleEngine()
        self.fallback_classifier = fallback_classifier or LightweightHeuristicClassifier()
        self.conditional_policy = conditional_policy

    def classify(self, query: str) -> StabilityResult:
        """Classify query stability with two-stage pipeline.

        Stage 1: Explainable rule check for high-confidence volatility and historical guards.
        Stage 2: Heuristic feature-scoring fallback for queries without definitive rule matches.

        Args:
            query: The user prompt or question string.

        Returns:
            StabilityResult with full prediction details, routing source, and rationale.
        """
        start_time = time.perf_counter()

        if not query or not query.strip():
            # Handle empty query defensively as UNCERTAIN -> DYNAMIC (bypass cache)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return StabilityResult(
                query=query,
                predicted_label=StabilityLabel.UNCERTAIN,
                effective_decision=StabilityLabel.DYNAMIC,
                is_cacheable=False,
                confidence=1.0,
                source=ClassificationSource.RULE,
                matched_rule="empty_query_guard",
                rationale="Empty or whitespace query cannot be cached.",
                metadata={"latency_ms": round(latency_ms, 4)},
            )

        # Stage 1: Rule-Based Evaluation
        rule_match = self.rule_engine.evaluate(query)

        if rule_match.matched and rule_match.label in (StabilityLabel.STABLE, StabilityLabel.DYNAMIC, StabilityLabel.CONDITIONALLY_STABLE):
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            raw_label = rule_match.label

            if raw_label == StabilityLabel.CONDITIONALLY_STABLE:
                if self.conditional_policy == ConditionalRoutingPolicy.CONSERVATIVE_BYPASS:
                    effective = StabilityLabel.DYNAMIC
                else:
                    effective = StabilityLabel.STABLE
            else:
                effective = raw_label

            return StabilityResult(
                query=query,
                predicted_label=raw_label,
                effective_decision=effective,
                is_cacheable=(effective == StabilityLabel.STABLE),
                confidence=rule_match.confidence,
                source=ClassificationSource.RULE,
                matched_rule=rule_match.rule_name,
                rationale=rule_match.rationale or f"Matched rule '{rule_match.rule_name}'.",
                metadata={
                    "latency_ms": round(latency_ms, 4),
                    "conditional_policy": self.conditional_policy.value,
                },
            )

        # Stage 2: Fallback Evaluation for Uncertain Queries
        fallback_pred = self.fallback_classifier.predict(query)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        raw_label = fallback_pred.label

        if raw_label == StabilityLabel.CONDITIONALLY_STABLE:
            if self.conditional_policy == ConditionalRoutingPolicy.CONSERVATIVE_BYPASS:
                effective = StabilityLabel.DYNAMIC
            else:
                effective = StabilityLabel.STABLE
        else:
            effective = raw_label

        # -----------------------------------------------------------------------
        # Sub-stage A safety override — uncertainty defaults to DYNAMIC
        # -----------------------------------------------------------------------
        # If the effective decision is STABLE but the fallback's confidence is
        # below STABLE_CONFIDENCE_THRESHOLD, override to DYNAMIC.  A low-confidence
        # STABLE prediction means the classifier is uncertain; per the project's
        # conservative-bias principle, uncertainty must route to bypass (DYNAMIC),
        # not cache (STABLE).  This is the single enforcement point for this rule.
        uncertainty_override_applied = False
        if effective == StabilityLabel.STABLE and fallback_pred.confidence < STABLE_CONFIDENCE_THRESHOLD:
            effective = StabilityLabel.DYNAMIC
            uncertainty_override_applied = True

        override_note = (
            f" [Sub-stage A: low-confidence STABLE override to DYNAMIC — "
            f"confidence {fallback_pred.confidence:.3f} < threshold {STABLE_CONFIDENCE_THRESHOLD:.2f}]"
            if uncertainty_override_applied
            else ""
        )

        return StabilityResult(
            query=query,
            predicted_label=raw_label,
            effective_decision=effective,
            is_cacheable=(effective == StabilityLabel.STABLE),
            confidence=fallback_pred.confidence,
            source=ClassificationSource.FALLBACK,
            matched_rule=None,
            rationale=fallback_pred.rationale + override_note,
            metadata={
                "latency_ms": round(latency_ms, 4),
                "fallback_metadata": fallback_pred.metadata,
                "rule_pass_result": rule_match.rule_name,
                "conditional_policy": self.conditional_policy.value,
                "stable_confidence_threshold": STABLE_CONFIDENCE_THRESHOLD,
                "uncertainty_override_applied": uncertainty_override_applied,
            },
        )
