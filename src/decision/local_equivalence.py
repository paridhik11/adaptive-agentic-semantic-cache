"""Phase 8 — Local Equivalence Layer.

Provides local pairwise semantic verification for borderline (AMBIGUOUS-tier) queries:
  1. Rule Guards:
     - Number match: strict equality of extracted numeric literals.
     - Negation match: strict polarity parity (negation words/contractions).
     - Entity match: proper noun/capitalized term consistency.
  2. Pairwise CrossEncoder Model:
     - Default checkpoint: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
     - Revision: '233902d25c440f23af6f7d6e94d2946bac0bee0a'
     - Training data: MS MARCO passage ranking (zero-shot transfer to question equivalence).
     - Score: logistic sigmoid over model logits, bounded in [0.0, 1.0].
  3. Optional remote LLM judge fallback:
     - Kept behind `fallback_to_llm` flag (default FALSE).

SAFETY INVARIANTS:
  - Any rule guard failure immediately yields BYPASS (fail-closed).
  - Score < threshold yields BYPASS unless fallback_to_llm is explicitly enabled.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from sentence_transformers import CrossEncoder

from src.decision.judge_call import JudgeResult, LLMJudge

DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CROSS_ENCODER_REVISION: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

# Common negation terms and contractions
NEGATION_TOKENS: Set[str] = {
    "not", "no", "never", "none", "neither", "nor", "without",
    "hardly", "barely", "scarcely",
    "cannot", "can't", "cant",
    "won't", "wont",
    "don't", "dont",
    "doesn't", "doesnt",
    "didn't", "didnt",
    "isn't", "isnt",
    "aren't", "arent",
    "wasn't", "wasnt",
    "weren't", "werent",
    "haven't", "havent",
    "hasn't", "hasnt",
    "hadn't", "hadnt",
    "shouldn't", "shouldnt",
    "wouldn't", "wouldnt",
    "couldn't", "couldnt",
}

_NUMERIC_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)*(?![A-Za-z0-9_])")
_WORD_RE = re.compile(r"\b[A-Za-z0-9_'/-]+\b")


def extract_numbers(text: str) -> Set[str]:
    """Extract all numeric tokens from text."""
    return set(_NUMERIC_RE.findall(text))


def extract_negations(text: str) -> Set[str]:
    """Extract all negation markers present in text."""
    tokens = {t.lower() for t in _WORD_RE.findall(text)}
    return tokens.intersection(NEGATION_TOKENS)


def extract_proper_entities(text: str) -> Set[str]:
    """Extract capitalized entity tokens occurring within the text.
    
    Excludes the first word of the query (which is capitalized by convention)
    unless the token is all-caps or title-cased throughout.
    """
    words = text.split()
    entities: Set[str] = set()
    for idx, w in enumerate(words):
        cleaned = re.sub(r"[^\w]", "", w)
        if not cleaned:
            continue
        # First word: only include if acronym/all-caps or contains digits
        if idx == 0:
            if cleaned.isupper() and len(cleaned) > 1:
                entities.add(cleaned.lower())
        else:
            if cleaned[0].isupper() or cleaned.isupper():
                entities.add(cleaned.lower())
    return entities


def evaluate_rule_guards(query_a: str, query_b: str) -> Tuple[bool, str]:
    """Evaluate deterministic rule guards between two queries.

    Checks:
      1. Number match: All numeric tokens must match exactly.
      2. Negation match: Polarity must match (same negation presence).
      3. Entity match: Proper noun entities must not disagree.

    Returns:
      (passed: bool, reason: str)
    """
    # 1. Number guard
    nums_a = extract_numbers(query_a)
    nums_b = extract_numbers(query_b)
    if nums_a != nums_b:
        return False, f"NUMBER_MISMATCH: query_a={sorted(nums_a)} vs query_b={sorted(nums_b)}"

    # 2. Negation guard
    negs_a = extract_negations(query_a)
    negs_b = extract_negations(query_b)
    if bool(negs_a) != bool(negs_b):
        return False, f"NEGATION_MISMATCH: query_a={sorted(negs_a)} vs query_b={sorted(negs_b)}"

    # 3. Entity guard
    ents_a = extract_proper_entities(query_a)
    ents_b = extract_proper_entities(query_b)
    if ents_a and ents_b and ents_a != ents_b:
        # If both specify distinct proper entities that don't match
        diff = ents_a.symmetric_difference(ents_b)
        return False, f"ENTITY_MISMATCH: differing_entities={sorted(diff)}"

    return True, "GUARDS_PASSED"


@dataclass(frozen=True)
class LocalEquivalenceResult:
    """Structured decision output from LocalEquivalenceDecisionStep."""

    decision: str  # "REUSE" or "BYPASS"
    is_safe: bool
    score: float
    threshold: float
    guards_passed: bool
    guard_reason: str
    model_name: str
    model_latency_ms: float
    fallback_to_llm_used: bool
    judge_result: Optional[JudgeResult] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "is_safe": self.is_safe,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "guards_passed": self.guards_passed,
            "guard_reason": self.guard_reason,
            "model_name": self.model_name,
            "model_latency_ms": round(self.model_latency_ms, 3),
            "fallback_to_llm_used": self.fallback_to_llm_used,
            "judge_result": self.judge_result.to_dict() if self.judge_result else None,
            "rationale": self.rationale,
        }


class LocalEquivalenceDecisionStep:
    """Pairwise equivalence classifier replacing remote LLM judge in AMBIGUOUS band.

    Pipeline:
      1. Deterministic rule guards (number, negation, entity). If failed -> BYPASS.
      2. Local CrossEncoder pairwise scoring.
      3. If score >= threshold -> REUSE.
      4. If score < threshold:
         - If fallback_to_llm=True -> invoke LLMJudge.
         - Else -> BYPASS.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        threshold: float = 0.70,
        fallback_to_llm: bool = False,
        judge: Optional[LLMJudge] = None,
        model_instance: Optional[CrossEncoder] = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.fallback_to_llm = fallback_to_llm
        self.judge = judge or (LLMJudge() if fallback_to_llm else None)
        self._model = model_instance

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def decide(
        self,
        query_a: str,
        query_b: str,
        similarity_score: float = 0.0,
        stability_confidence: float = 1.0,
        domain: str = "general",
    ) -> LocalEquivalenceResult:
        """Evaluate equivalence between cached query_a and incoming query_b."""
        # 1. Evaluate Rule Guards
        guards_ok, guard_reason = evaluate_rule_guards(query_a, query_b)
        if not guards_ok:
            return LocalEquivalenceResult(
                decision="BYPASS",
                is_safe=False,
                score=0.0,
                threshold=self.threshold,
                guards_passed=False,
                guard_reason=guard_reason,
                model_name=self.model_name,
                model_latency_ms=0.0,
                fallback_to_llm_used=False,
                rationale=f"Rule guard violation: {guard_reason}",
            )

        # 2. Local CrossEncoder inference
        t0 = time.perf_counter()
        raw_pred = self.model.predict([(query_a, query_b)])
        model_ms = (time.perf_counter() - t0) * 1000.0

        # Handle logit vs probability: if logit (can be negative or > 1), apply sigmoid
        val = float(raw_pred[0])
        if val < 0.0 or val > 1.0:
            score = float(torch.sigmoid(torch.tensor(val)).item())
        else:
            score = val

        # 3. Threshold check
        if score >= self.threshold:
            return LocalEquivalenceResult(
                decision="REUSE",
                is_safe=True,
                score=score,
                threshold=self.threshold,
                guards_passed=True,
                guard_reason=guard_reason,
                model_name=self.model_name,
                model_latency_ms=model_ms,
                fallback_to_llm_used=False,
                rationale=f"Local model score {score:.4f} >= threshold {self.threshold:.4f}",
            )

        # 4. Fallback check
        if self.fallback_to_llm and self.judge is not None:
            jr = self.judge.evaluate(
                query_a=query_a,
                query_b=query_b,
                similarity_score=similarity_score,
                domain=domain,
                stability_confidence=stability_confidence,
            )
            decision = "REUSE" if jr.is_safe else "BYPASS"
            return LocalEquivalenceResult(
                decision=decision,
                is_safe=jr.is_safe,
                score=score,
                threshold=self.threshold,
                guards_passed=True,
                guard_reason=guard_reason,
                model_name=self.model_name,
                model_latency_ms=model_ms,
                fallback_to_llm_used=True,
                judge_result=jr,
                rationale=f"Fallback LLM judge invoked: decision={decision}; reason={jr.rationale}",
            )

        return LocalEquivalenceResult(
            decision="BYPASS",
            is_safe=False,
            score=score,
            threshold=self.threshold,
            guards_passed=True,
            guard_reason=guard_reason,
            model_name=self.model_name,
            model_latency_ms=model_ms,
            fallback_to_llm_used=False,
            rationale=f"Local model score {score:.4f} < threshold {self.threshold:.4f}",
        )
