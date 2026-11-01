"""Answer-aware local semantic policy for safe cache reuse.

Combines hard structural requirement guards with local passage-level answer
verification. Candidates are evaluated against the actual cached answer,
not merely against the original query.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import threading
import time
from typing import Any, Callable

from src.policy.decomposed_applicability import requirement_guard
from src.policy.response_cache import ScopedResponseCache
from src.policy.semantic_runtime import ReuseDecision, digest

PASSAGE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PASSAGE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class LocalPassageScorer:
    """Local cross-encoder scoring query relevance against actual answer content."""

    def __init__(self, model: str = PASSAGE_MODEL, revision: str = PASSAGE_REVISION):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("passage scorer requires an immutable 40-character revision")
        import torch
        from sentence_transformers import CrossEncoder

        self.torch = torch
        self.model = CrossEncoder(
            model,
            revision=revision,
            local_files_only=True,
            device="cpu",
            trust_remote_code=False,
            activation_fn=torch.nn.Identity(),
            model_kwargs={"use_safetensors": True},
        )
        self.identity = f"{model}@{revision}"
        self._lock = threading.Lock()

    def score(self, query: str, passage: str) -> dict[str, float | int]:
        if not all(isinstance(x, str) and x.strip() for x in (query, passage)):
            raise ValueError("two nonempty strings required for passage scoring")
        tokens = self.model.tokenizer(query, passage, truncation=False)["input_ids"]
        limits = [self.model.tokenizer.model_max_length]
        max_seq = getattr(self.model, "max_seq_length", None)
        if max_seq is None:
            max_seq = getattr(self.model, "max_length", None)
        if max_seq is not None:
            limits.append(max_seq)
        if len(tokens) > min(limits):
            raise ValueError("query-passage pair exceeds model input window; no silent truncation")

        with self._lock:
            logit = float(self.model.predict([(query, passage)], show_progress_bar=False)[0])

        if not math.isfinite(logit):
            raise ValueError("nonfinite passage score logit")

        # Numerically stable sigmoid
        prob = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
        return {
            "logit": logit,
            "probability": prob,
            "input_tokens": len(tokens),
        }


@dataclass(frozen=True)
class PolicyAssessment:
    recommended_reuse: bool
    reason: str
    local_verifier_attempts: int
    local_verifier_ms: float
    scores: dict[str, Any]
    incoming_sha256: str
    candidate_sha256: str
    answer_sha256: str


class AnswerAwareSemanticPolicy:
    """Answer-aware verification policy combining hard guards and local neural scoring.

    Safety invariants:
    1. Similarity alone never authorizes reuse.
    2. Hard requirement guards (numbers, constraint polarity, conversion direction)
       fire before any neural inference.
    3. The neural verifier judges whether the *actual cached answer* addresses
       the *incoming question*, not question-to-question similarity.
    4. By default, `enabled=False`: the policy operates in observation/diagnostic
       mode until evaluation qualifies serving.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        scorer: LocalPassageScorer | None = None,
        threshold: float = 0.85,
        enabled: bool = False,
        promoter: Callable[[dict, dict, dict], None] | None = None,
    ):
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system prompt context must be nonempty string when specified")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite float in [0.0, 1.0]")
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")

        self.system_prompt = system_prompt
        self.scorer = scorer if scorer is not None else LocalPassageScorer()
        self.threshold = float(threshold)
        self.enabled = enabled
        self.promoter = promoter
        ctx_id = digest(system_prompt) if system_prompt is not None else "dynamic_pinned_match"
        self.identity = f"answer-aware-policy-v1:{self.scorer.identity}:th={self.threshold:.4f}:en={int(self.enabled)}:{ctx_id}"

    def assess(self, request: dict, candidate_request: dict, answer: dict) -> PolicyAssessment:
        started = time.perf_counter()
        req_hash = digest(request)
        cand_hash = digest(candidate_request)
        ans_hash = digest(answer)

        # Context boundary validation
        for body in (request, candidate_request):
            if not isinstance(body, dict):
                return PolicyAssessment(False, "unsupported_context_format", 0, 0.0, {}, req_hash, cand_hash, ans_hash)
            messages = body.get("messages", [])
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "system"
                or not isinstance(messages[0].get("content"), str)
                or (self.system_prompt is not None and messages[0]["content"] != self.system_prompt)
                or messages[1].get("role") != "user"
                or not isinstance(messages[1].get("content"), str)
                or not messages[1]["content"].strip()
            ):
                return PolicyAssessment(False, "outside_pinned_context", 0, 0.0, {}, req_hash, cand_hash, ans_hash)

        # Enforce that incoming request and cached candidate share the exact same context prompt
        if request["messages"][0]["content"] != candidate_request["messages"][0]["content"]:
            return PolicyAssessment(False, "context_mismatch", 0, 0.0, {}, req_hash, cand_hash, ans_hash)

        if not isinstance(answer, dict) or not ScopedResponseCache._admissible(answer, candidate_request):
            return PolicyAssessment(False, "inadmissible_answer", 0, 0.0, {}, req_hash, cand_hash, ans_hash)

        orig_query = candidate_request["messages"][-1]["content"]
        new_query = request["messages"][-1]["content"]

        # Stage 2: Hard structural requirement guards
        guard_failure = requirement_guard(orig_query, new_query)
        if guard_failure is not None:
            elapsed = (time.perf_counter() - started) * 1000.0
            return PolicyAssessment(
                False,
                f"guard_rejected:{guard_failure}",
                0,
                elapsed,
                {"guard": guard_failure},
                req_hash,
                cand_hash,
                ans_hash,
            )

        # Stage 4: Answer-aware verification
        answer_text = answer["choices"][0]["message"]["content"]
        try:
            score_data = self.scorer.score(new_query, answer_text)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            return PolicyAssessment(
                False,
                f"verifier_error:{type(exc).__name__}",
                1,
                elapsed,
                {"error": str(exc)},
                req_hash,
                cand_hash,
                ans_hash,
            )

        elapsed = (time.perf_counter() - started) * 1000.0
        prob = score_data["probability"]
        if prob >= self.threshold:
            return PolicyAssessment(
                True,
                "verified_answer_reuse",
                1,
                elapsed,
                score_data,
                req_hash,
                cand_hash,
                ans_hash,
            )
        else:
            return PolicyAssessment(
                False,
                "verifier_threshold_rejected",
                1,
                elapsed,
                score_data,
                req_hash,
                cand_hash,
                ans_hash,
            )

    def __call__(
        self,
        request: dict,
        candidate_request: dict,
        answer: dict,
        *,
        binding: dict,
        provider: Any = None,
    ) -> ReuseDecision:
        assessment = self.assess(request, candidate_request, answer)
        approved = self.enabled and assessment.recommended_reuse

        # Asynchronous promotion hook if qualified
        if approved and self.promoter is not None:
            try:
                self.promoter(request, candidate_request, answer)
            except Exception:
                pass

        return ReuseDecision(
            approved=approved,
            reason=assessment.reason,
            local_verifier_attempts=assessment.local_verifier_attempts,
            local_verifier_ms=assessment.local_verifier_ms,
            local_policy_recommendation=assessment.recommended_reuse,
        )
