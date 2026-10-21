"""Development-only, source-bound semantic reuse checks. No serving enable flag.

Separate requirement preservation, source support, and evidence coverage. The
scores are uncalibrated model outputs, not correctness probabilities. A selected
source span is not proof that all material requirements have been identified.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import re
import threading
import time

from src.policy.encoders import normalize
from src.policy.response_cache import ScopedResponseCache
from src.policy.semantic_runtime import ReuseDecision, digest

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
POLICY_VERSION = "decomposed-source-diagnostic-v1"


class LocalEntailmentScorer:
    """Pinned local three-class classifier with verified label metadata."""

    def __init__(self, model=NLI_MODEL, revision=NLI_REVISION):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("immutable NLI revision required")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
            local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision,
            local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cpu").eval()
        self.labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if set(self.labels) != {0, 1, 2} or set(self.labels.values()) != {"entailment", "neutral", "contradiction"}:
            raise ValueError("unknown NLI label mapping")
        self.identity = model + "@" + revision + ":" + digest(self.labels)
        self._lock = threading.Lock()

    def score(self, premise, hypothesis):
        encoded = self.tokenizer(premise, hypothesis, truncation=False, return_tensors="pt")
        count = encoded["input_ids"].shape[-1]
        if count > min(self.tokenizer.model_max_length, 512):
            raise ValueError("NLI input too long; no silent truncation")
        with self._lock, self.torch.inference_mode():
            logits = self.model(**encoded).logits[0].float()
            probabilities = self.torch.softmax(logits, dim=-1).tolist()
        scores = {self.labels[i]: float(p) for i, p in enumerate(probabilities)}
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
            raise ValueError("invalid NLI output")
        return {**scores, "input_tokens": count}


def numbers(text):
    return frozenset(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)*(?!\w)", text))


def call_names(text):
    return frozenset(re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", text))


def conversion_slots(text):
    # Generic grammatical slots, not a registry of question pairs or entities.
    patterns = [
        (r"\b(?:convert|change|turns?)\s+(.+?)\s+(?:into|to)\s+(.+?)(?:[?.!]|$)", False),
        (r"\b(?:make|create|obtain)\s+(.+?)\s+from\s+(.+?)(?:[?.!]|$)", True),
    ]
    stop = {"a", "an", "the", "existing", "given", "elements", "items"}
    for pattern, reverse in patterns:
        match = re.search(pattern, text.casefold())
        if match:
            parts = [frozenset(re.findall(r"\w+", x)) - stop for x in match.groups()]
            if all(parts):
                return tuple(reversed(parts)) if reverse else tuple(parts)
    return None


def constraints(text):
    patterns = {
        "negated_requirement": r"\b(?:not|no|never|without|avoid|excluding|except)\b",
        "recursive": r"\brecursiv\w*\b",
        "in_place": r"\bin[ -]place\b",
        "descending": r"\bdescending\b|\blargest to smallest\b|\bhighest to lowest\b",
        "ascending": r"\bascending\b|\bsmallest to largest\b|\blowest to highest\b",
        "preservation": r"\b(?:preserv\w*|retain\w*)\b",
    }
    return frozenset(name for name, pattern in patterns.items() if re.search(pattern, text, re.I))


def requirement_guard(original, incoming):
    if numbers(original) != numbers(incoming):
        return "numeric_or_version_requirement_changed"
    if constraints(incoming) - constraints(original):
        return "added_explicit_requirement"
    old, new = conversion_slots(original), conversion_slots(incoming)
    if old is not None and new is not None and (not old[0] & new[0] or not old[1] & new[1]):
        return "conversion_direction_or_entity_changed"
    return None


def task_statement(question):
    question = question.strip().rstrip("?.!")
    match = re.match(r"How (?:do|can|should) I (.+)", question, re.I)
    return "The requested task is to " + match.group(1) + "." if match else "The requested information is: " + question + "."


def split_claims(text):
    # Do not split decimal values, qualified API names or punctuation without
    # whitespace. Complex compound claims remain whole and can force abstention.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int


class EvidenceIndex:
    """Trusted verbatim source spans, selected using the ORIGINAL request only.

    The incoming question and cached answer cannot steer evidence selection.
    A span registry contains source text, never approved request pairs.
    """

    def __init__(self, *, context, spans, encoder, threshold=.50, margin=.03):
        if not isinstance(context, str) or not context.strip() or not spans or len(spans) > 256:
            raise ValueError("bounded nonempty source spans required")
        if (type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1
                or type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= 2):
            raise ValueError("invalid evidence retrieval thresholds")
        if not isinstance(getattr(encoder, "identity", None), str) or not encoder.identity:
            raise ValueError("identified encoder required")
        rows, seen = [], set()
        for span in spans:
            if (not isinstance(span, SourceSpan) or type(span.start) is not int or type(span.end) is not int
                    or not 0 <= span.start < span.end <= len(context) or (span.start, span.end) in seen):
                raise ValueError("invalid or duplicate source span")
            seen.add((span.start, span.end))
            text = context[span.start:span.end]
            if not text.strip():
                raise ValueError("empty source span")
            vector = tuple(normalize(encoder.encode(text)))
            if rows and len(vector) != len(rows[0][2]):
                raise ValueError("embedding dimension changed")
            rows.append((span, text, vector))
        self.context, self.rows, self.encoder = context, tuple(rows), encoder
        self.threshold, self.margin = threshold, margin
        self._lock = threading.Lock()
        self.identity = digest({"context": context, "spans": [(s.start, s.end) for s, _, _ in rows],
                                "encoder": encoder.identity, "threshold": threshold, "margin": margin})
        self.initial_embedding_calls = len(rows)

    def select(self, original):
        with self._lock:
            vector = tuple(normalize(self.encoder.encode(original)))
        if len(vector) != len(self.rows[0][2]):
            raise ValueError("embedding dimension changed")
        ranked = sorted(((sum(a*b for a,b in zip(vector, row[2])), i) for i,row in enumerate(self.rows)), reverse=True)
        score, index = ranked[0]
        if score < self.threshold or (len(ranked) > 1 and score-ranked[1][0] < self.margin):
            return None
        span, text, _ = self.rows[index]
        return {"text": text, "start": span.start, "end": span.end, "cosine": score,
                "text_sha256": digest(text), "context_sha256": digest(self.context)}


@dataclass(frozen=True)
class DecomposedAssessment:
    recommended_reuse: bool
    reason: str
    scores: dict
    evidence: dict | None
    work: dict
    binding: dict


class DecomposedDiagnosticPolicy:
    """Strict conjunctive diagnostic; serving interface always abstains."""

    def __init__(self, *, evidence_index, scorer, threshold=.95):
        if not isinstance(evidence_index, EvidenceIndex):
            raise ValueError("trusted evidence index required")
        if not callable(getattr(scorer, "score", None)) or not isinstance(getattr(scorer, "identity", None), str) or not scorer.identity:
            raise ValueError("identified local NLI scorer required")
        if type(threshold) not in (float, int) or not math.isfinite(threshold) or not .5 < threshold <= 1:
            raise ValueError("invalid entailment threshold")
        self.index, self.scorer, self.threshold = evidence_index, scorer, threshold
        self.identity = POLICY_VERSION + ":" + digest({"evidence": evidence_index.identity,
            "scorer": scorer.identity, "threshold": threshold})

    def assess(self, request, candidate_request, answer, *, collect_all_scores=False):
        """Development option observes later checks; never reverses a rejection."""
        if type(collect_all_scores) is not bool:
            raise ValueError("collect_all_scores must be boolean")
        started = time.perf_counter()
        scores, evidence, embeddings, nli = {}, None, 0, 0
        def result(approved, reason):
            return DecomposedAssessment(approved, reason, copy.deepcopy(scores), copy.deepcopy(evidence),
                {"embedding_attempts": embeddings, "nli_attempts": nli,
                 "elapsed_ms": (time.perf_counter()-started)*1000},
                {"incoming_request_sha256": digest(request), "candidate_request_sha256": digest(candidate_request),
                 "candidate_response_sha256": digest(answer), "context_sha256": digest(self.index.context),
                 "policy_identity": self.identity})
        for body in (request, candidate_request):
            if not isinstance(body, dict):
                return result(False, "unsupported_context")
            messages = body.get("messages")
            if (not isinstance(messages, list) or len(messages) != 2 or
                    messages[0] != {"role": "system", "content": self.index.context} or
                    not isinstance(messages[1], dict) or messages[1].get("role") != "user" or
                    not isinstance(messages[1].get("content"), str) or not messages[1]["content"].strip() or
                    type(body.get("n", 1)) is not int or body.get("n", 1) != 1):
                return result(False, "unsupported_context")
        def namespace(body):
            copy_body = copy.deepcopy(body)
            copy_body["messages"][-1]["content"] = "<query>"
            return copy_body
        if namespace(request) != namespace(candidate_request):
            return result(False, "request_scope_changed")
        if not isinstance(answer, dict) or not ScopedResponseCache._admissible(answer, candidate_request):
            return result(False, "incomplete_or_unsupported_answer")
        original, incoming = [body["messages"][-1]["content"] for body in (candidate_request, request)]
        guard = requirement_guard(original, incoming)
        if guard:
            return result(False, guard)
        content = answer["choices"][0]["message"]["content"]
        claims = split_claims(content)
        if len(claims) > 8:
            return result(False, "too_many_claims")
        try:
            embeddings += 1
            evidence = self.index.select(original)
            if evidence is None:
                return result(False, "evidence_retrieval_abstained")
            if not numbers(evidence["text"]) <= numbers(content):
                return result(False, "required_source_quantity_missing")
            if not call_names(evidence["text"]) <= call_names(content):
                return result(False, "required_source_api_missing")
            checks = [("original_to_incoming", task_statement(original), task_statement(incoming)),
                      ("incoming_to_original", task_statement(incoming), task_statement(original)),
                      ("evidence_coverage", content, evidence["text"])]
            checks += [("claim_support_"+str(i), self.index.context, claim) for i,claim in enumerate(claims)]
            first_failure = None
            for name, premise, hypothesis in checks:
                nli += 1
                values = self.scorer.score(premise, hypothesis)
                if not isinstance(values, dict):
                    raise ValueError("invalid NLI scores")
                probabilities = [values[k] for k in ("entailment", "neutral", "contradiction")]
                if (any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                        or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)):
                    raise ValueError("invalid NLI distribution")
                scores[name] = dict(values)
                # Short circuit saves work; remaining checks are missing, not successes.
                if values["entailment"] < self.threshold:
                    first_failure = first_failure or name + "_not_entailed"
                    if not collect_all_scores:
                        return result(False, first_failure)
            if first_failure:
                return result(False, first_failure)
        except Exception:
            return result(False, "local_check_failed")
        return result(True, "diagnostic_positive")

    def __call__(self, request, candidate_request, answer, *, binding, provider):
        assessment = self.assess(request, candidate_request, answer)
        return ReuseDecision(False, "diagnostic_only:"+assessment.reason,
            local_verifier_attempts=assessment.work["embedding_attempts"]+assessment.work["nli_attempts"],
            local_verifier_ms=assessment.work["elapsed_ms"],
            local_policy_recommendation=assessment.recommended_reuse)
