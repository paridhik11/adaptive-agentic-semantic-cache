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
        (r"\b(?:convert|change|turns?|transform|switch|cast|translate)\s+(.+?)\s+(?:into|to)\s+(.+?)(?:[?.!]|$)", False),
        (r"\b(?:make|create|obtain|build|construct|extract|derive)\s+(.+?)\s+from\s+(.+?)(?:[?.!]|$)", True),
        (r"\b(?:from\s+(.+?)\s+to\s+(.+?))(?:[?.!]|$)", False),
    ]
    stop = {"a", "an", "the", "existing", "given", "elements", "items", "in", "python"}
    for pattern, reverse in patterns:
        match = re.search(pattern, text.casefold())
        if match:
            parts = [frozenset(re.findall(r"\w+", x)) - stop for x in match.groups()]
            if all(parts):
                return tuple(reversed(parts)) if reverse else tuple(parts)
    return None


INVERSE_OPERATION_PAIRS = (
    # Encodings & Serialization
    (r"\b(?:encod\w*)\b", r"\b(?:decod\w*)\b"),
    (r"\b(?:serializ\w*)\b", r"\b(?:deserializ\w*)\b"),
    (r"\b(?:compress\w*)\b", r"\b(?:decompress\w*)\b"),
    (r"\b(?:encrypt\w*)\b", r"\b(?:decrypt\w*)\b"),
    (r"\b(?:pack\w*)\b", r"\b(?:unpack\w*)\b"),
    (r"\b(?:import\w*)\b", r"\b(?:export\w*)\b"),
    (r"\b(?:pars\w*)\b", r"\b(?:format\w*|stringif\w*)\b"),
    (r"\b(?:join\w*|combin\w*|merg\w*)\b", r"\b(?:split\w*|break\w*|partition\w*|divid\w*)\b"),
    (r"\b(?:zip\w*|archiv\w*)\b", r"\b(?:unzip\w*|extract\w*)\b"),
    # State & System
    (r"\b(?:install\w*)\b", r"\b(?:uninstall\w*)\b"),
    (r"\b(?:enable\w*|activat\w*)\b", r"\b(?:disable\w*|deactivat\w*)\b"),
    (r"\b(?:connect\w*)\b", r"\b(?:disconnect\w*)\b"),
    (r"\b(?:attach\w*)\b", r"\b(?:detach\w*)\b"),
    (r"\b(?:lock\w*)\b", r"\b(?:unlock\w*)\b"),
    (r"\b(?:open\w*)\b", r"\b(?:close\w*)\b"),
    (r"\b(?:start\w*|launch\w*)\b", r"\b(?:stop\w*|terminat\w*)\b"),
    (r"\b(?:mount\w*)\b", r"\b(?:unmount\w*)\b"),
    (r"\b(?:push\w*)\b", r"\b(?:pop\w*)\b"),
    # Sort order & Case
    (r"\b(?:ascend\w*|smallest to largest|lowest to highest)\b", r"\b(?:descend\w*|largest to smallest|highest to lowest)\b"),
    (r"\b(?:upper\w*|capital\w*)\b", r"\b(?:lower\w*)\b"),
    (r"\b(?:min\w*|minimum|smallest)\b", r"\b(?:max\w*|maximum|largest)\b"),
)


def inverse_operation_mismatch(original: str, incoming: str) -> bool:
    for pat_a, pat_b in INVERSE_OPERATION_PAIRS:
        has_a_orig = bool(re.search(pat_a, original, re.I))
        has_b_orig = bool(re.search(pat_b, original, re.I))
        has_a_inc = bool(re.search(pat_a, incoming, re.I))
        has_b_inc = bool(re.search(pat_b, incoming, re.I))
        if (has_a_orig and not has_b_orig and has_b_inc and not has_a_inc) or \
           (has_b_orig and not has_a_orig and has_a_inc and not has_b_inc):
            return True
    return False


def requires_executable_example(text: str) -> bool:
    patterns = [
        r"\b(?:executable|runnable|working|run\b|execut\w*)\s+(?:example|snippet|code|script|sample|demo)\b",
        r"\b(?:code\s+example|code\s+snippet|example\s+code|sample\s+code|demo\s+code)\b",
        r"\b(?:give|provide|show|include|write|with)\s+(?:an?\s+)?(?:executable|runnable|code\b|working\s+code)\b",
        r"\b(?:how\s+to\s+code|show\s+me\s+the\s+code|include\s+code|sample\s+snippet)\b",
    ]
    return any(re.search(pat, text, re.I) for pat in patterns)


def contains_executable_example(answer_text: str) -> bool:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return False
    # Fenced markdown code blocks ```...```
    if re.search(r"```(?:\w+)?\n[\s\S]+?\n```", answer_text):
        return True
    # Python interactive REPL sessions >>>
    if re.search(r"(?:^|\n)>>>\s+.+", answer_text):
        return True
    # Indented code blocks (4 spaces or tab) with python statements/keywords
    if re.search(r"(?:^|\n)(?:    |\t)(?:def |class |import |from |for |while |if |print\(|[a-zA-Z_]\w*\s*=).+", answer_text):
        return True
    # Inline code with function invocation or syntax: `func(...)` or `obj.method(...)`
    if re.search(r"`[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\([^`]*\)`", answer_text):
        return True
    return False


def answer_satisfies_requirements(incoming_query: str, answer_text: str) -> str | None:
    if requires_executable_example(incoming_query):
        if not contains_executable_example(answer_text):
            return "missing_explicit_executable_example"
    req_numbers = numbers(incoming_query)
    if req_numbers and not (req_numbers <= numbers(answer_text)):
        return "incoming_numeric_requirement_unmet_in_answer"
    return None


def constraints(text):
    patterns = {
        "negated_requirement": r"\b(?:not|no|never|without|avoid|excluding|except)\b",
        "recursive": r"\brecursiv\w*\b",
        "in_place": r"\bin[ -]place\b",
        "descending": r"\bdescending\b|\blargest to smallest\b|\bhighest to lowest\b",
        "ascending": r"\bascending\b|\bsmallest to largest\b|\blowest to highest\b",
        "preservation": r"\b(?:preserv\w*|retain\w*)\b",
        "reversal": r"\b(?:revers\w*|invers\w*|opposite)\b",
        "counting": r"\b(?:count\w*|frequency|tally)\b",
        "removal": r"\b(?:remov\w*|delet\w*|filter out|strip\w*|dedup\w*|unique|drop)\b",
        "reading": r"\b(?:read\w*|load\w*|fetch\w*|parse\w*)\b",
        "writing": r"\b(?:writ\w*|sav\w*|dump\w*|output)\b",
    }
    return frozenset(name for name, pattern in patterns.items() if re.search(pattern, text, re.I))


def requirement_guard(original, incoming):
    if numbers(original) != numbers(incoming):
        return "numeric_or_version_requirement_changed"
    if requires_executable_example(incoming) and not requires_executable_example(original):
        return "added_explicit_executable_example_requirement"
    if requires_executable_example(original) and not requires_executable_example(incoming):
        return "dropped_explicit_executable_example_requirement"
    if inverse_operation_mismatch(original, incoming):
        return "inverse_or_opposing_operation_mismatch"
    if constraints(incoming) - constraints(original):
        return "added_explicit_requirement"
    if constraints(original) - constraints(incoming):
        return "dropped_explicit_requirement"
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
