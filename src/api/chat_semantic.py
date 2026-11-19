"""Local semantic reuse for short standalone definitions in live chat."""
from collections import OrderedDict
import json
import re
import sqlite3
import time

from src.policy.decomposed_applicability import answer_satisfies_requirements, requirement_guard
from src.policy.encoders import normalize
from src.policy.hybrid_cache import request_boundary
from src.policy.response_cache import ResponseCacheResult, ScopedResponseCache, _serialize
from src.policy.semantic_runtime import digest

SIMILARITY_MIN = .84
CANDIDATE_MARGIN = .035
PAIR_MIN = .80
ANSWER_MIN = .85
MAX_CANDIDATES = 128
MAX_BYTES = 4 * 1024 * 1024
_DEFINITION = re.compile(
    r"(?:what (?:is|are) |define |explain (?:the (?:meaning|concept) of )?)(?P<subject>.+)|"
    r"what does (?P<meaning>.+) mean", re.IGNORECASE,
)
_QUALIFIER = re.compile(r"\b(?:with|without|using|in|versus|vs|between|except|include|example|examples|steps|compare)\b", re.I)


def definition_subject(text):
    """Restrict semantic reuse to short definitions with no extra instructions."""
    match = _DEFINITION.fullmatch(text.strip().rstrip("?.!").strip())
    if not match:
        return None
    subject = (match["subject"] or match["meaning"]).casefold()
    if _QUALIFIER.search(subject) or not re.fullmatch(r"[a-z][a-z0-9 -]*", subject):
        return None
    words = subject.split()
    if words and words[0] in ("a", "an", "the"):
        words = words[1:]
    return tuple(words) if 0 < len(words) <= 5 else None


def load_chat_models():
    from src.policy.encoders import DenseEncoder
    from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION, LocalPairScorer, LocalPassageScorer

    return DenseEncoder(ENCODER_MODEL, ENCODER_REVISION), LocalPairScorer(), LocalPassageScorer()


class ChatSemanticReuse:
    """Resolve guarded matches through the existing exact cache and source-bound index."""

    def __init__(self, exact, storage, models):
        self.exact, self.storage = exact, storage
        self.encoder, self.pair, self.answer = models
        self.warmed = OrderedDict()

    def process(self, body, *, tenant_id, provider, bypass=False):
        question = body["messages"][-1]["content"]
        subject = definition_subject(question) if len(body["messages"]) == 2 and not bypass else None
        if not subject:
            return self.exact.process(body, tenant_id=tenant_id, provider=provider, bypass=bypass), None, None

        namespace = digest({
            "tenant": tenant_id, "request": request_boundary(body),
            "models": [model.identity for model in (self.encoder, self.pair, self.answer)],
            "contract": "live-definition-reuse-v1",
        })
        source_key = self.exact.request_key(body, tenant_id)
        canonical = "what is a " + " ".join(subject)
        subject_terms = frozenset(subject)
        started = time.perf_counter()
        epoch, vector, match, decision = None, None, None, None

        def resolve(checked):
            nonlocal epoch, vector, match, decision
            try:
                epoch = self.storage.invalidation_epoch
                vector = normalize(self.encoder.encode(canonical))
                ranked = []
                for record in self.storage.recover_candidates(namespace, tenant_id, epoch):
                    identity, key, token, candidate_json, vector_json, response_json, _ = record
                    original = json.loads(candidate_json)
                    if self.exact.request_key(original, tenant_id) != key:
                        continue
                    saved_question = original["messages"][-1]["content"]
                    other_subject = definition_subject(saved_question)
                    if not other_subject or requirement_guard(saved_question, question):
                        continue
                    other_terms = frozenset(other_subject)
                    if subject_terms < other_terms or other_terms < subject_terms:
                        continue
                    existing = normalize(json.loads(vector_json))
                    similarity = sum(a * b for a, b in zip(vector, existing, strict=True))
                    ranked.append((similarity, identity, key, token, saved_question, other_subject, response_json))
                ranked.sort(key=lambda item: item[0], reverse=True)
                if not ranked or ranked[0][0] < SIMILARITY_MIN:
                    decision = "no_close_candidate"
                    return None
                if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < CANDIDATE_MARGIN:
                    decision = "ambiguous_candidates"
                    return None
                similarity, identity, key, token, saved_question, other_subject, response_json = ranked[0]
                answer = json.loads(response_json)
                if not self.exact._admissible(answer, checked):
                    return None
                content = answer["choices"][0]["message"]["content"]
                if answer_satisfies_requirements(question, content):
                    decision = "answer_not_verified"
                    return None
                canonical_source = "what is a " + " ".join(other_subject)
                pair = min(self.pair.score(canonical, canonical_source)["pair_score"],
                           self.pair.score(canonical_source, canonical)["pair_score"])
                if not pair >= PAIR_MIN:
                    decision = "meaning_not_verified"
                    return None
                if not self.answer.score(question, content)["probability"] >= ANSWER_MIN:
                    decision = "answer_not_verified"
                    return None
                if not self.storage.candidate_current(identity, key, token, namespace, tenant_id, epoch):
                    return None
                decision = "verified"
                match = {"matched_question": saved_question, "similarity": round(similarity, 4)}
                return ResponseCacheResult(answer, "HIT", "guarded_semantic_hit", 0, False,
                                           (time.perf_counter() - started) * 1_000)
            except Exception:
                # Any local model, shape or storage failure takes the fresh path.
                decision = "local_check_unavailable"
                return None

        result = self.exact.process(body, tenant_id=tenant_id, provider=provider, on_miss=resolve)
        warm_key = namespace, source_key
        if result.stored or (result.reason == "scoped_exact_hit" and warm_key not in self.warmed):
            try:
                # Existing exact answers can seed semantic reuse on their first
                # visit. Admission is tied to the original source and expiry.
                if epoch is None:
                    epoch = self.storage.invalidation_epoch
                if vector is None:
                    vector = normalize(self.encoder.encode(canonical))
                if self.storage.admit_candidate(
                    namespace=namespace, source_key=source_key, tenant=tenant_id, payload=_serialize(result.response),
                    epoch=epoch, request=_serialize(body), vector=vector, max_entries=MAX_CANDIDATES, max_bytes=MAX_BYTES,
                ):
                    self.warmed[warm_key] = None
                    self.warmed.move_to_end(warm_key)
                    if len(self.warmed) > MAX_CANDIDATES:
                        self.warmed.popitem(last=False)
            except Exception:
                pass  # A completed answer remains usable even if indexing fails.
        return ResponseCacheResult(result.response, result.status, result.reason, result.provider_calls,
                                   result.stored, (time.perf_counter() - started) * 1_000), match, decision
