"""Local reuse for stable requests in live chat."""
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
from functools import lru_cache
import json
from math import isfinite
import re
import sqlite3
from threading import Lock
import time

from src.api.chat_intents import direct_question, question_intent, reusable_fact_answer, wording_key
from src.api.chat_keywords import keyword_signature
from src.api.chat_reuse_checks import semantic_answer_reason
from src.policy.decomposed_applicability import requirement_guard
from src.policy.encoders import normalize
from src.policy.hybrid_cache import request_boundary
from src.policy.response_cache import ResponseCacheResult, ScopedResponseCache, _serialize
from src.policy.semantic_runtime import digest

SIMILARITY_MIN = .84
CANDIDATE_MARGIN = .035
PAIR_MIN = .80
ANSWER_MIN = .85
VERIFY_TOP_K = 3
MAX_CANDIDATES = 128
MAX_BYTES = 4 * 1024 * 1024
_DEFINITION = re.compile(
    r"what does (?P<meaning>.+) mean|explain what (?P<explained>.+) means|"
    r"(?:what (?:is meant by|is|are) |what['’]s |what do (?:you|u) mean by |"
    r"define |explain (?:the (?:meaning|concept) of )?)(?P<subject>.+)",
    re.IGNORECASE,
)
_QUALIFIER = re.compile(
    r"\b(?:with|without|using|in|for|per|under|within|after|before|at|and|or|not|"
    r"versus|vs|between|except|include|example|examples|steps|compare)\b", re.I)
_COMPARISON = re.compile(
    r"(?:compare|(?:what is|what['’]s|explain) the difference between|difference between) (.+?) and (.+)"
    r"|(.+?) (?:vs\.?|versus) (.+)", re.I)
_DIRECTIONAL_COMPARISON = re.compile(
    r"(?:how (?:does|do) (.+?) differ from (.+)|what distinguishes (.+?) from (.+)"
    r"|how (?:is|are) (.+?) different from (.+))", re.I)
_ANSWER_DEFINITION = re.compile(
    r"([a-z][a-z0-9 -]{0,99}?) (?:is|are|means|refers to|defines|describes|measures|quantifies|represents)\b(?!\s+(?:not|unclear|unknown)\b)", re.I)

_ALIAS_DEFINITION = re.compile(
    r"([a-z][a-z0-9 -]{0,80}?)\s*(?:\(\s*(?:short for|also known as) ([a-z][a-z0-9 -]{0,80})\)"
    r"|, (?:short for|also known as) ([a-z][a-z0-9 -]{0,80}),) "
    r"(?:is|are|refers to)\b(?!\s+(?:not|unclear|unknown)\b)", re.I)


def definition_subject(text, *, shorthand=False):
    """Restrict semantic reuse to short definitions with no extra instructions."""
    text = direct_question(text)
    statement = text.rstrip("?.!").strip()
    match = _DEFINITION.fullmatch(statement)
    if match:
        return _topic_words(match["subject"] or match["meaning"] or match["explained"])
    if shorthand and text.endswith("?"):
        return _topic_words(statement)
    return None


def _topic_words(subject):
    subject = subject.casefold()
    if _QUALIFIER.search(subject) or not re.fullmatch(r"[a-z][a-z0-9 -]{0,99}", subject):
        return None
    words = subject.split()
    if words and words[0] in ("a", "an", "the"):
        words = words[1:]
    return tuple(words) if 0 < len(words) <= 5 else None


def comparison_subjects(question):
    text = direct_question(question).rstrip("?.!").strip()
    match = _COMPARISON.fullmatch(text) or _DIRECTIONAL_COMPARISON.fullmatch(text)
    if not match:
        return None
    left, right = map(_topic_words, filter(None, match.groups()))
    return (left, right) if left and right and left != right else None


def _covers_comparison(subjects, answer):
    text = " ".join(answer.casefold().translate(str.maketrans("", "", "*_`")).split())
    return all(re.search(r"(?<![\w-])" + re.escape(" ".join(topic)) + r"(?![\w-])", text)
               for topic in subjects)


@lru_cache(maxsize=1)
def _spelling():
    from spellchecker import SpellChecker

    return SpellChecker(language="en")


# Some plural heads commonly denote a different concept (for example,
# safety glasses versus safety glass). Do not fold those forms together.
_NONINTERCHANGEABLE_PLURALS = frozenset(
    "arm content custom glass good ground manner mean premise spectacle work".split()
)
_IRREGULAR_PLURALS = {"analysis": "analyses", "criterion": "criteria", "index": "indices", "matrix": "matrices"}


def _same_head_number_variant(left, right):
    if left == right or not (left.isascii() and left.isalpha() and right.isascii() and right.isalpha()):
        return False
    if _IRREGULAR_PLURALS.get(left) == right or _IRREGULAR_PLURALS.get(right) == left:
        return True
    singular, plural = sorted((left, right), key=len)
    if len(singular) < 3 or singular in _NONINTERCHANGEABLE_PLURALS:
        return False
    if singular.endswith("y") and singular[-2] not in "aeiou":
        return plural == singular[:-1] + "ies"
    suffix = "es" if singular.endswith(("s", "x", "z", "ch", "sh")) else "s"
    return plural == singular + suffix


def _topic_match(typed, confirmed, *, source_topic=()):
    if typed == confirmed:
        return 2
    if len(typed) != len(confirmed):
        return 0
    # Treat number as a property of the last noun. All modifiers remain in
    # order, and at most one remaining word can require a spelling repair.
    typed_words = typed
    if _same_head_number_variant(typed[-1], confirmed[-1]):
        typed_words = (*typed[:-1], confirmed[-1])
    differences = [(i, left, right) for i, (left, right) in enumerate(zip(typed_words, confirmed)) if left != right]
    if not differences:
        return 1
    if len(differences) != 1 or len(typed) < 2:
        return 0
    index, left, right = differences[0]
    if (min(len(left), len(right)) < 8 or left[:5] != right[:5] or left[-1] != right[-1]
            or not left.isalpha() or not right.isalpha()):
        return 0
    edits = sum(max(j - i, n - m) for op, i, j, m, n in
                SequenceMatcher(None, left, right, autojunk=False).get_opcodes() if op != "equal")
    if edits > 2:
        return 0
    try:
        known = _spelling().known((left, right))
    except (ImportError, OSError):
        return 0
    if known == {right}:
        return 1
    # Technical terms can be absent from the dictionary. The answer can
    # corroborate one substituted letter when the user keeps the original
    # spelling in the same position. Do not conflate two new near-neighbors.
    if (not known and len(left) == len(right) >= 11 and edits == 1
            and len(source_topic) == len(typed) and source_topic[index] == left):
        return 1
    return 0


def confirmed_definition(question, answer):
    """Use only a topic explicitly defined at the start of the stored answer."""
    subject = definition_subject(question, shorthand=True)
    if subject is None:
        return None
    opening = answer.lstrip()[:200].translate(str.maketrans("", "", "*_`"))
    match = _ANSWER_DEFINITION.match(opening)
    confirmed = _topic_words(match[1]) if match else None
    return confirmed if confirmed and _topic_match(subject, confirmed, source_topic=subject) else None


def definition_alias(answer):
    """Recognize an explicit equivalence in the opening of a definition."""
    opening = answer.lstrip()[:240].translate(str.maketrans("", "", "*_`"))
    match = _ALIAS_DEFINITION.match(opening)
    if not match:
        return None
    head = _topic_words(match[1])
    alias_text = (match[2] or match[3]).strip()
    alias = _topic_words(alias_text)
    # A bare initialism can have several expansions. Only serve the named sense.
    if not head or not alias or (len(alias) == 1 and alias_text.isupper()):
        return None
    return head, alias


def _lookup_tag(body, signature):
    return digest({"contract": "ordered-keywords-v1", "boundary": request_boundary(body), "keywords": signature})


def _usable_keyword_answer(content):
    return not re.match(
        r"(?:i (?:do not know|don't know|cannot|can't|am not sure)|please clarify|could you clarify)\b",
        content.lstrip(), re.I,
    ) and not (content.rstrip().endswith("?") and len(content.split()) < 14)


def load_chat_models():
    from src.policy.encoders import DenseEncoder
    from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION, LocalPairScorer, LocalPassageScorer

    return DenseEncoder(ENCODER_MODEL, ENCODER_REVISION), LocalPairScorer(), LocalPassageScorer()


class ChatResponseCache(ScopedResponseCache):
    """Require answer evidence before admitting ambiguous topic shorthand."""

    @staticmethod
    def _admissible(response, request):
        if not ScopedResponseCache._admissible(response, request):
            return False
        question = request["messages"][-1]["content"]
        content = response["choices"][0]["message"]["content"]
        intent = question_intent(question)
        if intent:
            return bool(intent.subject) and reusable_fact_answer(content)
        comparison = comparison_subjects(question)
        if comparison:
            return question.lower().startswith(("compare ", "what ", "explain ")) or _covers_comparison(
                comparison, content)
        if keyword_signature(question):
            return _usable_keyword_answer(content)
        if not definition_subject(question) and (subject := definition_subject(question, shorthand=True)):
            alias = definition_alias(content)
            return bool((alias and subject in alias) or confirmed_definition(question, content))
        return True


class ChatSemanticReuse:
    """Resolve guarded matches through the existing exact cache and source-bound index."""

    def __init__(self, exact, storage, models=None):
        self.exact, self.storage = exact, storage
        self.models = models
        self.warmed = OrderedDict()
        self._warm_lock = Lock()
        self._model_lock = Lock()

    def backfill_keywords(self, tenant_id):
        def tag_request(source_key, raw_request):
            try:
                body = json.loads(raw_request)
                if len(body["messages"]) == 2 and self.exact.request_key(body, tenant_id) == source_key:
                    signature = keyword_signature(body["messages"][-1]["content"])
                    if signature:
                        return _lookup_tag(body, signature)
            except (ValueError, KeyError, IndexError, TypeError):
                pass
            return None

        try:
            self.storage.backfill_candidate_tags(tenant_id, tag_request, max_bytes=MAX_BYTES)
        except sqlite3.Error:
            pass  # Old records can be indexed when their exact questions are asked again.

    def _lookup_keywords(self, checked, tenant_id, signature, tag, epoch, rejections):
        question = checked["messages"][-1]["content"]
        for identity, key, token, raw_request, raw_answer, namespace in self.storage.recover_tagged_candidates(
            tag, tenant_id, epoch,
        ):
            original = json.loads(raw_request)
            if (self.exact.request_key(original, tenant_id) != key
                    or request_boundary(original) != request_boundary(checked)
                    or self.exact._eligibility_reason(original) is not None):
                continue
            saved = original["messages"][-1]["content"]
            # The signature preserves the ordered content and requirements.
            # Raw wording guards also match verbs in grammatical framing,
            # such as "written" in a request for the same formula.
            if keyword_signature(saved) != signature:
                continue
            answer = json.loads(raw_answer)
            if not self.exact._admissible(answer, checked):
                continue
            content = answer["choices"][0]["message"]["content"]
            rejection = semantic_answer_reason(saved, question, content, signature)
            if rejection or not _usable_keyword_answer(content):
                rejections[rejection or "unusable_answer"] += 1
                continue
            if self.storage.candidate_current(identity, key, token, namespace, tenant_id, epoch):
                return saved, answer
        return None

    def process(self, body, *, tenant_id, provider, bypass=False, diagnostics=None, mode="guarded"):
        if diagnostics is None:
            diagnostics = {}
        rejections = Counter()
        diagnostics.update(state="skipped", rejections=rejections, candidates=[],
                           embedding_ms=0.0, cross_encoder_ms=0.0, verification_ms=0.0)
        question = body["messages"][-1]["content"]
        if len(body["messages"]) != 2 or bypass or mode == "exact":
            return self.exact.process(body, tenant_id=tenant_id, provider=provider, bypass=bypass), None, None
        intent = question_intent(question)
        comparison = None if intent else comparison_subjects(question)
        subject = None if intent or comparison else definition_subject(question, shorthand=True)
        wording = None if intent else wording_key(question)
        signature = keyword_signature(question)
        tag = _lookup_tag(body, signature) if signature else None
        if not (subject or comparison or intent or wording or signature):
            return self.exact.process(body, tenant_id=tenant_id, provider=provider), None, None

        boundary = {"tenant": tenant_id, "request": request_boundary(body)}
        if intent:
            # Filter by the fact before scanning candidates. Precision is
            # checked against the actual cached answer after retrieval.
            boundary.update(contract="live-fact-reuse-v1", fact=[intent.relation, intent.subject, intent.rank])
        else:
            boundary.update(contract="live-definition-reuse-v1", models=[model.identity for model in self.models or ()])
        namespace = digest(boundary)
        source_key = self.exact.request_key(body, tenant_id)
        scope = request_boundary(body)
        explicit = definition_subject(question) if subject else None
        canonical = "what is a " + " ".join(subject or ())
        subject_terms = frozenset(subject or ())
        started = time.perf_counter()
        epoch, vector, match, decision = None, None, None, None

        def assess(checked):
            nonlocal epoch, vector, match, decision
            try:
                epoch = self.storage.invalidation_epoch
                if tag and not comparison:
                    keyword_hit = self._lookup_keywords(checked, tenant_id, signature, tag, epoch, rejections)
                    if keyword_hit:
                        saved, answer = keyword_hit
                        decision = "keyword_verified"
                        match = {"matched_question": saved, "similarity": None}
                        return ResponseCacheResult(answer, "HIT", "ordered_keyword_hit", 0, False,
                                                   (time.perf_counter() - started) * 1_000)
                ranked = []
                matches_by_topic = {}
                for record in self.storage.recover_candidates(namespace, tenant_id, epoch):
                    identity, key, token, candidate_json, vector_json, response_json, _ = record
                    original = json.loads(candidate_json)
                    if (self.exact.request_key(original, tenant_id) != key
                            or request_boundary(original) != scope):
                        rejections["source_binding_mismatch"] += 1
                        continue
                    if self.exact._eligibility_reason(original) is not None:
                        rejections["ineligible_source"] += 1
                        continue
                    saved_question = original["messages"][-1]["content"]
                    guard = requirement_guard(saved_question, question) if intent is None else None
                    if guard:
                        rejections[guard] += 1
                        continue
                    answer = json.loads(response_json)
                    if not self.exact._admissible(answer, checked):
                        rejections["inadmissible_answer"] += 1
                        continue
                    content = answer["choices"][0]["message"]["content"]
                    failure = semantic_answer_reason(saved_question, question, content, signature)
                    if failure:
                        rejections[failure] += 1
                        continue
                    equal_intent = intent is not None and intent.accepts(question_intent(saved_question), content)
                    if intent is not None and not equal_intent:
                        continue
                    if equal_intent:
                        matches_by_topic[intent] = (2, identity, key, token, saved_question, answer)
                        continue
                    if comparison:
                        if (comparison_subjects(saved_question) == comparison
                                and _covers_comparison(comparison, content)):
                            matches_by_topic[comparison] = (2, identity, key, token, saved_question, answer)
                        continue
                    if wording and wording_key(saved_question) == wording:
                        matches_by_topic[("wording", wording)] = (3, identity, key, token, saved_question, answer)
                        continue
                    if not subject:
                        continue
                    if signature and signature[0] == "definition":
                        source_signature = keyword_signature(saved_question)
                        if source_signature and source_signature[0] == "definition":
                            before, after = source_signature[1], signature[1]
                            if before != after and [w.casefold() for w in before] == [w.casefold() for w in after]:
                                continue  # Acronyms and mixed-case identifiers are distinct keywords.
                    alias = definition_alias(content)
                    if alias and alias[1] == subject and definition_subject(saved_question, shorthand=True) == alias[0]:
                        matches_by_topic[alias[0]] = (2, identity, key, token, saved_question, answer)
                        continue
                    confirmed = confirmed_definition(saved_question, content)
                    if confirmed:
                        source_topic = definition_subject(saved_question, shorthand=True)
                        strength = _topic_match(subject, confirmed, source_topic=source_topic)
                        if strength:
                            matches_by_topic[confirmed] = (strength, identity, key, token, saved_question, answer)
                    other_subject = definition_subject(saved_question)
                    if not self.models or not explicit or not other_subject:
                        continue
                    if subject != other_subject and sorted(subject) == sorted(other_subject):
                        continue  # The same terms in a different order can name a different concept.
                    other_terms = frozenset(other_subject)
                    if subject_terms < other_terms or other_terms < subject_terms:
                        continue
                    ranked.append((vector_json, identity, key, token, saved_question, other_subject, answer))
                if matches_by_topic:
                    strongest = max(item[0] for item in matches_by_topic.values())
                    matches = [item for item in matches_by_topic.values() if item[0] == strongest]
                    if len(matches) != 1:
                        decision = "ambiguous_candidates"
                        return None
                    strength, identity, key, token, saved_question, answer = matches[0]
                    if self.storage.candidate_current(identity, key, token, namespace, tenant_id, epoch):
                        kind = "intent" if intent else "comparison" if comparison else "definition"
                        if strength == 3:
                            kind = "wording"
                        decision = f"{kind}_verified"
                        match = {"matched_question": saved_question, "similarity": None}
                        if kind == "definition":
                            saved_topic = definition_subject(saved_question, shorthand=True)
                            if saved_topic and _same_head_number_variant(saved_topic[-1], subject[-1]):
                                decision = "definition_number_verified"
                                match["lexical_similarity"] = round(SequenceMatcher(
                                    None, " ".join(saved_topic), " ".join(subject), autojunk=False
                                ).ratio(), 4)
                        reason = f"{kind}_reuse_hit"
                        return ResponseCacheResult(answer, "HIT", reason, 0, False,
                                                   (time.perf_counter() - started) * 1_000)
                    return None
                if not ranked:
                    return None
                encoder, pair_scorer, answer_scorer = self.models
                with self._model_lock:
                    clock = time.perf_counter()
                    vector = normalize(encoder.encode(canonical))
                    diagnostics["embedding_ms"] += (time.perf_counter() - clock) * 1_000
                scored = []
                for vector_json, identity, key, token, saved_question, other_subject, answer in ranked:
                    existing = normalize(json.loads(vector_json))
                    similarity = sum(a * b for a, b in zip(vector, existing, strict=True))
                    scored.append((similarity, identity, key, token, saved_question, other_subject, answer))
                scored.sort(key=lambda item: item[0], reverse=True)
                verified = []
                for rank, candidate in enumerate(scored[:VERIFY_TOP_K], 1):
                    similarity, identity, key, token, saved_question, other_subject, answer = candidate
                    if verified and verified[0][0] - similarity >= CANDIDATE_MARGIN:
                        break
                    trace = {"rank": rank, "candidate": identity, "cosine": round(similarity, 6),
                             "scope": "matched", "requirements": "passed", "answer_sufficiency_label": None}
                    diagnostics["candidates"].append(trace)
                    if similarity < SIMILARITY_MIN:
                        trace["result"] = "below_cosine_threshold"
                        rejections["below_cosine_threshold"] += 1
                        continue
                    canonical_source = "what is a " + " ".join(other_subject)
                    content = answer["choices"][0]["message"]["content"]
                    with self._model_lock:
                        clock = time.perf_counter()
                        try:
                            forward = pair_scorer.score(canonical, canonical_source)["pair_score"]
                            reverse = pair_scorer.score(canonical_source, canonical)["pair_score"]
                            if not (isfinite(forward) and isfinite(reverse)):
                                raise ValueError("non-finite question score")
                            trace["pair_scores"] = [forward, reverse]
                            if not (forward >= PAIR_MIN and reverse >= PAIR_MIN):
                                decision = trace["result"] = "meaning_not_verified"
                                rejections[decision] += 1
                                continue
                            score = answer_scorer.score(question, content)["probability"]
                            if not isfinite(score):
                                raise ValueError("non-finite answer score")
                            trace["answer_score"] = score
                            if not score >= ANSWER_MIN:
                                decision = trace["result"] = "answer_not_verified"
                                rejections[decision] += 1
                                continue
                        finally:
                            diagnostics["cross_encoder_ms"] += (time.perf_counter() - clock) * 1_000
                    trace["result"] = "verified_candidate"
                    verified.append(candidate)
                if not verified:
                    decision = decision or "no_close_candidate"
                    return None
                # Ambiguity is a reason to abstain when both candidates independently
                # pass verification. A rejected answer does not block a qualified one.
                if len(verified) > 1 and verified[0][0] - verified[1][0] < CANDIDATE_MARGIN:
                    decision = "ambiguous_candidates"
                    rejections[decision] += 1
                    return None
                similarity, identity, key, token, saved_question, _, answer = verified[0]
                # A capped search cannot rule out an unexamined close candidate.
                if len(scored) > VERIFY_TOP_K and similarity - scored[VERIFY_TOP_K][0] < CANDIDATE_MARGIN:
                    decision = "ambiguous_candidates"
                    rejections["unexamined_close_candidate"] += 1
                    return None
                if not self.storage.candidate_current(identity, key, token, namespace, tenant_id, epoch):
                    decision = "source_expired_or_revoked"
                    rejections[decision] += 1
                    return None
                decision = "verified"
                match = {"matched_question": saved_question, "similarity": round(similarity, 4)}
                return ResponseCacheResult(answer, "HIT", "guarded_semantic_hit", 0, False,
                                           (time.perf_counter() - started) * 1_000)
            except Exception:
                # Any local model, shape or storage failure takes the fresh path.
                decision = "local_check_unavailable"
                rejections[decision] += 1
                return None

        def resolve(checked):
            nonlocal match
            diagnostics["state"] = "executed"
            clock = time.perf_counter()
            try:
                candidate = assess(checked)
            finally:
                diagnostics["verification_ms"] = (time.perf_counter() - clock) * 1_000
            diagnostics["outcome"] = decision or "no_qualified_candidate"
            if mode == "shadow" and candidate is not None:
                diagnostics["shadow_recommendation"] = True
                match = None
                return None
            return candidate

        result = self.exact.process(body, tenant_id=tenant_id, provider=provider, on_miss=resolve)
        warm_key = namespace, source_key
        with self._warm_lock:
            was_warmed = warm_key in self.warmed
        if result.stored or (result.reason == "scoped_exact_hit" and not was_warmed):
            try:
                # Existing exact answers can seed semantic reuse on their first
                # visit. Admission is tied to the original source and expiry.
                if epoch is None:
                    epoch = self.storage.invalidation_epoch
                if vector is None:
                    with self._model_lock:
                        vector = normalize(self.models[0].encode(canonical)) if self.models and explicit else []
                if self.storage.admit_candidate(
                    namespace=namespace, source_key=source_key, tenant=tenant_id, payload=_serialize(result.response),
                    epoch=epoch, request=_serialize(body), vector=vector, max_entries=MAX_CANDIDATES, max_bytes=MAX_BYTES,
                    lookup_tag=tag,
                ):
                    with self._warm_lock:
                        self.warmed[warm_key] = None
                        self.warmed.move_to_end(warm_key)
                        if len(self.warmed) > MAX_CANDIDATES:
                            self.warmed.popitem(last=False)
            except Exception:
                pass  # A completed answer remains usable even if indexing fails.
        return ResponseCacheResult(result.response, result.status, result.reason, result.provider_calls,
                                   result.stored, (time.perf_counter() - started) * 1_000), match, decision
