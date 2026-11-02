"""Conservative Hybrid Semantic Cache with Verified Promotion.

Incorporates principles from recent literature on verified and guarded semantic
caching (vCache, Krites) without claiming their formal proofs:
1. Exact-first scoped lookup (O(1) hashing with version and context bound).
2. Dense semantic top-k retrieval (MiniLM dense embeddings).
3. Hard structural guards before neural inference:
   - System prompt / context alignment guard
   - Source revision and policy revision guard
   - Direction reversal guard
   - Negation and requirement polarity guard
   - Numeric and version constraint guard
4. Answer-aware passage verification (MS MARCO cross-encoder on actual answers).
5. Cache-entry-specific decisions:
   - Specific reviewed records can veto or approve entries.
6. Candidate-promotion queue for asynchronous verification:
   - Generated candidates enter a promotion queue.
   - ONLY candidates that pass all guards and neural verification are promoted
     into the serving cache.
   - Unverified entries are NEVER admitted to the serving cache.
7. Explicit revocation and TTL handling:
   - Epoch-based namespace invalidation
   - Tenant-level revocation
   - Entry-level revocation tombstones
   - TTL expiration enforcement
8. Fail-closed fresh fallback on any uncertain request.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import queue
import re
import threading
import time
from typing import Any, Callable

from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy
from src.policy.decomposed_applicability import answer_satisfies_requirements, requirement_guard
from src.policy.encoders import DenseEncoder, normalize
from src.policy.local_models import (
    ENCODER_MODEL, ENCODER_REVISION,
    PASSAGE_MODEL, PASSAGE_REVISION,
    LocalPassageScorer,
)
from src.policy.response_cache import (
    EligibilityDecision, ResponseCacheResult, ScopedResponseCache, _serialize
)
from src.policy.semantic_runtime import (
    KnowledgeScope, ReuseDecision, ServingCacheResult, approval_binding, digest
)
from src.policy.work_ledger import WorkLedger


@dataclass(frozen=True)
class EntryReviewRecord:
    binding_hash: str
    candidate_id: str
    approved: bool
    reason: str
    review_count: int = 1
    updated_at: float = field(default_factory=time.time)


def request_boundary(body: dict) -> dict:
    """Extract canonical request boundary context (model, parameters, history)."""
    scoped = copy.deepcopy(body)
    if isinstance(scoped.get("messages"), list) and scoped["messages"]:
        scoped["messages"][-1] = {
            "role": scoped["messages"][-1].get("role", "user"),
            "content": "<retrieval-query>",
        }
    return scoped


@dataclass
class CandidatePromotionItem:
    candidate_id: str
    tenant_id: str
    scope: dict
    request: dict
    response: dict
    vector: tuple[float, ...]
    epoch: int = 1
    expires_at: float = 0.0
    enqueued_at: float = field(default_factory=time.time)
    verified: bool = False
    verification_reason: str = "unverified"
    verification_ms: float = 0.0


@dataclass(frozen=True)
class ServingCandidate:
    candidate_id: str
    tenant_id: str
    scope_hash: str
    request_json: str
    response_json: str
    vector: tuple[float, ...]
    epoch: int
    expires_at: float
    verified: bool
    verification_identity: str
    size: int


class CandidatePromotionQueue:
    """Thread-safe queue for asynchronous candidate verification and promotion.
    
    Safety Invariant: Unverified entries NEVER enter the serving cache.
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.Queue[CandidatePromotionItem] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._total_enqueued = 0
        self._total_promoted = 0
        self._total_rejected = 0

    def enqueue(self, item: CandidatePromotionItem) -> bool:
        if not isinstance(item, CandidatePromotionItem):
            raise TypeError("item must be CandidatePromotionItem")
        try:
            self._queue.put_nowait(item)
            with self._lock:
                self._total_enqueued += 1
            return True
        except queue.Full:
            return False

    def qsize(self) -> int:
        return self._queue.qsize()

    def drain_and_verify(
        self,
        verifier: Callable[[dict, dict, dict], tuple[bool, str, float]],
        max_items: int | None = None,
    ) -> list[CandidatePromotionItem]:
        """Verify queued candidates; only approved items get verified=True."""
        processed = []
        count = 0
        while not self._queue.empty():
            if max_items is not None and count >= max_items:
                break
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            approved, reason, elapsed_ms = verifier(item.request, item.request, item.response)
            item.verified = approved
            item.verification_reason = reason
            item.verification_ms = elapsed_ms

            with self._lock:
                if approved:
                    self._total_promoted += 1
                else:
                    self._total_rejected += 1

            processed.append(item)
            self._queue.task_done()
            count += 1

        return processed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "enqueued": self._total_enqueued,
                "promoted": self._total_promoted,
                "rejected": self._total_rejected,
                "pending": self._queue.qsize(),
            }


class ConservativeHybridCache:
    """Conservative Hybrid Semantic Cache combining exact lookup, structural guards,
    entry-specific reviews, candidate promotion queue, and answer-aware neural verification.
    """

    def __init__(
        self,
        exact_cache: ScopedResponseCache,
        *,
        scope: KnowledgeScope,
        encoder: DenseEncoder | None = None,
        passage_scorer: LocalPassageScorer | None = None,
        mode: str = "exact",
        retrieval_threshold: float = 0.60,
        verification_threshold: float = 0.85,
        top_k: int = 3,
        ttl_seconds: float = 86400.0,
        max_serving_candidates: int = 1000,
        async_promotion: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        if not isinstance(exact_cache, ScopedResponseCache):
            raise TypeError("exact_cache must be ScopedResponseCache")
        if mode not in {"off", "exact", "shadow", "guarded"}:
            raise ValueError("mode must be 'off', 'exact', 'shadow', or 'guarded'")
        if not (0.0 <= retrieval_threshold <= 1.0):
            raise ValueError("retrieval_threshold must be in [0.0, 1.0]")
        if not (0.0 <= verification_threshold <= 1.0):
            raise ValueError("verification_threshold must be in [0.0, 1.0]")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive integer")

        self.exact = exact_cache
        self.default_scope = scope
        self.mode = mode
        self.encoder = encoder if encoder is not None else DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
        self.passage_scorer = passage_scorer if passage_scorer is not None else LocalPassageScorer(PASSAGE_MODEL, PASSAGE_REVISION)
        self.retrieval_threshold = float(retrieval_threshold)
        self.verification_threshold = float(verification_threshold)
        self.top_k = top_k
        self.ttl_seconds = float(ttl_seconds)
        self.max_serving_candidates = max_serving_candidates
        self.async_promotion = async_promotion
        self.clock = clock

        self._lock = threading.RLock()
        self._tenant_scopes: dict[str, KnowledgeScope] = {}
        self._tenant_epochs: dict[str, int] = {}
        self._serving_candidates: dict[str, ServingCandidate] = {}
        self._entry_reviews: dict[str, EntryReviewRecord] = {}
        self._revoked_keys: set[str] = set()
        self._binding_to_exact_keys: dict[str, set[str]] = {}
        self._embedding_cache: dict[str, tuple[float, ...]] = {}
        self._passage_score_cache: dict[tuple[str, str], dict] = {}
        self._max_cache_size = 2048

        self.promotion_queue = CandidatePromotionQueue()
        self.answer_policy = AnswerAwareSemanticPolicy(
            system_prompt=None,
            scorer=self.passage_scorer,
            threshold=self.verification_threshold,
            enabled=True,
        )

    def _encode_query(self, query: str, ledger: WorkLedger | None = None) -> tuple[float, ...]:
        with self._lock:
            if query in self._embedding_cache:
                return self._embedding_cache[query]
        if ledger is not None:
            with ledger.measure("embedding"):
                raw_vec = self.encoder.encode(query)
        else:
            raw_vec = self.encoder.encode(query)
        normalized_vec = tuple(normalize(raw_vec))
        with self._lock:
            if len(self._embedding_cache) >= self._max_cache_size:
                del self._embedding_cache[next(iter(self._embedding_cache))]
            self._embedding_cache[query] = normalized_vec
        return normalized_vec

    def _score_passage(self, query: str, passage: str, ledger: WorkLedger | None = None) -> dict:
        key = (query, passage)
        with self._lock:
            if key in self._passage_score_cache:
                return self._passage_score_cache[key]
        if ledger is not None:
            with ledger.measure("applicability"):
                score_data = self.passage_scorer.score(query, passage)
        else:
            score_data = self.passage_scorer.score(query, passage)
        with self._lock:
            if len(self._passage_score_cache) >= self._max_cache_size:
                del self._passage_score_cache[next(iter(self._passage_score_cache))]
            self._passage_score_cache[key] = score_data
        return score_data

    def update_scope(self, scope: KnowledgeScope, tenant_id: str = "default"):
        with self._lock:
            self._tenant_scopes[tenant_id] = scope

    def get_scope(self, tenant_id: str = "default") -> KnowledgeScope:
        with self._lock:
            return self._tenant_scopes.get(tenant_id, self.default_scope)

    def get_epoch(self, tenant_id: str = "default") -> int:
        with self._lock:
            return self._tenant_epochs.get(tenant_id, 1)

    def revoke_tenant(self, tenant_id: str = "default"):
        """Explicitly revokes all cached entries for a tenant by incrementing epoch."""
        with self._lock:
            current = self._tenant_epochs.get(tenant_id, 1)
            self._tenant_epochs[tenant_id] = current + 1
            # Purge in-memory serving candidates for this tenant
            stale_keys = [k for k, v in self._serving_candidates.items() if v.tenant_id == tenant_id]
            for k in stale_keys:
                del self._serving_candidates[k]
        self.exact.invalidate(tenant_id=tenant_id)

    def revoke_entry(self, binding_hash: str):
        """Explicit tombstone revocation for a specific candidate binding across both tiers."""
        with self._lock:
            self._revoked_keys.add(binding_hash)
            if binding_hash in self._entry_reviews:
                rec = self._entry_reviews[binding_hash]
                self._entry_reviews[binding_hash] = EntryReviewRecord(
                    binding_hash=binding_hash,
                    candidate_id=rec.candidate_id,
                    approved=False,
                    reason="explicitly_revoked",
                    review_count=rec.review_count + 1,
                    updated_at=self.clock(),
                )
            if binding_hash in self._binding_to_exact_keys:
                for k in self._binding_to_exact_keys[binding_hash]:
                    self.exact.invalidate_key(k)
            stale = []
            for cid, c in self._serving_candidates.items():
                try:
                    cand_req = json.loads(c.request_json)
                    cand_resp = json.loads(c.response_json)
                    cand_scope = asdict(self.get_scope(c.tenant_id))
                    b = digest(approval_binding(cand_req, cand_req, cand_resp, tenant_id=c.tenant_id, scope=cand_scope))
                    if b == binding_hash:
                        stale.append(cid)
                except Exception:
                    pass
            for cid in stale:
                del self._serving_candidates[cid]

    def record_entry_decision(self, binding_hash: str, candidate_id: str, approved: bool, reason: str):
        """Records an authoritative review decision for a specific candidate binding."""
        with self._lock:
            existing = self._entry_reviews.get(binding_hash)
            count = (existing.review_count + 1) if existing else 1
            self._entry_reviews[binding_hash] = EntryReviewRecord(
                binding_hash=binding_hash,
                candidate_id=candidate_id,
                approved=approved,
                reason=reason,
                review_count=count,
                updated_at=self.clock(),
            )
            if not approved:
                self._revoked_keys.add(binding_hash)
                if binding_hash in self._binding_to_exact_keys:
                    for k in self._binding_to_exact_keys[binding_hash]:
                        self.exact.invalidate_key(k)
                stale = []
                for cid, c in self._serving_candidates.items():
                    try:
                        cand_req = json.loads(c.request_json)
                        cand_resp = json.loads(c.response_json)
                        cand_scope = asdict(self.get_scope(c.tenant_id))
                        b = digest(approval_binding(cand_req, cand_req, cand_resp, tenant_id=c.tenant_id, scope=cand_scope))
                        if b == binding_hash:
                            stale.append(cid)
                    except Exception:
                        pass
                for cid in stale:
                    del self._serving_candidates[cid]

    def _scope_namespace(self, tenant_id: str, scope: KnowledgeScope | dict, request: dict | None = None) -> str:
        data = {
            "tenant": tenant_id,
            "scope": asdict(scope) if isinstance(scope, KnowledgeScope) else dict(scope),
            "contract": "hybrid-cache-v2",
        }
        if request is not None:
            data["request_boundary"] = request_boundary(request)
        return digest(data)

    def admit_candidate(
        self,
        request: dict,
        response: dict,
        *,
        tenant_id: str = "default",
        force_sync: bool = False,
    ) -> bool:
        """Admit a new request/response pair.
        
        If async_promotion is True and not force_sync:
          Enqueues the candidate for asynchronous verification.
        If async_promotion is False or force_sync:
          Verifies immediately; admits ONLY if verification passes.
        """
        if not ScopedResponseCache._admissible(response, request):
            return False

        user_content = request["messages"][-1]["content"] if request.get("messages") else ""
        if not user_content:
            return False

        scope = self.get_scope(tenant_id)
        epoch = self.get_epoch(tenant_id)
        namespace = self._scope_namespace(tenant_id, scope, request)
        candidate_id = digest({"tenant": tenant_id, "req": request, "resp": response, "epoch": epoch})
        expires_at = self.clock() + self.ttl_seconds

        try:
            vector = self._encode_query(user_content)
        except Exception:
            return False

        binding = approval_binding(request, request, response, tenant_id=tenant_id, scope=asdict(scope))
        binding_h = digest(binding)
        exact_k = ScopedResponseCache.request_key(request, tenant_id, asdict(scope))

        if self.async_promotion and not force_sync:
            item = CandidatePromotionItem(
                candidate_id=candidate_id,
                tenant_id=tenant_id,
                scope=asdict(scope),
                request=copy.deepcopy(request),
                response=copy.deepcopy(response),
                vector=vector,
                epoch=epoch,
                expires_at=expires_at,
                enqueued_at=self.clock(),
            )
            with self._lock:
                self._binding_to_exact_keys.setdefault(binding_h, set()).add(exact_k)
            return self.promotion_queue.enqueue(item)

        # Synchronous verification: candidate must pass before admission
        ref_text = response["choices"][0]["message"]["content"]
        try:
            score_info = self._score_passage(user_content, ref_text)
            prob = score_info.get("probability", 0.0)
            if prob < self.verification_threshold:
                return False
        except Exception:
            return False

        # Promoted to serving cache
        req_json = _serialize(request)
        resp_json = _serialize(response)
        size = len(req_json.encode()) + len(resp_json.encode()) + len(vector) * 8

        cand = ServingCandidate(
            candidate_id=candidate_id,
            tenant_id=tenant_id,
            scope_hash=namespace,
            request_json=req_json,
            response_json=resp_json,
            vector=vector,
            epoch=epoch,
            expires_at=expires_at,
            verified=True,
            verification_identity=self.passage_scorer.identity,
            size=size,
        )

        with self._lock:
            # Revalidate lifecycle under lock to prevent post-revocation or expired admission
            cur_epoch = self.get_epoch(tenant_id)
            if cur_epoch != epoch:
                return False
            cur_scope = self.get_scope(tenant_id)
            if cur_scope != scope:
                return False
            if binding_h in self._revoked_keys:
                return False
            if self.clock() >= expires_at:
                return False

            self._serving_candidates[candidate_id] = cand
            self._binding_to_exact_keys.setdefault(binding_h, set()).add(exact_k)
            # Evict if over limit
            while len(self._serving_candidates) > self.max_serving_candidates:
                first_key = next(iter(self._serving_candidates))
                del self._serving_candidates[first_key]

        return True

    def process_promotion_queue(self, max_items: int | None = None) -> int:
        """Processes queued candidates; admits only verified ones into serving store."""
        def verifier(req, cand_req, resp):
            q = req["messages"][-1]["content"]
            ans = resp["choices"][0]["message"]["content"]
            t0 = time.perf_counter()
            try:
                score_info = self.passage_scorer.score(q, ans)
                prob = score_info.get("probability", 0.0)
                app = prob >= self.verification_threshold
                reason = "passage_verified" if app else "passage_below_threshold"
            except Exception as exc:
                app = False
                reason = f"verifier_error:{type(exc).__name__}"
            el = (time.perf_counter() - t0) * 1000.0
            return app, reason, el

        processed = self.promotion_queue.drain_and_verify(verifier, max_items=max_items)
        admitted = 0
        with self._lock:
            for item in processed:
                if item.verified:
                    now = self.clock()
                    # 1. Check expiration
                    if item.expires_at <= now:
                        continue
                    # 2. Check tenant epoch
                    current_epoch = self.get_epoch(item.tenant_id)
                    if item.epoch != current_epoch:
                        continue
                    # 3. Check scope
                    current_scope = asdict(self.get_scope(item.tenant_id))
                    if item.scope != current_scope:
                        continue
                    # 4. Check revocation
                    binding = approval_binding(item.request, item.request, item.response,
                                               tenant_id=item.tenant_id, scope=item.scope)
                    binding_h = digest(binding)
                    if binding_h in self._revoked_keys:
                        continue

                    req_json = _serialize(item.request)
                    resp_json = _serialize(item.response)
                    size = len(req_json.encode()) + len(resp_json.encode()) + len(item.vector) * 8
                    namespace = self._scope_namespace(item.tenant_id, self.get_scope(item.tenant_id), item.request)
                    cand = ServingCandidate(
                        candidate_id=item.candidate_id,
                        tenant_id=item.tenant_id,
                        scope_hash=namespace,
                        request_json=req_json,
                        response_json=resp_json,
                        vector=item.vector,
                        epoch=item.epoch,
                        expires_at=item.expires_at,
                        verified=True,
                        verification_identity=self.passage_scorer.identity,
                        size=size,
                    )
                    exact_k = ScopedResponseCache.request_key(item.request, item.tenant_id, item.scope)
                    self._serving_candidates[item.candidate_id] = cand
                    self._binding_to_exact_keys.setdefault(binding_h, set()).add(exact_k)
                    while len(self._serving_candidates) > self.max_serving_candidates:
                        first_key = next(iter(self._serving_candidates))
                        del self._serving_candidates[first_key]
                    admitted += 1
        return admitted

    def process(
        self,
        request: dict,
        *,
        tenant_id: str = "default",
        provider: Callable[[dict], dict],
        bypass: bool = False,
        ledger: WorkLedger | None = None,
    ) -> ServingCacheResult:
        started = time.perf_counter()
        scope = self.get_scope(tenant_id)
        epoch = self.get_epoch(tenant_id)
        namespace = self._scope_namespace(tenant_id, scope, request)
        now = self.clock()

        diagnosis: dict[str, Any] = {
            "mode": f"hybrid_{self.mode}",
            "stage": "exact_lookup",
            "tier": "NONE",
            "candidate": None,
            "similarity": None,
            "guard_rejections": [],
            "entry_decision": None,
            "local_verifier_attempts": 0,
            "local_verifier_ms": 0.0,
            "local_policy_recommendation": None,
        }

        if self.mode == "off" or bypass:
            t0 = time.perf_counter()
            resp = provider(copy.deepcopy(request))
            el = (time.perf_counter() - t0) * 1000.0
            diagnosis.update(stage="cache_off", tier="NONE")
            return ServingCacheResult(resp, "BYPASS", "cache_off", 1, False, el, "NONE", diagnosis)

        incoming_query = request["messages"][-1]["content"] if request.get("messages") else ""

        def resolve_semantic(checked: dict) -> ResponseCacheResult | None:
            if not incoming_query:
                return None

            # 1. Candidate check BEFORE neural inference (avoid unnecessary neural scoring)
            with self._lock:
                valid_candidates = [
                    c for c in self._serving_candidates.values()
                    if c.tenant_id == tenant_id
                    and c.scope_hash == namespace
                    and c.epoch == epoch
                    and c.expires_at > now
                    and c.verified
                ]

            if not valid_candidates:
                diagnosis["stage"] = "no_candidates_in_scope"
                return None

            # 2. Cheap structural guards BEFORE dense encoding
            incoming_msgs = checked.get("messages", [])
            compatible_candidates = []
            for c in valid_candidates:
                try:
                    cand_req = json.loads(c.request_json)
                except Exception:
                    continue
                if checked.get("model") != cand_req.get("model"):
                    continue
                if checked.get("response_format") != cand_req.get("response_format"):
                    continue
                cand_msgs = cand_req.get("messages", [])
                if len(incoming_msgs) != len(cand_msgs):
                    continue
                if incoming_msgs[:-1] != cand_msgs[:-1]:
                    continue
                compatible_candidates.append((c, cand_req))

            if not compatible_candidates:
                diagnosis["stage"] = "no_compatible_candidates_in_scope"
                return None

            # 3. Dense Semantic Top-K Retrieval using cached/reused embeddings
            try:
                query_vec = self._encode_query(incoming_query, ledger)
            except Exception as exc:
                diagnosis["guard_rejections"].append(f"encoder_failure:{type(exc).__name__}")
                return None

            # Cosine search over compatible candidates
            scored_candidates = []
            for c, cand_req in compatible_candidates:
                sim = sum(a * b for a, b in zip(query_vec, c.vector))
                if sim >= self.retrieval_threshold:
                    scored_candidates.append((sim, c, cand_req))

            if ledger is not None:
                with ledger.measure("search"):
                    scored_candidates.sort(key=lambda x: -x[0])
            else:
                scored_candidates.sort(key=lambda x: -x[0])

            top_candidates = scored_candidates[:self.top_k]
            if not top_candidates:
                diagnosis["stage"] = "threshold_rejected"
                diagnosis["similarity"] = scored_candidates[0][0] if scored_candidates else 0.0
                return None

            # Evaluate top candidates
            for sim, cand, cand_req in top_candidates:
                cand_resp = json.loads(cand.response_json)
                cand_q = cand_req["messages"][-1]["content"]
                cand_ans = cand_resp["choices"][0]["message"]["content"]

                binding = approval_binding(checked, cand_req, cand_resp, tenant_id=tenant_id, scope=asdict(scope))
                binding_h = digest(binding)

                # 4. Direction & Requirement Guards
                guard_reason = requirement_guard(cand_q, incoming_query)
                if guard_reason is not None:
                    diagnosis["guard_rejections"].append(f"guard_rejected:{guard_reason}")
                    continue

                # 4b. Answer Requirement Satisfaction Guard
                ans_reason = answer_satisfies_requirements(incoming_query, cand_ans)
                if ans_reason is not None:
                    diagnosis["guard_rejections"].append(f"answer_unmet_requirement:{ans_reason}")
                    continue

                # 5. Entry Review Decision Check
                if binding_h in self._revoked_keys:
                    diagnosis["guard_rejections"].append(f"entry_revoked:{binding_h}")
                    continue

                entry_review = self._entry_reviews.get(binding_h)
                if entry_review is not None:
                    if not entry_review.approved:
                        diagnosis["guard_rejections"].append(f"entry_veto:{entry_review.reason}")
                        continue
                    with self._lock:
                        now_check = self.clock()
                        cur_epoch = self.get_epoch(tenant_id)
                        cur_scope = self.get_scope(tenant_id)
                        if (
                            cur_epoch != cand.epoch
                            or cur_epoch != epoch
                            or cand.candidate_id not in self._serving_candidates
                            or binding_h in self._revoked_keys
                            or now_check >= cand.expires_at
                            or cur_scope != scope
                        ):
                            diagnosis["guard_rejections"].append("stale_before_entry_hit")
                            continue

                        if self.mode == "shadow":
                            diagnosis.update(
                                stage="shadow_entry_reviewed_candidate",
                                candidate=cand.candidate_id,
                                similarity=sim,
                                tier="SEMANTIC",
                                entry_decision=entry_review.reason,
                                local_policy_recommendation=True,
                            )
                            return None

                        diagnosis.update(
                            stage="entry_reviewed_approved",
                            candidate=cand.candidate_id,
                            similarity=sim,
                            tier="SEMANTIC",
                            entry_decision=entry_review.reason,
                            local_policy_recommendation=True,
                        )
                        el = (time.perf_counter() - started) * 1000.0
                        return ResponseCacheResult(cand_resp, "HIT", "entry_reviewed_hit", 0, False, el)

                # 6. Answer-Aware Passage Verification with cached scoring
                t_ver0 = time.perf_counter()
                try:
                    score_data = self._score_passage(incoming_query, cand_ans, ledger)
                    ver_ms = (time.perf_counter() - t_ver0) * 1000.0
                    diagnosis["local_verifier_attempts"] += 1
                    diagnosis["local_verifier_ms"] += ver_ms

                    prob = score_data.get("probability", 0.0)
                    if prob >= self.verification_threshold:
                        with self._lock:
                            now_check = self.clock()
                            cur_epoch = self.get_epoch(tenant_id)
                            cur_scope = self.get_scope(tenant_id)
                            if (
                                cur_epoch != cand.epoch
                                or cur_epoch != epoch
                                or cand.candidate_id not in self._serving_candidates
                                or binding_h in self._revoked_keys
                                or now_check >= cand.expires_at
                                or cur_scope != scope
                            ):
                                diagnosis["guard_rejections"].append("stale_during_verification")
                                continue

                            if self.mode == "shadow":
                                diagnosis.update(
                                    stage="shadow_candidate",
                                    candidate=cand.candidate_id,
                                    similarity=sim,
                                    tier="SEMANTIC",
                                    local_policy_recommendation=True,
                                )
                                return None

                            diagnosis.update(
                                stage="verified_semantic_hit",
                                candidate=cand.candidate_id,
                                similarity=sim,
                                tier="SEMANTIC",
                                local_policy_recommendation=True,
                            )
                            el = (time.perf_counter() - started) * 1000.0
                            return ResponseCacheResult(cand_resp, "HIT", "verified_semantic_hit", 0, False, el)
                    else:
                        diagnosis["guard_rejections"].append(f"verifier_low_score:{prob:.4f}<{self.verification_threshold}")
                except Exception as exc:
                    ver_ms = (time.perf_counter() - t_ver0) * 1000.0
                    diagnosis["local_verifier_attempts"] += 1
                    diagnosis["local_verifier_ms"] += ver_ms
                    diagnosis["guard_rejections"].append(f"verifier_failure:{type(exc).__name__}")
                    continue

            diagnosis["stage"] = "all_candidates_rejected_fallback"
            return None

        # Execute scoped exact cache with on_miss semantic hook (guarded or shadow mode)
        semantic_active = self.mode in ("guarded", "shadow") and not bypass
        exact_result = self.exact.process(
            request,
            tenant_id=tenant_id,
            provider=provider,
            bypass=bypass,
            key_context=asdict(scope),
            on_miss=resolve_semantic if semantic_active else None,
        )

        if exact_result.status == "HIT":
            if exact_result.reason in ("verified_semantic_hit", "entry_reviewed_hit"):
                tier = "SEMANTIC"
            else:
                tier = "EXACT"
                # Cross-tier revocation consistency check
                binding = approval_binding(request, request, exact_result.response, tenant_id=tenant_id, scope=asdict(scope))
                binding_h = digest(binding)
                with self._lock:
                    cur_epoch = self.get_epoch(tenant_id)
                    cur_scope = self.get_scope(tenant_id)
                    if cur_epoch != epoch or cur_scope != scope or binding_h in self._revoked_keys:
                        self.exact.invalidate_key(self.exact.request_key(request, tenant_id, asdict(scope)))
                        fresh_response = provider(copy.deepcopy(request))
                        total_elapsed = (time.perf_counter() - started) * 1000.0
                        diagnosis.update(stage="exact_entry_revoked_fallback", tier="NONE")
                        return ServingCacheResult(
                            fresh_response,
                            "MISS",
                            "exact_entry_revoked",
                            1,
                            False,
                            total_elapsed,
                            "NONE",
                            diagnosis,
                        )
                diagnosis.update(stage="exact_hit", tier="EXACT")
            total_elapsed = (time.perf_counter() - started) * 1000.0
            return ServingCacheResult(
                exact_result.response,
                "HIT",
                exact_result.reason,
                exact_result.provider_calls,
                exact_result.stored,
                total_elapsed,
                tier,
                diagnosis,
            )

        # Cache Miss: admitted to semantic candidates synchronously in guarded or shadow mode
        if exact_result.stored and self.mode in ("guarded", "shadow"):
            try:
                self.admit_candidate(request, exact_result.response, tenant_id=tenant_id)
            except Exception as exc:
                diagnosis["guard_rejections"].append(f"admission_failure:{type(exc).__name__}")

        total_elapsed = (time.perf_counter() - started) * 1000.0
        diagnosis.setdefault("stage", "fallback_fresh_execution")
        return ServingCacheResult(
            exact_result.response,
            "MISS",
            exact_result.reason,
            exact_result.provider_calls,
            exact_result.stored,
            total_elapsed,
            "NONE",
            diagnosis,
        )


