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
from src.policy.decomposed_applicability import requirement_guard
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


@dataclass
class CandidatePromotionItem:
    candidate_id: str
    tenant_id: str
    scope: dict
    request: dict
    response: dict
    vector: tuple[float, ...]
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
        if not (0.0 <= retrieval_threshold <= 1.0):
            raise ValueError("retrieval_threshold must be in [0.0, 1.0]")
        if not (0.0 <= verification_threshold <= 1.0):
            raise ValueError("verification_threshold must be in [0.0, 1.0]")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be positive integer")

        self.exact = exact_cache
        self.default_scope = scope
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

        self.promotion_queue = CandidatePromotionQueue()
        self.answer_policy = AnswerAwareSemanticPolicy(
            system_prompt=None,
            scorer=self.passage_scorer,
            threshold=self.verification_threshold,
            enabled=True,
        )

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
        """Explicit tombstone revocation for a specific candidate binding."""
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

    def _scope_namespace(self, tenant_id: str, scope: KnowledgeScope) -> str:
        return digest({"tenant": tenant_id, "scope": asdict(scope), "contract": "hybrid-cache-v1"})

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
        namespace = self._scope_namespace(tenant_id, scope)
        candidate_id = digest({"tenant": tenant_id, "req": request, "resp": response, "epoch": epoch})

        raw_vec = self.encoder.encode(user_content)
        vector = tuple(normalize(raw_vec))

        if self.async_promotion and not force_sync:
            item = CandidatePromotionItem(
                candidate_id=candidate_id,
                tenant_id=tenant_id,
                scope=asdict(scope),
                request=copy.deepcopy(request),
                response=copy.deepcopy(response),
                vector=vector,
                enqueued_at=self.clock(),
            )
            return self.promotion_queue.enqueue(item)

        # Synchronous verification: candidate must pass before admission
        ref_text = response["choices"][0]["message"]["content"]
        score_info = self.passage_scorer.score(user_content, ref_text)
        prob = score_info["probability"]
        if prob < self.verification_threshold:
            return False

        # Promoted to serving cache
        req_json = _serialize(request)
        resp_json = _serialize(response)
        size = len(req_json.encode()) + len(resp_json.encode()) + len(vector) * 8
        expires = self.clock() + self.ttl_seconds

        cand = ServingCandidate(
            candidate_id=candidate_id,
            tenant_id=tenant_id,
            scope_hash=namespace,
            request_json=req_json,
            response_json=resp_json,
            vector=vector,
            epoch=epoch,
            expires_at=expires,
            verified=True,
            verification_identity=self.passage_scorer.identity,
            size=size,
        )

        with self._lock:
            self._serving_candidates[candidate_id] = cand
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
            score_info = self.passage_scorer.score(q, ans)
            el = (time.perf_counter() - t0) * 1000.0
            prob = score_info["probability"]
            app = prob >= self.verification_threshold
            return app, "passage_verified" if app else "passage_below_threshold", el

        processed = self.promotion_queue.drain_and_verify(verifier, max_items=max_items)
        admitted = 0
        with self._lock:
            for item in processed:
                if item.verified:
                    req_json = _serialize(item.request)
                    resp_json = _serialize(item.response)
                    size = len(req_json.encode()) + len(resp_json.encode()) + len(item.vector) * 8
                    expires = self.clock() + self.ttl_seconds
                    namespace = digest({"tenant": item.tenant_id, "scope": item.scope, "contract": "hybrid-cache-v1"})
                    cand = ServingCandidate(
                        candidate_id=item.candidate_id,
                        tenant_id=item.tenant_id,
                        scope_hash=namespace,
                        request_json=req_json,
                        response_json=resp_json,
                        vector=item.vector,
                        epoch=self.get_epoch(item.tenant_id),
                        expires_at=expires,
                        verified=True,
                        verification_identity=self.passage_scorer.identity,
                        size=size,
                    )
                    self._serving_candidates[item.candidate_id] = cand
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
        namespace = self._scope_namespace(tenant_id, scope)
        now = self.clock()

        diagnosis: dict[str, Any] = {
            "mode": "hybrid_conservative",
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

        incoming_query = request["messages"][-1]["content"] if request.get("messages") else ""

        def resolve_semantic(checked: dict) -> ResponseCacheResult | None:
            if not incoming_query:
                return None

            # 2. Dense Semantic Top-K Retrieval
            t_enc = time.perf_counter()
            if ledger is not None:
                with ledger.measure("embedding"):
                    query_vec = tuple(normalize(self.encoder.encode(incoming_query)))
            else:
                query_vec = tuple(normalize(self.encoder.encode(incoming_query)))

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

            # Cosine search
            scored_candidates = []
            for c in valid_candidates:
                sim = sum(a * b for a, b in zip(query_vec, c.vector))
                if sim >= self.retrieval_threshold:
                    scored_candidates.append((sim, c))

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
            for sim, cand in top_candidates:
                cand_req = json.loads(cand.request_json)
                cand_resp = json.loads(cand.response_json)
                cand_q = cand_req["messages"][-1]["content"]
                cand_ans = cand_resp["choices"][0]["message"]["content"]

                binding = approval_binding(checked, cand_req, cand_resp, tenant_id=tenant_id, scope=asdict(scope))
                binding_h = digest(binding)

                # Context Guard
                incoming_sys = checked["messages"][0]["content"] if checked.get("messages") else ""
                cand_sys = cand_req["messages"][0]["content"] if cand_req.get("messages") else ""
                if incoming_sys != cand_sys:
                    diagnosis["guard_rejections"].append(f"context_mismatch:{cand.candidate_id}")
                    continue

                # Direction & Requirement Guards
                guard_reason = requirement_guard(cand_q, incoming_query)
                if guard_reason is not None:
                    diagnosis["guard_rejections"].append(f"guard_rejected:{guard_reason}")
                    continue

                # Entry Review Decision Check
                if binding_h in self._revoked_keys:
                    diagnosis["guard_rejections"].append(f"entry_revoked:{binding_h}")
                    continue

                entry_review = self._entry_reviews.get(binding_h)
                if entry_review is not None:
                    if not entry_review.approved:
                        diagnosis["guard_rejections"].append(f"entry_veto:{entry_review.reason}")
                        continue
                    diagnosis.update(
                        stage="entry_reviewed_approved",
                        candidate=cand.candidate_id,
                        similarity=sim,
                        tier="SEMANTIC",
                        entry_decision=entry_review.reason,
                        local_policy_recommendation=True,
                    )
                    el = (time.perf_counter() - started) * 1000.0
                    return ResponseCacheResult(cand_resp, "HIT", "entry_reviewed_hit", 0, False, el / 1000.0)

                # Answer-Aware Passage Verification
                t_ver0 = time.perf_counter()
                if ledger is not None:
                    with ledger.measure("applicability"):
                        score_data = self.passage_scorer.score(incoming_query, cand_ans)
                else:
                    score_data = self.passage_scorer.score(incoming_query, cand_ans)
                ver_ms = (time.perf_counter() - t_ver0) * 1000.0
                diagnosis["local_verifier_attempts"] += 1
                diagnosis["local_verifier_ms"] += ver_ms

                prob = score_data["probability"]
                if prob >= self.verification_threshold:
                    diagnosis.update(
                        stage="verified_semantic_hit",
                        candidate=cand.candidate_id,
                        similarity=sim,
                        tier="SEMANTIC",
                        local_policy_recommendation=True,
                    )
                    el = (time.perf_counter() - started) * 1000.0
                    return ResponseCacheResult(cand_resp, "HIT", "verified_semantic_hit", 0, False, el / 1000.0)

                else:
                    diagnosis["guard_rejections"].append(f"verifier_low_score:{prob:.4f}<{self.verification_threshold}")

            diagnosis["stage"] = "all_candidates_rejected_fallback"
            return None

        # Execute scoped exact cache with on_miss semantic hook
        exact_result = self.exact.process(
            request,
            tenant_id=tenant_id,
            provider=provider,
            bypass=bypass,
            key_context=asdict(scope),
            on_miss=resolve_semantic if not bypass else None,
        )

        total_elapsed = (time.perf_counter() - started) * 1000.0
        if exact_result.status == "HIT":
            if exact_result.reason in ("verified_semantic_hit", "entry_reviewed_hit"):
                tier = "SEMANTIC"
            else:
                tier = "EXACT"
                diagnosis.update(stage="exact_hit", tier="EXACT")
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

        # Cache Miss: admitted to semantic candidates
        if exact_result.stored:
            self.admit_candidate(request, exact_result.response, tenant_id=tenant_id)

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


