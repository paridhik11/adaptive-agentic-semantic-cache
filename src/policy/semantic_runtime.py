"""Opt-in semantic miss resolution around the existing exact-response runtime.

Candidates are never approved by similarity alone. Exact storage remains the
source of fresh-answer admission. The default index is process-local; optional
SQLite candidate recovery binds records to the exact source admission.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from contextlib import nullcontext

from src.policy.encoders import normalize
from src.policy.response_cache import EligibilityDecision, ResponseCacheResult, ScopedResponseCache, _serialize
from src.policy.sqlite_store import SQLiteResponseStore


def digest(value):
    return hashlib.sha256(_serialize(value).encode()).hexdigest()


@dataclass(frozen=True)
class KnowledgeScope:
    knowledge_base: str
    revision: str
    access_scope: str
    policy_revision: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() or len(value) > 1024
               for value in asdict(self).values()):
            raise ValueError("knowledge/access/policy scope must be explicit bounded text")


class PinnedContextPolicy:
    """Operator opt-in to a fixed static-document prompt, not a safety label.

    The complete pinned evidence/instructions must be in the configured prompt.
    Arbitrary history, personal/dynamic requests and creative requests bypass.
    Correctness of generated answers still requires separate evaluation.
    """
    _unsupported = re.compile(
        r"\b(today|tomorrow|yesterday|now|latest|current|my|mine|password|"
        r"write|compose|draft|invent|brainstorm|random|generate)\b|@", re.I)

    def __init__(self, system_prompt):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("a fixed evidence-bearing system prompt is required")
        self.system_prompt = system_prompt

    def __call__(self, body):
        messages = body.get("messages", [])
        if (len(messages) != 2 or messages[0] != {"role": "system", "content": self.system_prompt}
                or messages[1].get("role") != "user"):
            return EligibilityDecision(False, "outside_pinned_document_context")
        text = messages[1].get("content")
        if not isinstance(text, str) or len(text) > 8000 or self._unsupported.search(text):
            return EligibilityDecision(False, "unsupported_document_request")
        return EligibilityDecision(True, "pinned_document_context")


@dataclass(frozen=True)
class ReuseDecision:
    approved: bool
    reason: str
    local_verifier_attempts: int = 0
    local_verifier_ms: float = 0.0
    local_policy_recommendation: bool | None = None


BINDING_FIELDS = ("scope_sha256", "incoming_request_sha256", "candidate_request_sha256", "candidate_response_sha256")


def approval_binding(request, candidate_request, answer, *, tenant_id, scope):
    """Bind approval to the full actual answer and trusted request/source scope."""
    return {
        "scope_sha256": digest({"tenant_id": tenant_id, "scope": scope}),
        "incoming_request_sha256": digest(request),
        "candidate_request_sha256": digest(candidate_request),
        "candidate_response_sha256": digest(answer),
    }


class AnswerBoundApprovals:
    """Immutable operator-supplied approvals, not authenticated human labels.

    Unknown, changed or incomplete bindings abstain. The declaration in the
    file is provenance metadata, not proof of reviewer independence/correctness.
    """
    def __init__(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {"schema_version", "provenance", "approvals"}
                or payload["schema_version"] != "answer-bound-approvals-v1"
                or payload["provenance"] not in {"authored_fixture", "declared_human_review"}
                or not isinstance(payload["approvals"], list)):
            raise ValueError("invalid answer-bound approval file")
        accepted, seen = set(), set()
        for row in payload["approvals"]:
            if (not isinstance(row, dict) or set(row) != {*BINDING_FIELDS, "approved"}
                    or type(row["approved"]) is not bool
                    or any(not isinstance(row[field], str) or not re.fullmatch("[0-9a-f]{64}", row[field])
                           for field in BINDING_FIELDS)):
                raise ValueError("invalid answer-bound approval row")
            key = tuple(row[field] for field in BINDING_FIELDS)
            if key in seen:
                raise ValueError("duplicate answer-bound approval")
            seen.add(key)
            if row["approved"]:
                accepted.add(key)
        self._accepted = frozenset(accepted)
        self.identity = "answer-bound-approvals-v1:" + digest(payload)
        self.provenance = payload["provenance"]

    def __call__(self, request, candidate_request, answer, *, binding, provider):
        return ReuseDecision(tuple(binding[field] for field in BINDING_FIELDS) in self._accepted,
                             "operator_answer_binding")


@dataclass(frozen=True)
class ServingCacheResult(ResponseCacheResult):
    tier: str
    diagnosis: dict


@dataclass(frozen=True)
class _Candidate:
    tenant_id: str
    scope_hash: str
    request_json: str
    response_json: str
    vector: tuple[float, ...]
    expires: float
    size: int
    source_key: str | None = None
    source_token: str | None = None


class GuardedSemanticCache:
    """Exact-first cache with shadow/explicit approval modes and bounded index.

    There is no similarity-only serving mode. Approval is a configured trusted
    policy signal, never an independently evaluated correctness guarantee.
    """
    def __init__(self, exact: ScopedResponseCache, *, scope: KnowledgeScope, encoder=None,
                 mode="shadow", approver=None, threshold=0.85, margin=0.05,
                 max_entries=256, max_bytes=8 * 1024 * 1024, max_dimensions=4096,
                 clock=time.monotonic, persistent_candidates=False, local_approval_only=False):
        if not isinstance(exact, ScopedResponseCache) or not isinstance(scope, KnowledgeScope):
            raise TypeError("exact cache and trusted knowledge scope are required")
        if mode not in {"off", "exact", "shadow", "guarded"}:
            raise ValueError("cache mode must be off, exact, shadow or guarded")
        if mode in {"shadow", "guarded"} and (not callable(getattr(encoder, "encode", None))
                or not isinstance(getattr(encoder, "identity", None), str) or not encoder.identity):
            raise ValueError("semantic modes require an identified encoder")
        if mode == "guarded" and (not callable(approver) or not isinstance(getattr(approver, "identity", None), str)
                                  or not approver.identity):
            raise ValueError("guarded mode requires an identified explicit approver")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in (threshold, margin)) or not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("invalid retrieval thresholds")
        if any(type(value) is not int or value < 1 for value in (max_entries, max_bytes, max_dimensions)):
            raise ValueError("semantic index bounds must be positive integers")
        self.exact, self.encoder, self.approver = exact, encoder, approver
        if type(local_approval_only) is not bool:
            raise ValueError('local_approval_only must be boolean')
        self.local_approval_only = local_approval_only
        self.semantic_mode = mode
        self.threshold, self.margin = threshold, margin
        self.max_entries, self.max_bytes, self.max_dimensions = max_entries, max_bytes, max_dimensions
        if type(persistent_candidates) is not bool:
            raise ValueError("persistent_candidates must be boolean")
        if persistent_candidates and not isinstance(exact._storage, SQLiteResponseStore):
            raise ValueError("persistent candidates require SQLite exact storage")
        self._persistent = exact._storage if persistent_candidates else None
        # Source TTL is authoritative; do not mix monotonic and persisted wall time.
        self.clock = self._persistent.clock if self._persistent is not None else clock
        self._default_scope, self._scopes = scope, {}
        self._lock, self._encoder_lock = threading.RLock(), threading.Lock()
        self._entries, self._bytes, self._generation = OrderedDict(), 0, None

    @property
    def size(self):
        return self.exact.size

    @property
    def semantic_size(self):
        with self._lock:
            self._sync_generation()
            self._expire()
            return len(self._entries)

    def _clear(self):
        self._entries.clear()
        self._bytes = 0

    def _sync_generation(self):
        generation = self.exact.invalidation_epoch
        if generation != self._generation:
            self._clear()
            self._generation = generation
        return generation

    def _expire(self):
        now = self.clock()
        for key in list(self._entries):
            if self._entries[key].expires <= now:
                self._bytes -= self._entries.pop(key).size

    def invalidate(self, *, tenant_id=None):
        with self._lock:
            # SQLite's epoch is also observed by other process-local indexes.
            try:
                return self.exact.invalidate(tenant_id=tenant_id)
            finally:
                self._clear()

    def update_scope(self, scope: KnowledgeScope, *, tenant_id=None):
        if not isinstance(scope, KnowledgeScope):
            raise TypeError("trusted KnowledgeScope required")
        if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id.strip()):
            raise ValueError("tenant_id must be nonempty")
        with self._lock:
            self.invalidate(tenant_id=tenant_id)
            if tenant_id is None:
                self._default_scope = scope
                self._scopes.clear()
            else:
                self._scopes[tenant_id] = scope

    def _recover(self, namespace, tenant, scope, generation):
        records = self._persistent.recover_candidates(namespace, tenant, generation)
        # Remove this namespace's memory view; preserve object identity for unchanged
        # rows so overlapping approvals do not revoke each other spuriously.
        previous = {k: v for k, v in self._entries.items() if v.scope_hash == namespace}
        for key in previous:
            self._bytes -= self._entries.pop(key).size
        for identity, source_key, token, request_json, vector_json, response_json, expires in records:
            request, answer = json.loads(request_json), json.loads(response_json)
            raw = json.loads(vector_json)
            if (not isinstance(raw, list) or not 0 < len(raw) <= self.max_dimensions
                    or not ScopedResponseCache._admissible(answer, request)
                    or ScopedResponseCache.request_key(request, tenant, scope) != source_key):
                raise ValueError("invalid durable candidate")
            vector = tuple(normalize(raw))
            size = sum(len(x.encode()) for x in (request_json, response_json)) + len(vector)*8 + len(_serialize(scope).encode())
            candidate = _Candidate(tenant, namespace, request_json, response_json, vector, expires, size, source_key, token)
            self._entries[identity] = previous[identity] if previous.get(identity) == candidate else candidate
            self._bytes += size
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            self._bytes -= self._entries.popitem(last=False)[1].size

    def candidate_namespace(self, body, tenant_id, scope):
        scoped = copy.deepcopy(body)
        if isinstance(scoped.get('messages'), list) and scoped['messages']:
            scoped['messages'][-1]['content'] = '<retrieval-query>'
        return digest({'tenant':tenant_id,'scope':scope,'request':scoped,
                       'encoder':self.encoder.identity,
                       'policy':getattr(self.approver,'identity',None),
                       'threshold':self.threshold,'margin':self.margin,
                       'contract':'source-bound-candidates-v1'})

    def process(self, request, *, tenant_id, provider, bypass=False, ledger=None):
        started = time.perf_counter()
        calls, generated_at, vector = 0, None, None
        with self._lock:
            scope = asdict(self._scopes.get(tenant_id, self._default_scope))
        body = json.loads(_serialize(dict(request)))
        diagnosis = {"mode": self.semantic_mode, "stage": "exact_lookup", "candidate": None,
                     "similarity": None, "verifier_calls": 0, "approval_signal": None,
                     "approver_identity": getattr(self.approver, "identity", None),
                     "local_verifier_attempts": 0, "local_verifier_ms": 0.0,
                     "local_policy_recommendation": None}

        def metered(outgoing):
            nonlocal calls
            calls += 1
            return provider(copy.deepcopy(outgoing))

        def generate(outgoing):
            nonlocal generated_at
            generated_at = self.clock()
            return metered(outgoing)

        def resolve(checked):
            nonlocal vector
            if checked["messages"][-1]["role"] != "user":
                diagnosis["stage"] = "unsupported_semantic_scope"
                return None
            try:
                with self._encoder_lock, ledger.measure('embedding') if ledger is not None else nullcontext():
                    raw = self.encoder.encode(checked["messages"][-1]["content"])
                    if not hasattr(raw, "__len__") or not 0 < len(raw) <= self.max_dimensions:
                        raise ValueError("embedding dimensions exceed bound")
                    vector = tuple(normalize(raw))
                with self._lock, ledger.measure('search') if ledger is not None else nullcontext():
                    self._sync_generation()
                    self._expire()
                    if self._persistent is not None:
                        self._recover(namespace, tenant_id, scope, generation)
                    candidates = []
                    for key, entry in self._entries.items():
                        if entry.scope_hash != namespace or entry.tenant_id != tenant_id:
                            continue
                        if len(entry.vector) != len(vector):
                            raise ValueError("embedding dimension changed")
                        candidates.append((sum(a * b for a, b in zip(vector, entry.vector)), key, entry))
                candidates.sort(key=lambda item: (-item[0], item[1]))
                if not candidates:
                    diagnosis["stage"] = "no_candidate"
                    return None
                score, key, candidate = candidates[0]
                diagnosis.update(similarity=score, candidate=key)
                if score < self.threshold:
                    diagnosis["stage"] = "threshold_rejected"
                    return None
                if len(candidates) > 1 and score - candidates[1][0] < self.margin:
                    diagnosis["stage"] = "margin_rejected"
                    return None
                candidate_request, answer = json.loads(candidate.request_json), json.loads(candidate.response_json)
                binding = approval_binding(checked, candidate_request, answer, tenant_id=tenant_id, scope=scope)
                diagnosis["binding"] = binding
                if self.semantic_mode == "shadow":
                    diagnosis["stage"] = "shadow_candidate"
                    return None
                before = calls
                try:
                    def forbidden_remote(_):
                        raise ValueError('remote approval is disabled on the document hit path')
                    with ledger.measure('applicability') if ledger is not None else nullcontext():
                        verdict = self.approver(copy.deepcopy(checked), candidate_request, copy.deepcopy(answer),
                                                binding=copy.deepcopy(binding),
                                                provider=forbidden_remote if self.local_approval_only else metered)
                finally:
                    diagnosis["verifier_calls"] = calls - before
                if not isinstance(verdict, ReuseDecision) or type(verdict.approved) is not bool or not isinstance(verdict.reason, str):
                    raise ValueError("invalid approval contract")
                if (type(verdict.local_verifier_attempts) is not int or verdict.local_verifier_attempts < 0
                        or type(verdict.local_verifier_ms) not in (int, float)
                        or not math.isfinite(verdict.local_verifier_ms) or verdict.local_verifier_ms < 0
                        or (verdict.local_policy_recommendation is not None and type(verdict.local_policy_recommendation) is not bool)):
                    raise ValueError("invalid local verification work")
                diagnosis.update(local_verifier_attempts=verdict.local_verifier_attempts,
                                 local_verifier_ms=verdict.local_verifier_ms,
                                 local_policy_recommendation=verdict.local_policy_recommendation)
                diagnosis["approval_signal"] = verdict.approved
                if not verdict.approved:
                    diagnosis["stage"] = "approval_rejected"
                    return None
                with self._lock:
                    if (self._sync_generation() != generation or self._entries.get(key) is not candidate
                            or candidate.expires <= self.clock()
                            or (self._persistent is not None and not self._persistent.candidate_current(
                                key, candidate.source_key, candidate.source_token, namespace, tenant_id, generation))
                            or self.exact._eligibility_reason(checked) is not None):
                        diagnosis["stage"] = "candidate_revoked"
                        return None
                    self._entries.move_to_end(key)
                    diagnosis["stage"] = "semantic_hit"
                    return ServingCacheResult(answer, "HIT", "guarded_semantic_hit", calls, False,
                                              (time.perf_counter() - started) * 1000, "SEMANTIC", copy.deepcopy(diagnosis))
            except Exception:
                # No retrieval or approval failure may authorize semantic reuse.
                vector = None
                diagnosis["stage"] = "semantic_processing_failed"
                return None

        generation, namespace = None, None
        semantic_active = self.semantic_mode in {"shadow", "guarded"} and not bypass
        if semantic_active:
            try:
                with self._lock:
                    generation = self._sync_generation()
                namespace = self.candidate_namespace(body, tenant_id, scope)
            except (sqlite3.Error, ValueError, TypeError):
                semantic_active = False
                diagnosis["stage"] = "semantic_scope_unavailable"
        result = self.exact.process(body, tenant_id=tenant_id, provider=generate,
                                    bypass=bypass or self.semantic_mode == "off", key_context=scope,
                                    on_miss=resolve if semantic_active else None)
        if result.status == "MISS" and result.stored and vector is not None and generated_at is not None:
            try:
                request_json, response_json = _serialize(body), _serialize(result.response)
                if self._persistent is not None:
                    self._persistent.admit_candidate(namespace=namespace,
                        source_key=ScopedResponseCache.request_key(body, tenant_id, scope),
                        tenant=tenant_id, payload=response_json, epoch=generation,
                        request=request_json, vector=vector,
                        max_entries=self.max_entries, max_bytes=self.max_bytes)
                    with self._lock:
                        self._recover(namespace, tenant_id, scope, generation)
                    return ServingCacheResult(copy.deepcopy(result.response), result.status, result.reason, calls,
                                              result.stored, (time.perf_counter()-started)*1000,
                                              "PROVIDER", copy.deepcopy(diagnosis))
                # Conservative origin: never extend the source response's TTL.
                expires = generated_at + self.exact._ttl_seconds
                key = digest({"tenant": tenant_id, "scope": scope, "request": body, "response": result.response})
                size = len(request_json.encode()) + len(response_json.encode()) + len(vector) * 8 + len(_serialize(scope).encode())
                with self._lock:
                    if self._sync_generation() == generation and expires > self.clock() and size <= self.max_bytes:
                        if key in self._entries:
                            self._bytes -= self._entries.pop(key).size
                        self._entries[key] = _Candidate(tenant_id, namespace, request_json, response_json, vector, expires, size)
                        self._bytes += size
                        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                            self._bytes -= self._entries.popitem(last=False)[1].size
            except (sqlite3.Error, TypeError, ValueError):
                diagnosis["index_admission_failed"] = True
        if result.status == "HIT" and not isinstance(result, ServingCacheResult):
            diagnosis["stage"] = "exact_hit"
        tier = result.tier if isinstance(result, ServingCacheResult) else "EXACT" if result.status == "HIT" else "PROVIDER"
        return ServingCacheResult(copy.deepcopy(result.response), result.status, result.reason, calls,
                                  result.stored, (time.perf_counter() - started) * 1000,
                                  tier, copy.deepcopy(diagnosis))
