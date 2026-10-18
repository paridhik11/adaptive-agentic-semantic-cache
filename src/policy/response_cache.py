"""Scoped exact-response cache for the recovery runtime.

Unlike the historical pair/stream harnesses, this component calls a provider
on a miss and stores its complete response. Eligibility runs before every
lookup and admission. No normalization or semantic fast path is enabled.
The caller supplies an explicit stability/eligibility policy; absence of a
policy cannot silently make dynamic requests cacheable.

Tenant identity must come from a trusted application boundary. This module
provides namespace isolation, not authentication. Storage is bounded and
thread-safe within one process; optional SQLite storage persists answers
and coordinates invalidation across local processes. Streaming is unsupported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class EligibilityDecision:
    cacheable: bool
    reason: str


@dataclass(frozen=True)
class ResponseCacheResult:
    response: JsonObject
    status: str  # HIT, MISS or BYPASS
    reason: str
    provider_calls: int
    stored: bool
    total_latency_ms: float


@dataclass(frozen=True)
class _Entry:
    tenant_id: str
    response_json: str
    expires_at: float
    payload_bytes: int


def _serialize(value: Any) -> str:
    """Stable JSON serialization; reject NaN and non-JSON request values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class ScopedResponseCache:
    """Cache complete non-streaming responses under the full request scope.

    All request fields (including every message, model, generation parameters,
    response format and metadata) participate in the key. Text is preserved
    verbatim. Unsupported request types are forwarded without caching.

    Fixed lock stripes bound coordination memory and coalesce cacheable
    identical misses. Invalidation increments an epoch so an in-flight
    generation cannot repopulate invalidated data. No global lock is held
    while calling the upstream provider.
    """

    def __init__(
        self,
        *,
        eligibility: Callable[[JsonObject], EligibilityDecision],
        ttl_seconds: float = 3600.0,
        max_entries: int = 1024,
        max_bytes: int = 64 * 1024 * 1024,
        temperature_cap: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        storage=None,
    ) -> None:
        if not callable(eligibility):
            raise TypeError("an explicit eligibility policy is required")
        if isinstance(ttl_seconds, bool) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if isinstance(temperature_cap, bool) or not math.isfinite(temperature_cap) or not 0 <= temperature_cap <= 2:
            raise ValueError("temperature_cap must be finite and in [0, 2]")
        self._storage = storage
        self._eligibility = eligibility
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._payload_bytes = 0
        self._temperature_cap = temperature_cap
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._stripes = tuple(threading.Lock() for _ in range(64))
        self._epoch = 0

    def _eligibility_reason(self, request: JsonObject) -> str | None:
        temperature = request.get("temperature")
        # An omitted temperature uses a provider-dependent default, not zero.
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or not 0 <= temperature <= self._temperature_cap):
            return "temperature_not_eligible"
        if request.get("tools") or request.get("functions"):
            return "tool_requests_not_supported"
        if any(not isinstance(m.get("content"), str) for m in request["messages"]):
            return "non_text_messages_not_supported"
        try:
            decision = self._eligibility(copy.deepcopy(request))
        except Exception:
            return "eligibility_policy_failed"
        if (not isinstance(decision, EligibilityDecision)
                or type(decision.cacheable) is not bool
                or not isinstance(decision.reason, str)):
            return "invalid_eligibility_decision"
        return None if decision.cacheable else decision.reason or "policy_bypass"

    @staticmethod
    def _admissible(response: JsonObject, request: JsonObject) -> bool:
        if "error" in response or not isinstance(response.get("model"), str):
            return False
        choices = response.get("choices")
        expected = request.get("n", 1)
        if (type(expected) is not int or expected < 1 or not isinstance(choices, list)
                or len(choices) != expected):
            return False
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                return False
            message = choice.get("message")
            if (not isinstance(message, dict) or message.get("role") != "assistant"
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                    or message.get("tool_calls") or message.get("function_call")
                    or message.get("refusal")):
                return False
        return True

    def process(
        self,
        request: Mapping[str, Any],
        *,
        tenant_id: str,
        provider: Callable[[JsonObject], Mapping[str, Any]],
        bypass: bool = False,
        key_context: Mapping[str, Any] | None = None,
        on_miss: Callable[[JsonObject], ResponseCacheResult | None] | None = None,
    ) -> ResponseCacheResult:
        """Return a stored answer or execute the supplied provider exactly once.

        Provider exceptions propagate to the caller and are never admitted.
        key_context is trusted server-side scope and is never forwarded upstream.
        on_miss is a trusted extension invoked only after eligible exact lookup.
        Its hits are never readmitted as fresh generations or given a new TTL.
        Usage inside a cached response describes its original generation;
        ``provider_calls`` distinguishes a hit from newly consumed work.
        """
        started = time.perf_counter()
        if type(bypass) is not bool:
            raise ValueError("bypass must be a boolean")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("a trusted non-empty tenant_id is required")
        body = json.loads(_serialize(dict(request)))  # snapshot caller-owned data
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise ValueError("request.model must be a non-empty string")
        messages = body.get("messages")
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(m, dict) or not isinstance(m.get("role"), str)
                       for m in messages)):
            raise ValueError("request.messages must contain message objects")
        if body.get("stream"):
            raise ValueError("streaming is not supported by this runtime yet")

        def result(response: JsonObject, status: str, reason: str,
                   calls: int, stored: bool) -> ResponseCacheResult:
            return ResponseCacheResult(response, status, reason, calls, stored,
                                       (time.perf_counter() - started) * 1000.0)

        def generate() -> JsonObject:
            response = provider(copy.deepcopy(body))
            if not isinstance(response, Mapping):
                raise TypeError("provider must return a response mapping")
            return copy.deepcopy(dict(response))

        reason = "cache_disabled_by_caller" if bypass else self._eligibility_reason(body)
        if reason is not None:
            return result(generate(), "BYPASS", reason, 1, False)

        key_data = {"tenant_id": tenant_id, "request": body}
        if key_context is not None:
            key_data["context"] = json.loads(_serialize(dict(key_context)))
        key = hashlib.sha256(_serialize(key_data).encode()).hexdigest()
        stripe = self._stripes[int(key[:8], 16) % len(self._stripes)]
        with stripe:
            # A queued request must recheck policy after the preceding call.
            reason = self._eligibility_reason(body)
            if reason is not None:
                return result(generate(), "BYPASS", reason, 1, False)
            if self._storage is not None:
                try:
                    payload, epoch = self._storage.lookup(key)
                    if payload is not None:
                        cached = json.loads(payload)
                        _serialize(cached)  # reject non-finite persisted JSON
                        if not isinstance(cached, dict) or not self._admissible(cached, body):
                            raise ValueError("invalid stored response")
                        return result(cached, "HIT", "scoped_exact_hit", 0, False)
                except (sqlite3.Error, ValueError, TypeError):
                    return result(generate(), "BYPASS", "cache_storage_unavailable", 1, False)
            else:
                with self._lock:
                    now = self._clock()
                    entry = self._entries.get(key)
                    if entry is not None and entry.expires_at > now:
                        self._entries.move_to_end(key)
                        return result(json.loads(entry.response_json), "HIT", "scoped_exact_hit", 0, False)
                    if entry is not None:
                        self._payload_bytes -= entry.payload_bytes
                        del self._entries[key]
                    epoch = self._epoch
            if on_miss is not None:
                resolved = on_miss(copy.deepcopy(body))
                if resolved is not None:
                    if (not isinstance(resolved, ResponseCacheResult) or resolved.status != "HIT"
                            or resolved.stored is not False
                            or type(resolved.provider_calls) is not int or resolved.provider_calls < 0
                            or not self._admissible(resolved.response, body)):
                        raise ValueError("invalid miss resolution")
                    return resolved
            response = generate()
            # Recheck after generation: policy can revoke eligibility while
            # upstream work is running. Invalid responses are never cached.
            reason = self._eligibility_reason(body)
            response_json = None
            if reason is None and self._admissible(response, body):
                try:
                    response_json = _serialize(response)
                except (TypeError, ValueError):
                    pass
            stored = False
            payload_bytes = len(response_json.encode("utf-8")) if response_json is not None else 0
            if self._storage is not None:
                if response_json is not None:
                    try:
                        stored = self._storage.admit(key, tenant_id, response_json, epoch,
                            ttl=self._ttl_seconds, max_entries=self._max_entries, max_bytes=self._max_bytes)
                    except sqlite3.Error:
                        return result(response, "MISS", "cache_storage_unavailable", 1, False)
            else:
                with self._lock:
                    if response_json is not None and payload_bytes <= self._max_bytes and epoch == self._epoch:
                        self._entries[key] = _Entry(tenant_id, response_json,
                                                   self._clock() + self._ttl_seconds, payload_bytes)
                        self._payload_bytes += payload_bytes
                        self._entries.move_to_end(key)
                        while len(self._entries) > self._max_entries or self._payload_bytes > self._max_bytes:
                            _, evicted = self._entries.popitem(last=False)
                            self._payload_bytes -= evicted.payload_bytes
                        stored = True
            return result(response, "MISS", "generated_and_stored" if stored else "response_not_admitted", 1, stored)

    def invalidate(self, *, tenant_id: str | None = None) -> int:
        """Remove a tenant namespace or all entries; block in-flight admission."""
        if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id.strip()):
            raise ValueError("tenant_id must be non-empty")
        if self._storage is not None:
            return self._storage.invalidate(tenant_id)
        with self._lock:
            self._epoch += 1
            keys = [key for key, entry in self._entries.items()
                    if tenant_id is None or entry.tenant_id == tenant_id]
            for key in keys:
                self._payload_bytes -= self._entries[key].payload_bytes
                del self._entries[key]
            return len(keys)

    @property
    def invalidation_epoch(self) -> int:
        """Generation token shared with a local SQLite store when configured."""
        if self._storage is not None:
            return self._storage.invalidation_epoch
        with self._lock:
            return self._epoch

    @property
    def size(self) -> int:
        if self._storage is not None:
            return self._storage.size
        with self._lock:
            now = self._clock()
            expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
            for key in expired:
                self._payload_bytes -= self._entries[key].payload_bytes
                del self._entries[key]
            return len(self._entries)
