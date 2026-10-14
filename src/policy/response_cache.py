"""Scoped exact-response cache for the recovery runtime.

Unlike the historical pair/stream harnesses, this component calls a provider
on a miss and stores its complete response. Eligibility runs before every
lookup and admission. No normalization or semantic fast path is enabled.
The caller supplies an explicit stability/eligibility policy; absence of a
policy cannot silently make dynamic requests cacheable.

Tenant identity must come from a trusted application boundary. This module
provides namespace isolation, not authentication. Storage is bounded and
thread-safe within one process; persistence and streaming remain separate
gateway work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
        temperature_cap: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(eligibility):
            raise TypeError("an explicit eligibility policy is required")
        if isinstance(ttl_seconds, bool) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(temperature_cap, bool) or not math.isfinite(temperature_cap) or not 0 <= temperature_cap <= 2:
            raise ValueError("temperature_cap must be finite and in [0, 2]")
        self._eligibility = eligibility
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
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
    ) -> ResponseCacheResult:
        """Return a stored answer or execute the supplied provider exactly once.

        Provider exceptions propagate to the caller and are never admitted.
        Usage inside a cached response describes its original generation;
        ``provider_calls`` distinguishes a hit from newly consumed work.
        """
        started = time.perf_counter()
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

        reason = self._eligibility_reason(body)
        if reason is not None:
            return result(generate(), "BYPASS", reason, 1, False)

        key = hashlib.sha256(_serialize({"tenant_id": tenant_id, "request": body}).encode()).hexdigest()
        stripe = self._stripes[int(key[:8], 16) % len(self._stripes)]
        with stripe:
            # A queued request must recheck policy after the preceding call.
            reason = self._eligibility_reason(body)
            if reason is not None:
                return result(generate(), "BYPASS", reason, 1, False)
            with self._lock:
                now = self._clock()
                entry = self._entries.get(key)
                if entry is not None and entry.expires_at > now:
                    self._entries.move_to_end(key)
                    return result(json.loads(entry.response_json), "HIT", "scoped_exact_hit", 0, False)
                if entry is not None:
                    del self._entries[key]
                epoch = self._epoch

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
            with self._lock:
                if response_json is not None and epoch == self._epoch:
                    self._entries[key] = _Entry(tenant_id, response_json,
                                               self._clock() + self._ttl_seconds)
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
                    stored = True
            return result(response, "MISS", "generated_and_stored" if stored else "response_not_admitted", 1, stored)

    def invalidate(self, *, tenant_id: str | None = None) -> int:
        """Remove a tenant namespace or all entries; block in-flight admission."""
        if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id.strip()):
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            self._epoch += 1
            keys = [key for key, entry in self._entries.items()
                    if tenant_id is None or entry.tenant_id == tenant_id]
            for key in keys:
                del self._entries[key]
            return len(keys)

    @property
    def size(self) -> int:
        with self._lock:
            now = self._clock()
            expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
            for key in expired:
                del self._entries[key]
            return len(self._entries)
