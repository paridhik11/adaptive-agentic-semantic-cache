"""Gateway request policy, validation, and bounded aggregate work accounting."""
from __future__ import annotations

from collections import Counter
import json
import math
import re
import threading
import time

from src.classifier.models import StabilityLabel
from src.classifier.rules import StabilityRuleEngine
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache


class InvalidRequest(ValueError):
    pass


class GatewayBusy(Exception):
    pass


class ConservativeRulesPolicy:
    """Opt-in heuristic for single-user static questions; no trained fallback.

    This is not calibrated safety certification. Contextual, creative, personal,
    uncertain, and unsupported requests bypass. Complete request scope remains
    in the exact key even when a deployment injects a different eligibility policy.
    """

    _personal = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+?\d[\d ()-]{7,}\d)")
    _creative = re.compile(r"\b(write|compose|draft|invent|brainstorm|create|generate|random)\b", re.I)

    def __init__(self):
        self._rules = StabilityRuleEngine()

    def __call__(self, body: dict) -> EligibilityDecision:
        messages = body["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            return EligibilityDecision(False, "context_requires_review")
        text = messages[0]["content"]
        if len(text.split()) > 128:
            return EligibilityDecision(False, "long_request")
        if self._personal.search(text) or self._creative.search(text):
            return EligibilityDecision(False, "personal_or_generative_request")
        rule = self._rules.evaluate(text)
        eligible = rule.matched and rule.label == StabilityLabel.STABLE
        return EligibilityDecision(eligible, "rule_confirmed_static" if eligible else "not_rule_confirmed_static")


def validate_request(body: object, allowed_models: frozenset[str], max_output_tokens: int) -> dict:
    if not isinstance(body, dict):
        raise InvalidRequest("body must be a JSON object")
    fields = {"model", "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens",
              "frequency_penalty", "presence_penalty", "seed", "stop", "n", "stream", "response_format"}
    if set(body) - fields:
        raise InvalidRequest("unsupported request field")
    if not isinstance(body.get("model"), str) or body["model"] not in allowed_models:
        raise InvalidRequest("model is not enabled")
    messages = body.get("messages")
    if (not isinstance(messages, list) or not 1 <= len(messages) <= 128
            or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                   or m["role"] not in ("system", "developer", "user", "assistant")
                   or not isinstance(m["content"], str) or not m["content"].strip() for m in messages)):
        raise InvalidRequest("messages must contain supported text role/content objects")
    if "stream" in body and body["stream"] is not False:
        raise InvalidRequest("streaming is not supported in this gateway milestone")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise InvalidRequest("only n=1 is supported")
    for field, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                             ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
        if field in body:
            value = body[field]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise InvalidRequest("invalid generation parameter")
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise InvalidRequest("specify only one output token limit")
    for field in ("max_tokens", "max_completion_tokens"):
        if field in body and (type(body[field]) is not int or not 1 <= body[field] <= max_output_tokens):
            raise InvalidRequest("output token limit is outside gateway bounds")
    if not {"max_tokens", "max_completion_tokens"}.intersection(body):
        raise InvalidRequest("an explicit output token limit is required")
    if "seed" in body and (type(body["seed"]) is not int or not -(2**63) <= body["seed"] < 2**63):
        raise InvalidRequest("invalid seed")
    if "stop" in body:
        stop = body["stop"]
        if not (isinstance(stop, str) and 1 <= len(stop) <= 256 or
                isinstance(stop, list) and 1 <= len(stop) <= 4 and
                all(isinstance(s, str) and 1 <= len(s) <= 256 for s in stop)):
            raise InvalidRequest("invalid stop sequence")
    if "response_format" in body and body["response_format"] not in ({"type": "text"}, {"type": "json_object"}):
        raise InvalidRequest("unsupported response_format")
    return body


class WorkMetrics:
    """Fixed-cardinality counters and histogram; no prompts, tenants, or keys."""

    buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0)

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = Counter()
        self._latency = {status: [0] * (len(self.buckets) + 2)
                         for status in ("HIT", "MISS", "BYPASS", "ERROR")}

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[key] += amount

    def usage(self, response: dict) -> None:
        usage = response.get("usage")
        fields = ("prompt_tokens", "completion_tokens")
        if not isinstance(usage, dict) or any(type(usage.get(f)) is not int or usage[f] < 0 for f in fields):
            self.increment("upstream_usage_missing_total")
            return
        with self._lock:
            for field in fields:
                self._counts["upstream_" + field + "_total"] += usage[field]

    def finish(self, status: str, elapsed_seconds: float) -> None:
        with self._lock:
            row = self._latency[status]
            for i, boundary in enumerate(self.buckets):
                row[i] += int(elapsed_seconds <= boundary)
            row[-2] += 1
            row[-1] += elapsed_seconds

    def render(self) -> str:
        with self._lock:
            lines = []
            for key in ("upstream_attempts_total", "upstream_errors_total", "upstream_usage_missing_total",
                        "upstream_prompt_tokens_total", "upstream_completion_tokens_total",
                        "requests_rejected_busy_total", "stream_completed_total", "stream_failed_total",
                        "stream_cancelled_total", "stream_cleanup_errors_total", "stream_first_text_microseconds_total", "stream_first_text_count"):
                lines += [f"# TYPE cache_gateway_{key} counter", f"cache_gateway_{key} {self._counts[key]}"]
            lines.append("# TYPE cache_gateway_processing_seconds histogram")
            for status, row in self._latency.items():
                for bound, count in zip(self.buckets, row):
                    lines.append(f'cache_gateway_processing_seconds_bucket{{status="{status}",le="{bound}"}} {count}')
                lines += [f'cache_gateway_processing_seconds_bucket{{status="{status}",le="+Inf"}} {row[-2]}',
                          f'cache_gateway_processing_seconds_count{{status="{status}"}} {row[-2]}',
                          f'cache_gateway_processing_seconds_sum{{status="{status}"}} {row[-1]}']
            return "\n".join(lines) + "\n"


class GatewayService:
    def __init__(self, cache: ScopedResponseCache, provider, *, allowed_models: set[str],
                 max_inflight: int = 32, max_output_tokens: int = 4096):
        if not allowed_models or any(not isinstance(m, str) or not m.strip() for m in allowed_models):
            raise ValueError("at least one model must be enabled")
        if type(max_inflight) is not int or max_inflight < 1 or type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("gateway limits must be positive integers")
        if not callable(provider):
            raise TypeError("provider must be callable")
        self.cache, self.provider = cache, provider
        self.allowed_models = frozenset(allowed_models)
        self.max_output_tokens = max_output_tokens
        self.metrics = WorkMetrics()
        self._slots = threading.BoundedSemaphore(max_inflight)

    def open_stream(self, body: dict, tenant_id: str):
        from src.api.streaming import StreamSession
        if not isinstance(tenant_id,str) or not tenant_id.strip():raise InvalidRequest('trusted tenant required')
        if not isinstance(body,dict) or body.get('stream') is not True:raise InvalidRequest('stream must be true')
        checked = dict(body)
        options=checked.pop('stream_options',None)
        if options is not None and options != {'include_usage':True}:raise InvalidRequest('only include_usage=true is supported')
        checked['stream']=False
        validate_request(checked,self.allowed_models,self.max_output_tokens)
        if not callable(getattr(self.provider,'open_stream',None)):raise InvalidRequest('provider does not support streaming')
        if not self._slots.acquire(blocking=False):
            self.metrics.increment('requests_rejected_busy_total')
            raise GatewayBusy()
        started=time.perf_counter()
        self.metrics.increment('upstream_attempts_total')
        source=None
        try:
            outgoing={**checked,'stream':True,'stream_options':{'include_usage':True}}
            source=self.provider.open_stream(outgoing)
            return StreamSession(source,self.metrics,self._slots.release,started)
        except Exception:
            if source is not None and hasattr(source,'close'):
                try:source.close()
                except Exception:self.metrics.increment('stream_cleanup_errors_total')
            self.metrics.increment('upstream_errors_total')
            self.metrics.increment('upstream_usage_missing_total')
            self.metrics.finish('ERROR',time.perf_counter()-started)
            self._slots.release()
            raise

    def complete(self, body: dict, tenant_id: str, *, bypass: bool = False):
        validate_request(body, self.allowed_models, self.max_output_tokens)
        if not self._slots.acquire(blocking=False):
            self.metrics.increment("requests_rejected_busy_total")
            raise GatewayBusy()
        started = time.perf_counter()
        status = "ERROR"
        try:
            def generate(request):
                self.metrics.increment("upstream_attempts_total")
                try:
                    response = self.provider(request)
                    if not isinstance(response, dict):
                        raise TypeError("provider returned a non-object")
                    json.dumps(response, allow_nan=False)
                    self.metrics.usage(response)
                    return response
                except Exception:
                    self.metrics.increment("upstream_errors_total")
                    raise
            result = self.cache.process(body, tenant_id=tenant_id, provider=generate, bypass=bypass)
            status = result.status
            return result
        finally:
            self.metrics.finish(status, time.perf_counter() - started)
            self._slots.release()
