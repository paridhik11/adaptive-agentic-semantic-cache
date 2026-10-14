"""Bounded, non-streaming chat provider adapters. No implicit retries."""
from __future__ import annotations

import hashlib
import json
import math
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def _reject_constant(value):
    raise ValueError("non-finite upstream JSON")


class ProviderError(Exception):
    """Sanitized upstream error; raw provider bodies never reach clients."""

    def __init__(self, code: str, status: int = 502):
        super().__init__(code)
        self.code, self.status = code, status


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTPChatProvider:
    """POST to one operator-configured endpoint; never forward client auth."""

    mode = "live"

    def __init__(self, endpoint: str, *, api_key: str | None = None,
                 timeout_seconds: float = 60, max_response_bytes: int = 4_194_304):
        url = urlsplit(endpoint)
        loopback_http = url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
        if (url.scheme != "https" and not loopback_http) or not url.hostname:
            raise ValueError("upstream requires HTTPS, or HTTP on literal loopback/localhost")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("upstream URL must not contain credentials, query, or fragment")
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("provider timeout must be finite and positive")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        if api_key is not None and (not isinstance(api_key, str) or not api_key or "\n" in api_key or "\r" in api_key):
            raise ValueError("invalid upstream credential")
        self.endpoint, self.api_key = endpoint, api_key
        self.timeout_seconds, self.max_response_bytes = timeout_seconds, max_response_bytes
        self._opener = build_opener(_NoRedirect())

    def __call__(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.endpoint, data=json.dumps(body, allow_nan=False).encode(),
                          headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise ProviderError("upstream_status")
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ProviderError("upstream_response_too_large")
            parsed = json.loads(raw, parse_constant=_reject_constant)
            if not isinstance(parsed, dict) or "error" in parsed or not isinstance(parsed.get("choices"), list):
                raise ProviderError("invalid_upstream_response")
            return parsed
        except HTTPError as exc:
            exc.close()
            raise ProviderError("upstream_rate_limited" if exc.code == 429 else "upstream_status",
                                503 if exc.code == 429 else 502) from None
        except (TimeoutError, socket.timeout):
            raise ProviderError("upstream_timeout", 504) from None
        except URLError as exc:
            timeout = isinstance(exc.reason, (TimeoutError, socket.timeout))
            raise ProviderError("upstream_timeout" if timeout else "upstream_unavailable",
                                504 if timeout else 502) from None
        except (ValueError, UnicodeError):
            raise ProviderError("invalid_upstream_json") from None


class FixtureChatProvider:
    """Functional smoke-test fixture, not an LLM or latency/cost benchmark."""

    mode = "simulated_fixture"

    def __call__(self, body: dict) -> dict:
        text = "SIMULATED FIXTURE: " + body["messages"][-1]["content"]
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:20]
        prompt_tokens = sum(len(m["content"].split()) for m in body["messages"])
        completion_tokens = len(text.split())
        return {"id": "fixture-" + digest, "object": "chat.completion", "created": 0,
                "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}}
