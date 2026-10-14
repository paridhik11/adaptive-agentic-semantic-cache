"""Real loopback HTTP transport tests; no external provider or paid calls."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.api.gateway import create_app
from src.api.providers import HTTPChatProvider, FixtureChatProvider, ProviderError
from src.api.service import GatewayService
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache


@contextmanager
def upstream(kind="ok"):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"body": body, "authorization": self.headers.get("Authorization"), "path": self.path})
            if kind == "redirect":
                self.send_response(302)
                self.send_header("Location", "/should-not-follow")
                self.end_headers()
                return
            if kind == "rate":
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b'{"error":"upstream-secret-body"}')
                return
            if kind == "slow":
                time.sleep(0.1)
            self.send_response(200)
            self.end_headers()
            payload = b"invalid JSON" if kind == "invalid" else json.dumps(FixtureChatProvider()(body)).encode()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


BODY = {"model": "local-model", "messages": [{"role": "user", "content": "What is gravity?"}],
        "temperature": 0, "max_tokens": 64}


def test_gateway_to_loopback_provider_preserves_request_and_separates_credentials():
    with upstream() as (url, requests):
        provider = HTTPChatProvider(url, api_key="fixture-upstream-key")
        cache = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "fixture"))
        service = GatewayService(cache, provider, allowed_models={"local-model"})
        client = TestClient(create_app(service, tenant_tokens={"fixture-client-credential": "tenant"}))
        headers = {"Authorization": "Bearer fixture-client-credential"}
        first = client.post("/v1/chat/completions", json=BODY, headers=headers)
        second = client.post("/v1/chat/completions", json=BODY, headers=headers)
        assert first.status_code == second.status_code == 200
        assert first.headers["x-cache"] == "MISS"
        assert second.headers["x-cache"] == "HIT"
        assert len(requests) == 1
        assert requests[0]["body"] == BODY
        assert requests[0]["authorization"] == "Bearer fixture-upstream-key"


@pytest.mark.parametrize("kind,code,status", [("redirect", "upstream_status", 502),
                                               ("rate", "upstream_rate_limited", 503),
                                               ("invalid", "invalid_upstream_json", 502),
                                               ("slow", "upstream_timeout", 504)])
def test_transport_failures_are_bounded_sanitized_and_not_retried(kind, code, status):
    with upstream(kind) as (url, requests):
        provider = HTTPChatProvider(url, timeout_seconds=0.03 if kind == "slow" else 2)
        with pytest.raises(ProviderError) as exc:
            provider(BODY)
        assert exc.value.code == code
        assert exc.value.status == status
        assert "upstream-secret-body" not in str(exc.value)
        assert len(requests) == 1


def test_response_byte_limit():
    with upstream() as (url, _):
        with pytest.raises(ProviderError, match="upstream_response_too_large"):
            HTTPChatProvider(url, max_response_bytes=16)(BODY)


@pytest.mark.parametrize("url", ["http://example.com/v1/chat/completions", "file:///tmp/secret",
                                 "https://user:password@example.com/v1/chat/completions",
                                 "https://example.com/v1/chat/completions?key=secret"])
def test_invalid_endpoint_configuration_rejected(url):
    with pytest.raises(ValueError):
        HTTPChatProvider(url)
