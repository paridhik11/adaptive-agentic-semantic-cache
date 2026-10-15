"""Authenticated non-streaming chat gateway; start with app_from_env factory."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sqlite3

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from src.api.providers import FixtureChatProvider, HTTPChatProvider, ProviderError
from src.api.service import ConservativeRulesPolicy, GatewayBusy, GatewayService, InvalidRequest
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.sqlite_store import SQLiteResponseStore


def _reject_constant(value):
    raise ValueError("non-finite JSON")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(raw):
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


class Credentials:
    def __init__(self, tokens: dict[str, str], admin_token: str | None):
        if not isinstance(tokens, dict) or not tokens:
            raise ValueError("configure at least one tenant credential")
        self._tokens = []
        for token, tenant in tokens.items():
            self._validate(token)
            if not isinstance(tenant, str) or not tenant.strip():
                raise ValueError("tenant IDs must be non-empty strings")
            self._tokens.append((self._digest(token), tenant))
        if admin_token is not None:
            self._validate(admin_token)
            if admin_token in tokens:
                raise ValueError("admin and tenant credentials must be different")
        self._admin = self._digest(admin_token) if admin_token else None

    @staticmethod
    def _validate(token):
        if not isinstance(token, str) or not 16 <= len(token) <= 512 or any(c.isspace() for c in token):
            raise ValueError("credentials must contain 16–512 non-whitespace characters")

    @staticmethod
    def _digest(value):
        return hashlib.sha256(value.encode()).digest()

    def authenticate(self, request: Request, *, admin: bool = False):
        headers = request.headers.getlist("authorization")
        if len(headers) != 1:
            return None
        scheme, sep, token = headers[0].partition(" ")
        if not sep or scheme.lower() != "bearer" or not 16 <= len(token) <= 512:
            return None
        digest = self._digest(token)
        if admin:
            return bool(self._admin and secrets.compare_digest(digest, self._admin))
        for expected, tenant in self._tokens:
            if secrets.compare_digest(digest, expected):
                return tenant
        return None


def create_app(service: GatewayService, *, tenant_tokens: dict[str, str],
               admin_token: str | None = None, max_request_bytes: int = 65_536) -> FastAPI:
    if type(max_request_bytes) is not int or max_request_bytes < 1:
        raise ValueError("max_request_bytes must be a positive integer")
    credentials = Credentials(tenant_tokens, admin_token)
    mode = getattr(service.provider, "mode", "custom")
    if mode not in ("live", "simulated_fixture", "custom"):
        raise ValueError("unsupported provider mode")
    app = FastAPI(title="Adaptive response cache gateway", docs_url=None,
                  redoc_url=None, openapi_url=None)

    def error(code: str, status: int, message: str | None = None):
        headers = {"Cache-Control": "no-store", "X-Evaluation-Mode": mode}
        if status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse({"error": {"type": code, "message": message or code}}, status_code=status, headers=headers)

    @app.get("/health")
    def health():
        return {"status": "ok", "provider_mode": mode, "semantic_reuse_enabled": False}

    @app.get("/metrics")
    def metrics(request: Request):
        if not credentials.authenticate(request, admin=True):
            return error("unauthorized", 401)
        info = '# TYPE cache_gateway_info gauge\n' + f'cache_gateway_info{{provider_mode="{mode}"}} 1\n'
        return PlainTextResponse(info + service.metrics.render(), media_type="text/plain; version=0.0.4",
                                 headers={"Cache-Control": "no-store"})

    @app.delete("/v1/cache")
    def invalidate(request: Request):
        tenant = credentials.authenticate(request)
        if tenant is None:
            return error("unauthorized", 401)
        try:
            removed = service.cache.invalidate(tenant_id=tenant)
        except sqlite3.Error:
            return error("cache_invalidation_failed", 503)
        return JSONResponse({"removed": removed},
                            headers={"Cache-Control": "no-store"})

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        tenant = credentials.authenticate(request)
        if tenant is None:
            return error("unauthorized", 401)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return error("application_json_required", 415)
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            return error("content_encoding_not_supported", 415)
        if request.headers.get("x-cache-mode", "default") not in ("default", "bypass"):
            return error("invalid_cache_mode", 400)
        async def read_body():
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > max_request_bytes:
                    raise OverflowError()
                raw.extend(chunk)
            return bytes(raw)
        try:
            raw = await asyncio.wait_for(read_body(), timeout=10)
            body = _strict_json(raw)
        except OverflowError:
            return error("request_too_large", 413)
        except (ValueError, UnicodeError, RecursionError):
            return error("invalid_json", 400)
        except (TimeoutError, ClientDisconnect):
            return error("request_body_timeout_or_disconnect", 408)
        try:
            result = await run_in_threadpool(service.complete, body, tenant,
                                            bypass=request.headers.get("x-cache-mode") == "bypass")
            return JSONResponse(result.response, headers={
                "X-Cache": result.status, "X-Cache-Tier": "EXACT" if result.status == "HIT" else "PROVIDER",
                "X-Cache-Reason": result.reason, "X-Provider-Calls": str(result.provider_calls),
                "X-Evaluation-Mode": mode, "Cache-Control": "no-store"})
        except InvalidRequest as exc:
            return error("invalid_request_error", 400, str(exc))
        except GatewayBusy:
            return error("gateway_busy", 503)
        except ProviderError as exc:
            return error(exc.code, exc.status)
        except Exception:
            return error("upstream_or_processing_error", 502)

    return app


def app_from_env() -> FastAPI:
    """No import-time credentials, models, network calls, or automatic .env load."""
    tokens = _strict_json(os.environ["GATEWAY_TENANTS_JSON"])
    mode = os.environ.get("GATEWAY_PROVIDER", "fixture")
    if mode == "fixture":
        provider = FixtureChatProvider()
        models = {"fixture-model"}
    elif mode == "http":
        provider = HTTPChatProvider(os.environ["UPSTREAM_CHAT_URL"],
                                    api_key=os.environ.get("UPSTREAM_API_KEY"),
                                    timeout_seconds=float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "60")))
        models = {m.strip() for m in os.environ["GATEWAY_MODELS"].split(",") if m.strip()}
    else:
        raise ValueError("GATEWAY_PROVIDER must be fixture or http")
    policy_name = os.environ.get("GATEWAY_CACHE_POLICY", "bypass")
    if policy_name == "rules":
        policy = ConservativeRulesPolicy()
    elif policy_name == "bypass":
        policy = lambda _: EligibilityDecision(False, "deployment_cache_disabled")
    else:
        raise ValueError("GATEWAY_CACHE_POLICY must be bypass or rules")
    storage = None
    if os.environ.get("GATEWAY_CACHE_DB"):
        version = os.environ.get("GATEWAY_CACHE_VERSION", "").strip()
        if not version:
            raise ValueError("GATEWAY_CACHE_VERSION is required with persistent storage")
        # Version must change when model aliases, policy code or upstream data change.
        scope = {"version": version, "provider": mode, "policy": policy_name,
                 "endpoint": os.environ.get("UPSTREAM_CHAT_URL") if mode == "http" else None,
                 "models": sorted(models),
                 "ttl": os.environ.get("GATEWAY_TTL_SECONDS", "3600"),
                 "entries": os.environ.get("GATEWAY_MAX_ENTRIES", "1024"),
                 "bytes": os.environ.get("GATEWAY_MAX_CACHE_BYTES", "67108864")}
        namespace = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        storage = SQLiteResponseStore(os.environ["GATEWAY_CACHE_DB"], namespace=namespace)
    cache = ScopedResponseCache(eligibility=policy, storage=storage,
                                ttl_seconds=float(os.environ.get("GATEWAY_TTL_SECONDS", "3600")),
                                max_entries=int(os.environ.get("GATEWAY_MAX_ENTRIES", "1024")),
                                max_bytes=int(os.environ.get("GATEWAY_MAX_CACHE_BYTES", "67108864")))
    service = GatewayService(cache, provider, allowed_models=models,
                             max_inflight=int(os.environ.get("GATEWAY_MAX_INFLIGHT", "32")),
                             max_output_tokens=int(os.environ.get("GATEWAY_MAX_OUTPUT_TOKENS", "4096")))
    return create_app(service, tenant_tokens=tokens, admin_token=os.environ.get("GATEWAY_ADMIN_TOKEN"))
