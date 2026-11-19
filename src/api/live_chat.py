"""General-purpose local chat with a live provider and a scoped exact cache."""
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from src.api.demo import ObservedProvider, workbench_shell
from src.api.gateway import _strict_json
from src.api.service import ConservativeRulesPolicy
from src.benchmark.control import RunPaused
from src.benchmark.free_pilot import FreeOpenRouterProvider, JournaledProvider, PilotJournal
from src.classifier.models import StabilityLabel
from src.classifier.rules import StabilityRuleEngine
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.sqlite_store import SQLiteResponseStore

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
RUN_LIMIT = 30
MAX_QUESTION = 4_000
MAX_ANSWER_TOKENS = 1_024
TENANT = "local-general-chat"
INSTRUCTION = (
    "Answer clearly and directly. Use the conversation when provided. "
    "For current information that you cannot verify, say so. Do not invent citations."
)

_DEFINITION = re.compile(
    r"^(?:explain|define|compare|what (?:is|are)|how does)\b", re.IGNORECASE
)
_FRESHNESS = re.compile(
    r"\b(?:today|current(?:ly)?|latest|now|recent|update|weather|forecast|news|stock price|"
    r"yesterday|tomorrow|this (?:week|month|year)|my|our)\b", re.IGNORECASE
)

_MESSAGES = {
    "run_attempt_budget_exhausted": "The 30-attempt allowance is used up. Eligible cached answers still work.",
    "project_minute_quota": "The shared per-minute allowance was reached. Try again after the displayed time.",
    "project_daily_quota": "The shared daily allowance was reached. Try again after the displayed time.",
    "upstream_rate_limit_pause": "OpenRouter rate-limited this journal. Try again after the displayed time.",
    "incomplete_response_inspect_journal": "The model did not return a complete answer. Try a shorter question.",
    "upstream_attempt_failed_requires_inspection": "OpenRouter did not return an answer. Check the key and selected free model, then try again.",
    "uncertain_attempt_requires_review": "A previous provider attempt has uncertain status. Inspect the shared journal.",
}


def request_from_payload(value, model):
    if not isinstance(value, dict) or set(value) - {"question", "history", "cache"}:
        raise ValueError("Send a question, optional history, and cache preference.")
    question = value.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
        raise ValueError("Enter a question under 4,000 characters.")
    history = value.get("history", [])
    enabled = value.get("cache", True)
    if type(enabled) is not bool or not isinstance(history, list) or len(history) > 6 or len(history) % 2:
        raise ValueError("Invalid follow-up context or cache preference.")
    for index, item in enumerate(history):
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] != ("assistant" if index % 2 else "user")
                or not isinstance(item["content"], str) or not item["content"].strip()
                or len(item["content"]) > 8_000):
            raise ValueError("Invalid follow-up context.")
    return {
        "model": model, "temperature": 0, "max_tokens": MAX_ANSWER_TOKENS,
        "messages": [{"role": "system", "content": INSTRUCTION}, *history,
                     {"role": "user", "content": question.strip()}],
    }, enabled


def create_live_chat_app(directory, *, journal_path, api_key, model=DEFAULT_MODEL, benchmark_directory=None, transport=None):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Set OPENROUTER_API_KEY in PowerShell before launching live chat.")
    upstream = FreeOpenRouterProvider(api_key, models={model}, reasoning_enabled=False, transport=transport)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal = PilotJournal(journal_path, run_id="general-chat-v1", fingerprint="general-chat-free-v1", max_calls=RUN_LIMIT)
    provider = JournaledProvider(upstream, journal)
    observed = ObservedProvider(provider)
    policy = ConservativeRulesPolicy()
    stability = StabilityRuleEngine()

    def eligibility(body):
        # Keep the full system prompt in the cache key, and route conversations
        # through the provider. Common educational definitions can reuse exactly.
        turns = body["messages"][1:]
        decision = policy({"messages": turns})
        if len(turns) != 1:
            return decision
        question = turns[0]["content"]
        if _FRESHNESS.search(question) or stability.evaluate(question).label == StabilityLabel.DYNAMIC:
            return EligibilityDecision(False, "freshness_requires_provider")
        if decision.reason == "not_rule_confirmed_static" and _DEFINITION.match(question):
            return EligibilityDecision(True, "standalone_definition")
        return decision

    cache = ScopedResponseCache(
        eligibility=eligibility,
        storage=SQLiteResponseStore(directory / "chat.sqlite", namespace="general-chat-v1"),
    )
    lock = threading.RLock()
    session = secrets.token_urlsafe(32)
    hits = 0
    app, benchmark_link = workbench_shell(
        "Trace · Live chat", session, benchmark_directory, {"workbench.css", "live_chat.css", "live_chat.js"},
    )

    def stats():
        return {"model": model, "used": sum(journal.status().values()), "limit": RUN_LIMIT, "hits": hits}

    @app.get("/", response_class=HTMLResponse)
    def page():
        html = Path(__file__).with_name("live_chat.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__SESSION__", session).replace("__BENCHMARK_LINK__", benchmark_link))

    @app.get("/api/chat")
    def info():
        with lock:
            return stats()

    def execute(body, enabled):
        nonlocal hits
        with lock:
            before = provider.new_calls
            started = time.perf_counter()
            upstream_ms = None

            def generate(request):
                nonlocal upstream_ms
                contact = time.perf_counter()
                try:
                    with journal.transaction() as db:
                        if db.execute(
                            "SELECT 1 FROM pilot_calls WHERE run=? AND state='reserved' LIMIT 1",
                            (journal.run_id,),
                        ).fetchone():
                            raise RunPaused("uncertain_attempt_requires_review")
                        provider.ordinal = db.execute(
                            "SELECT COALESCE(MAX(ordinal),-1)+1 FROM pilot_calls WHERE run=?", (journal.run_id,)
                        ).fetchone()[0]
                    replayed = provider.replayed_calls
                    result = observed(request)
                    if provider.replayed_calls != replayed:
                        raise RunPaused("concurrent_provider_attempt")
                    return result
                except sqlite3.Error:
                    raise RunPaused("quota_journal_unavailable_or_write_uncertain") from None
                finally:
                    upstream_ms = (time.perf_counter() - contact) * 1_000

            try:
                result = cache.process(body, tenant_id=TENANT, provider=generate, bypass=not enabled)
            except RunPaused as exc:
                return JSONResponse({
                    "error": _MESSAGES.get(exc.reason, "The provider request stopped. Check the shared quota journal."),
                    "retry_at": exc.retry_at, "provider_calls": provider.new_calls - before,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                }, status_code=503)

            hits += result.status == "HIT"
            calls = provider.new_calls - before
            usage = result.response.get("usage")
            usage = usage.get("completion_tokens") if isinstance(usage, dict) else None
            return {
                "answer": result.response["choices"][0]["message"]["content"],
                "cache_status": result.status, "cache_reason": result.reason, "stored": result.stored,
                "provider_calls": calls, "provider_ms": upstream_ms if calls else None,
                "latency_ms": result.total_latency_ms,
                "generated_tokens": (usage if type(usage) is int and usage >= 0 else None) if calls else 0,
                **stats(),
            }

    @app.post("/api/chat")
    async def chat(request: Request):
        try:
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                raise ValueError("Send JSON.")
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 200_000:
                    raise ValueError("Chat request is too large.")
            payload = _strict_json(bytes(raw))
            body, enabled = request_from_payload(payload, model)
        except (ValueError, UnicodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return await run_in_threadpool(execute, body, enabled)

    return app
