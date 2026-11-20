"""General-purpose chat with a live provider and scoped response reuse."""
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from src.api.chat_intents import direct_question, question_intent, scope_reply
from src.api.chat_keywords import educational_keywords, keyword_signature
from src.api.chat_semantic import ChatResponseCache, ChatSemanticReuse, comparison_subjects, definition_subject
from src.api.demo import workbench_shell
from src.api.gateway import _strict_json
from src.api.providers import ProviderError
from src.api.service import ConservativeRulesPolicy
from src.benchmark.control import RunPaused
from src.benchmark.free_pilot import FreeOpenRouterProvider, JournaledProvider, PilotJournal
from src.classifier.models import StabilityLabel
from src.classifier.rules import StabilityRuleEngine
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.sqlite_store import SQLiteResponseStore

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_JOURNAL = Path("runs/live-quota.sqlite")
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
    r"yesterday|tomorrow|(?:this|last|next) (?:week|month|quarter|year)|as of|"
    r"20\d{2}|fy\s*\d{2,4}|q[1-4]|ytd|qtd|mtd|my|our)\b", re.IGNORECASE
)

_MESSAGES = {
    "run_attempt_budget_exhausted": "The 30-attempt allowance is used up. Eligible cached answers still work.",
    "project_minute_quota": "The shared per-minute allowance was reached. Try again after the displayed time.",
    "project_daily_quota": "The shared daily allowance was reached. Try again after the displayed time.",
    "upstream_rate_limit_pause": "OpenRouter rate-limited this journal. Try again after the displayed time.",
    "upstream_attempt_failed_requires_inspection": "OpenRouter did not return an answer. Check the key and selected free model, then try again.",
    "uncertain_attempt_requires_review": "Another server is using the journal, or an earlier request was interrupted. Wait for the active request or inspect the journal.",
    "quota_journal_unavailable_or_write_uncertain": "The quota journal is unavailable. Check the journal path and file permissions.",
    "upstream_incomplete_answer": "The model did not return a complete answer. Try asking a shorter question.",
    "upstream_auth_rejected": "OpenRouter rejected the API key. Restart the server with a valid key.",
    "upstream_access_denied": "OpenRouter denied access. Check the key permissions and free-model availability.",
    "upstream_model_unavailable": "The selected model or route is unavailable. Restart with another explicit free model using --model.",
    "upstream_timeout": "OpenRouter took too long to answer. You can submit the question again.",
    "upstream_unavailable": "Trace could not reach OpenRouter. Check your connection and try again.",
}


def request_from_payload(value, model):
    if not isinstance(value, dict) or set(value) - {"question", "history", "cache", "previous_question"}:
        raise ValueError("Send a question, optional history, and cache preference.")
    question = value.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
        raise ValueError("Enter a question under 4,000 characters.")
    question.encode("utf-8")
    previous = value.get("previous_question", "")
    if not isinstance(previous, str) or len(previous) > MAX_QUESTION:
        raise ValueError("Invalid previous question.")
    previous.encode("utf-8")
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
        item["content"].encode("utf-8")
    return {
        "model": model, "temperature": 0, "max_tokens": MAX_ANSWER_TOKENS,
        "messages": [{"role": "system", "content": INSTRUCTION}, *history,
                     {"role": "user", "content": question.strip()}],
    }, enabled, previous


def open_chat_journal(path):
    return PilotJournal(path, run_id="general-chat-v1", fingerprint="general-chat-free-v1", max_calls=RUN_LIMIT)


def create_live_chat_app(directory, *, journal_path, api_key, model=DEFAULT_MODEL, benchmark_directory=None, transport=None, semantic_models=None, semantic_mode="guarded"):
    if semantic_mode not in {"exact", "guarded", "shadow"}:
        raise ValueError("Invalid chat semantic mode.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Set OPENROUTER_API_KEY in PowerShell before launching live chat.")
    upstream = FreeOpenRouterProvider(api_key, models={model}, reasoning_enabled=False, transport=transport)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal = open_chat_journal(journal_path)
    policy = ConservativeRulesPolicy()
    stability = StabilityRuleEngine()

    def eligibility(body):
        turns = body["messages"][1:]
        if len(turns) != 1:
            return policy({"messages": turns})
        question = direct_question(turns[0]["content"])
        decision = policy({"messages": [{"role": "user", "content": question}]})
        rule = stability.evaluate(question)
        intent = question_intent(question)
        # The shared role rule treats an office without a temporal anchor
        # as current. An explicit, parsed ordinal supplies that anchor here.
        ranked_role = intent and intent.rank and rule.rule_name == "dynamic_answer_type_current_role"
        if _FRESHNESS.search(question) or (rule.label == StabilityLabel.DYNAMIC and not ranked_role):
            return EligibilityDecision(False, "freshness_requires_provider")
        if intent and intent.subject and decision.reason == "not_rule_confirmed_static":
            return EligibilityDecision(True, "standalone_structured_question")
        if decision.reason == "not_rule_confirmed_static" and (
                _DEFINITION.match(question) or educational_keywords(keyword_signature(question))
                or definition_subject(question, shorthand=True)
                or comparison_subjects(question)):
            return EligibilityDecision(True, "standalone_educational_request")
        return decision

    storage = SQLiteResponseStore(directory / "chat.sqlite", namespace="general-chat-v1")
    cache = ChatResponseCache(eligibility=eligibility, storage=storage)
    semantic = ChatSemanticReuse(cache, storage, semantic_models)
    semantic.backfill_keywords(TENANT)
    provider_lock = threading.Lock()
    stats_lock = threading.Lock()
    session = secrets.token_urlsafe(32)
    exact_hits = semantic_hits = 0
    app, benchmark_link = workbench_shell(
        "Trace · Live chat", session, benchmark_directory, {"workbench.css", "live_chat.css", "live_chat.js"},
    )

    def stats():
        try:
            used = sum(journal.status().values())
        except sqlite3.Error:
            used = None
        return {
            "model": model, "used": used, "limit": RUN_LIMIT,
            "hits": exact_hits + semantic_hits, "exact_hits": exact_hits, "semantic_hits": semantic_hits,
            "semantic_enabled": semantic_models is not None, "semantic_mode": semantic_mode,
        }

    @app.get("/", response_class=HTMLResponse)
    def page():
        html = Path(__file__).with_name("live_chat.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__SESSION__", session).replace("__BENCHMARK_LINK__", benchmark_link))

    @app.get("/api/chat")
    def info():
        with stats_lock:
            return stats()

    def complete_answer(request):
        response = upstream(request)
        if not isinstance(response, dict) or not ScopedResponseCache._admissible(response, request):
            raise ProviderError("upstream_incomplete_answer")
        return response

    def execute(body, enabled, previous):
        nonlocal exact_hits, semantic_hits
        provider = JournaledProvider(complete_answer, journal, append=True)
        started = time.perf_counter()
        upstream_ms = None
        resolved_question = None
        if len(body["messages"]) == 2:
            question = body["messages"][-1]["content"]
            intent = question_intent(question)
            prior = question_intent(previous) if previous else None
            if prior and prior.rank and not prior.subject and not intent and (name := scope_reply(question)):
                resolved_question = prior.scoped_question(name)
            elif intent and intent.rank and not intent.subject:
                if prior and prior.rank and prior.relation == intent.relation and prior.subject:
                    resolved_question = intent.scoped_question(prior.subject)
                else:
                    with stats_lock:
                        counters = stats()
                    return {
                        "answer": "Which country or organization do you mean? You can reply with its name in automatic context.",
                        "cache_status": "CLARIFY", "cache_reason": "missing_subject", "stored": False,
                        "cache_tier": None, "semantic_match": None, "semantic_decision": None,
                        "resolved_question": None, "provider_calls": 0, "provider_ms": None,
                        "generated_tokens": 0, "latency_ms": (time.perf_counter() - started) * 1_000,
                        **counters,
                    }
            if resolved_question:
                body["messages"][-1]["content"] = resolved_question

        def generate(request):
            nonlocal upstream_ms
            with provider_lock:
                contact = time.perf_counter()
                try:
                    return provider(request)
                finally:
                    upstream_ms = (time.perf_counter() - contact) * 1_000

        try:
            verification = {}
            result, match, semantic_decision = semantic.process(
                body, tenant_id=TENANT, provider=generate, bypass=not enabled,
                diagnostics=verification, mode=semantic_mode)
        except RunPaused as exc:
            code = (exc.diagnostics or {}).get("code", exc.reason)
            message = _MESSAGES.get(code) or _MESSAGES.get(exc.reason)
            return JSONResponse({
                "error": message or "The provider request stopped. Check the shared quota journal.",
                "retry_at": exc.retry_at, "provider_calls": provider.new_calls,
                "latency_ms": (time.perf_counter() - started) * 1_000,
            }, status_code=503)

        with stats_lock:
            semantic_hits += match is not None
            exact_hits += result.status == "HIT" and match is None
            counters = stats()
        calls = provider.new_calls
        usage = result.response.get("usage")
        usage = usage.get("completion_tokens") if isinstance(usage, dict) else None
        return {
            "answer": result.response["choices"][0]["message"]["content"],
            "cache_status": result.status, "cache_reason": result.reason, "stored": result.stored,
            "cache_tier": "SEMANTIC" if match else "EXACT" if result.status == "HIT" else None,
            "semantic_match": match, "semantic_decision": semantic_decision,
            "semantic_verification": verification,
            "resolved_question": resolved_question,
            "provider_calls": calls, "provider_ms": upstream_ms if calls else None,
            "latency_ms": (time.perf_counter() - started) * 1_000,
            "generated_tokens": (usage if type(usage) is int and usage >= 0 else None) if calls else 0,
            **counters,
        }

    @app.post("/api/chat")
    async def chat(request: Request):
        try:
            if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                raise ValueError("Send JSON.")
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 200_000:
                    raise ValueError("Chat request is too large.")
            payload = _strict_json(bytes(raw))
            body, enabled, previous = request_from_payload(payload, model)
        except UnicodeError:
            return JSONResponse({"error": "Use valid Unicode text for the question and conversation."}, status_code=400)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return await run_in_threadpool(execute, body, enabled, previous)

    return app
