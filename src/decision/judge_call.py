"""Phase 3 — Judge Call: LLM-based intent verification for AMBIGUOUS-tier traffic.

When non-LLM signals (cosine similarity, stability confidence, category history)
cannot distinguish semantic equivalence from inverted algorithmic intent in
the ambiguous band (e.g. PAIR-035: string-to-int vs. int-to-string at similarity
0.9961), an LLM judge is invoked with the full query text and context.

CRITICAL SAFETY INVARIANT — FAIL-CLOSED:
-----------------------------------------
If the judge call times out, encounters a network disconnect, receives malformed
output, suffers an API/authentication failure, or errors in any way:
    DECISION DEFAULTS TO BYPASS (NEVER REUSE).
A judge-call failure must NEVER fail open. Missed reuse adds regeneration
latency; false positive reuse creates a corrupted or dangerous cache hazard.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pricing and Model Constants
# ---------------------------------------------------------------------------
# Source: OpenRouter Free Tier / Google AI Studio Free Tier ($0.00)
DEFAULT_JUDGE_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
FALLBACK_JUDGE_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
GEMINI_DEFAULT_JUDGE_MODEL: str = "gemini-2.5-flash"
GEMINI_FALLBACK_JUDGE_MODEL: str = "gemini-3.6-flash"
DEFAULT_TIMEOUT_SECONDS: float = 15.0

PRICE_PER_1M_INPUT_TOKENS: float = 0.0   # $0.00 (Free tier)
PRICE_PER_1M_OUTPUT_TOKENS: float = 0.0  # $0.00 (Free tier)


# ---------------------------------------------------------------------------
# Prompt Templates (Step 0 Contract Preserved Verbatim)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are a strict, conservative Semantic Cache Reuse Judge for an enterprise AI gateway.
Your task is to determine whether it is completely safe to serve the pre-computed cached answer of Query A in response to incoming Query B.

CRITICAL SAFETY POLICY:
1. Serving a cached answer for a query with different intent causes a CACHE HAZARD (incorrect, corrupted, or dangerous output for the user).
2. Missed reuse (bypassing the cache when queries are equivalent) only adds slight latency and regeneration cost.
3. Therefore, CACHE HAZARD IS UNACCEPTABLE. When in any doubt, or if there is any nuance, inversion, differing constraint, differing parameter, or differing environment, you MUST decide BYPASS.
4. Watch out especially for:
   - Algorithmic or intent inversions (e.g., ascending vs. descending sort, string-to-int vs. int-to-string, push vs. pull, create vs. drop, serialize vs. parse, install vs. uninstall, encrypt vs. decrypt).
   - In-place modification vs. returning a new copy / preserving order vs. discarding order.
   - Differing targets, tools, dialects, or environments (e.g., Linux vs. PowerShell, PostgreSQL vs. SQLite, Nginx vs. Apache, Node.js vs. Python).
   - Scope discrepancies (e.g., general definition of all codes vs. specific error code 429).
   - Temporal or historical constraints (e.g., current capital vs. Heian period capital)."""

USER_PROMPT_TEMPLATE: str = """Evaluate the following query pair for cache reuse safety:

--- CONTEXT ---
Domain: {domain}
Embedding Similarity Score: {similarity_score:.4f}
Stability Confidence: {stability_confidence:.4f}
Domain Historical Reuse Success Rate: {category_history_rate:.4f}

--- QUERIES ---
Cached Query (Query A): "{query_a}"
Incoming Query (Query B): "{query_b}"

--- DECISION TASK ---
Can the cached answer produced for Query A be served to the user asking Query B with 100% semantic and technical correctness?

Respond strictly using the structured schema:
- decision: "REUSE" only if Query A's answer fully and safely answers Query B without semantic hazard; otherwise "BYPASS".
- is_safe: true if REUSE, false if BYPASS.
- confidence: your confidence score between 0.0 and 1.0.
- rationale: a concise explanation (1-2 sentences) of the semantic comparison and why reuse is safe or hazardous."""


# ---------------------------------------------------------------------------
# Structured Output Schema (Pydantic)
# ---------------------------------------------------------------------------

class JudgeDecisionEnum(str, Enum):
    REUSE = "REUSE"
    BYPASS = "BYPASS"


class JudgeOutputSchema(BaseModel):
    """Pydantic model enforced via structured JSON output and schema validation."""

    decision: JudgeDecisionEnum = Field(
        description="Final decision: 'REUSE' if Query A's cached response is completely safe to serve for Query B, else 'BYPASS'."
    )
    is_safe: bool = Field(
        description="True if safe for reuse, False if hazardous or ambiguous."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in [0.0, 1.0].",
    )
    rationale: str = Field(
        description="Concise 1-2 sentence explanation of the semantic comparison and safety rationale."
    )


# ---------------------------------------------------------------------------
# Judge Result Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeResult:
    """Structured outcome of an LLM judge evaluation.

    Attributes:
        decision: "REUSE" or "BYPASS".
        is_safe: True if decision is REUSE, False if BYPASS.
        confidence: Judge's confidence score in [0.0, 1.0].
        rationale: Explanatory reasoning for the decision.
        prompt_tokens: Input tokens used in the call.
        completion_tokens: Output tokens generated by the model.
        total_tokens: Total tokens (prompt + completion).
        latency_ms: Measured round-trip latency in milliseconds.
        model: Model name/version used for the call.
        cost_usd: Computed dollar cost ($0.0 for free tier).
        fallback_triggered: True if error/timeout occurred and fail-closed fallback was used.
        error: Error message string if fallback was triggered, else None.
        raw_response: Raw response text string from the model.
        retries_used: Count of rate limit / transient retries triggered.
        request_id: Provider generation / request ID for independent verification.
    """

    decision: str
    is_safe: bool
    confidence: float
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    cost_usd: Optional[float] = 0.0
    fallback_triggered: bool = False
    error: Optional[str] = None
    raw_response: Optional[str] = None
    retries_used: int = 0
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a dictionary for logging and reporting."""
        return {
            "decision": self.decision,
            "is_safe": self.is_safe,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else 0.0,
            "fallback_triggered": self.fallback_triggered,
            "error": self.error,
            "raw_response": self.raw_response,
            "retries_used": self.retries_used,
            "request_id": self.request_id,
        }


# ---------------------------------------------------------------------------
# API Key and JSON Parsing Helpers
# ---------------------------------------------------------------------------

def _clean_and_parse_json(raw_text: Optional[str]) -> Optional[JudgeOutputSchema]:
    """Parse and validate JSON string against JudgeOutputSchema, stripping code fences if present."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return JudgeOutputSchema.model_validate_json(cleaned)
    except Exception:
        return None


def _resolve_api_key(backend: str = "openrouter", explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from explicit argument, environment, or .env file."""
    if explicit_key:
        return explicit_key

    env_var = "OPENROUTER_API_KEY" if backend == "openrouter" else "GEMINI_API_KEY"
    key = os.environ.get(env_var)
    if key:
        return key
    if backend == "gemini":
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            return key

    # Check .env in current working directory or repository root
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith(f"{env_var}="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
                        elif backend == "gemini" and line.startswith("GOOGLE_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# LLMJudge Class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Invokes LLM judge for AMBIGUOUS-tier query pairs with structured outputs.

    Supports OpenRouter (default, free tier open-weight models via OpenAI SDK)
    and Google Gemini (free tier via google-genai SDK).

    Matches the hook signature of JudgeCallable (Callable[[float, float, str], float])
    via its __call__ method, while exposing rich judge() evaluation for pair inspection.

    Safety: Guarantees fail-closed fallback to BYPASS on any network error,
    timeout, malformed output, rate limit, or authentication failure.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        rate_limit_delay_seconds: float = 4.0,
    ) -> None:
        if backend is not None:
            self.backend = backend.lower()
        elif client is not None and hasattr(client, "_mock_children"):
            if "models" in client._mock_children and "chat" not in client._mock_children:
                self.backend = "gemini"
            elif "chat" in client._mock_children and "models" not in client._mock_children:
                self.backend = "openrouter"
            else:
                self.backend = "openrouter"
        elif client is not None and hasattr(client, "models") and not hasattr(client, "chat"):
            self.backend = "gemini"
        elif os.environ.get("JUDGE_BACKEND"):
            self.backend = os.environ.get("JUDGE_BACKEND").lower()
        else:
            self.backend = "openrouter"

        if model is not None:
            self.model = model
        elif self.backend == "openrouter":
            self.model = os.environ.get("OPENROUTER_MODEL") or DEFAULT_JUDGE_MODEL
        else:
            self.model = os.environ.get("GEMINI_MODEL") or GEMINI_DEFAULT_JUDGE_MODEL

        self.timeout = timeout
        self.api_key = _resolve_api_key(backend=self.backend, explicit_key=api_key)
        self._client = client
        # Disable pacing delay on mocks to keep unit tests fast
        if client is not None and (
            hasattr(client, "_mock_children")
            or hasattr(client, "mock_calls")
            or type(client).__name__ in ("MagicMock", "Mock", "NonCallableMagicMock")
        ):
            self.rate_limit_delay_seconds = 0.0
        else:
            self.rate_limit_delay_seconds = rate_limit_delay_seconds

    @property
    def client(self) -> Any:
        """Lazily initialize OpenRouter (OpenAI SDK) or Google GenAI client if not injected."""
        if self._client is None:
            if not self.api_key:
                return None
            if self.backend == "openrouter":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=self.api_key,
                        default_headers={
                            "HTTP-Referer": "https://github.com/paridhik11/adaptive-agentic-semantic-cache",
                            "X-Title": "Adaptive Agentic Semantic Cache",
                        },
                    )
                except Exception:
                    return None
            else:
                try:
                    from google import genai
                    self._client = genai.Client(api_key=self.api_key)
                except Exception:
                    return None
        return self._client

    def format_prompt(
        self,
        query_a: str,
        query_b: str,
        domain: str,
        similarity_score: float,
        stability_confidence: float = 1.0,
        category_history_rate: float = 0.5,
    ) -> str:
        """Render the user prompt template with all context variables."""
        return USER_PROMPT_TEMPLATE.format(
            query_a=query_a,
            query_b=query_b,
            domain=domain,
            similarity_score=similarity_score,
            stability_confidence=stability_confidence,
            category_history_rate=category_history_rate,
        )

    def judge(
        self,
        query_a: str,
        query_b: str,
        domain: str,
        similarity_score: float,
        stability_confidence: float = 1.0,
        category_history_rate: float = 0.5,
    ) -> JudgeResult:
        """Evaluate a pair for cache reuse safety using structured output.

        Returns JudgeResult. On ANY failure, safety block, or timeout,
        strictly returns a fail-closed BYPASS decision with fallback_triggered=True.
        """
        if self.backend == "gemini":
            return self._judge_gemini(
                query_a=query_a,
                query_b=query_b,
                domain=domain,
                similarity_score=similarity_score,
                stability_confidence=stability_confidence,
                category_history_rate=category_history_rate,
            )
        return self._judge_openrouter(
            query_a=query_a,
            query_b=query_b,
            domain=domain,
            similarity_score=similarity_score,
            stability_confidence=stability_confidence,
            category_history_rate=category_history_rate,
        )

    def _judge_openrouter(
        self,
        query_a: str,
        query_b: str,
        domain: str,
        similarity_score: float,
        stability_confidence: float = 1.0,
        category_history_rate: float = 0.5,
    ) -> JudgeResult:
        """OpenRouter backend execution with robust schema validation and backoff."""
        t0 = time.perf_counter()

        client = self.client
        if client is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return JudgeResult(
                decision=JudgeDecisionEnum.BYPASS.value,
                is_safe=False,
                confidence=0.0,
                rationale="Judge unavailable: missing OpenRouter API key or client; safe fallback to BYPASS",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=elapsed_ms,
                model=self.model,
                cost_usd=0.0,
                fallback_triggered=True,
                error="Missing API key or uninitialized client",
                request_id=None,
            )

        user_content = self.format_prompt(
            query_a=query_a,
            query_b=query_b,
            domain=domain,
            similarity_score=similarity_score,
            stability_confidence=stability_confidence,
            category_history_rate=category_history_rate,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        max_retries = 3
        backoff_delays = [15.0, 30.0, 60.0]
        retries_used = 0
        completion = None

        try:
            for attempt in range(max_retries + 1):
                try:
                    completion = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        timeout=self.timeout,
                    )
                    break
                except Exception as exc:
                    err_str = str(exc).lower()
                    # Check for rate limit or transient server error
                    is_429 = "429" in err_str or "rate limit" in err_str or "ratelimit" in err_str
                    is_transient = any(
                        code in err_str
                        for code in ["500", "502", "503", "504", "520", "521", "522", "524", "529", "overloaded", "unavailable", "server error", "connection"]
                    )
                    if (is_429 or is_transient) and attempt < max_retries:
                        retries_used += 1
                        delay = backoff_delays[attempt]
                        try:
                            m = re.search(r"retry after (\d+(\.\d+)?)", err_str) or re.search(r"retry in (\d+(\.\d+)?)s", err_str)
                            if m:
                                delay = max(delay, float(m.group(1)) + 2.0)
                        except Exception:
                            pass
                        time.sleep(delay)
                        continue
                    # Non-transient errors fail closed immediately
                    raise

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Inter-call pacing delay
            if self.rate_limit_delay_seconds > 0:
                time.sleep(self.rate_limit_delay_seconds)

            request_id = getattr(completion, "id", None)
            model_version = getattr(completion, "model", None) or self.model

            # Token tracking
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            usage = getattr(completion, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

            # Choice validation
            choice = completion.choices[0] if (hasattr(completion, "choices") and completion.choices) else None
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            finish_reason_str = str(finish_reason or "").upper()

            if finish_reason and any(bad in finish_reason_str for bad in ["FILTER", "CONTENT_FILTER", "SAFETY"]):
                return JudgeResult(
                    decision=JudgeDecisionEnum.BYPASS.value,
                    is_safe=False,
                    confidence=0.0,
                    rationale=f"Model response blocked by safety filter (finish_reason={finish_reason_str}); safe fallback to BYPASS",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=elapsed_ms,
                    model=model_version,
                    cost_usd=0.0,
                    fallback_triggered=True,
                    error=f"Candidate blocked with finish_reason={finish_reason_str}",
                    raw_response=None,
                    retries_used=retries_used,
                    request_id=request_id,
                )

            raw_text = choice.message.content if (choice and hasattr(choice, "message") and choice.message) else None

            # Parse structured response
            parsed = _clean_and_parse_json(raw_text)

            # If not valid JSON matching schema, attempt exactly ONE retry asking model to reformat
            if parsed is None:
                retry_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": raw_text or ""},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not adhere to the required JSON schema.\n"
                            "Respond strictly with a single valid JSON object containing exactly these fields:\n"
                            '{"decision": "REUSE" | "BYPASS", "is_safe": bool, "confidence": float, "rationale": str}\n'
                            "Do not include any explanation or text outside the JSON object."
                        ),
                    },
                ]
                try:
                    retry_completion = client.chat.completions.create(
                        model=self.model,
                        messages=retry_messages,
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        timeout=self.timeout,
                    )
                    retry_choice = retry_completion.choices[0] if (hasattr(retry_completion, "choices") and retry_completion.choices) else None
                    retry_raw_text = retry_choice.message.content if (retry_choice and hasattr(retry_choice, "message") and retry_choice.message) else None
                    parsed = _clean_and_parse_json(retry_raw_text)
                    if parsed is not None:
                        raw_text = retry_raw_text
                        retry_usage = getattr(retry_completion, "usage", None)
                        if retry_usage is not None:
                            prompt_tokens += getattr(retry_usage, "prompt_tokens", 0) or 0
                            completion_tokens += getattr(retry_usage, "completion_tokens", 0) or 0
                            total_tokens = prompt_tokens + completion_tokens
                except Exception:
                    parsed = None

            if parsed is None:
                return JudgeResult(
                    decision=JudgeDecisionEnum.BYPASS.value,
                    is_safe=False,
                    confidence=0.0,
                    rationale="Judge returned unparseable or refused response failing schema validation; safe fallback to BYPASS",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=elapsed_ms,
                    model=model_version,
                    cost_usd=0.0,
                    fallback_triggered=True,
                    error="Parsed output is None or invalid JSON after retry",
                    raw_response=raw_text,
                    retries_used=retries_used,
                    request_id=request_id,
                )

            # Conservative safety enforcement: even if model output was REUSE,
            # if is_safe is False, coerce decision to BYPASS
            decision = parsed.decision.value
            is_safe = parsed.is_safe and (decision == JudgeDecisionEnum.REUSE.value)
            if not is_safe:
                decision = JudgeDecisionEnum.BYPASS.value

            return JudgeResult(
                decision=decision,
                is_safe=is_safe,
                confidence=float(parsed.confidence),
                rationale=parsed.rationale,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=elapsed_ms,
                model=model_version,
                cost_usd=0.0,
                fallback_triggered=False,
                error=None,
                raw_response=raw_text,
                retries_used=retries_used,
                request_id=request_id,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = f"{exc.__class__.__name__}: {str(exc)}"
            return JudgeResult(
                decision=JudgeDecisionEnum.BYPASS.value,
                is_safe=False,
                confidence=0.0,
                rationale=f"Judge call error ({err_msg}); safe fallback to BYPASS",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=elapsed_ms,
                model=self.model,
                cost_usd=0.0,
                fallback_triggered=True,
                error=err_msg,
                retries_used=retries_used,
                request_id=None,
            )

    def _judge_gemini(
        self,
        query_a: str,
        query_b: str,
        domain: str,
        similarity_score: float,
        stability_confidence: float = 1.0,
        category_history_rate: float = 0.5,
    ) -> JudgeResult:
        """Google Gemini backend execution (preserved for backwards compatibility)."""
        t0 = time.perf_counter()

        client = self.client
        if client is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return JudgeResult(
                decision=JudgeDecisionEnum.BYPASS.value,
                is_safe=False,
                confidence=0.0,
                rationale="Judge unavailable: missing Gemini API key or client; safe fallback to BYPASS",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=elapsed_ms,
                model=self.model,
                cost_usd=0.0,
                fallback_triggered=True,
                error="Missing API key or uninitialized client",
                request_id=None,
            )

        user_content = self.format_prompt(
            query_a=query_a,
            query_b=query_b,
            domain=domain,
            similarity_score=similarity_score,
            stability_confidence=stability_confidence,
            category_history_rate=category_history_rate,
        )

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=JudgeOutputSchema,
                temperature=0.0,
            )
        except Exception:
            config = None

        try:
            max_retries = 3
            completion = None
            backoff_delays = [15.0, 30.0, 60.0]
            retries_used = 0
            for attempt in range(max_retries + 1):
                try:
                    completion = client.models.generate_content(
                        model=self.model,
                        contents=user_content,
                        config=config,
                    )
                    break
                except Exception as exc:
                    err_str = str(exc).lower()
                    is_429 = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                    if is_429 and attempt < max_retries:
                        retries_used += 1
                        delay = backoff_delays[attempt]
                        try:
                            m = re.search(r"retry in (\d+(\.\d+)?)s", err_str)
                            if m:
                                delay = max(delay, float(m.group(1)) + 2.0)
                        except Exception:
                            pass
                        time.sleep(delay)
                        continue
                    raise

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if self.rate_limit_delay_seconds > 0:
                time.sleep(self.rate_limit_delay_seconds)

            if hasattr(completion, "candidates") and completion.candidates:
                cand = completion.candidates[0]
                finish_reason = getattr(cand, "finish_reason", None)
                finish_reason_str = str(finish_reason.value if hasattr(finish_reason, "value") else finish_reason)
                if finish_reason and "STOP" not in finish_reason_str.upper():
                    return JudgeResult(
                        decision=JudgeDecisionEnum.BYPASS.value,
                        is_safe=False,
                        confidence=0.0,
                        rationale=f"Gemini candidate blocked or incomplete (finish_reason={finish_reason_str}); safe fallback to BYPASS",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=elapsed_ms,
                        model=self.model,
                        cost_usd=0.0,
                        fallback_triggered=True,
                        error=f"Candidate blocked with finish_reason={finish_reason_str}",
                        raw_response=getattr(completion, "text", None),
                        request_id=None,
                    )

            parsed = _clean_and_parse_json(getattr(completion, "text", None))
            if parsed is None:
                raw_parsed = getattr(completion, "parsed", None)
                if isinstance(raw_parsed, JudgeOutputSchema):
                    parsed = raw_parsed

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            usage = getattr(completion, "usage_metadata", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
                total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)

            model_version = getattr(completion, "model_version", None) or self.model
            raw_text = getattr(completion, "text", None)

            if parsed is None:
                return JudgeResult(
                    decision=JudgeDecisionEnum.BYPASS.value,
                    is_safe=False,
                    confidence=0.0,
                    rationale="Judge returned unparseable or refused response; safe fallback to BYPASS",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=elapsed_ms,
                    model=model_version,
                    cost_usd=0.0,
                    fallback_triggered=True,
                    error="Parsed output is None or invalid JSON",
                    raw_response=raw_text,
                    request_id=None,
                )

            decision = parsed.decision.value
            is_safe = parsed.is_safe and (decision == JudgeDecisionEnum.REUSE.value)
            if not is_safe:
                decision = JudgeDecisionEnum.BYPASS.value

            return JudgeResult(
                decision=decision,
                is_safe=is_safe,
                confidence=float(parsed.confidence),
                rationale=parsed.rationale,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=elapsed_ms,
                model=model_version,
                cost_usd=0.0,
                fallback_triggered=False,
                error=None,
                raw_response=raw_text,
                retries_used=retries_used,
                request_id=None,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = f"{exc.__class__.__name__}: {str(exc)}"
            return JudgeResult(
                decision=JudgeDecisionEnum.BYPASS.value,
                is_safe=False,
                confidence=0.0,
                rationale=f"Judge call error ({err_msg}); safe fallback to BYPASS",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=elapsed_ms,
                model=self.model,
                cost_usd=0.0,
                fallback_triggered=True,
                error=err_msg,
                retries_used=retries_used,
                request_id=None,
            )

    def __call__(
        self,
        similarity_score: float,
        stability_confidence: float,
        domain: str,
        query_a: str = "",
        query_b: str = "",
        category_history_rate: float = 0.5,
    ) -> float:
        """Implements compatibility with JudgeCallable = Callable[[float, float, str], float].

        Returns the judge's confidence score in [0.0, 1.0] if safe REUSE,
        or 0.0 if BYPASS or if query texts are not supplied.
        """
        if not query_a or not query_b:
            return 0.0

        res = self.judge(
            query_a=query_a,
            query_b=query_b,
            domain=domain,
            similarity_score=similarity_score,
            stability_confidence=stability_confidence,
            category_history_rate=category_history_rate,
        )
        return res.confidence if res.is_safe else 0.0
