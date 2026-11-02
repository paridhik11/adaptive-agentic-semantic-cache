"""Opt-in Live Benchmark Runner with Hard Spend Cap and Safety Guards.

Safety Invariants:
1. Live execution strictly requires explicit `--execute-live` CLI flag.
2. Spend cap defaults to $0.00; non-zero spend requires explicit `--spend-cap`.
3. Free-only transport: defaults to free-tier model (`nvidia/nemotron-3-super-120b-a12b:free`).
4. Never print, hardcode, paste or pass secrets in command arguments or logs.
5. Preserves existing provider quota journals without reset.
6. Persistent per-request receipts with token accounting and price provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FREE_MODELS = frozenset({
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
})

DEFAULT_FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
PROVIDER_PRICING = {
    # Free-tier models on OpenRouter (Pricing verified 2026-11-01)
    "nvidia/nemotron-3-super-120b-a12b:free": {
        "prompt_per_1m": 0.00,
        "completion_per_1m": 0.00,
        "pricing_source": "OpenRouter free model catalog",
        "pricing_date": "2026-11-01",
    },
    "meta-llama/llama-3.2-3b-instruct:free": {
        "prompt_per_1m": 0.00,
        "completion_per_1m": 0.00,
        "pricing_source": "OpenRouter free model catalog",
        "pricing_date": "2026-11-01",
    },
}


def compute_cost(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    pricing = PROVIDER_PRICING.get(model)
    if not pricing:
        raise ValueError(f"Unknown pricing for model '{model}'. Invented pricing fallback is prohibited.")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens * pricing["prompt_per_1m"] + completion_tokens * pricing["completion_per_1m"]) / 1_000_000.0


def sanitize_outbound_request(request_body: dict, model: str, max_output_tokens: int) -> dict:
    """Build an explicit allowlisted outbound request body rather than forwarding arbitrary caller fields.
    
    Rejects forbidden auxiliary routing, caller-controlled provider settings, and unsupported parameters.
    Binds the verified free-only model and disables paid fallback.
    """
    if not isinstance(request_body, dict):
        raise TypeError("request_body must be a dictionary")

    # 1. Reject forbidden auxiliary routing and caller-controlled provider settings
    forbidden = {"provider", "route", "models", "plugins", "tools", "web_search_options", "stream"}
    found_forbidden = forbidden.intersection(request_body.keys())
    if found_forbidden:
        raise ValueError(f"Forbidden request parameters rejected: {sorted(found_forbidden)}")

    # 2. Reject unsupported arbitrary keys
    allowed_keys = {"model", "messages", "max_tokens", "temperature", "response_format"}
    unsupported = set(request_body.keys()) - allowed_keys
    if unsupported:
        raise ValueError(f"Unsupported request parameters rejected: {sorted(unsupported)}")

    # 3. Validate messages
    messages = request_body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Request must contain a non-empty 'messages' list")
    sanitized_messages = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("Each message must have 'role' and 'content' strings")
        sanitized_messages.append({"role": str(m["role"]), "content": str(m["content"])})

    # 4. Validate and clamp max_tokens
    req_max_tokens = request_body.get("max_tokens")
    if req_max_tokens is not None:
        if not isinstance(req_max_tokens, int) or req_max_tokens <= 0:
            raise ValueError(f"Invalid max_tokens value: {req_max_tokens}")
        final_max_tokens = min(req_max_tokens, max_output_tokens)
    else:
        final_max_tokens = max_output_tokens

    # 5. Build explicit allowlisted outbound body binding verified free model and disabling paid fallback
    outbound = {
        "model": model,
        "messages": sanitized_messages,
        "max_tokens": final_max_tokens,
        "provider": {
            "allow_fallbacks": False,
            "max_price": {"prompt": 0.0, "completion": 0.0, "request": 0.0},
            "require_parameters": True,
        },
    }

    if "temperature" in request_body:
        temp = request_body["temperature"]
        if not isinstance(temp, (int, float)) or not 0.0 <= temp <= 2.0:
            raise ValueError(f"Invalid temperature value: {temp}")
        outbound["temperature"] = float(temp)

    if "response_format" in request_body:
        rf = request_body["response_format"]
        if not isinstance(rf, dict) or "type" not in rf:
            raise ValueError(f"Invalid response_format: {rf}")
        outbound["response_format"] = rf

    return outbound


class LiveBenchmarkRunner:
    """Hermetic opt-in runner for live provider evaluation with durable quota safeguards."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_FREE_MODEL,
        execute_live: bool = False,
        spend_cap_usd: float = 0.00,
        max_attempts: int = 5,
        max_output_tokens: int = 1024,
        journal_path: Path | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ):
        if model not in FREE_MODELS:
            raise ValueError(f"Model '{model}' is not in the free-only allowlist: {sorted(FREE_MODELS)}")
        self.model = model
        self.execute_live = execute_live
        self.spend_cap_usd = float(spend_cap_usd)
        self.max_attempts = int(max_attempts)
        self.max_output_tokens = int(max_output_tokens)
        self.journal_path = journal_path or (ROOT / "runs/openrouter-free.sqlite")
        self.output_dir = output_dir or (ROOT / "artifacts/live_evaluation/receipts")
        self.run_id = run_id or "live-benchmark-run-v1"

        self.current_spent_usd = 0.00
        self.attempts_made = 0
        self.receipts: list[dict] = []
        self._journal = None
        self._request_ordinals: dict[str, int] = {}

    def preflight_check(self) -> dict:
        api_key_present = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        pricing = PROVIDER_PRICING.get(self.model)

        journal_exists = self.journal_path.exists()

        status = "READY_OPT_IN" if self.execute_live else "DRY_RUN_ONLY"
        blockers = []

        if not self.execute_live:
            blockers.append("Execution flag --execute-live not provided (safety default)")
        if self.model not in FREE_MODELS:
            blockers.append(f"Model {self.model} not in free-only allowlist")
        if not pricing:
            blockers.append(f"Model {self.model} pricing is unverified; unknown pricing rejected")
        if self.execute_live and not api_key_present:
            blockers.append("No API key detected in environment (OPENROUTER_API_KEY)")
        if self.execute_live and not journal_exists:
            blockers.append(f"Durable quota journal does not exist at {self.journal_path}; shared journal required")
        if self.execute_live and self.spend_cap_usd <= 0.00 and pricing and pricing["prompt_per_1m"] != 0.00:
            blockers.append("Spend cap is $0.00 on a non-free model; execution blocked")

        return {
            "status": status if not blockers else "BLOCKED",
            "execute_live_flag": self.execute_live,
            "model": self.model,
            "is_free_model": pricing.get("prompt_per_1m") == 0.00 if pricing else False,
            "pricing": pricing,
            "spend_cap_usd": self.spend_cap_usd,
            "max_attempts": self.max_attempts,
            "max_output_tokens": self.max_output_tokens,
            "api_key_configured": api_key_present,
            "quota_journal_preserved": journal_exists,
            "journal_path": str(self.journal_path),
            "blockers": blockers,
        }

    def execute_request(self, request_body: dict, request_id: str) -> dict:
        preflight = self.preflight_check()
        if preflight["status"] == "BLOCKED":
            raise RuntimeError(f"Live execution blocked: {preflight['blockers']}")

        # 1. Strict request body sanitization and allowlisting
        outbound_body = sanitize_outbound_request(request_body, self.model, self.max_output_tokens)

        # 2. Durable quota reservation with restart and concurrency protection
        from src.benchmark.free_pilot import PilotJournal, RunPaused
        from src.benchmark.schema import digest

        if self._journal is None:
            self._journal = PilotJournal(
                self.journal_path,
                run_id=self.run_id,
                fingerprint=digest({"model": self.model, "cap": self.max_attempts}),
                max_calls=min(self.max_attempts, 30),
            )

        ordinal = self._request_ordinals.setdefault(request_id, len(self._request_ordinals))

        try:
            res_state, res_payload = self._journal.reserve(ordinal, outbound_body)
        except RunPaused as paused:
            # Uncertain attempt, rate limit, or exhausted budget: never repeat automatically
            receipt = {
                "request_id": request_id,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "status": "RESERVATION_PAUSED",
                "reason": str(paused),
                "elapsed_ms": 0.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "usage_missing": True,
                "spend_usd": None,
            }
            self.receipts.append(receipt)
            self._persist_single_receipt(receipt)
            raise RuntimeError(f"Journal reservation paused/blocked: {paused}") from paused

        # Handle existing completed attempt without dispatching new network call
        if res_state == "complete":
            receipt = {
                "request_id": request_id,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "status": "REPLAY_COMPLETE",
                "elapsed_ms": 0.0,
                "prompt_tokens": res_payload.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": res_payload.get("usage", {}).get("completion_tokens"),
                "usage_missing": False,
                "spend_usd": 0.0,
                "price_source": "journal_replay",
            }
            self.receipts.append(receipt)
            self._persist_single_receipt(receipt)
            return res_payload

        # Handle existing recorded error without repeating call
        if res_state == "error":
            receipt = {
                "request_id": request_id,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "status": "REPLAY_ERROR",
                "error": res_payload.get("error", "Previous attempt recorded error"),
                "elapsed_ms": 0.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "usage_missing": True,
                "spend_usd": None,
            }
            self.receipts.append(receipt)
            self._persist_single_receipt(receipt)
            raise RuntimeError(f"Previous attempt at ordinal {ordinal} failed with recorded error: {res_payload}")

        if self.attempts_made >= self.max_attempts:
            raise RuntimeError(f"Hard maximum provider attempts reached ({self.max_attempts})")

        t0 = time.perf_counter()
        self.attempts_made += 1

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("API key missing from environment")

        import urllib.request
        req_data = json.dumps(outbound_body).encode("utf-8")
        http_req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=req_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/paridhik11/adaptive-agentic-semantic-cache",
                "X-Title": "Adaptive Semantic Cache Live Benchmark",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_req, timeout=30) as resp:
                raw_resp = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            try:
                self._journal.finish(ordinal, "error", {"error": str(exc)})
            except Exception:
                pass
            receipt = {
                "request_id": request_id,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "status": "ERROR",
                "error": str(exc),
                "elapsed_ms": elapsed,
                "prompt_tokens": None,
                "completion_tokens": None,
                "usage_missing": True,
                "spend_usd": None,
            }
            self.receipts.append(receipt)
            self._persist_single_receipt(receipt)
            raise

        elapsed = (time.perf_counter() - t0) * 1000.0
        try:
            self._journal.finish(ordinal, "complete", raw_resp)
        except Exception:
            pass
        usage = raw_resp.get("usage", {})
        p_tok = usage.get("prompt_tokens")
        c_tok = usage.get("completion_tokens")
        usage_missing = p_tok is None or c_tok is None

        call_cost = compute_cost(self.model, p_tok, c_tok)
        if call_cost is not None:
            self.current_spent_usd += call_cost

        if self.spend_cap_usd > 0.0 and self.current_spent_usd > self.spend_cap_usd:
            raise RuntimeError(f"Hard spend cap reached: ${self.current_spent_usd:.6f} > ${self.spend_cap_usd:.2f}")

        receipt = {
            "request_id": request_id,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "status": "SUCCESS",
            "elapsed_ms": elapsed,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "usage_missing": usage_missing,
            "spend_usd": call_cost,
            "price_source": PROVIDER_PRICING.get(self.model, {}).get("pricing_source", "unknown"),
            "price_date": PROVIDER_PRICING.get(self.model, {}).get("pricing_date", "unknown"),
        }
        self.receipts.append(receipt)
        self._persist_single_receipt(receipt)
        return raw_resp

    def _persist_single_receipt(self, receipt: dict):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            req_file = self.output_dir / f"receipt_{receipt['request_id']}.json"
            with open(req_file, "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
            log_file = self.output_dir / "live_receipts.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt) + "\n")
        except Exception:
            pass

    def save_receipts(self, path: Path | None = None) -> Path:
        target = path or (self.output_dir / f"live_receipt_{int(time.time())}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model,
                "spend_cap_usd": self.spend_cap_usd,
                "actual_spent_usd": self.current_spent_usd,
                "attempts_made": self.attempts_made,
                "receipts": self.receipts,
            }, f, indent=2)
        return target

    def __call__(self, body: dict) -> dict:
        req_id = f"workflow-attempt-{self.attempts_made + 1}"
        return self.execute_request(body, req_id)

    def run_agent_workflow(self, question: str = "How do I convert a list to a tuple in Python?") -> dict:
        """Execute complete documentation agent workflow using the live provider runner."""
        from src.agents.document_agent import DocumentAgent, DocumentTools
        from src.agents.document_answer_cache import CompletedDocumentCache
        from src.api.service import GatewayService
        from src.policy.encoders import DenseEncoder
        from src.policy.hybrid_cache import ConservativeHybridCache
        from src.policy.local_models import (
            ENCODER_MODEL, ENCODER_REVISION,
            PASSAGE_MODEL, PASSAGE_REVISION,
            LocalPassageScorer,
        )
        from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
        from src.policy.semantic_runtime import KnowledgeScope
        from src.policy.sqlite_store import SQLiteResponseStore

        docs = {
            "tuple_conv": "Tuples and lists can be converted using tuple() and list() constructors in Python.",
        }
        tools = DocumentTools(docs)
        scope = KnowledgeScope("agent-live-docs", tools.revision, "live-tenant", "live-v1")
        tenant = "live-tenant"
        principal = "live-operator"

        db_dir = ROOT / "runs/live_agent_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        tasks_db = db_dir / "tasks.sqlite"
        answers_db = db_dir / "answers.sqlite"

        exact_storage = SQLiteResponseStore(answers_db, namespace="live-agent")
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(True, "live_policy"),
            storage=exact_storage,
        )
        encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
        scorer = LocalPassageScorer(PASSAGE_MODEL, PASSAGE_REVISION)
        hybrid = ConservativeHybridCache(
            exact,
            scope=scope,
            encoder=encoder,
            passage_scorer=scorer,
            mode="guarded",
        )
        ans_cache = CompletedDocumentCache(hybrid)
        service = GatewayService(
            ScopedResponseCache(eligibility=lambda _: EligibilityDecision(False, "off")),
            self,
            allowed_models={self.model},
        )
        agent = DocumentAgent(
            service,
            tools,
            tasks_db,
            model=self.model,
            answer_cache=ans_cache,
            stage_accounting=True,
        )

        tid = agent.create(question, tenant=tenant, principal=principal, standalone_document_task=True)
        for _ in range(25):
            state = agent.advance(tid, tenant=tenant, principal=principal)
            if state["status"] in {"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}:
                break

        if state["status"] == "AWAITING_APPROVAL":
            state = agent.decide(tid, "approve", tenant=tenant, principal=principal)

        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-live", action="store_true", default=False,
                        help="Explicit execution flag required to enable live API inference")
    parser.add_argument("--spend-cap", type=float, default=0.00,
                        help="Hard spend cap in USD (default: 0.00)")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Hard maximum provider attempts (default: 5)")
    parser.add_argument("--model", type=str, default=DEFAULT_FREE_MODEL,
                        help=f"Provider model (default: {DEFAULT_FREE_MODEL})")
    parser.add_argument("--check-only", action="store_true", default=False,
                        help="Perform preflight and safety check only, then exit")
    parser.add_argument("--agent-workflow", action="store_true", default=False,
                        help="Run complete documentation agent workflow instead of isolated completion")
    args = parser.parse_args()

    runner = LiveBenchmarkRunner(
        model=args.model,
        execute_live=args.execute_live,
        spend_cap_usd=args.spend_cap,
        max_attempts=args.max_attempts,
    )

    preflight = runner.preflight_check()
    print("\n=== LIVE BENCHMARK RUNNER PREFLIGHT REPORT ===")
    print(json.dumps(preflight, indent=2))

    if args.check_only or not args.execute_live:
        print("\nSafety Notice: Live inference is NOT authorized; running in safety check mode.")
        print("To authorize live execution, pass: --execute-live --spend-cap <amount>")
        return 0

    print("\nExecuting authorized live evaluation...")
    if args.agent_workflow:
        print("Running complete agent workflow benchmark...")
        workflow_res = runner.run_agent_workflow()
        print(f"Agent task terminal status: {workflow_res['status']}")
    else:
        sample_queries = [
            "How do I convert a list to a tuple in Python?",
            "How do I convert a list to a tuple in Python?",
            "How can I turn a Python list into a tuple?",
        ]
        for idx, query in enumerate(sample_queries):
            body = {
                "model": runner.model,
                "messages": [
                    {"role": "system", "content": "Answer concisely using official documentation."},
                    {"role": "user", "content": query},
                ],
                "max_tokens": runner.max_output_tokens,
            }
            try:
                print(f"[{idx+1}/{len(sample_queries)}] Executing query: {query}")
                resp = runner.execute_request(body, f"live-task-{idx+1}")
                print(f" -> Success: {resp.get('choices', [{}])[0].get('message', {}).get('content', '')[:60]}...")
            except Exception as exc:
                print(f" -> Paused/Failed: {exc}")
                break
    saved_path = runner.save_receipts()
    print(f"\nLive execution completed. Receipts saved to {saved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
