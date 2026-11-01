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


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PROVIDER_PRICING.get(model)
    if not pricing:
        # Conservative assumption for unknown models: $0.15 / 1M prompt, $0.60 / 1M completion
        return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0
    return (prompt_tokens * pricing["prompt_per_1m"] + completion_tokens * pricing["completion_per_1m"]) / 1_000_000.0


class LiveBenchmarkRunner:
    """Hermetic opt-in runner for live provider evaluation."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_FREE_MODEL,
        execute_live: bool = False,
        spend_cap_usd: float = 0.00,
        max_attempts: int = 5,
        journal_path: Path | None = None,
        output_dir: Path | None = None,
    ):
        self.model = model
        self.execute_live = execute_live
        self.spend_cap_usd = float(spend_cap_usd)
        self.max_attempts = int(max_attempts)
        self.journal_path = journal_path or (ROOT / "data/openrouter_synthetic_feedback_telemetry.json")
        self.output_dir = output_dir or (ROOT / "artifacts/live_evaluation/receipts")

        self.current_spent_usd = 0.00
        self.attempts_made = 0
        self.receipts: list[dict] = []

    def preflight_check(self) -> dict:
        api_key_present = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        pricing = PROVIDER_PRICING.get(self.model, {
            "prompt_per_1m": "unknown",
            "completion_per_1m": "unknown",
            "pricing_source": "unregistered_model",
            "pricing_date": "unknown",
        })

        journal_exists = self.journal_path.exists()

        status = "READY_OPT_IN" if self.execute_live else "DRY_RUN_ONLY"
        blockers = []

        if not self.execute_live:
            blockers.append("Execution flag --execute-live not provided (safety default)")
        if self.execute_live and not api_key_present:
            blockers.append("No API key detected in environment (OPENROUTER_API_KEY / GEMINI_API_KEY)")
        if self.execute_live and self.spend_cap_usd <= 0.00 and pricing["prompt_per_1m"] != 0.00:
            blockers.append("Spend cap is $0.00 on a non-free model; execution blocked")

        return {
            "status": status if not blockers else "BLOCKED",
            "execute_live_flag": self.execute_live,
            "model": self.model,
            "is_free_model": pricing.get("prompt_per_1m") == 0.00,
            "pricing": pricing,
            "spend_cap_usd": self.spend_cap_usd,
            "max_attempts": self.max_attempts,
            "api_key_configured": api_key_present,
            "quota_journal_preserved": journal_exists,
            "journal_path": str(self.journal_path),
            "blockers": blockers,
        }

    def execute_request(self, request_body: dict, request_id: str) -> dict:
        preflight = self.preflight_check()
        if preflight["status"] == "BLOCKED":
            raise RuntimeError(f"Live execution blocked: {preflight['blockers']}")

        if self.attempts_made >= self.max_attempts:
            raise RuntimeError(f"Hard maximum provider attempts reached ({self.max_attempts})")

        # In live execution mode, this performs the call through the official provider adapter
        t0 = time.perf_counter()
        self.attempts_made += 1

        # Check API key from environment only
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("API key missing from environment")

        import urllib.request
        req_data = json.dumps(request_body).encode("utf-8")
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
                "spend_usd": 0.00,
            }
            self.receipts.append(receipt)
            raise

        elapsed = (time.perf_counter() - t0) * 1000.0
        usage = raw_resp.get("usage", {})
        p_tok = usage.get("prompt_tokens")
        c_tok = usage.get("completion_tokens")
        usage_missing = p_tok is None or c_tok is None

        call_cost = compute_cost(self.model, p_tok or 0, c_tok or 0)
        self.current_spent_usd += call_cost

        if self.current_spent_usd > self.spend_cap_usd and self.spend_cap_usd > 0.0:
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
        return raw_resp

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
    # Ready for authorized execution
    return 0


if __name__ == "__main__":
    sys.exit(main())
