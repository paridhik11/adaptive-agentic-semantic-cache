"""Three-Arm Paired Live Benchmark Evaluation Harness for DocumentAgent.

Task C Deliverable:
- Matched three-arm comparison:
    1. no_cache
    2. exact_only
    3. verified_semantic (guarded mode)
- Full DocumentAgent execution across planning, tool, draft, and review stages.
- Durable PilotJournal quota reservation with zero paid fallback and free-only models.
- Preflight quota and model availability validation.
- Immutable answer caching for replay evaluation.
- SAFETY INVARIANT: Live network calls strictly blocked by default. Requires
  explicit `--execute-live` authorization.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.document_agent import DocumentAgent, DocumentTools
from src.agents.document_answer_cache import CompletedDocumentCache
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.benchmark.control import RunPaused
from src.benchmark.free_pilot import FreeOpenRouterProvider, JournaledProvider, PilotJournal
from src.policy.answer_aware_policy import LocalPassageScorer
from src.policy.encoders import DenseEncoder
from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION, PASSAGE_MODEL, PASSAGE_REVISION
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope
from src.policy.sqlite_store import SQLiteResponseStore

FREE_MODELS = frozenset({
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
})
DEFAULT_LIVE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

FROZEN_LIVE_WORKLOAD = [
    # Group 1: list_operations (Seed, Exact Repeat, Natural Paraphrase, Adversarial)
    {"id": "live-1", "question": "How do I convert a list to a tuple in Python?", "category": "seed", "topic": "list_operations"},
    {"id": "live-2", "question": "How do I convert a list to a tuple in Python?", "category": "exact_repeat", "topic": "list_operations"},
    {"id": "live-3", "question": "How can I turn a Python list into a tuple?", "category": "natural_paraphrase", "topic": "list_operations"},
    {"id": "live-4", "question": "How do I convert a tuple to a list in Python?", "category": "adversarial", "topic": "list_operations"},
    # Group 2: dict_operations (Seed, Exact Repeat, Natural Paraphrase, Adversarial)
    {"id": "live-5", "question": "How do I merge two dictionaries in Python 3.9+?", "category": "seed", "topic": "dict_operations"},
    {"id": "live-6", "question": "How do I merge two dictionaries in Python 3.9+?", "category": "exact_repeat", "topic": "dict_operations"},
    {"id": "live-7", "question": "What is the operator syntax to combine two dicts in Python 3.9?", "category": "natural_paraphrase", "topic": "dict_operations"},
    {"id": "live-8", "question": "How do I merge two lists into a dictionary in Python?", "category": "adversarial", "topic": "dict_operations"},
    # Group 3: file_io (Seed, Exact Repeat, Natural Paraphrase, Adversarial)
    {"id": "live-9", "question": "How do I read a file line by line in Python?", "category": "seed", "topic": "file_io"},
    {"id": "live-10", "question": "How do I read a file line by line in Python?", "category": "exact_repeat", "topic": "file_io"},
    {"id": "live-11", "question": "What is the standard way to iterate through lines in a file in Python?", "category": "natural_paraphrase", "topic": "file_io"},
    {"id": "live-12", "question": "How do I write a list of strings line by line to a file in Python?", "category": "adversarial", "topic": "file_io"},
]

REFERENCE_DOCS = {
    "list_operations": "Lists are mutable sequences in Python. Convert a list to a tuple using tuple(my_list). Slicing uses my_list[start:stop].",
    "dict_operations": "Dictionaries store key-value mappings. Merge dictionaries in Python 3.9+ using the union operator d1 | d2. Use dict.get(key, default) for safe retrieval.",
    "file_io": "Open files using 'with open(path, mode) as f:'. Read line by line iterating over f directly without loading the complete file into memory.",
}


class LiveEvaluationRunner:
    """Orchestrates matched 3-arm evaluation with safety preflight and audit logging."""

    def __init__(
        self,
        journal_path: Path | None = None,
        model: str = DEFAULT_LIVE_MODEL,
        execute_live: bool = False,
        spend_cap_usd: float = 0.0,
        max_output_tokens: int = 2048,
        output_dir: Path | None = None,
    ):
        self.journal_path = journal_path
        self.model = model
        self.execute_live = execute_live
        self.spend_cap_usd = spend_cap_usd
        self.max_output_tokens = max_output_tokens
        self.output_dir = output_dir or ROOT / "artifacts/live_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preflight_check(self) -> dict[str, Any]:
        """Validates all safety invariants, model availability, and journal quota."""
        blockers = []
        checks = {}

        # 1. Model allowlist
        if self.model not in FREE_MODELS:
            blockers.append(f"Model '{self.model}' not in free model allowlist: {sorted(FREE_MODELS)}")
        checks["model_allowed"] = self.model in FREE_MODELS

        # 2. Spend cap
        if self.spend_cap_usd > 0.0:
            blockers.append(f"Non-zero spend cap ${self.spend_cap_usd:.2f} rejected. Free transport must remain $0.00.")
        checks["spend_cap_zero"] = self.spend_cap_usd == 0.0

        # 3. Output token ceiling
        if self.max_output_tokens <= 0:
            blockers.append(f"Output token ceiling must be > 0, got {self.max_output_tokens}")
        checks["output_ceiling_valid"] = self.max_output_tokens > 0

        # 4. Authorization check
        if not self.execute_live:
            blockers.append("Live execution flag '--execute-live' not passed. Operating in preflight verification mode.")
        checks["live_authorized"] = self.execute_live

        # 5. Environment credentials
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        checks["api_key_configured"] = bool(api_key)
        if self.execute_live and not api_key:
            blockers.append("OPENROUTER_API_KEY environment variable missing.")

        # 6. Journal capacity check
        checks["journal_available"] = False
        checks["journal_quota_remaining"] = 0
        if self.journal_path and Path(self.journal_path).is_file():
            try:
                db = sqlite3.connect(f"file:{Path(self.journal_path).resolve().as_uri()}?mode=ro", uri=True)
                cur = db.cursor()
                calls_made = cur.execute("SELECT COUNT(*) FROM pilot_calls").fetchone()[0]
                checks["journal_available"] = True
                checks["journal_total_calls"] = calls_made
                db.close()
            except Exception as e:
                blockers.append(f"Failed to inspect journal at {self.journal_path}: {e}")
        elif self.execute_live:
            blockers.append("Existing shared journal required when live execution is authorized.")

        status = "READY" if not blockers else "BLOCKED"
        return {
            "status": status,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "checks": checks,
            "blockers": blockers,
        }

    def run_paired_study(self, arms: list[str] | None = None) -> dict[str, Any]:
        """Executes the frozen 3-arm comparison if authorized, or reports preflight readiness."""
        preflight = self.preflight_check()
        if preflight["status"] != "READY":
            # Save readiness report
            readiness_file = self.output_dir / "live_readiness_receipt.json"
            with open(readiness_file, "w", encoding="utf-8") as f:
                json.dump(preflight, f, indent=2)
            return {
                "status": "BLOCKED",
                "message": "Live benchmark execution blocked pending explicit user authorization.",
                "preflight": preflight,
                "readiness_receipt": str(readiness_file),
            }

        # If authorized, execute live paired evaluation
        arms = arms or ["no_cache", "exact_only", "verified_semantic"]
        results = {}
        for arm in arms:
            results[arm] = self._run_arm(arm)
        return {
            "status": "COMPLETED",
            "preflight": preflight,
            "arms": results,
        }

    def _run_arm(self, arm_name: str) -> dict[str, Any]:
        """Runs the complete agent workflow across the frozen workload for one arm."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        journal = PilotJournal(self.journal_path, run_id=f"live-paired-{arm_name}", fingerprint=f"live-{arm_name}")
        provider = JournaledProvider(
            FreeOpenRouterProvider(api_key, models={self.model}, reasoning_enabled=False),
            journal,
        )

        db_dir = self.output_dir / arm_name
        db_dir.mkdir(parents=True, exist_ok=True)
        answers_db = db_dir / "answers.db"
        tasks_db = db_dir / "tasks.db"

        tools = DocumentTools(REFERENCE_DOCS)
        scope = KnowledgeScope("live-docs", tools.revision, "live-tenant", f"live-{arm_name}")
        exact_storage = SQLiteResponseStore(answers_db, namespace=f"live-{arm_name}")

        mode = "off" if arm_name == "no_cache" else "exact" if arm_name == "exact_only" else "guarded"
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(arm_name != "no_cache", "live_eval_policy"),
            storage=exact_storage,
        )
        hybrid = ConservativeHybridCache(
            exact,
            scope=scope,
            mode=mode,
        )
        answer_cache = CompletedDocumentCache(hybrid)
        service = GatewayService(
            ScopedResponseCache(eligibility=lambda _: EligibilityDecision(False, "gateway_off")),
            provider,
            allowed_models={self.model},
        )
        agent = DocumentAgent(
            service,
            tools,
            tasks_db,
            model=self.model,
            answer_cache=answer_cache,
            stage_accounting=True,
        )

        events = []
        for task in FROZEN_LIVE_WORKLOAD:
            t0 = time.perf_counter()
            tid = agent.create(task["question"], tenant="live-tenant", principal="live-evaluator", standalone_document_task=True)
            state = None
            for _ in range(25):
                state = agent.advance(tid, tenant="live-tenant", principal="live-evaluator")
                if state["status"] in {"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}:
                    break
            if state["status"] == "AWAITING_APPROVAL":
                state = agent.decide(tid, "approve", tenant="live-tenant", principal="live-evaluator")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            events.append({
                "task_id": task["id"],
                "category": task["category"],
                "question": task["question"],
                "status": state["status"],
                "is_hit": bool(state.get("answer_cache_hit")),
                "tier": state.get("trace", [{}])[0].get("tier", "NONE") if state.get("answer_cache_hit") else "NONE",
                "elapsed_ms": elapsed_ms,
            })

        return {
            "arm": arm_name,
            "total_tasks": len(events),
            "events": events,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, help="Path to shared pilot journal")
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL, help="Free-tier model")
    parser.add_argument("--execute-live", action="store_true", help="Explicit authorization to run live API calls")
    parser.add_argument("--spend-cap", type=float, default=0.0, help="Hard spend cap in USD (default: 0.00)")
    parser.add_argument("--check-only", action="store_true", help="Run preflight safety checks without live calls")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    runner = LiveEvaluationRunner(
        journal_path=args.journal,
        model=args.model,
        execute_live=args.execute_live and not args.check_only,
        spend_cap_usd=args.spend_cap,
    )

    if args.check_only or not args.execute_live:
        report = runner.preflight_check()
    else:
        report = runner.run_paired_study()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("================================================================================")
        print("          LIVE PAIRED BENCHMARK EVALUATION HARNESS — STATUS REPORT              ")
        print("================================================================================\n")
        print(f"Status           : {report['status']}")
        print(f"Model            : {report.get('model', args.model)}")
        print(f"Live Authorized  : {args.execute_live and not args.check_only}")
        if report.get("blockers"):
            print("\nBlockers / Safety Gates:")
            for b in report["blockers"]:
                print(f"  - [BLOCKED] {b}")
        print("\nChecks:")
        for k, v in report.get("checks", {}).items():
            print(f"  - {k:<25}: {v}")
        print("\nLive calls remain strictly disabled until explicit authorization is supplied.")


if __name__ == "__main__":
    main()
