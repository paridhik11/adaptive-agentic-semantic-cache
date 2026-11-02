"""Three-Arm Paired Live Benchmark Evaluation Harness for DocumentAgent.

Task C & Pilot Preparation Deliverable:
- Matched three-arm comparison:
    1. no_cache (mode='off', fresh execution on all tasks)
    2. exact_only (mode='exact', safe release default)
    3. shadow_semantic (mode='shadow', experimental semantic reuse in diagnostic shadow mode)
- Frozen Real-Provider Pilot Workload (4 tasks across 2 groups: 2 seeds, 2 exact repeats):
    - Exact proposed provider calls:
        - no_cache: 4 tasks x 3 stages = 12 calls
        - exact_only: 2 cold x 3 stages + 2 exact hits x 0 = 6 calls
        - shadow_semantic: 2 cold x 3 stages + 2 exact hits x 0 = 6 calls
        - Total proposed calls = 24 calls (within 30-call single run cap and 40-call daily cap).
- Shared PilotJournal quota enforcement across all arms in a single durable database.
- Full DocumentAgent execution across planning, drafting, review, and SQLite persistence.
- Zero paid fallback, free-only allowlist (:free suffix required), max price 0.00.
- Stable rate limit handling via fail-closed RunPaused without hidden retries.
- Actual OpenRouter token usage accounting from response receipts.
- Immutable receipt logging with response SHA-256 digests.
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

# ---------------------------------------------------------------------------
# Workload Definitions
# ---------------------------------------------------------------------------

# 1. Primary Pilot Workload: 4 frozen tasks (2 seeds + 2 exact repeats)
# Designed for safe, bounded live validation within a single 30-call quota window.
FROZEN_PILOT_WORKLOAD = [
    # Group 1: list_operations (Seed, Exact Repeat)
    {
        "id": "pilot-1",
        "question": "How do I convert a list to a tuple in Python?",
        "category": "seed",
        "topic": "list_operations",
        "expected_calls": 3,
    },
    {
        "id": "pilot-2",
        "question": "How do I convert a list to a tuple in Python?",
        "category": "exact_repeat",
        "topic": "list_operations",
        "expected_calls": 0,  # on exact_only and shadow_semantic
    },
    # Group 2: dict_operations (Seed, Exact Repeat)
    {
        "id": "pilot-3",
        "question": "How do I merge two dictionaries in Python 3.9+?",
        "category": "seed",
        "topic": "dict_operations",
        "expected_calls": 3,
    },
    {
        "id": "pilot-4",
        "question": "How do I merge two dictionaries in Python 3.9+?",
        "category": "exact_repeat",
        "topic": "dict_operations",
        "expected_calls": 0,  # on exact_only and shadow_semantic
    },
]

# 2. Extended Workload (12 tasks across 3 groups for larger follow-up studies)
FROZEN_EXTENDED_WORKLOAD = [
    # Group 1: list_operations
    {"id": "live-1", "question": "How do I convert a list to a tuple in Python?", "category": "seed", "topic": "list_operations"},
    {"id": "live-2", "question": "How do I convert a list to a tuple in Python?", "category": "exact_repeat", "topic": "list_operations"},
    {"id": "live-3", "question": "How can I turn a Python list into a tuple?", "category": "natural_paraphrase", "topic": "list_operations"},
    {"id": "live-4", "question": "How do I convert a tuple to a list in Python?", "category": "adversarial", "topic": "list_operations"},
    # Group 2: dict_operations
    {"id": "live-5", "question": "How do I merge two dictionaries in Python 3.9+?", "category": "seed", "topic": "dict_operations"},
    {"id": "live-6", "question": "How do I merge two dictionaries in Python 3.9+?", "category": "exact_repeat", "topic": "dict_operations"},
    {"id": "live-7", "question": "What is the operator syntax to combine two dicts in Python 3.9?", "category": "natural_paraphrase", "topic": "dict_operations"},
    {"id": "live-8", "question": "How do I merge two lists into a dictionary in Python?", "category": "adversarial", "topic": "dict_operations"},
    # Group 3: file_io
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
    """Orchestrates matched 3-arm evaluation with safety preflight, shared quota, and audit receipts."""

    def __init__(
        self,
        journal_path: Path | None = None,
        model: str = DEFAULT_LIVE_MODEL,
        execute_live: bool = False,
        spend_cap_usd: float = 0.0,
        max_output_tokens: int = 2048,
        use_pilot_workload: bool = True,
        output_dir: Path | None = None,
        max_study_budget: int = 30,
    ):
        self.output_dir = output_dir or ROOT / "artifacts/live_evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = journal_path or (self.output_dir / "pilot_journal.sqlite")
        self.model = model
        self.execute_live = execute_live
        self.spend_cap_usd = spend_cap_usd
        self.max_output_tokens = max_output_tokens
        self.use_pilot_workload = use_pilot_workload
        self.workload = FROZEN_PILOT_WORKLOAD if use_pilot_workload else FROZEN_EXTENDED_WORKLOAD
        self.max_study_budget = max_study_budget

    def calculate_proposed_requests(self) -> dict[str, Any]:
        """Calculates expected provider requests across the three matched arms."""
        if self.use_pilot_workload:
            # 4 tasks: 2 seeds, 2 exact repeats
            # no_cache: 4 tasks x 3 calls = 12 calls
            # exact_only: 2 seeds x 3 calls + 2 hits x 0 = 6 calls
            # shadow_semantic: 2 seeds x 3 calls + 2 hits x 0 = 6 calls (shadow mode does not substitute semantic answers)
            calls_no_cache = 4 * 3
            calls_exact = 2 * 3
            calls_shadow = 2 * 3
            total = calls_no_cache + calls_exact + calls_shadow
            return {
                "workload_type": "pilot_4_tasks",
                "total_tasks_per_arm": 4,
                "proposed_calls_no_cache": calls_no_cache,
                "proposed_calls_exact_only": calls_exact,
                "proposed_calls_shadow_semantic": calls_shadow,
                "total_proposed_provider_requests": total,
                "budget_cap": self.max_study_budget,
                "fits_within_budget": total <= self.max_study_budget,
            }
        else:
            calls_no_cache = 12 * 3  # 36 calls
            calls_exact = 9 * 3      # 27 calls
            calls_shadow = 9 * 3     # 27 calls
            total = calls_no_cache + calls_exact + calls_shadow
            return {
                "workload_type": "extended_12_tasks",
                "total_tasks_per_arm": 12,
                "proposed_calls_no_cache": calls_no_cache,
                "proposed_calls_exact_only": calls_exact,
                "proposed_calls_shadow_semantic": calls_shadow,
                "total_proposed_provider_requests": total,
                "budget_cap": self.max_study_budget,
                "fits_within_budget": total <= self.max_study_budget,
            }

    def preflight_check(self) -> dict[str, Any]:
        """Validates all safety invariants, model availability, and journal quota."""
        blockers = []
        checks = {}
        proposed = self.calculate_proposed_requests()

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

        # 4. Budget cap validation
        if not proposed["fits_within_budget"]:
            blockers.append(
                f"Proposed calls ({proposed['total_proposed_provider_requests']}) exceed max study budget ({self.max_study_budget})."
            )
        checks["budget_cap_valid"] = proposed["fits_within_budget"]

        # 5. Authorization check
        if not self.execute_live:
            blockers.append("Live execution flag '--execute-live' not passed. Operating in preflight verification mode.")
        checks["live_authorized"] = self.execute_live

        # 6. Environment credentials
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        checks["api_key_configured"] = bool(api_key)
        if self.execute_live and not api_key:
            blockers.append("OPENROUTER_API_KEY environment variable missing.")

        # 7. Journal capacity check
        checks["journal_available"] = False
        checks["journal_path"] = str(self.journal_path)
        checks["journal_total_calls"] = 0
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
            # When live is authorized, the journal file will be initialized
            checks["journal_available"] = True

        status = "READY" if not blockers else "BLOCKED"
        receipt = {
            "status": status,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "proposed_requests": proposed,
            "checks": checks,
            "blockers": blockers,
        }

        # Save preflight receipt
        receipt_path = self.output_dir / "live_readiness_receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

        return receipt

    def run_paired_study(self, arms: list[str] | None = None) -> dict[str, Any]:
        """Executes the frozen 3-arm comparison if authorized, or reports preflight readiness."""
        preflight = self.preflight_check()
        if preflight["status"] != "READY":
            return {
                "status": "BLOCKED",
                "message": "Live benchmark execution blocked pending explicit user authorization.",
                "preflight": preflight,
                "readiness_receipt": str(self.output_dir / "live_readiness_receipt.json"),
            }

        # The genuine 3 matched arms:
        # no_cache (baseline) vs exact_only (safe default) vs shadow_semantic (diagnostic shadow mode)
        arms = arms or ["no_cache", "exact_only", "shadow_semantic"]
        base_run_id = f"live-study-{int(time.time())}"
        results = {}
        cumulative_calls = 0

        for arm in arms:
            arm_res = self._run_arm(arm, base_run_id)
            results[arm] = arm_res
            cumulative_calls += arm_res["total_provider_calls"]
            if cumulative_calls > self.max_study_budget:
                raise RunPaused(f"Study budget exceeded: {cumulative_calls} > {self.max_study_budget}")

        output_data = {
            "status": "COMPLETED",
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "preflight": preflight,
            "model": self.model,
            "workload_type": "pilot" if self.use_pilot_workload else "extended",
            "arms": results,
            "cumulative_provider_calls": cumulative_calls,
        }

        results_file = self.output_dir / "live_paired_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        return output_data

    def _run_arm(self, arm_name: str, base_run_id: str) -> dict[str, Any]:
        """Runs the complete agent workflow across the frozen workload for one arm."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        # Shared journal database across arms; unique sub-run ID per arm for clean ordinals
        journal = PilotJournal(
            self.journal_path,
            run_id=f"{base_run_id}-{arm_name}",
            fingerprint=f"live-fingerprint-{arm_name}",
            max_calls=30,
            daily_limit=40,
            minute_limit=10,
        )
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

        # Mode configuration:
        # no_cache -> mode="off"
        # exact_only -> mode="exact"
        # shadow_semantic -> mode="shadow" (evaluates neural cross-encoder in background; serves exact only)
        mode = "off" if arm_name == "no_cache" else "exact" if arm_name == "exact_only" else "shadow"
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
        total_prompt_tok = 0
        total_comp_tok = 0

        for task in self.workload:
            t0 = time.perf_counter()
            calls_before = provider.new_calls
            tid = agent.create(task["question"], tenant="live-tenant", principal="live-evaluator", standalone_document_task=True)
            state = None
            for _ in range(25):
                state = agent.advance(tid, tenant="live-tenant", principal="live-evaluator")
                if state["status"] in {"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}:
                    break
            if state["status"] == "AWAITING_APPROVAL":
                state = agent.decide(tid, "approve", tenant="live-tenant", principal="live-evaluator")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            task_calls = provider.new_calls - calls_before

            answer_text = state.get("answer", "")
            ans_sha = hashlib.sha256(answer_text.encode("utf-8")).hexdigest() if answer_text else None

            # Actual token usage accumulated in DocumentAgent stage metrics
            p_tok = state.get("prompt_tokens", 0)
            c_tok = state.get("completion_tokens", 0)
            total_prompt_tok += p_tok
            total_comp_tok += c_tok

            events.append({
                "task_id": task["id"],
                "category": task["category"],
                "question": task["question"],
                "status": state["status"],
                "is_hit": bool(state.get("answer_cache_hit")),
                "tier": state.get("trace", [{}])[0].get("tier", "NONE") if state.get("answer_cache_hit") else "NONE",
                "provider_calls": task_calls,
                "prompt_tokens_actual": p_tok,
                "completion_tokens_actual": c_tok,
                "total_tokens_actual": p_tok + c_tok,
                "answer_sha256": ans_sha,
                "elapsed_ms": elapsed_ms,
            })

        return {
            "arm": arm_name,
            "mode": mode,
            "total_tasks": len(events),
            "total_provider_calls": provider.new_calls,
            "replayed_calls": provider.replayed_calls,
            "total_prompt_tokens_actual": total_prompt_tok,
            "total_completion_tokens_actual": total_comp_tok,
            "total_tokens_actual": total_prompt_tok + total_comp_tok,
            "events": events,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, help="Path to shared pilot journal database")
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL, help="Free-tier model allowlist route")
    parser.add_argument("--execute-live", action="store_true", help="Explicit authorization to run live API calls")
    parser.add_argument("--spend-cap", type=float, default=0.0, help="Hard spend cap in USD (default: 0.00)")
    parser.add_argument("--extended-workload", action="store_true", help="Use 12-task extended workload instead of 4-task pilot")
    parser.add_argument("--check-only", action="store_true", help="Run preflight safety checks without live calls")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    runner = LiveEvaluationRunner(
        journal_path=args.journal,
        model=args.model,
        execute_live=args.execute_live and not args.check_only,
        spend_cap_usd=args.spend_cap,
        use_pilot_workload=not args.extended_workload,
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

        prop = report.get("proposed_requests", {})
        if prop:
            print("\nProposed Real-Provider Pilot Plan:")
            print(f"  - Workload Profile         : {prop.get('workload_type')} ({prop.get('total_tasks_per_arm')} tasks/arm)")
            print(f"  - Arm 1 (no_cache)         : {prop.get('proposed_calls_no_cache')} provider calls")
            print(f"  - Arm 2 (exact_only)       : {prop.get('proposed_calls_exact_only')} provider calls")
            print(f"  - Arm 3 (shadow_semantic)  : {prop.get('proposed_calls_shadow_semantic')} provider calls (shadow mode)")
            print(f"  - Total Proposed Requests  : {prop.get('total_proposed_provider_requests')} calls (budget cap: {prop.get('budget_cap')})")
            print(f"  - Fits Within Budget       : {prop.get('fits_within_budget')}")

        if report.get("blockers"):
            print("\nBlockers / Safety Gates Active:")
            for b in report["blockers"]:
                print(f"  - [BLOCKED] {b}")
        print("\nSafety & Invariant Checks:")
        for k, v in report.get("checks", {}).items():
            print(f"  - {k:<25}: {v}")
        print("\nPreflight receipt written to: artifacts/live_evaluation/live_readiness_receipt.json")
        print("Live API calls remain strictly blocked pending explicit authorization.\n")


if __name__ == "__main__":
    main()
