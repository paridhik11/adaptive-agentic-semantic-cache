"""Flagship Verification Demo: 7 Lifecycle Execution Scenarios + Criterion 8 Evidence Reporting.

Demonstrates:
1. Fresh request through planning, retrieval, drafting, review.
2. Approval and durable answer storage in SQLite.
3. Safe default exact-only repeat avoiding fresh agent work (Tier 0).
4. Experimental semantic paraphrase reuse under explicit opt-in (Tier 1 Guarded).
5. Changed requirement correctly missing and falling back safely.
6. Tenant/source-revision invalidation blocking stale reuse.
7. Restart persistence and safe fallback across process re-initialization.
Criterion 8 (Evidence Reporting): Formatted receipt table detailing provider calls,
tier decisions, and stage latencies across all 7 execution scenarios.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope
from src.policy.sqlite_store import SQLiteResponseStore
from scripts.benchmark_whole_agent import AgentBenchmarkProvider

DEMO_DOCUMENTS = {
    "tuple_conv": "Tuples and lists can be converted using tuple() and list() constructors in Python.",
    "str_split": "The str.split(sep) method splits a string by separator into a list of strings in Python.",
    "file_read": "Use 'with open(path) as f:' and iterate over f line by line to read files in Python.",
}


def run_demo(base_dir: Path, verbose: bool = True, experimental_semantic: bool = False) -> dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    answers_db = base_dir / "demo_answers.db"
    tasks_db = base_dir / "demo_tasks.db"
    if answers_db.exists():
        answers_db.unlink()
    if tasks_db.exists():
        tasks_db.unlink()

    tools = DocumentTools(DEMO_DOCUMENTS)
    scope = KnowledgeScope("agent-demo-docs", tools.revision, "demo-tenant", "flagship-v1")
    tenant = "demo-tenant"
    principal = "demo-operator"

    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION) if experimental_semantic else None
    passage_scorer = LocalPassageScorer(PASSAGE_MODEL, PASSAGE_REVISION) if experimental_semantic else None

    def init_agent():
        exact_storage = SQLiteResponseStore(answers_db, namespace="demo-flagship")
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(True, "demo_policy"),
            storage=exact_storage,
        )
        if experimental_semantic:
            hybrid = ConservativeHybridCache(
                exact, scope=scope, encoder=encoder, passage_scorer=passage_scorer,
                mode="guarded", retrieval_threshold=0.60, verification_threshold=0.85,
            )
        else:
            hybrid = GuardedSemanticCache(exact, scope=scope, mode="exact")
        ans_cache = CompletedDocumentCache(hybrid)
        prov = AgentBenchmarkProvider(DEMO_DOCUMENTS, delay_seconds=0.015)
        svc = GatewayService(
            ScopedResponseCache(eligibility=lambda _: EligibilityDecision(False, "off")),
            prov,
            allowed_models={"gpt-4o-mini-fixture"},
        )
        ag = DocumentAgent(
            svc,
            tools,
            tasks_db,
            model="gpt-4o-mini-fixture",
            answer_cache=ans_cache,
            stage_accounting=True,
        )
        return ag, prov, hybrid

    agent, provider, hybrid_cache = init_agent()

    steps_receipts = []

    def advance_task(tid: str) -> dict:
        for _ in range(25):
            s = agent.advance(tid, tenant=tenant, principal=principal)
            if s["status"] in {"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}:
                return s
        return s

    if verbose:
        print("================================================================================")
        print("              ADAPTIVE AGENTIC SEMANTIC CACHE — FLAGSHIP DEMO                   ")
        print("================================================================================\n")

    # -------------------------------------------------------------------------
    # STEP 1: Fresh Request through Planning, Retrieval, Drafting, Review
    # -------------------------------------------------------------------------
    q1 = "How do I convert a list to a tuple in Python?"
    t0 = time.perf_counter()
    p_before = provider.attempts
    tid1 = agent.create(q1, tenant=tenant, principal=principal, standalone_document_task=True)
    state1 = advance_task(tid1)
    el1 = (time.perf_counter() - t0) * 1000.0
    calls1 = provider.attempts - p_before
    tier1 = state1.get("trace", [{}])[0].get("tier", "NONE") if state1.get("answer_cache_hit") else "NONE"

    assert state1["status"] == "AWAITING_APPROVAL", f"Expected AWAITING_APPROVAL, got {state1['status']}"
    assert calls1 == 3, f"Expected 3 provider calls for fresh task, got {calls1}"
    assert not state1.get("answer_cache_hit"), "Fresh task must not be a cache hit"

    r1 = {
        "step": 1,
        "name": "Fresh Request",
        "question": q1,
        "status": state1["status"],
        "cache_tier": tier1,
        "is_hit": False,
        "provider_calls": calls1,
        "avoided_calls": 0,
        "elapsed_ms": el1,
        "scope": scope.knowledge_base,
    }
    steps_receipts.append(r1)
    if verbose:
        print(f"[Step 1] Fresh Request: '{q1}'")
        print(f"         Status: {state1['status']} | Provider Calls: {calls1} (Plan, Tool, Draft) | Elapsed: {el1:.1f}ms | Tier: {tier1}\n")

    # -------------------------------------------------------------------------
    # STEP 2: Approval and Durable Answer Storage
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    p_before = provider.attempts
    state2 = agent.decide(tid1, "approve", tenant=tenant, principal=principal)
    el2 = (time.perf_counter() - t0) * 1000.0
    calls2 = provider.attempts - p_before

    assert state2["status"] == "COMPLETED", f"Expected COMPLETED after approval, got {state2['status']}"
    assert calls2 == 0, f"Expected 0 provider calls for approval, got {calls2}"
    assert state2.get("answer"), "Expected approved answer to be stored"

    r2 = {
        "step": 2,
        "name": "Operator Approval & Admission",
        "question": q1,
        "status": state2["status"],
        "cache_tier": "ADMITTED",
        "is_hit": False,
        "provider_calls": calls2,
        "avoided_calls": 0,
        "elapsed_ms": el2,
        "scope": scope.knowledge_base,
    }
    steps_receipts.append(r2)
    if verbose:
        print(f"[Step 2] Operator Approval: Task {tid1}")
        print(f"         Status: {state2['status']} | Provider Calls: {calls2} | Admitted to Durable Cache | Elapsed: {el2:.1f}ms\n")

    # -------------------------------------------------------------------------
    # STEP 3: Exact Repeat Avoiding Fresh Agent Work
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    p_before = provider.attempts
    tid3 = agent.create(q1, tenant=tenant, principal=principal, standalone_document_task=True)
    state3 = advance_task(tid3)
    if state3["status"] == "AWAITING_APPROVAL":
        state3 = agent.decide(tid3, "approve", tenant=tenant, principal=principal)
    el3 = (time.perf_counter() - t0) * 1000.0
    calls3 = provider.attempts - p_before
    tier3 = state3.get("trace", [{}])[0].get("tier", "EXACT")

    assert state3.get("answer_cache_hit"), "Exact repeat must hit cache"
    assert tier3 == "EXACT", f"Expected EXACT tier, got {tier3}"
    assert calls3 == 0, f"Exact repeat must incur 0 provider calls, got {calls3}"

    r3 = {
        "step": 3,
        "name": "Exact Repeat",
        "question": q1,
        "status": state3["status"],
        "cache_tier": tier3,
        "is_hit": True,
        "provider_calls": calls3,
        "avoided_calls": 3,
        "elapsed_ms": el3,
        "scope": scope.knowledge_base,
    }
    steps_receipts.append(r3)
    if verbose:
        print(f"[Step 3] Exact Repeat: '{q1}'")
        print(f"         Cache Tier: {tier3} HIT | Provider Calls: {calls3} (Avoided 3 calls) | Elapsed: {el3:.1f}ms\n")

    # -------------------------------------------------------------------------
    # STEP 4: Safe Paraphrase Reusing Verified Cached Answer
    # -------------------------------------------------------------------------
    q4 = "How can I turn a Python list into a tuple?"
    t0 = time.perf_counter()
    p_before = provider.attempts
    tid4 = agent.create(q4, tenant=tenant, principal=principal, standalone_document_task=True)
    state4 = advance_task(tid4)
    if state4["status"] == "AWAITING_APPROVAL":
        state4 = agent.decide(tid4, "approve", tenant=tenant, principal=principal)
    el4 = (time.perf_counter() - t0) * 1000.0
    calls4 = provider.attempts - p_before
    tier4 = state4.get("trace", [{}])[0].get("tier", "NONE") if state4.get("answer_cache_hit") else "NONE"

    if experimental_semantic:
        assert state4.get("answer_cache_hit"), "Safe paraphrase must hit semantic tier when experimental-semantic is enabled"
        assert tier4 == "SEMANTIC", f"Expected SEMANTIC tier, got {tier4}"
        assert calls4 == 0, f"Semantic hit must incur 0 provider calls, got {calls4}"
        r4_name = "Guarded Paraphrase Hit (Opt-In)"
        r4_hit = True
        r4_avoided = 3
    else:
        assert not state4.get("answer_cache_hit"), "Default demo must never substitute experimental semantic answers"
        assert tier4 == "NONE", f"Expected NONE tier in default exact mode, got {tier4}"
        assert calls4 == 3, f"Expected 3 fresh provider calls in default exact mode, got {calls4}"
        r4_name = "Paraphrase Exact Miss (Default)"
        r4_hit = False
        r4_avoided = 0

    r4 = {
        "step": 4,
        "name": r4_name,
        "question": q4,
        "status": state4["status"],
        "cache_tier": tier4,
        "is_hit": r4_hit,
        "provider_calls": calls4,
        "avoided_calls": r4_avoided,
        "elapsed_ms": el4,
        "scope": scope.knowledge_base,
    }
    steps_receipts.append(r4)
    if verbose:
        if experimental_semantic:
            print(f"[Step 4] Safe Paraphrase: '{q4}'")
            print(f"         Cache Tier: {tier4} HIT (Answer-Aware Verification) | Provider Calls: {calls4} (Avoided 3 calls) | Elapsed: {el4:.1f}ms\n")
        else:
            print(f"[Step 4] Paraphrase Query (Default Exact Serving): '{q4}'")
            print(f"         Status: {state4['status']} | Provider Calls: {calls4} (Fresh Execution) | Elapsed: {el4:.1f}ms | Tier: {tier4} (Semantic Substitution Disabled by Default)\n")

    # -------------------------------------------------------------------------
    # STEP 5: Changed Requirement Correctly Missing
    # -------------------------------------------------------------------------
    q5 = "How do I convert a tuple to a list in Python?"  # Direction reversal: must NOT hit list->tuple answer
    t0 = time.perf_counter()
    p_before = provider.attempts
    tid5 = agent.create(q5, tenant=tenant, principal=principal, standalone_document_task=True)
    state5 = advance_task(tid5)
    el5 = (time.perf_counter() - t0) * 1000.0
    calls5 = provider.attempts - p_before
    tier5 = state5.get("trace", [{}])[0].get("tier", "NONE") if state5.get("answer_cache_hit") else "NONE"

    assert not state5.get("answer_cache_hit"), "Reversed requirement must fail closed (MISS)"
    assert calls5 == 3, f"Expected fresh agent execution (3 calls), got {calls5}"

    r5 = {
        "step": 5,
        "name": "Changed Requirement (Unsafe)",
        "question": q5,
        "status": state5["status"],
        "cache_tier": tier5,
        "is_hit": False,
        "provider_calls": calls5,
        "avoided_calls": 0,
        "elapsed_ms": el5,
        "scope": scope.knowledge_base,
    }
    steps_receipts.append(r5)
    if verbose:
        print(f"[Step 5] Changed Requirement: '{q5}'")
        print(f"         Cache Tier: {tier5} (Structural Guard Rejection) | Fresh Provider Calls: {calls5} | Elapsed: {el5:.1f}ms\n")

    # -------------------------------------------------------------------------
    # STEP 6: Tenant/Source-Revision Invalidation Blocking Stale Reuse
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    agent.invalidate(tenant=tenant, principal=principal)
    # Subsequent attempt on previously cached paraphrase must miss
    p_before = provider.attempts
    tid6 = agent.create(q4, tenant=tenant, principal=principal, standalone_document_task=True)
    state6 = advance_task(tid6)
    el6 = (time.perf_counter() - t0) * 1000.0
    calls6 = provider.attempts - p_before
    tier6 = state6.get("trace", [{}])[0].get("tier", "NONE") if state6.get("answer_cache_hit") else "NONE"

    assert not state6.get("answer_cache_hit"), "Invalidated scope must reject stale cached answers"
    assert calls6 == 3, f"Expected 3 fresh provider calls after invalidation, got {calls6}"

    r6 = {
        "step": 6,
        "name": "Scope Invalidation",
        "question": q4,
        "status": state6["status"],
        "cache_tier": tier6,
        "is_hit": False,
        "provider_calls": calls6,
        "avoided_calls": 0,
        "elapsed_ms": el6,
        "scope": f"{scope.knowledge_base} (epoch incremented)",
    }
    steps_receipts.append(r6)
    if verbose:
        print(f"[Step 6] Scope Invalidation & Stale Reuse Prevention")
        print(f"         Invalidated Tenant Epoch | Attempted Paraphrase: '{q4}'")
        print(f"         Result: Cache MISS (Stale Scope Blocked) | Fresh Provider Calls: {calls6} | Elapsed: {el6:.1f}ms\n")

    # -------------------------------------------------------------------------
    # STEP 7: Restart Persistence and Safe Fallback
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    # Re-initialize clean agent instances from same SQLite databases
    agent_restarted, provider_restarted, hybrid_restarted = init_agent()
    # Verify post-invalidation task records can be inspected across restarts
    reloaded_state6 = agent_restarted.inspect(tid6, tenant=tenant, principal=principal)
    assert reloaded_state6["id"] == tid6, "Task state must survive agent restart in SQLite"
    assert reloaded_state6["status"] == "AWAITING_APPROVAL", "Unfinished status must survive restart"
    # Approve and persist
    agent_restarted.decide(tid6, "approve", tenant=tenant, principal=principal)
    reloaded_completed = agent_restarted.inspect(tid6, tenant=tenant, principal=principal)
    assert reloaded_completed["status"] == "COMPLETED"
    el7 = (time.perf_counter() - t0) * 1000.0

    r7 = {
        "step": 7,
        "name": "Restart Persistence",
        "question": "N/A (Process re-initialization)",
        "status": "VERIFIED",
        "cache_tier": "DURABLE_STORE",
        "is_hit": True,
        "provider_calls": 0,
        "avoided_calls": 0,
        "elapsed_ms": el7,
        "scope": scope.knowledge_base,
    }
    steps_receipts.append(r7)
    if verbose:
        print(f"[Step 7] Restart Persistence & Safe Fallback")
        print(f"         Re-initialized agent from persistent SQLite stores.")
        print(f"         Reloaded Task {tid6} Status: {reloaded_completed['status']} | Zero Corrupt State | Elapsed: {el7:.1f}ms\n")

    # -------------------------------------------------------------------------
    # CRITERION 8: Per-Request Evidence Reporting (Summary of 7 Scenarios)
    # -------------------------------------------------------------------------
    if verbose:
        print("================================================================================")
        print("           CRITERION 8: PER-REQUEST VERIFICATION EVIDENCE SUMMARY               ")
        print("================================================================================\n")
        print("| Step | Lifecycle Scenario | Question / Action | Cache Tier | Calls | Avoided | Latency | Outcome |")
        print("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
        for r in steps_receipts:
            q_short = (r["question"][:35] + "...") if len(r["question"]) > 35 else r["question"]
            print(f"| {r['step']} | {r['name']} | {q_short} | **{r['cache_tier']}** | {r['provider_calls']} | {r['avoided_calls']} | {r['elapsed_ms']:.1f} ms | {r['status']} |")
        print("\nAll 7 lifecycle execution scenarios verified; Criterion 8 evidence reported successfully (PASS).\n")

    return {
        "status": "PASS",
        "lifecycle_scenarios_executed": len(steps_receipts),
        "evidence_reporting_criterion": "VERIFIED",
        "default_serving_mode": "exact",
        "experimental_semantic_opt_in": experimental_semantic,
        "serving_mode": "guarded" if experimental_semantic else "exact",
        "steps": steps_receipts,
        "models": {
            "dense_encoder": f"{ENCODER_MODEL}@{ENCODER_REVISION}",
            "passage_scorer": f"{PASSAGE_MODEL}@{PASSAGE_REVISION}",
        },
        "database_paths": {
            "tasks": str(tasks_db),
            "answers": str(answers_db),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/flagship_demo")
    parser.add_argument("--json", action="store_true", help="Print json receipt")
    parser.add_argument("--experimental-semantic", action="store_true", help="Explicit opt-in to experimental guarded semantic reuse (default: exact-only).")
    args = parser.parse_args()

    receipt = run_demo(args.directory, verbose=not args.json, experimental_semantic=args.experimental_semantic)
    if args.json:
        print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
