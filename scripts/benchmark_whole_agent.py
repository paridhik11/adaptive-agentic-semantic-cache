"""Milestone C: Whole-Agent Latency and Provider Accounting Benchmark.

Instruments the complete Project 15 documentation agent workflow across:
- Supervisor/planning
- Tool dispatch (search and read)
- Specialist draft generation
- Reviewer verification
- Human approval / admission
- Cache hits (exact and verified semantic)
- Provider attempts, token accounting (prompt & completion), and stage latencies.

Compares matched configurations:
  Arm 1: No cache (mode='off')
  Arm 2: Exact-only (mode='exact')
  Arm 3: Verified-semantic (Conservative hybrid cache with Answer-Aware verification)

Reports both All-Traffic and Eligible-Traffic results.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.document_agent import DocumentAgent, DocumentTools
from src.agents.document_answer_cache import CompletedDocumentCache
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy, LocalPassageScorer
from src.policy.encoders import DenseEncoder
from src.policy.local_models import (
    ENCODER_MODEL, ENCODER_REVISION,
    PASSAGE_MODEL, PASSAGE_REVISION,
)
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, ReuseDecision, digest
from src.policy.sqlite_store import SQLiteResponseStore
from src.policy.work_ledger import WorkLedger


class AgentBenchmarkProvider(FixtureChatProvider):
    """Accurate provider simulation for supervisor, writer, and reviewer stages."""

    def __init__(self, documents: dict[str, str], delay_seconds: float = 0.02):
        self.documents = documents
        self.delay = delay_seconds
        self.attempts = 0
        self.errors = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def __call__(self, body: dict) -> dict:
        self.attempts += 1
        messages = body.get("messages", [])
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""

        # Determine agent workflow stage from prompt
        if "Plan 1-8" in system or "steps" in system or "planning" in system.lower():
            stage = "planning"
            # Find relevant document
            matched_doc = next(iter(self.documents.keys()))
            for doc_id, doc_text in self.documents.items():
                if any(w in doc_text.lower() for w in user.lower().split() if len(w) > 3):
                    matched_doc = doc_id
                    break
            content = json.dumps({
                "steps": [
                    {"id": "read_doc", "tool": "read", "arguments": {"document_id": matched_doc}, "depends_on": []}
                ]
            })
        elif "reviewer" in system.lower() or "check correctness" in system.lower():
            stage = "review"
            content = json.dumps({"approved": True, "feedback": "Draft correctly cites source reference."})
        else:
            stage = "generation"
            # Extract doc content or generate answer
            doc_snippet = next(iter(self.documents.values()))[:200]
            content = f"Based on documentation: {user.strip('?.')}. Reference evidence: {doc_snippet}"

        if self.delay > 0:
            time.sleep(self.delay)

        prompt_tokens = len(system.split()) + len(user.split()) + 15
        completion_tokens = len(content.split()) + 5

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return {
            "id": f"call-{hashlib.sha256(content.encode()).hexdigest()[:12]}",
            "model": body.get("model", "gpt-4o-mini-fixture"),
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def build_workload() -> list[dict]:
    """Diverse 24-task workload testing whole-agent workflow:
    - 6 seed questions
    - 6 exact repeats (eligible for exact cache)
    - 6 safe paraphrases (eligible for semantic cache)
    - 6 adversarial variations (direction reversal, negation, numeric change - must NOT hit semantic cache)
    """
    seeds = [
        ("tuple_conv", "How do I convert a list to a tuple in Python?", "Tuples and lists can be converted using tuple() and list() constructors."),
        ("dict_comp", "How do I write a dictionary comprehension in Python?", "Dictionary comprehensions use the syntax {key: value for item in iterable}."),
        ("str_split", "How do I split a string by delimiter in Python?", "The str.split(sep) method splits a string by separator into a list of strings."),
        ("file_read", "How do I read a file line by line in Python?", "Use 'with open(path) as f:' and iterate over f line by line."),
        ("list_sort", "How do I sort a list in Python?", "Use list.sort() to sort in place or sorted(list) to return a new sorted list."),
        ("set_unique", "How do I remove duplicates from a list in Python?", "Convert the list to a set with set(my_list), or use dict.fromkeys() to preserve order."),
    ]

    tasks = []
    # 1. Seeds (cold)
    for qid, q, ans in seeds:
        tasks.append({
            "task_id": f"{qid}:seed",
            "family": "seed",
            "question": q,
            "eligible": True,
            "expected_hit": False,
            "doc_id": qid,
            "doc_text": ans,
        })

    # 2. Exact repeats
    for qid, q, ans in seeds:
        tasks.append({
            "task_id": f"{qid}:exact",
            "family": "exact_repeat",
            "question": q,
            "eligible": True,
            "expected_hit": True,
            "expected_tier": "EXACT",
            "doc_id": qid,
            "doc_text": ans,
        })

    # 3. Safe paraphrases
    paraphrase_map = {
        "tuple_conv": "How can I turn a Python list into a tuple?",
        "dict_comp": "What is the syntax for dict comprehension in Python?",
        "str_split": "In Python, how can I divide a string by a delimiter?",
        "file_read": "How to iterate over lines in a file in Python?",
        "list_sort": "What is the method to sort elements of a list in Python?",
        "set_unique": "How can I filter out duplicate items from a list in Python?",
    }
    for qid, q, ans in seeds:
        tasks.append({
            "task_id": f"{qid}:paraphrase",
            "family": "safe_paraphrase",
            "question": paraphrase_map[qid],
            "eligible": True,
            "expected_hit": True,
            "expected_tier": "SEMANTIC",
            "doc_id": qid,
            "doc_text": ans,
        })

    # 4. Adversarial variations (unsafe - must fail closed and execute fresh agent work)
    adversarial_map = {
        "tuple_conv": "How do I convert a tuple to a list in Python?",  # Direction reversal
        "dict_comp": "How do I write a dictionary comprehension without using curly braces in Python?",  # Negation
        "str_split": "How do I split a string into exactly 100 character chunks in Python?",  # Numeric constraint
        "file_read": "How do I write a file line by line in Python?",  # Action reversal
        "list_sort": "How do I sort a list without using any built-in functions in Python?",  # Negation
        "set_unique": "How do I count duplicate occurrences in a list in Python?",  # Changed task
    }
    for qid, q, ans in seeds:
        tasks.append({
            "task_id": f"{qid}:adversarial",
            "family": "adversarial_constraint",
            "question": adversarial_map[qid],
            "eligible": False,  # Not eligible for safe reuse of the seed answer
            "expected_hit": False,
            "doc_id": qid,
            "doc_text": ans,
        })

    return tasks


def run_arm(
    arm_name: str,
    workload: list[dict],
    documents: dict[str, str],
    db_dir: Path,
    encoder: DenseEncoder,
    passage_scorer: LocalPassageScorer,
) -> dict:
    db_dir.mkdir(parents=True, exist_ok=True)
    answers_db = db_dir / f"{arm_name}_answers.db"
    tasks_db = db_dir / f"{arm_name}_tasks.db"
    if answers_db.exists():
        answers_db.unlink()
    if tasks_db.exists():
        tasks_db.unlink()

    provider = AgentBenchmarkProvider(documents, delay_seconds=0.015)
    tools = DocumentTools(documents)
    scope = KnowledgeScope("agent-python-docs", tools.revision, "public-docs", f"whole-agent-{arm_name}")

    if arm_name == "no_cache":
        mode = "off"
        approver = None
    elif arm_name == "exact_only":
        mode = "exact"
        approver = None
    elif arm_name == "verified_semantic":
        mode = "guarded"
        approver = AnswerAwareSemanticPolicy(
            scorer=passage_scorer,
            threshold=0.85,
            enabled=True,
        )

    exact = ScopedResponseCache(
        eligibility=lambda _: EligibilityDecision(True, "whole_agent_benchmark"),
        storage=SQLiteResponseStore(answers_db, namespace=f"agent-{arm_name}"),
    )
    outer = GuardedSemanticCache(
        exact,
        scope=scope,
        encoder=encoder if mode == "guarded" else None,
        mode=mode,
        approver=approver,
        persistent_candidates=(mode in {"shadow", "guarded"}),
        local_approval_only=True,
        threshold=0.60,
        margin=0.0,
    )

    answer_cache = CompletedDocumentCache(outer)
    service = GatewayService(
        ScopedResponseCache(eligibility=lambda _: EligibilityDecision(False, "agent_stage_cache_off")),
        provider,
        allowed_models={"gpt-4o-mini-fixture"},
    )

    agent = DocumentAgent(
        service,
        tools,
        tasks_db,
        model="gpt-4o-mini-fixture",
        answer_cache=answer_cache,
        stage_accounting=True,
    )

    task_records = []
    t_arm0 = time.perf_counter()

    for item in workload:
        t0 = time.perf_counter()
        prov_before = provider.attempts
        p_tokens_before = provider.total_prompt_tokens
        c_tokens_before = provider.total_completion_tokens

        # Create task with standalone_document_task=True for whole-answer lookup
        tid = agent.create(
            item["question"],
            tenant="tenant-main",
            principal="benchmark-user",
            standalone_document_task=True,
        )

        # Advance through states until terminal or awaiting approval
        state = None
        for step_idx in range(25):
            state = agent.advance(tid, tenant="tenant-main", principal="benchmark-user")
            if state["status"] in {"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        prov_calls = provider.attempts - prov_before
        p_tokens = provider.total_prompt_tokens - p_tokens_before
        c_tokens = provider.total_completion_tokens - c_tokens_before

        is_hit = bool(state.get("answer_cache_hit", False))
        tier = "NONE"
        if is_hit:
            tier = state["trace"][0].get("tier", "EXACT")

        # Invariant check: Cache hit must show zero remote generation calls attributable to that hit
        if is_hit and prov_calls != 0:
            raise RuntimeError(f"Violation: Cache hit on task {item['task_id']} incurred {prov_calls} provider calls!")

        # Count stage events
        work_events = state.get("work", [])
        stages_run = [e["stage"] for e in work_events]

        tool_calls = sum(1 for e in stages_run if e == "tool")
        planning_calls = sum(1 for e in stages_run if e == "planning")
        generation_calls = sum(1 for e in stages_run if e == "generation")
        review_calls = sum(1 for e in stages_run if e == "review")

        # Operator approval to admit qualified answers for subsequent reuse
        if state["status"] == "AWAITING_APPROVAL":
            agent.decide(tid, "approve", tenant="tenant-main", principal="benchmark-user")

        task_records.append({
            "task_id": item["task_id"],
            "family": item["family"],
            "eligible": item["eligible"],
            "is_hit": is_hit,
            "tier": tier,
            "provider_calls": prov_calls,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "elapsed_ms": elapsed_ms,
            "stages": {
                "planning": planning_calls,
                "tool": tool_calls,
                "generation": generation_calls,
                "review": review_calls,
            },
        })

    arm_elapsed_s = time.perf_counter() - t_arm0

    # Summary metrics (All traffic and Eligible traffic)
    def compute_summary(records):
        n = len(records)
        hits = sum(r["is_hit"] for r in records)
        exact_hits = sum(r["is_hit"] and r["tier"] == "EXACT" for r in records)
        semantic_hits = sum(r["is_hit"] and r["tier"] == "SEMANTIC" for r in records)
        misses = n - hits

        prov_calls_total = sum(r["provider_calls"] for r in records)
        prompt_tokens_total = sum(r["prompt_tokens"] for r in records)
        comp_tokens_total = sum(r["completion_tokens"] for r in records)
        total_tokens = sum(r["total_tokens"] for r in records)

        latencies = sorted(r["elapsed_ms"] for r in records)
        p50 = latencies[int(n * 0.50)] if latencies else 0.0
        p95 = latencies[int(n * 0.95)] if latencies else 0.0
        mean_l = sum(latencies) / n if n else 0.0

        hit_lats = sorted(r["elapsed_ms"] for r in records if r["is_hit"])
        miss_lats = sorted(r["elapsed_ms"] for r in records if not r["is_hit"])

        return {
            "total_tasks": n,
            "hits_total": hits,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "misses": misses,
            "provider_calls_total": prov_calls_total,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": comp_tokens_total,
            "total_tokens_consumed": total_tokens,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_mean_ms": mean_l,
            "hit_p50_ms": hit_lats[int(len(hit_lats) * 0.50)] if hit_lats else 0.0,
            "miss_p50_ms": miss_lats[int(len(miss_lats) * 0.50)] if miss_lats else 0.0,
        }

    all_traffic_summary = compute_summary(task_records)
    eligible_traffic_summary = compute_summary([r for r in task_records if r["eligible"]])

    return {
        "arm": arm_name,
        "elapsed_seconds": arm_elapsed_s,
        "all_traffic": all_traffic_summary,
        "eligible_traffic": eligible_traffic_summary,
        "task_records": task_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/whole_agent_benchmark_results.json")
    args = parser.parse_args()

    workload = build_workload()
    documents = {item["doc_id"]: item["doc_text"] for item in workload if "doc_id" in item}

    print("Pre-loading neural models for whole-agent evaluation...", flush=True)
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    passage_scorer = LocalPassageScorer(PASSAGE_MODEL, PASSAGE_REVISION)

    arms = ["no_cache", "exact_only", "verified_semantic"]
    db_base = ROOT / "runs/agent_bench_dbs"

    benchmark_output = {}

    for arm in arms:
        print(f"\n--- Running Arm: {arm} ({len(workload)} tasks) ---", flush=True)
        res = run_arm(arm, workload, documents, db_base, encoder, passage_scorer)
        benchmark_output[arm] = res
        all_s = res["all_traffic"]
        elig_s = res["eligible_traffic"]
        print(f"[{arm}] ALL TRAFFIC: {all_s['total_tasks']} tasks | Hits: {all_s['hits_total']} (Exact: {all_s['exact_hits']}, Sem: {all_s['semantic_hits']}) | Provider Calls: {all_s['provider_calls_total']} | Total Tokens: {all_s['total_tokens_consumed']} | Latency p50: {all_s['latency_p50_ms']:.1f}ms")
        print(f"[{arm}] ELIGIBLE TRAFFIC: {elig_s['total_tasks']} tasks | Hits: {elig_s['hits_total']} | Avoided Calls: {benchmark_output['no_cache']['eligible_traffic']['provider_calls_total'] - elig_s['provider_calls_total']} | Latency p50: {elig_s['latency_p50_ms']:.1f}ms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)

    print(f"\n=== WHOLE-AGENT BENCHMARK RESULTS SAVED TO {args.output} ===")

    # Print markdown comparison table
    print("\n### Whole-Agent Comparative Accounting\n")
    print("| Arm | Traffic Scope | Total Tasks | Hits (Exact / Sem) | Provider Calls | Tokens Consumed | Calls Avoided | Token Savings | Latency p50 (ms) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    no_calls_all = benchmark_output["no_cache"]["all_traffic"]["provider_calls_total"]
    no_tokens_all = benchmark_output["no_cache"]["all_traffic"]["total_tokens_consumed"]
    no_calls_elig = benchmark_output["no_cache"]["eligible_traffic"]["provider_calls_total"]
    no_tokens_elig = benchmark_output["no_cache"]["eligible_traffic"]["total_tokens_consumed"]

    for arm in arms:
        a = benchmark_output[arm]["all_traffic"]
        e = benchmark_output[arm]["eligible_traffic"]
        call_sav_all = no_calls_all - a["provider_calls_total"]
        tok_sav_all = no_tokens_all - a["total_tokens_consumed"]
        call_sav_elig = no_calls_elig - e["provider_calls_total"]
        tok_sav_elig = no_tokens_elig - e["total_tokens_consumed"]

        pct_tok_all = (tok_sav_all / no_tokens_all * 100) if no_tokens_all else 0.0
        pct_tok_elig = (tok_sav_elig / no_tokens_elig * 100) if no_tokens_elig else 0.0

        print(f"| **{arm}** | All Traffic | {a['total_tasks']} | {a['hits_total']} ({a['exact_hits']}/{a['semantic_hits']}) | {a['provider_calls_total']} | {a['total_tokens_consumed']:,} | {call_sav_all} | {tok_sav_all:,} ({pct_tok_all:.1f}%) | {a['latency_p50_ms']:.1f} ms |")
        print(f"| **{arm}** | Eligible | {e['total_tasks']} | {e['hits_total']} ({e['exact_hits']}/{e['semantic_hits']}) | {e['provider_calls_total']} | {e['total_tokens_consumed']:,} | {call_sav_elig} | {tok_sav_elig:,} ({pct_tok_elig:.1f}%) | {e['latency_p50_ms']:.1f} ms |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
