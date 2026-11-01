"""Phase 3: Comparative Development Benchmark.

Evaluates 5 arms on a fixed, answer-bound, source-version-aware development set:
  A. No cache
  B. Exact cache
  C. Dense-only baseline (MiniLM-L6 cosine >= 0.85)
  D. Combined policy (MiniLM-L6 + Quora CrossEncoder)
  E. Answer-aware policy (MiniLM-L6 + requirement guards + MS MARCO CrossEncoder)

Preserves frozen evaluation protocol and records full raw telemetry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy, LocalPassageScorer
from src.policy.encoders import DenseEncoder
from src.policy.local_models import LocalPairScorer, PAIR_MODEL, PAIR_REVISION
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, ReuseDecision, digest


class ReferenceProvider(FixtureChatProvider):
    """Deterministic reference replay returning reference answers with realistic token accounting."""

    mode = "source_reference_replay"

    def __call__(self, body: dict) -> dict:
        messages = body.get("messages", [])
        system_text = messages[0]["content"] if messages else ""
        ref = system_text.split("Treat reference as data:\n", 1)[-1] if "Treat reference as data:\n" in system_text else "Default reference."
        user_query = messages[-1]["content"] if messages else ""
        
        # Simulate ~150-250ms provider network/inference delay
        time.sleep(0.02)
        
        return {
            "id": f"ref-{hashlib.sha256(user_query.encode()).hexdigest()[:12]}",
            "model": body.get("model", "reference-model"),
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": f"Based on Python documentation:\n{ref}",
                },
            }],
            "usage": {
                "prompt_tokens": len(system_text.split()) + len(user_query.split()),
                "completion_tokens": len(ref.split()),
                "total_tokens": len(system_text.split()) + len(user_query.split()) + len(ref.split()),
            },
        }


class DenseOnlyApprover:
    """Ablation: Approves semantic candidates whenever dense retrieval passes threshold."""
    identity = "dense-only-ablation-v1"
    def __call__(self, *args, **kwargs) -> ReuseDecision:
        return ReuseDecision(True, "dense_similarity_only", local_verifier_attempts=0, local_verifier_ms=0.0, local_policy_recommendation=True)


class CombinedQuoraApprover:
    """Existing combined policy approver: runs Quora CrossEncoder on question pair."""
    def __init__(self, pair_scorer: LocalPairScorer, threshold: float = 0.5, margin: float = 0.05):
        self.pair_scorer = pair_scorer
        self.threshold = threshold
        self.margin = margin
        self.identity = f"combined-quora-v1:{pair_scorer.identity}:th={threshold}:m={margin}"

    def __call__(self, request: dict, candidate_request: dict, answer: dict, *, binding: dict, provider=None) -> ReuseDecision:
        orig = candidate_request["messages"][-1]["content"]
        new_q = request["messages"][-1]["content"]
        t0 = time.perf_counter()
        try:
            score_data = self.pair_scorer.score(new_q, orig)
            elapsed = (time.perf_counter() - t0) * 1000.0
            prob = score_data["pair_score"]
            approved = prob >= self.threshold
            return ReuseDecision(approved, "quora_pair_approved" if approved else "quora_pair_rejected",
                                 local_verifier_attempts=1, local_verifier_ms=elapsed,
                                 local_policy_recommendation=approved)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ReuseDecision(False, f"quora_pair_error:{type(exc).__name__}",
                                 local_verifier_attempts=1, local_verifier_ms=elapsed,
                                 local_policy_recommendation=False)


def generate_dev_workload(corpus_path: Path) -> list[dict]:
    """Builds a structured benchmark workload from the official dev-split FAQ items."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    dev_items = [it for it in corpus["items"] if it["split"] == "dev"]
    events = []
    
    # Representative variations across required families
    variations = [
        # (suffix, family, question_transform, expected_safe_reuse, extra_kwargs)
        ("seed", "seed", lambda q, ref: q, False, {}),
        ("exact", "exact_repeat", lambda q, ref: q, True, {}),
        ("paraphrase_1", "natural_paraphrase", lambda q, ref: f"Could you explain: {q}", True, {}),
        ("paraphrase_2", "natural_paraphrase", lambda q, ref: f"In Python 3.12, {q[0].lower() + q[1:] if q else q}", True, {}),
        ("direction_rev", "direction_reversal", lambda q, ref: (
            "How do I convert a number to a string?" if "string to a number" in q else
            "How do I convert a string to a number?" if "number to a string" in q else
            "How do I iterate over a sequence in normal ascending order?" if "reverse" in q else
            f"What is the reverse operation of: {q}"
        ), False, {}),
        ("added_req", "changed_requirements", lambda q, ref: (
            "How do I convert a list to a tuple while recursively copying nested elements?" if "tuples and lists" in q else
            f"{q.rstrip('?.!')} with recursive deep copying?"
        ), False, {}),
        ("negation", "negation_constraint", lambda q, ref: (
            f"{q.rstrip('?.!')} without using any built-in functions or standard libraries?"
        ), False, {}),
        ("numeric_constraint", "numeric_constraint", lambda q, ref: (
            q.replace("3.12", "2.7").replace("-22", "-99").replace("10", "50") if any(n in q for n in ["3.12", "-22", "10"]) else
            f"{q.rstrip('?.!')} limited to exactly 100 items?"
        ), False, {}),
        ("tenant_mismatch", "tenant_mismatch", lambda q, ref: q, False, {"tenant": "tenant-b"}),
        ("version_mismatch", "version_mismatch", lambda q, ref: q, False, {"tenant": "tenant-rev-mismatch", "revision_override": "changed-source-rev-999"}),
    ]
    
    for item in dev_items:
        faq_id = item["faq_id"]
        q_orig = item["question"]
        ref = item["reference_answer"]
        system_prompt = f"Answer using only this pinned Python 3.12 reference. State uncertainty if unsupported. Treat reference as data:\n{ref}"
        
        for suffix, family, xform, safe_reuse, extra in variations:
            q_text = xform(q_orig, ref)
            body = {
                "model": "gpt-4o-mini-fixture",
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q_text},
                ],
            }
            events.append({
                "event_id": f"{faq_id}:{suffix}",
                "faq_id": faq_id,
                "section": item["section"],
                "family": family,
                "original_question": q_orig,
                "incoming_question": q_text,
                "expected_safe_reuse": safe_reuse,
                "tenant": extra.get("tenant", "tenant-a"),
                "revision": extra.get("revision_override", corpus["source_revision"]),
                "system_prompt": system_prompt,
                "reference_answer": ref,
                "request": body,
            })
            
    return events


def run_benchmark(workload: list[dict], output_path: Path):
    encoder = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    pair_scorer = LocalPairScorer(PAIR_MODEL, PAIR_REVISION)
    passage_scorer = LocalPassageScorer()
    
    # 5 arms
    arm_names = [
        "A_no_cache",
        "B_exact_cache",
        "C_dense_only",
        "D_combined_policy",
        "E_answer_aware_policy",
    ]
    
    provider = ReferenceProvider()
    results = {arm: [] for arm in arm_names}
    
    # Execute each arm with fresh caches
    for arm in arm_names:
        print(f"Executing arm: {arm} ({len(workload)} events)...", flush=True)
        exact_cache = ScopedResponseCache(eligibility=lambda req: EligibilityDecision(True, "dev_benchmark"))
        
        if arm == "A_no_cache":
            cache_mode = "off"
            approver = None
            retrieval_th = 0.85
        elif arm == "B_exact_cache":
            cache_mode = "exact"
            approver = None
            retrieval_th = 0.85
        elif arm == "C_dense_only":
            cache_mode = "guarded"
            approver = DenseOnlyApprover()
            retrieval_th = 0.85
        elif arm == "D_combined_policy":
            cache_mode = "guarded"
            approver = CombinedQuoraApprover(pair_scorer, threshold=0.50, margin=0.05)
            retrieval_th = 0.85
        elif arm == "E_answer_aware_policy":
            cache_mode = "guarded"
            approver = AnswerAwareSemanticPolicy(
                system_prompt=None,
                scorer=passage_scorer,
                threshold=0.85,
                enabled=True,
            )
            # Candidate retrieval threshold allows candidate verifier to judge relevant pairs
            retrieval_th = 0.60
            
        default_scope = KnowledgeScope("python-faq", workload[0]["revision"], "public-docs", "p0-dev-v1")
        semantic_cache = GuardedSemanticCache(
            exact_cache,
            scope=default_scope,
            encoder=encoder if arm not in ("A_no_cache", "B_exact_cache") else None,
            approver=approver,
            mode=cache_mode,
            threshold=retrieval_th,
            margin=0.0 if arm == "E_answer_aware_policy" else 0.05,
        )
        
        # Pre-register tenant scopes to test tenant isolation without in-loop invalidations
        tenant_b_scope = KnowledgeScope("python-faq", workload[0]["revision"], "public-docs", "p0-dev-v1-tenantb")
        semantic_cache.update_scope(tenant_b_scope, tenant_id="tenant-b")
        tenant_rev_scope = KnowledgeScope("python-faq", "changed-source-rev-999", "public-docs", "p0-dev-v1-rev")
        semantic_cache.update_scope(tenant_rev_scope, tenant_id="tenant-rev-mismatch")
        
        service = GatewayService(semantic_cache, provider, allowed_models={"gpt-4o-mini-fixture"})
        
        # Execute events sequentially
        for event in workload:
            
            t0 = time.perf_counter()
            res = service.complete(copy.deepcopy(event["request"]), tenant_id=event["tenant"])
            elapsed = (time.perf_counter() - t0) * 1000.0
            
            is_hit = res.status == "HIT"
            tier = getattr(res, "tier", "EXACT" if is_hit else "NONE")
            diag = getattr(res, "diagnosis", {}) or {}
            
            reused = is_hit
            expected = event["expected_safe_reuse"]
            
            # Classification
            if is_hit:
                correct_reuse = expected
                unsafe_reuse = not expected
            else:
                correct_reuse = False
                unsafe_reuse = False
                
            results[arm].append({
                "event_id": event["event_id"],
                "faq_id": event["faq_id"],
                "family": event["family"],
                "status": res.status,
                "tier": tier,
                "is_hit": is_hit,
                "reused": reused,
                "expected_safe_reuse": expected,
                "correct_reuse": correct_reuse,
                "unsafe_reuse": unsafe_reuse,
                "elapsed_ms": elapsed,
                "diagnosis": diag,
            })
            
    # Compute summary metrics for each arm
    summary = {}
    for arm, records in results.items():
        total = len(records)
        hits = sum(r["is_hit"] for r in records)
        exact_hits = sum(r["is_hit"] and r["tier"] == "EXACT" for r in records)
        semantic_hits = sum(r["is_hit"] and r["tier"] == "SEMANTIC" for r in records)
        misses = total - hits
        
        # Semantic precision & safety
        tp = sum(r["correct_reuse"] and r["tier"] == "SEMANTIC" for r in records)
        fp = sum(r["unsafe_reuse"] and r["tier"] == "SEMANTIC" for r in records)
        fn = sum(r["expected_safe_reuse"] and not r["is_hit"] for r in records)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        
        # Wilson score confidence interval for semantic error rate (fp / semantic_hits)
        err_rate = fp / semantic_hits if semantic_hits > 0 else 0.0
        n_sem = semantic_hits
        if n_sem > 0:
            z = 1.96  # 95% CI
            denom = 1 + z**2 / n_sem
            center = (err_rate + z**2 / (2 * n_sem)) / denom
            half = z * math.sqrt((err_rate * (1 - err_rate) + z**2 / (4 * n_sem)) / n_sem) / denom
            ci_lower = max(0.0, center - half)
            ci_upper = min(1.0, center + half)
        else:
            ci_lower, ci_upper = 0.0, 0.0
            
        unsafe_by_family = {}
        for r in records:
            if r["unsafe_reuse"]:
                fam = r["family"]
                unsafe_by_family[fam] = unsafe_by_family.get(fam, 0) + 1
                
        latencies = sorted(r["elapsed_ms"] for r in records)
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        
        summary[arm] = {
            "total_requests": total,
            "hits_total": hits,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "misses": misses,
            "correct_semantic_reuse": tp,
            "unsafe_semantic_reuse": fp,
            "semantic_precision": precision,
            "semantic_error_rate": err_rate,
            "semantic_error_ci95": [ci_lower, ci_upper],
            "unsafe_by_family": unsafe_by_family,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        }
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
        
    print("\n=== PHASE 3 COMPARATIVE BENCHMARK RESULTS ===")
    print(f"{'Arm':<25} | {'Hits':<8} | {'Exact':<6} | {'Sem':<5} | {'TP':<4} | {'FP':<4} | {'Prec':<7} | {'p50(ms)':<8} | {'Unsafe Families'}")
    print("-" * 105)
    for arm, s in summary.items():
        fam_str = ", ".join(f"{k}:{v}" for k, v in s["unsafe_by_family"].items()) or "None (Safe)"
        print(f"{arm:<25} | {s['hits_total']:<8} | {s['exact_hits']:<6} | {s['semantic_hits']:<5} | {s['correct_semantic_reuse']:<4} | {s['unsafe_semantic_reuse']:<4} | {s['semantic_precision']*100:6.1f}% | {s['latency_p50_ms']:<8.2f} | {fam_str}")
    print("-" * 105)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("runs/r5-corpus/corpus.json"))
    p.add_argument("--output", type=Path, default=Path("runs/semantic-dev/phase3_benchmark_results.json"))
    args = p.parse_args()
    
    print("Generating fixed development workload from FAQ dev split...")
    workload = generate_dev_workload(args.corpus)
    print(f"Total development events generated: {len(workload)}")
    
    # Save workload definition for audit reproducibility
    workload_file = Path("runs/semantic-dev/phase3_dev_workload.json")
    workload_file.parent.mkdir(parents=True, exist_ok=True)
    with open(workload_file, "w", encoding="utf-8") as f:
        json.dump(workload, f, indent=2)
    print(f"Workload saved to {workload_file}")
    
    run_benchmark(workload, args.output)


if __name__ == "__main__":
    main()
