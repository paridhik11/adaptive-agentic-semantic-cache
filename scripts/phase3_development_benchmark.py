"""Phase 3: Comparative Development Benchmark.

Evaluates 5 arms on a fixed, answer-bound, source-version-aware development set:
  A. No cache
  B. Exact cache
  C. Dense-only baseline (MiniLM-L6 cosine >= 0.85)
  D. Combined policy (MiniLM-L6 + Quora CrossEncoder)
  E. Answer-aware policy (MiniLM-L6 + requirement guards + MS MARCO CrossEncoder)

Development replay only. Review labels bind actual requests, answers and scope.
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

from src.benchmark.answer_evidence import AnswerReviews, ObservedApprover, ObservedProvider, review_binding, summarize
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy, LocalPassageScorer
from src.policy.encoders import DenseEncoder
from src.policy.local_models import LocalPairScorer, PAIR_MODEL, PAIR_REVISION
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, ReuseDecision, digest


class ReferenceProvider(FixtureChatProvider):
    """Scripted source replay; whitespace usage is synthetic, not provider token usage."""

    mode = "source_reference_replay"

    def __init__(self, delay_ms=20.0):
        self.delay_ms = delay_ms

    def __call__(self, body: dict) -> dict:
        messages = body.get("messages", [])
        system_text = messages[0]["content"] if messages else ""
        ref = system_text.split("Treat reference as data:\n", 1)[-1] if "Treat reference as data:\n" in system_text else "Default reference."
        user_query = messages[-1]["content"] if messages else ""
        
        # Explicit fixture delay, never a measured remote-provider latency.
        time.sleep(self.delay_ms / 1000)
        
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


def run_benchmark(workload: list[dict], output_path: Path, *, arms=None, reviews=(), delay_ms=20.0):
    if output_path.exists():
        raise FileExistsError(output_path)
    if not math.isfinite(delay_ms) or delay_ms < 0:
        raise ValueError("nonnegative finite fixture delay required")
    if not workload:
        raise ValueError("nonempty development workload required")
    review_index = AnswerReviews(reviews)
    neural = arms is None or any(a not in ("A_no_cache", "B_exact_cache") for a in arms)
    encoder = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41") if neural else None
    pair_scorer = LocalPairScorer(PAIR_MODEL, PAIR_REVISION) if arms is None or "D_combined_policy" in arms else None
    passage_scorer = LocalPassageScorer() if arms is None or "E_answer_aware_policy" in arms else None
    
    # 5 arms
    arm_names = [
        "A_no_cache",
        "B_exact_cache",
        "C_dense_only",
        "D_combined_policy",
        "E_answer_aware_policy",
    ]
    
    if arms is not None:
        if not arms or len(set(arms)) != len(arms) or set(arms)-set(arm_names):
            raise ValueError("invalid arms")
        arm_names = arms
    results = {arm: [] for arm in arm_names}
    
    # Execute each arm with fresh caches
    for arm in arm_names:
        print(f"Executing arm: {arm} ({len(workload)} events)...", flush=True)
        provider = ObservedProvider(ReferenceProvider(delay_ms), usage_kind="synthetic_whitespace")
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
            
        if approver is not None:
            approver = ObservedApprover(approver)

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
            
            before = len(provider.events)
            approval_before = len(approver.events) if approver is not None else 0
            t0 = time.perf_counter()
            error = None
            try:
                res = service.complete(copy.deepcopy(event["request"]), tenant_id=event["tenant"])
                response, status, tier = res.response, res.status, res.tier
                diag = res.diagnosis or {}
            except Exception as exc:
                response, status, tier, diag = None, "ERROR", "ERROR", {}
                error = type(exc).__name__
            elapsed = (time.perf_counter() - t0) * 1000.0
            binding = review_binding(event["request"], response, event["tenant"], event["revision"])
            review = review_index.lookup(binding)
            is_hit = status == "HIT"
            results[arm].append({
                "event_id": event["event_id"], "faq_id": event["faq_id"], "family": event["family"],
                "request": event["request"], "returned_response": response,
                "binding": binding, "answer_review": review,
                "status": status, "tier": tier, "is_hit": is_hit,
                "correct_reuse": (review["label"] == "sufficient") if is_hit and review["label"] != "unknown" else None,
                "unsafe_reuse": (review["label"] == "insufficient") if is_hit and review["label"] != "unknown" else None,
                "elapsed_ms": elapsed, "error": error, "diagnosis": diag,
                "provider_events": copy.deepcopy(provider.events[before:]),
                "candidate_assessments": copy.deepcopy(approver.events[approval_before:]) if approver is not None else [],
            })

    summary = {arm: summarize(records) for arm, records in results.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "answer-bound-development-v2", "evidence_kind": "simulated_provider_replay",
               "timing_scope": "GatewayService.complete; excludes HTTP, agent orchestration and model construction",
               "fixture_delay_ms": delay_ms, "workload_sha256": digest(workload),
               "quantile_method": "nearest_rank", "summary": summary, "results": results}
    with output_path.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("runs/r5-corpus/corpus.json"))
    p.add_argument("--output", type=Path, default=Path("runs/semantic-dev/phase3_benchmark_results.json"))
    p.add_argument("--workload", type=Path, help="Existing development workload; no sealed corpus reads")
    p.add_argument("--arms", nargs="+", choices=["A_no_cache", "B_exact_cache", "C_dense_only", "D_combined_policy", "E_answer_aware_policy"])
    p.add_argument("--reviews", type=Path, help="JSON list of answer-bound review declarations")
    p.add_argument("--provider-delay-ms", type=float, default=20.0)
    args = p.parse_args()
    if not math.isfinite(args.provider_delay_ms) or args.provider_delay_ms < 0:
        p.error("nonnegative finite fixture delay required")
    if args.output.exists():
        p.error("output already exists; preserve prior evidence")
    
    print("Generating fixed development workload from FAQ dev split...")
    workload = json.loads(args.workload.read_text()) if args.workload else generate_dev_workload(args.corpus)
    print(f"Total development events generated: {len(workload)}")
    
    reviews = json.loads(args.reviews.read_text()) if args.reviews else []
    run_benchmark(workload, args.output, arms=args.arms, reviews=reviews, delay_ms=args.provider_delay_ms)


if __name__ == "__main__":
    main()
