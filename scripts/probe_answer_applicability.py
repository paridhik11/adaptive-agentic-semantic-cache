"""Exposed authored development study, never independent answer-quality evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policy.answer_applicability import (ANSWER_MODEL, ANSWER_REVISION, CHAT_MODEL, CHAT_REVISION,
    DiagnosticAnswerPolicy, LocalYesNoScorer, LocalChatYesNoScorer)
from src.policy.encoders import DenseEncoder
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION
from src.policy.response_cache import ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, PinnedContextPolicy

CONTEXT = ("Answer only from this authored development reference, not a full documentation source. "
    "In Python, tuple(items) returns a tuple containing the items of an iterable. "
    "list(items) returns a list containing the items of an iterable. Neither operation recursively "
    "converts nested containers. Tuples cannot be modified in place. "
    "sorted(items) returns a new ascending sorted list without modifying the input. "
    "sorted(items, reverse=True) returns a new descending sorted list. "
    "To convert Celsius C to Fahrenheit F, compute F = C * 9 / 5 + 32. "
    "To convert Fahrenheit F to Celsius C, compute C = (F - 32) * 5 / 9. "
    "Celsius to Kelvin is K = C + 273.15.")
LIST_QUERY = "How do I convert a Python list into a tuple?"
LIST_ANSWER = "Use tuple(items) to create a tuple from the items in your list. This is a shallow conversion."
TEMP_QUERY = "How do I convert Celsius to Fahrenheit?"
TEMP_ANSWER = "Compute F = C * 9 / 5 + 32, where C is the temperature in Celsius."
SORT_QUERY = "How do I sort a Python list without changing it?"
SORT_ANSWER = "Use sorted(items) to return a new ascending sorted list, leaving the original list unchanged."
# All labels are the implementation author's development expectations, not human
# or independent model reviews. These groups are spent/exposed immediately.
CASES = [
    ("list-positive", "list-tuple", LIST_QUERY, "How can I make a tuple from an existing Python list?", LIST_ANSWER, True),
    ("list-positive-2", "list-tuple", LIST_QUERY, "What function turns list elements into a tuple?", LIST_ANSWER, True),
    ("temperature-positive", "temperature", TEMP_QUERY, "Given Celsius, how can I express the temperature in Fahrenheit?", TEMP_ANSWER, True),
    ("sort-positive", "sort", SORT_QUERY, "How can I get a sorted copy of a list?", SORT_ANSWER, True),
    ("direction", "list-tuple", LIST_QUERY, "How do I convert a Python tuple into a list?", LIST_ANSWER, False),
    ("recursion", "list-tuple", LIST_QUERY, "How do I recursively convert a nested Python list into tuples?", LIST_ANSWER, False),
    ("new-constraint", "list-tuple", LIST_QUERY, "How can I turn a list into a tuple without creating another object?", LIST_ANSWER, False),
    ("negation", "list-tuple", LIST_QUERY, "Why should I not convert a Python list into a tuple?", LIST_ANSWER, False),
    ("units", "temperature", TEMP_QUERY, "How do I convert Celsius to Kelvin?", TEMP_ANSWER, False),
    ("temperature-direction", "temperature", TEMP_QUERY, "How do I convert Fahrenheit to Celsius?", TEMP_ANSWER, False),
    ("sort-direction", "sort", SORT_QUERY, "How do I sort a Python list in descending order?", SORT_ANSWER, False),
    ("wrong-actual-answer", "list-tuple", LIST_QUERY, "How can I make a tuple from a list?", "Use list(items) to create a tuple.", False),
    ("unsupported-actual-answer", "list-tuple", LIST_QUERY, "How can I make a tuple from a list?", "Use tuple(items). The resulting tuple can then be modified in place.", False),
    ("irrelevant-actual-answer", "list-tuple", LIST_QUERY, "How can I make a tuple from a list?", TEMP_ANSWER, False),
    ("incomplete-actual-answer", "temperature", TEMP_QUERY, "How do I convert Celsius to Fahrenheit?", "Multiply Celsius by 9 / 5.", False),
]


def request(query):
    return {"model": "authored-reference-fixture", "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "system", "content": CONTEXT}, {"role": "user", "content": query}]}


def response(content):
    return {"model": "authored-reference-fixture", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}]}


def run(scorer_name="flan"):
    scorer = LocalYesNoScorer() if scorer_name == "flan" else LocalChatYesNoScorer()
    policy = DiagnosticAnswerPolicy(system_prompt=CONTEXT, scorer=scorer)
    rows = []
    for case_id, group, original, incoming, answer, expected in CASES:
        assessment = policy.assess(request(incoming), request(original), response(answer))
        rows.append({"id": case_id, "intent_group": group, "authored_expected_applicable": expected,
                     "assessment": asdict(assessment)})
        print(case_id, assessment.reason, flush=True)

    # Exercise real dense lookup + exact cache with this diagnostic approver.
    # Its positive recommendation must NEVER bypass generation on a cache miss.
    calls = []
    def provider(body):
        calls.append(body["messages"][-1]["content"])
        return response(LIST_ANSWER)
    cache = GuardedSemanticCache(ScopedResponseCache(eligibility=PinnedContextPolicy(CONTEXT)),
        scope=KnowledgeScope("authored-dev", "v1", "public", policy.identity),
        encoder=DenseEncoder(ENCODER_MODEL, ENCODER_REVISION), mode="guarded", approver=policy,
        threshold=.80, margin=.05)
    seed = cache.process(request(LIST_QUERY), tenant_id="dev", provider=provider)
    repeated = cache.process(request(LIST_QUERY), tenant_id="dev", provider=provider)
    variant = cache.process(request(CASES[0][3]), tenant_id="dev", provider=provider)
    assert seed.status == "MISS" and repeated.tier == "EXACT" and variant.status == "MISS"
    assert len(calls) == 2 and variant.diagnosis["verifier_calls"] == 0
    assert variant.diagnosis["local_verifier_attempts"] == 2
    negatives = [r for r in rows if not r["authored_expected_applicable"]]
    positives = [r for r in rows if r["authored_expected_applicable"]]
    return {"schema_version": "answer-applicability-dev-v1", "evidence_type": "authored_development_model_diagnostics",
            "model": scorer.identity, "policy": policy.identity, "threshold": policy.threshold,
            "minimum_yes_mass": policy.minimum_yes_mass,
            "rows": rows, "summary": {"cases": len(rows), "authored_intent_groups": len({r['intent_group'] for r in rows}),
                "positive_cases": len(positives), "negative_cases": len(negatives),
                "positive_recommendations": sum(r["assessment"]["recommended_reuse"] for r in positives),
                "negative_recommendations": sum(r["assessment"]["recommended_reuse"] for r in negatives)},
            "cache_integration": {"seed": asdict(seed), "exact_repeat": asdict(repeated),
                "paraphrase": asdict(variant), "fixture_provider_calls": len(calls)},
            "external_generation_calls": 0, "independent_human_reviews": 0,
            "serving_enabled": False, "final_quality_gate": "not_evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--scorer", choices=["flan", "qwen"], default="flan")
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-dev/answer-applicability.json"))
    args = parser.parse_args()
    if args.prepare:
        from huggingface_hub import snapshot_download
        selected = (ANSWER_MODEL, ANSWER_REVISION) if args.scorer == "flan" else (CHAT_MODEL, CHAT_REVISION)
        for name, revision in [selected, (ENCODER_MODEL, ENCODER_REVISION)]:
            snapshot_download(name, revision=revision, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.jinja", "README.md"],
                              ignore_patterns=["onnx/*", "openvino/*"], max_workers=4)
    def forbidden(*args, **kwargs):
        raise RuntimeError("network connection forbidden during development inference")
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    result = run(args.scorer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
