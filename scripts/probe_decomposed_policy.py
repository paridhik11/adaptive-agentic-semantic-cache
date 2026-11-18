"""Declared development replay plus broader authored cases; no final quality test."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_answer_applicability import CASES, CONTEXT, LIST_QUERY, LIST_ANSWER
from src.policy.decomposed_applicability import (DecomposedDiagnosticPolicy, EvidenceIndex,
    LocalEntailmentScorer, SourceSpan, NLI_MODEL, NLI_REVISION)
from src.policy.encoders import DenseEncoder
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION
from src.policy.response_cache import ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, PinnedContextPolicy, ReuseDecision, digest

# Each tuple: group, trusted authored source, seed question, positive paraphrase,
# changed-requirement negative, complete seed answer, wrong-answer negative.
# These are builder-authored expectations, not independent final labels.
TRANSFER = [
    ("length", "len(items) returns the number of elements in a Python list.",
     "How do I find the length of a Python list?", "How can I count the elements in a Python list?",
     "How do I find the largest element of a Python list?", "Use len(items) to get the number of list elements.",
     "Use max(items) to get the number of list elements."),
    ("integer-string", "str(value) converts an integer value to its string representation.",
     "How do I convert an integer to a string?", "How can I make a string from an integer?",
     "How do I convert a string to an integer?", "Use str(value) to obtain the string representation of the integer.",
     "Use int(value) to convert the integer to a string."),
    ("string-integer", "int(text) parses a decimal integer from a string containing a valid decimal integer.",
     "How do I convert a decimal integer string to an integer?", "How can I parse an integer from a decimal string?",
     "How do I convert a decimal integer string to an integer without using int?", "Use int(text) to parse the decimal integer string.",
     "Use float(text) to parse the string as an integer."),
    ("dictionary-default", "mapping.get(key, default) returns the key's value if the key exists, and default otherwise.",
     "How do I get a dictionary value with a default for missing keys?", "How can I retrieve a value or a fallback when the dictionary key is absent?",
     "How do I remove a dictionary key and return its value?", "Use mapping.get(key, default) to return the value or the default if the key is absent.",
     "Use mapping.pop(key) to read the value without removing it."),
    ("append", "items.append(value) adds value as a single element at the end of the list, even when value is itself a list.",
     "How do I add a list as one element at the end of another list?", "How can I append an entire list as a single nested element?",
     "How do I add each element of one list to another list?", "Use items.append(value) to add the entire value as one element at the end.",
     "Use items.extend(value) to add the whole list as one element."),
    ("descending", "sorted(values, reverse=True) returns a new list sorted in descending order.",
     "How do I sort values in descending order into a new list?", "How can I get a new list ordered from largest to smallest?",
     "How do I sort values in ascending order into a new list?", "Use sorted(values, reverse=True) for a new list sorted in descending order.",
     "Use sorted(values) for a new list sorted in descending order."),
    ("basename", "os.path.basename(path) returns the final component of a filesystem path.",
     "How do I get the final component of a filesystem path?", "How can I extract the last part of a file path?",
     "How do I get the parent directory of a filesystem path?", "Use os.path.basename(path) to get the final path component.",
     "Use os.path.dirname(path) to get the final component."),
    ("json-parse", "json.loads(text) parses a JSON document in a string into the corresponding Python object.",
     "How do I convert a JSON string into a Python object?", "How can I parse JSON text into a Python object?",
     "How do I convert a Python object into a JSON string?", "Use json.loads(text) to parse a JSON string into a Python object.",
     "Use json.dumps(text) to parse a JSON string into a Python object."),
    ("deduplication", "set(items) creates a set of distinct elements from an iterable; it does not preserve the original element order.",
     "How do I remove duplicates from an iterable using a set?", "How can I get the distinct elements of an iterable as a set?",
     "How do I remove duplicates from an iterable while preserving its original order?", "Use set(items) to obtain distinct elements. The set does not preserve the input order.",
     "Use set(items) to obtain distinct elements while preserving the input order."),
    ("sum", "sum(values) returns the total of the numeric values in an iterable.",
     "How do I find the total of numbers in an iterable?", "How can I add together all the numbers in an iterable?",
     "How do I find the average of numbers in an iterable?", "Use sum(values) to get the total of the numbers.",
     "Use len(values) to get the total of the numbers."),
]


def request(context, query):
    return {"model": "authored-reference-fixture", "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "system", "content": context}, {"role": "user", "content": query}]}


def response(text):
    return {"model": "authored-reference-fixture", "choices": [{"finish_reason": "stop",
            "message": {"role": "assistant", "content": text}}]}


def source_spans(context):
    # Sentence boundaries preserve decimal numbers and API names. These spans
    # are trusted source fragments; selection remains automatic and local.
    return [SourceSpan(m.start(), m.end()) for m in re.finditer(r".+?(?:\.(?=\s|$)|$)", context)
            if m.group().strip()]


def cases():
    rows = [{"cohort": "legacy_development_replay", "id": i, "group": "legacy-"+g,
             "context": CONTEXT, "original": original, "incoming": incoming,
             "answer": answer, "expected": expected}
            for i,g,original,incoming,answer,expected in CASES]
    for group, context, original, positive, negative, answer, wrong in TRANSFER:
        for suffix,incoming,text,expected in [("positive",positive,answer,True),
                ("changed-request",negative,answer,False), ("wrong-answer",positive,wrong,False)]:
            rows.append({"cohort": "broader_authored_development", "id": group+"-"+suffix,
                "group": group, "context": context, "original": original,
                "incoming": incoming, "answer": text, "expected": expected})
    return rows


def run(*, collect_all_scores=False):
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    scorer = LocalEntailmentScorer()
    policies, rows = {}, []
    for case in cases():
        context = case["context"]
        if context not in policies:
            policies[context] = DecomposedDiagnosticPolicy(evidence_index=EvidenceIndex(
                context=context, spans=source_spans(context), encoder=encoder), scorer=scorer)
        assessment = policies[context].assess(request(context,case["incoming"]),
            request(context,case["original"]), response(case["answer"]),
            collect_all_scores=collect_all_scores)
        rows.append({"id":case["id"],"group":case["group"],"cohort":case["cohort"],
            "authored_expected_applicable":case["expected"],"case_sha256":digest(case),
            "assessment":asdict(assessment)})
        print(case["id"],assessment.reason,flush=True)
    summaries = {}
    for cohort in ["legacy_development_replay", "broader_authored_development"]:
        subset = [r for r in rows if r["cohort"] == cohort]
        summaries[cohort] = {"cases":len(subset),"groups":len({r["group"] for r in subset}),
            "positives":sum(r["authored_expected_applicable"] for r in subset),
            "negatives":sum(not r["authored_expected_applicable"] for r in subset),
            "positive_recommendations":sum(r["authored_expected_applicable"] and r["assessment"]["recommended_reuse"] for r in subset),
            "negative_recommendations":sum(not r["authored_expected_applicable"] and r["assessment"]["recommended_reuse"] for r in subset)}

    # Isolated fixture-only cache experiment. This adapter is intentionally NOT
    # exported from the policy or exposed by the application. Sockets are blocked.
    class OfflineExperimentApprover:
        def __init__(self, policy):
            self.policy = policy
            self.identity = "offline-fixture-only:"+policy.identity
        def __call__(self, incoming, candidate, answer, *, binding, provider):
            a = self.policy.assess(incoming,candidate,answer)
            return ReuseDecision(a.recommended_reuse,"offline-fixture-experiment:"+a.reason,
                local_verifier_attempts=a.work["embedding_attempts"]+a.work["nli_attempts"],
                local_verifier_ms=a.work["elapsed_ms"],local_policy_recommendation=a.recommended_reuse)
    experiments=[]
    for case in cases():
        policy=policies[case['context']]
        calls=[]
        def fixture_provider(body):
            calls.append(digest(body))
            return response(case['answer'] if len(calls)==1 else
                            "Fresh scripted fixture response; its answer quality is not evaluated.")
        cache=GuardedSemanticCache(ScopedResponseCache(eligibility=PinnedContextPolicy(case['context'])),
            scope=KnowledgeScope("authored-dev","v1","public",policy.identity),encoder=encoder,
            mode="guarded",approver=OfflineExperimentApprover(policy),threshold=.80,margin=.05)
        seed=cache.process(request(case['context'],case['original']),tenant_id="dev",provider=fixture_provider)
        result=cache.process(request(case['context'],case['incoming']),tenant_id="dev",provider=fixture_provider)
        experiments.append({'id':case['id'],'authored_expected_applicable':case['expected'],
            'seed_status':seed.status,'result_status':result.status,'tier':result.tier,
            'diagnosis':result.diagnosis,'fixture_provider_calls':len(calls),
            'incoming_provider_calls':result.provider_calls,
            'returned_seed_response':result.response==seed.response})
        print('cache',case['id'],result.tier,flush=True)
    return {"schema_version":"decomposed-development-v1","evidence_type":"builder_authored_model_assisted_development",
        "collect_all_scores":collect_all_scores,
        "cases_sha256":digest(cases()),"models":{"encoder":encoder.identity,"nli":scorer.identity},
        "thresholds":{"nli":.95,"evidence_cosine":.50,"evidence_margin":.03},
        "rows":rows,"summaries":summaries,
        "initial_source_embedding_calls":sum(p.index.initial_embedding_calls for p in policies.values()),
        "offline_cache_experiments":experiments,
        "cache_summary":{
            "semantic_hits":sum(r['tier']=='SEMANTIC' for r in experiments),
            "unsafe_semantic_hits":sum(r['tier']=='SEMANTIC' and not r['authored_expected_applicable'] for r in experiments),
            "exact_hits":sum(r['tier']=='EXACT' for r in experiments),
            "known_bad_seed_exact_reuses":sum(r['tier']=='EXACT' and not r['authored_expected_applicable'] for r in experiments),
            "fixture_provider_calls":sum(r['fixture_provider_calls'] for r in experiments)},
        "production_serving_enabled":False,"external_generation_calls":0,"independent_human_reviews":0,
        "final_quality_gate":"not_evaluated"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--collect-all-scores",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("runs/semantic-dev/decomposed-v1.json"))
    args=parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve previous evidence and choose a fresh path")
    if args.prepare:
        from huggingface_hub import snapshot_download
        for model,revision in [(NLI_MODEL,NLI_REVISION),(ENCODER_MODEL,ENCODER_REVISION)]:
            snapshot_download(model,revision=revision,allow_patterns=["*.json","*.txt","*.safetensors","*.model","README.md"],
                              ignore_patterns=["onnx/*","openvino/*"],max_workers=4)
    def forbidden(*args,**kwargs):
        raise RuntimeError("network connections forbidden during local experiment")
    socket.socket.connect=forbidden
    socket.create_connection=forbidden
    result=run(collect_all_scores=args.collect_all_scores)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        handle.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summaries"],indent=2))


if __name__ == "__main__":
    main()
