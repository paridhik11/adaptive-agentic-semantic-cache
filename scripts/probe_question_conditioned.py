"""Day 3 authored development replay; no human labels or production approval."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_decomposed_policy import cases as previous_cases, request, response
from src.policy.decomposed_applicability import LocalEntailmentScorer, NLI_MODEL, NLI_REVISION
from src.policy.encoders import DenseEncoder
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION
from src.policy.question_conditioned import (LocalSpanReader, QuestionConditionedPolicy,
    QA_MODEL, QA_REVISION, QA_MARGINS, SUPPORT_THRESHOLDS, recommend)
from src.policy.response_cache import ScopedResponseCache
from src.policy.semantic_runtime import GuardedSemanticCache, KnowledgeScope, PinnedContextPolicy, ReuseDecision, digest

NEW_GROUPS = [
 ('lowercase','text.lower() returns a lowercase copy of the string.',
  'How do I change a string to lowercase?', 'How can I obtain a lowercase version of a string?',
  'How do I change a string to uppercase?', 'Use text.lower() to get a lowercase copy.',
  'Use text.upper() to get a lowercase copy.'),
 ('uppercase','text.upper() returns an uppercase copy of the string.',
  'How do I change a string to uppercase?', 'How can I get an uppercase version of a string?',
  'How do I change a string to lowercase?', 'Use text.upper() to get an uppercase copy.',
  'Use text.lower() to get an uppercase copy.'),
 ('strip','text.strip() returns a copy with leading and trailing whitespace removed.',
  'How do I remove whitespace at both ends of a string?', 'How can I trim leading and trailing whitespace?',
  'How do I remove all spaces inside a string?', 'Use text.strip() to remove leading and trailing whitespace.',
  'Use text.lstrip() to remove whitespace from both ends.'),
 ('split','text.split() returns a list of words separated by runs of whitespace.',
  'How do I split a string into words on whitespace?', 'How can I get a list of whitespace-separated words?',
  'How do I join a list of words into a string?', 'Use text.split() for a list of whitespace-separated words.',
  'Use text.strip() to split the string into a list of words.'),
 ('join','separator.join(parts) concatenates strings in parts with separator between them.',
  'How do I combine strings with a separator between them?', 'How can I join string parts with a delimiter?',
  'How do I split a string at each separator?', 'Use separator.join(parts) to combine the strings with separators.',
  'Use parts.append(separator) to combine strings into one string.'),
 ('replace','text.replace(old, new) returns a copy with all occurrences of old replaced by new.',
  'How do I replace all occurrences of a substring?', 'How can I substitute a new substring for every occurrence of an old one?',
  'How do I replace only the first occurrence of a substring?', 'Use text.replace(old, new) to replace all occurrences.',
  'Use text.find(old) to replace all occurrences with new.'),
 ('absolute','abs(value) returns the absolute value of a number.',
  'How do I find the absolute value of a number?', 'How can I obtain a number\'s magnitude without its sign?',
  'How do I change the sign of a number?', 'Use abs(value) to get the absolute value.',
  'Use round(value) to get the absolute value.'),
 ('rounding','round(value) rounds a number to the nearest integer, with ties going to the even choice.',
  'How do I round a number to the nearest integer?', 'How can I get the nearest integer using rounding?',
  'How do I always round a number upward?', 'Use round(value) to round to the nearest integer; ties go to even.',
  'Use abs(value) to round to the nearest integer.'),
 ('reverse-copy','list(reversed(items)) creates a new list in the reverse of the original order.',
  'How do I get a reversed copy of a list?', 'How can I make a new list with the elements in reverse order?',
  'How do I reverse a list in place?', 'Use list(reversed(items)) to create a new reversed list.',
  'Use sorted(items) to create a list in reverse input order.'),
 ('all-predicate','all(values) returns True if every element is truthy, and True for an empty iterable.',
  'How do I check whether every value is truthy?', 'How can I test that all elements evaluate to true?',
  'How do I check whether at least one value is truthy?', 'Use all(values) to test whether every element is truthy; an empty iterable returns True.',
  'Use any(values) to test whether every element is truthy.'),
]


def cases():
    rows = previous_cases()
    for group, context, original, positive, negative, answer, wrong in NEW_GROUPS:
        for suffix, incoming, text, expected in [('positive',positive,answer,True),
                ('changed-request',negative,answer,False),('wrong-answer',positive,wrong,False)]:
            rows.append({'cohort':'additional_authored_development','id':group+'-'+suffix,
                'group':group,'context':context,'original':original,'incoming':incoming,
                'answer':text,'expected':expected})
    return rows


def counts(rows, qa_margin=0., support=.95):
    positives = sum(r['expected'] for r in rows)
    accepted = [r for r in rows if recommend(r['assessment']['features'], qa_margin, support)]
    correct = sum(r['expected'] for r in accepted)
    return {'cases':len(rows),'intent_groups':len({r['group'] for r in rows}),
        'positives':positives,'negatives':len(rows)-positives,'accepted':len(accepted),
        'correct_accepts':correct,'incorrect_accepts':len(accepted)-correct,
        'observed_authored_precision':correct/len(accepted) if accepted else None,
        'positive_coverage':correct/positives if positives else None,
        'population_confidence_interval':None,
        'interval_reason':'purposive builder-authored correlated development fixtures, not an independent probability sample'}


def run(*, span_mode='strict', question_form='raw', intent_gate=False):
    import torch
    torch.set_num_threads(2)
    started = time.perf_counter()
    reader, scorer = LocalSpanReader(), LocalEntailmentScorer()
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    setup_ms = (time.perf_counter()-started)*1000
    policies, rows, experiments = {}, [], []
    for case in cases():
        policy = policies.setdefault(case['context'], QuestionConditionedPolicy(
            context=case['context'], reader=reader, scorer=scorer, span_mode=span_mode,
            question_form=question_form,intent_gate=intent_gate))
        a = policy.assess(request(case['context'],case['incoming']),
                          request(case['context'],case['original']),response(case['answer']))
        rows.append({**case,'assessment':asdict(a)})
        print(case['id'],a.reason,flush=True)

    class FixtureApprover:
        def __init__(self, policy):
            self.policy, self.identity = policy, 'isolated-fixture:'+policy.identity
        def __call__(self, incoming, candidate, answer, *, binding, provider):
            a = self.policy.assess(incoming,candidate,answer)
            return ReuseDecision(a.recommended_reuse,'fixture:'+a.reason,
                local_verifier_attempts=a.work['qa_attempts']+a.work['nli_attempts'],
                local_verifier_ms=a.work['elapsed_ms'],local_policy_recommendation=a.recommended_reuse)
    for case in cases():
        calls = []
        def provider(body):
            calls.append(digest(body))
            return response(case['answer'] if len(calls)==1 else 'Fresh scripted fixture response; answer quality is not evaluated.')
        policy = policies[case['context']]
        cache = GuardedSemanticCache(ScopedResponseCache(eligibility=PinnedContextPolicy(case['context'])),
            scope=KnowledgeScope('authored-day3','v1','public',policy.identity),encoder=encoder,
            mode='guarded',approver=FixtureApprover(policy),threshold=.80,margin=.05)
        seed = cache.process(request(case['context'],case['original']),tenant_id='dev',provider=provider)
        result = cache.process(request(case['context'],case['incoming']),tenant_id='dev',provider=provider)
        experiments.append({'id':case['id'],'expected':case['expected'],'seed_status':seed.status,
            'status':result.status,'tier':result.tier,'diagnosis':result.diagnosis,
            'provider_calls_including_seed':len(calls),'incoming_provider_calls':result.provider_calls})
    cohorts = sorted({r['cohort'] for r in rows})
    return {'schema':'day3-question-conditioned-v1','evidence_type':'builder_authored_model_assisted_development',
        'span_mode':span_mode,'question_form':question_form,'intent_gate':intent_gate,
        'case_sha256':digest(cases()),'models':{'reader':reader.identity,'scorer':scorer.identity,'encoder':encoder.identity},
        'setup_ms':setup_ms,'rows':rows,'summary':counts(rows),
        'cohorts':{c:counts([r for r in rows if r['cohort']==c]) for c in cohorts},
        'curve':[{'qa_margin':q,'support_threshold':s,**counts(rows,q,s)} for q in QA_MARGINS for s in SUPPORT_THRESHOLDS],
        'first_failure_counts':dict(Counter(r['assessment']['reason'] for r in rows)),
        'cache_experiments':experiments,
        'cache_summary':{'semantic_hits':sum(r['tier']=='SEMANTIC' for r in experiments),
            'unsafe_semantic_hits':sum(r['tier']=='SEMANTIC' and not r['expected'] for r in experiments),
            'exact_hits':sum(r['tier']=='EXACT' for r in experiments),
            'known_bad_seed_exact_hits':sum(r['tier']=='EXACT' and not r['expected'] for r in experiments),
            'fixture_provider_calls':sum(r['provider_calls_including_seed'] for r in experiments)},
        'production_serving_enabled':False,'human_reviews':0,'external_generation_calls':0,
        'socket_connections_blocked':True,'final_quality_gate':'not_evaluated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--span-mode',choices=['strict','api-anchor'],default='strict')
    parser.add_argument('--question-form',choices=['raw','function'],default='raw')
    parser.add_argument('--require-intent-entailment',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence and choose a fresh path')
    if args.prepare:
        from huggingface_hub import snapshot_download
        for model, revision in [(QA_MODEL,QA_REVISION),(NLI_MODEL,NLI_REVISION),(ENCODER_MODEL,ENCODER_REVISION)]:
            snapshot_download(model,revision=revision,allow_patterns=['*.json','*.txt','*.safetensors','*.model'],
                ignore_patterns=['onnx/*','openvino/*'],max_workers=4)
    def forbidden(*args, **kwargs):
        raise RuntimeError('network forbidden in local development experiment')
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    result = run(span_mode=args.span_mode,question_form=args.question_form,
                 intent_gate=args.require_intent_entailment)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':result['summary'],'cache':result['cache_summary']},indent=2))


if __name__ == '__main__':
    main()
