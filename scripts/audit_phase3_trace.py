"""Reconcile every spent development trace without rerunning or relabeling it."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.answer_evidence import AnswerReviews, review_binding, percentile
from src.policy.semantic_runtime import digest


def audit(raw, workload, reviews=()):
    review_index = AnswerReviews(reviews)
    events = {e['event_id']: e for e in workload}
    # Reconstruct only deterministic fixture outputs and require full-response hash equality.
    responses = {}
    for event in workload:
        q = event['request']['messages'][-1]['content']
        ref = event['reference_answer']
        system = event['request']['messages'][0]['content']
        response = {'id': 'ref-'+hashlib.sha256(q.encode()).hexdigest()[:12],
                    'model': event['request']['model'],
                    'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                        'role': 'assistant', 'content': 'Based on Python documentation:\n'+ref}}],
                    'usage': {'prompt_tokens': len(system.split())+len(q.split()),
                              'completion_tokens': len(ref.split()),
                              'total_tokens': len(system.split())+len(q.split())+len(ref.split())}}
        responses[digest(response)] = response
    summaries, packet = {}, []
    for arm, rows in raw['results'].items():
        if [r['event_id'] for r in rows] != [e['event_id'] for e in workload]:
            raise ValueError('unaligned ordered workload')
        semantic = [r for r in rows if r['is_hit'] and r['tier']=='SEMANTIC']
        recovered = 0
        for row in semantic:
            event = events[row['event_id']]
            binding = row['diagnosis'].get('binding', {})
            if binding.get('incoming_request_sha256') != digest(event['request']):
                raise ValueError('incoming request binding mismatch')
            response = responses.get(binding.get('candidate_response_sha256'))
            recovered += response is not None
            packet.append({'arm':arm, 'event_id': row['event_id'], 'faq_id': row['faq_id'],
                           'request': event['request'], 'returned_response': response,
                           'recovery': 'deterministic_fixture_full_response_hash_match' if response else 'unavailable',
                           'binding': review_binding(event['request'], response, event['tenant'], event['revision']) if response else None,
                           'candidate_binding': binding, 'label': 'unknown', 'reviewer': None,
                           'rationale': 'No independent actual-answer review supplied'})
        arm_packet = [r for r in packet if r['arm']==arm]
        for record in arm_packet:
            review = review_index.lookup(record['binding']) if record['binding'] else {'label':'unknown','reviewer':None,'rationale':'answer unavailable'}
            record.update({k:review[k] for k in ('label','reviewer','rationale')})
        counts = {label:sum(r['label']==label for r in arm_packet) for label in ('sufficient','insufficient','unknown')}
        summaries[arm] = {'requests':len(rows), 'exact_hits':sum(r['is_hit'] and r['tier']=='EXACT' for r in rows),
                          'semantic_hits':len(semantic), 'semantic_answers_hash_recovered':recovered,
                          'semantic_answers_unknown':counts['unknown'], 'answer_review_counts':counts,
                          'semantic_precision':counts['sufficient']/len(semantic) if semantic and not counts['unknown'] else None,
                          'fixture_provider_calls_inferred':sum(not r['is_hit'] for r in rows),
                          'inference_assumption':'one fixture invocation per non-hit; not directly recorded',
                          'billable_tokens':None, 'monetary_savings':None,
                          'historical_fixture_latency_p50_ms':percentile([r['elapsed_ms'] for r in rows],.5),
                          'historical_fixture_latency_p95_ms':percentile([r['elapsed_ms'] for r in rows],.95)}
    return {'evidence_kind':'historical_simulated_provider_trace_audit', 'summary':summaries, 'review_packet':packet}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True); p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--reviews',type=Path); args=p.parse_args()
    result=audit(json.loads(args.raw.read_text()),json.loads(args.workload.read_text()),json.loads(args.reviews.read_text()) if args.reviews else ())
    result['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.raw,args.workload)}
    if args.reviews: result['input_sha256'][str(args.reviews)]=hashlib.sha256(args.reviews.read_bytes()).hexdigest()
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
