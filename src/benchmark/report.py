"""Candidate-bound safety and paired serving metrics; missing labels stay missing."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random

from src.benchmark.schema import binding, digest, finite, nonempty


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * q
    low = math.floor(at)
    return values[low] + (values[math.ceil(at)] - values[low]) * (at - low)


def interval(errors, n):
    if not n:
        return None
    from scipy.stats import beta
    return [0.0 if errors == 0 else float(beta.ppf(.025, errors, n-errors+1)),
            1.0 if errors == n else float(beta.ppf(.975, errors+1, n-errors))]


def cluster_interval(rows, value, *, seed=17, samples=1000):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(value(row))
    blocks = list(groups.values())
    if len(blocks) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        selected = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        means.append(sum(sum(b) for b in selected) / sum(len(b) for b in selected))
    return [quantile(means, .025), quantile(means, .975)]


def validate_trace(trace):
    if trace.get('schema_version') != 'answer-trace-v1':
        raise ValueError('expected answer-trace-v1')
    if trace.get('evaluation_mode') not in ('simulated_fixture', 'live'):
        raise ValueError('unknown evaluation mode')
    if trace.get('nature') not in ('synthetic_fixture', 'natural', 'constructed_natural_queries'):
        raise ValueError('unknown workload nature')
    if trace.get('split') not in ('dev', 'test', 'train'):
        raise ValueError('missing trace split')
    if not isinstance(trace.get('events'), list) or not trace['events']:
        raise ValueError('empty trace')
    seen = {}
    for event in trace['events']:
        for field in ('event_id', 'tenant_id', 'source', 'group_id'):
            nonempty(event.get(field), field)
        if event['event_id'] in seen:
            raise ValueError('duplicate event_id')
        if event.get('split') != trace['split']:
            raise ValueError('mixed split trace')
        if event.get('request_sha256') != digest(event.get('request')):
            raise ValueError('request digest mismatch')
        if type(event.get('stored')) is not bool:
            raise ValueError('admission state must be recorded')
        if event['stored'] and event.get('status') != 'MISS':
            raise ValueError('only a miss can admit a response')
        if event.get('status') not in ('HIT', 'MISS', 'BYPASS', 'ERROR'):
            raise ValueError('unknown cache status')
        for arm, hash_key in (('cache', 'response_sha256'), ('fresh', 'fresh_response_sha256')):
            data = event[arm]
            finite(data['latency_ms'], 'latency_ms')
            if type(data.get('provider_calls')) is not int or data['provider_calls'] < 0:
                raise ValueError('invalid provider call count')
            if not isinstance(data.get('usage'), list) or len(data['usage']) > data['provider_calls']:
                raise ValueError('usage count exceeds attempts')
            for usage in data['usage']:
                if usage is not None and (not isinstance(usage, dict) or any(
                    type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens'))):
                    raise ValueError('invalid usage')
            if data.get('error') is not None and data.get('response') is not None:
                raise ValueError('failed arm cannot supply an answer')
            if data.get('error') is None and not isinstance(data.get('response'), dict):
                raise ValueError('successful arm requires response')
            if event.get(hash_key) != digest(data.get('response')):
                raise ValueError('response digest mismatch')
        verification = event.get('verification')
        if verification is not None:
            if not isinstance(verification,dict) or type(verification.get('calls')) is not int or not 0<=verification['calls']<=1:
                raise ValueError('invalid verifier attempt count')
            if type(verification.get('approved')) is not bool:
                raise ValueError('invalid verifier decision')
            finite(verification.get('latency_ms'), 'verifier latency')
            usage = verification.get('usage')
            if not isinstance(usage,list) or len(usage)>verification['calls']:
                raise ValueError('invalid verifier usage')
            for item in usage:
                if item is not None and (not isinstance(item,dict) or any(type(item.get(k)) is not int or item[k]<0 for k in ('prompt_tokens','completion_tokens'))):
                    raise ValueError('invalid verifier token count')
            if verification['error'] is not None and verification['approved']:
                raise ValueError('failed verifier cannot approve')
        if event['fresh']['provider_calls'] != 1:
            raise ValueError('fresh baseline requires exactly one provider attempt')
        if event['status'] != 'HIT' and event['cache']['provider_calls'] != 1:
            raise ValueError('non-hit requires exactly one provider attempt')
        if (event['status'] == 'ERROR') != (event['cache']['error'] is not None):
            raise ValueError('cache error and status disagree')
        if event['status'] == 'HIT':
            source = seen.get(event.get('candidate_event_id'))
            if source is None or source['status'] not in ('MISS',) or source['cache']['error'] is not None or not source['stored']:
                raise ValueError('hit must reference an earlier generated candidate')
            if source['tenant_id'] != event['tenant_id']:
                raise ValueError('candidate crosses tenant boundary')
            from src.benchmark.semantic import scope
            if scope(source['request'],event['tenant_id'])!=scope(event['request'],event['tenant_id']):
                raise ValueError('candidate crosses generation or context scope')
            nonexact=source['request_sha256']!=event['request_sha256']
            matcher=trace.get('config',{}).get('matcher','exact')
            if nonexact and matcher not in ('similarity','verified'):
                raise ValueError('exact policy cannot claim a non-exact hit')
            if nonexact and matcher=='verified' and (verification is None or not verification['approved'] or verification['error'] is not None):
                raise ValueError('verified reuse requires an affirmative recorded verdict')
            if source['response_sha256'] != event['response_sha256']:
                raise ValueError('candidate answer does not match returned answer')
            if event['cache']['provider_calls'] != 0 or event['cache']['error'] is not None:
                raise ValueError('hit must have zero generation calls and no error')
        elif event.get('candidate_event_id') is not None:
            raise ValueError('non-hit cannot claim a reused candidate')
        seen[event['event_id']] = event
    return seen


def review_template(trace):
    validate_trace(trace)
    return {'schema_version': 'answer-reviews-v1', 'trace_sha256': digest(trace),
            'reviews': [{**binding(event), 'reviewer': '', 'method': 'human',
                         'safe_reuse': None, 'answer_correct': None, 'fresh_answer_correct': None,
                         'notes': ''} for event in trace['events']]}


def validate_reviews(trace, reviews):
    events = validate_trace(trace)
    if reviews is None:
        return {}
    if reviews.get('schema_version') != 'answer-reviews-v1' or reviews.get('trace_sha256') != digest(trace):
        raise ValueError('reviews belong to a different trace')
    labels = {}
    for row in reviews['reviews']:
        identity = row.get('event_id')
        if identity in labels or identity not in events:
            raise ValueError('duplicate or unknown review event')
        event = events[identity]
        if any(row.get(k) != v for k, v in binding(event).items()):
            raise ValueError('review is not bound to the actual answer/candidate')
        fields = ('safe_reuse', 'answer_correct', 'fresh_answer_correct')
        if any(row.get(k) is not None and type(row[k]) is not bool for k in fields):
            raise ValueError('review labels must be boolean or null')
        if row.get('method') not in ('human', 'independent_judge', 'fixture'):
            raise ValueError('unknown review method')
        if any(row.get(k) is not None for k in fields):
            nonempty(row.get('reviewer'), 'reviewer')
        if row.get('safe_reuse') is not None and event['status'] != 'HIT':
            raise ValueError('safe_reuse only labels actual hits')
        for field, arm in (('answer_correct', 'cache'), ('fresh_answer_correct', 'fresh')):
            if row.get(field) is not None and event[arm]['error'] is not None:
                raise ValueError('failed arm cannot have an answer quality label')
        if row.get('method') == 'fixture' and trace['evaluation_mode'] != 'simulated_fixture':
            raise ValueError('fixture labels cannot certify live output')
        labels[identity] = row
    return labels


def summarize(trace, reviews=None):
    seen = validate_trace(trace)
    labels = validate_reviews(trace, reviews)
    events = trace['events']
    hits = [e for e in events if e['status'] == 'HIT']
    nonexact = [e for e in hits if e['request_sha256'] != seen[e['candidate_event_id']]['request_sha256']]
    def safety(rows):
        labeled = [e for e in rows if labels.get(e['event_id'], {}).get('safe_reuse') is not None]
        errors = sum(not labels[e['event_id']]['safe_reuse'] for e in labeled)
        n, total = len(labeled), len(rows)
        return {'accepted': total, 'reviewed': n, 'unreviewed': total-n, 'incorrect': errors,
                'reviewed_incorrect_rate': errors/n if n else None,
                'reviewed_clopper_pearson_95': interval(errors, n),
                'reviewed_group_bootstrap_95': cluster_interval(labeled, lambda e: int(not labels[e['event_id']]['safe_reuse'])),
                'all_accepted_error_identification_bounds': [errors/total, (errors+total-n)/total] if total else None,
                'review_methods': dict(Counter(labels[e['event_id']]['method'] for e in labeled))}
    def arm_stats(arm):
        rows = [e[arm] for e in events]
        successful = [r for r in rows if r['error'] is None]
        attempts = sum(r['provider_calls'] for r in rows)
        usage_rows = [u for r in rows for u in r['usage'] if u is not None]
        missing = attempts-len(usage_rows)
        return {'requests': len(rows), 'successful_answers': len(successful), 'errors': len(rows)-len(successful),
                'provider_attempts': attempts, 'attempts_without_usage': missing,
                'observed_prompt_tokens': sum(u['prompt_tokens'] for u in usage_rows),
                'observed_completion_tokens': sum(u['completion_tokens'] for u in usage_rows),
                'successful_answer_latency_ms': {k: quantile([r['latency_ms'] for r in successful], q)
                    for k,q in (('p50',.5),('p95',.95),('p99',.99))},
                'all_attempt_latency_ms_mean': sum(r['latency_ms'] for r in rows)/len(rows)}
    cache, fresh = arm_stats('cache'), arm_stats('fresh')
    pairs = [e for e in events if e['cache']['error'] is None and e['fresh']['error'] is None]
    latency_delta = lambda e: e['cache']['latency_ms']-e['fresh']['latency_ms']
    quality = [e for e in pairs if all(labels.get(e['event_id'], {}).get(k) is not None
                for k in ('answer_correct', 'fresh_answer_correct'))]
    quality_delta = lambda e: int(labels[e['event_id']]['answer_correct'])-int(labels[e['event_id']]['fresh_answer_correct'])
    verification = [e['verification'] for e in events if e.get('verification') is not None]
    verifier_calls = sum(v['calls'] for v in verification)
    verifier_usage = [u for v in verification for u in v['usage'] if u is not None]
    verifier_missing = verifier_calls-len(verifier_usage)
    verifier_tokens = {k:sum(u[k] for u in verifier_usage) for k in ('prompt_tokens','completion_tokens')}
    token_delta = None
    if not cache['attempts_without_usage'] and not fresh['attempts_without_usage'] and not verifier_missing:
        token_delta = {k: fresh['observed_'+k]-cache['observed_'+k]-verifier_tokens[k] for k in ('prompt_tokens', 'completion_tokens')}
    latency_valid=trace.get('latency_valid',True)
    if not latency_valid:
        for stats in (cache,fresh):
            stats['successful_answer_latency_ms']={k:None for k in ('p50','p95','p99')}
            stats['all_attempt_latency_ms_mean']=None
    return {'schema_version': 'answer-report-v1', 'latency_valid':latency_valid, 'trace_sha256': digest(trace),
            'evaluation_mode': trace['evaluation_mode'], 'nature': trace['nature'], 'split': trace['split'],
            'timing_scope': trace['timing_scope'], 'requests': len(events),
            'status_counts': dict(Counter(e['status'] for e in events)),
            'overall_reuse': safety(hits), 'nonexact_reuse': safety(nonexact),
            'by_source': {source: safety([e for e in hits if e['source']==source]) for source in sorted({e['source'] for e in events})},
            'cache_arm': cache, 'fresh_arm': fresh,
            'verifier_work': {'provider_attempts':verifier_calls,'attempts_without_usage':verifier_missing,
                'observed_tokens':verifier_tokens,'errors':sum(v['error'] is not None for v in verification)},
            'total_cache_provider_attempts':cache['provider_attempts']+verifier_calls,
            'provider_attempts_avoided': fresh['provider_attempts']-cache['provider_attempts']-verifier_calls,
            'token_difference_fresh_minus_cache': token_delta,
            'paired_latency': {'complete_pairs': len(pairs), 'mean_cache_minus_fresh_ms': sum(map(latency_delta,pairs))/len(pairs) if pairs and latency_valid else None,
                              'group_bootstrap_95': cluster_interval(pairs,latency_delta) if latency_valid else None},
            'paired_answer_quality': {'reviewed_pairs': len(quality), 'unreviewed_or_incomplete_pairs': len(events)-len(quality),
                'mean_cache_minus_fresh_accuracy': sum(map(quality_delta,quality))/len(quality) if quality else None,
                'group_bootstrap_95': cluster_interval(quality,quality_delta)},
            'safety_certified': False,
            'limitations': ['Development evidence does not certify a sealed safety gate.',
                'Binomial intervals assume independent trials; repeated groups violate that assumption.',
                'Group bootstrap is exploratory; few or identical groups can yield misleadingly narrow intervals.',
                'Reviewed-only estimates may be biased; unreviewed hits are not counted as safe.',
                'No recall is reported without an independently labeled opportunity denominator.',
                ('Fixture calls and tokens are simulated; monetary savings are not inferred.' if trace['evaluation_mode']=='simulated_fixture' else 'Token counts are reported by the live provider; monetary savings are not inferred.'),
                'Fresh and cache arms are serial and randomized within each pair, not simultaneous traffic.']}
