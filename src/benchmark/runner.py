"""Paired serial serving experiment. No automatic external API execution."""
from __future__ import annotations
import copy
import random
import time

from src.api.service import ConservativeRulesPolicy, validate_request
from src.policy.response_cache import ScopedResponseCache
from src.benchmark.schema import digest, validate_workload
from src.benchmark.control import RunPaused
from src.benchmark.diagnostics import response_diagnostics


def usage(response):
    value = response.get('usage', {}) if isinstance(response, dict) else {}
    fields = ('prompt_tokens', 'completion_tokens')
    if any(type(value.get(k)) is not int or value[k] < 0 for k in fields):
        return None
    return {k: value[k] for k in fields}


def run_workload(workload, provider, *, split='dev', seed=17, ttl_seconds=3600,
                 max_entries=1024, max_bytes=67108864, policy="exact", encoder=None,
                 verifier=None, threshold=.85, margin=.05, observer=None):
    validate_workload(workload)
    rows = [r for r in workload['requests'] if r['split'] == split]
    if not rows:
        raise ValueError('selected split is empty')
    if split != 'dev':
        raise ValueError('recovery runner currently permits development only; no unregistered test runs')
    models = frozenset(row['request'].get('model') for row in rows)
    if any(not isinstance(m, str) or not m.strip() for m in models):
        raise ValueError('all requests require a nonempty model')
    # Validate the entire workload before making any provider calls.
    for row in rows:
        validate_request(row['request'], models, 4096)
    cache = ScopedResponseCache(eligibility=ConservativeRulesPolicy(), ttl_seconds=ttl_seconds,
                                max_entries=max_entries, max_bytes=max_bytes)
    if policy not in ('exact','similarity','verified'):
        raise ValueError('unknown benchmark policy')
    if policy != 'exact':
        from src.benchmark.semantic import ExperimentalSemanticCache, LexicalEncoder
        encoder = encoder or LexicalEncoder()
        cache = ExperimentalSemanticCache(eligibility=ConservativeRulesPolicy(), encoder=encoder,
            mode=policy, verifier=verifier, threshold=threshold, margin=margin,
            ttl_seconds=ttl_seconds,max_entries=max_entries,max_bytes=max_bytes)
    rng, origins, events = random.Random(seed), {}, []
    provider_mode = getattr(provider, 'mode', 'unspecified')
    if provider_mode not in ('live', 'simulated_fixture'):
        raise ValueError('provider must declare live or simulated_fixture mode')
    for row in rows:
        body, tenant = copy.deepcopy(row['request']), row['tenant_id']
        key = digest({'tenant_id': tenant, 'request': body})
        arms = ['cache', 'fresh']
        rng.shuffle(arms)
        guard=cache if policy=='exact' else cache.guard
        eligibility_reason=guard._eligibility_reason(body)
        record = {**copy.deepcopy(row), 'request_sha256': digest(body), 'order': arms,
            'eligibility':{'cacheable':eligibility_reason is None,'bypass_reason':eligibility_reason}}
        for arm in arms:
            if observer is not None:
                try:observer({'kind':'arm_started','event_id':row['event_id'],'arm':arm,
                    'question':body['messages'][-1]['content']})
                except Exception:pass
            started = time.perf_counter()
            calls, consumed, response, result, error = 0, [], None, None, None
            diagnostics=None
            def measured(request):
                nonlocal calls
                calls += 1
                value = provider(copy.deepcopy(request))
                consumed.append(usage(value))
                return value
            try:
                if arm == 'cache':
                    result = cache.process(body, tenant_id=tenant, provider=measured,
                        **({"event_id":row["event_id"]} if policy != "exact" else {}))
                    response = result.response
                else:
                    response = measured(body)
                diagnostics=response_diagnostics(response,body)
                if not isinstance(response, dict):
                    raise TypeError('invalid response')
                digest(response)  # fail invalid JSON before recording success
                if not ScopedResponseCache._admissible(response, body):
                    raise ValueError('incomplete answer')
            except RunPaused:
                raise
            except Exception:
                # No provider bodies/credentials in errors. Attempts still count.
                error = diagnostics['rejection_reason'] if diagnostics is not None and diagnostics['rejection_reason'] else 'provider_or_processing_error'
                response = None
            record[arm] = {'latency_ms': (time.perf_counter() - started) * 1000,
                           'provider_calls': calls, 'usage': consumed, 'error': error,
                           'response': response, 'response_diagnostics':diagnostics}
            if observer is not None:
                try:observer({'kind':'arm_finished','event_id':row['event_id'],'arm':arm,
                    'status':(result.status if result else 'GENERATED') if error is None else 'ERROR',
                    'provider_calls':calls,'latency_ms':record[arm]['latency_ms'],'error':error,
                    'answer':response['choices'][0]['message']['content'][:2000] if response is not None else None})
                except Exception:pass
            if arm == 'cache':
                record['decision_diagnostics']=getattr(result,'diagnosis',None)
                record['stored'] = result.stored if result is not None and error is None else False
                record['status'] = result.status if result is not None and error is None else 'ERROR'
                record['reason'] = result.reason if result is not None else error
                record['candidate_event_id'] = (getattr(result,'candidate_event_id',None) if policy != 'exact' else origins.get(key)) if record['status'] == 'HIT' else None
                record['verification'] = getattr(result,'verification',getattr(cache,'last_verification',None))
                record['similarity'] = getattr(result,'similarity',None)
                if result is not None and result.stored:
                    origins[key] = row['event_id']
        record['response_sha256'] = digest(record['cache']['response'])
        record['fresh_response_sha256'] = digest(record['fresh']['response'])
        events.append(record)
    return {'schema_version': 'answer-trace-v1', 'evaluation_mode': provider_mode,
            'workload_sha256': digest(workload), 'workload_name': workload['name'],
            'nature': workload['nature'], 'provenance': workload['provenance'], 'license': workload['license'],
            'split': split, 'config': {'seed': seed, 'policy': 'conservative-rules', 'matcher': policy,
                'encoder':getattr(encoder,'identity',None), 'verifier':getattr(verifier,'identity',None),
                'threshold':threshold if policy!='exact' else None,'margin':margin if policy!='exact' else None,
                'ttl_seconds': ttl_seconds, 'max_entries': max_entries, 'max_bytes': max_bytes},
            'timing_scope': 'serial in-process complete serving including provider, lookup and admission; excludes HTTP gateway transport',
            'events': events}
