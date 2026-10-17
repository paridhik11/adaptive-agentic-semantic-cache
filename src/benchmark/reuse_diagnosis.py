"""Explain recorded cache outcomes without inventing safety labels or opportunities."""
from collections import Counter,defaultdict
from src.benchmark.schema import digest
from src.benchmark.report import summarize


def diagnose(trace):
    report=summarize(trace)  # Existing strict trace/binding/work validation.
    terminal=Counter();reasons=Counter();stages=Counter();sources=defaultdict(Counter)
    eligibility=Counter();admission=Counter();seen=set();prior_same=Counter()
    coverage=Counter();expirations=evictions=0
    for event in trace['events']:
        status=event['status'];terminal[status]+=1
        reasons[event.get('reason') or 'unrecorded']+=1
        sources[event['source']][status]+=1
        key=digest({'tenant':event['tenant_id'],'request':event['request']})
        if key in seen:prior_same[status]+=1
        seen.add(key)
        decision=event.get('eligibility')
        if decision is None:
            eligibility['unrecorded']+=1
        elif (not isinstance(decision,dict) or type(decision.get('cacheable')) is not bool
              or (decision['cacheable'] and decision.get('bypass_reason') is not None)
              or (not decision['cacheable'] and not isinstance(decision.get('bypass_reason'),str))):
            raise ValueError('invalid recorded eligibility')
        elif decision['cacheable']:
            eligibility['eligible']+=1
        else:
            eligibility['bypass:'+decision['bypass_reason']]+=1
        admission['stored' if event['stored'] else 'not_stored']+=1
        diagnostic=event['cache'].get('response_diagnostics')
        if diagnostic is not None and not diagnostic['admissible_complete_answer']:
            admission['incomplete_or_rejected_response']+=1
        d=event.get('decision_diagnostics')
        if d is None:
            coverage['semantic_decision_unrecorded_or_not_applicable']+=1
        else:
            if (not isinstance(d,dict) or not isinstance(d.get('stage'),str)
                or any(type(d.get(k)) is not int or d[k]<0 for k in ('expired_entries','evicted_entries'))):
                raise ValueError('invalid semantic diagnostics')
            coverage['semantic_decision_recorded']+=1
            stages[d['stage']]+=1
            expirations+=d['expired_entries'];evictions+=d['evicted_entries']
    count=len(trace['events'])
    if sum(terminal.values())!=count or dict(terminal)!=report['status_counts']:
        raise ValueError('terminal accounting does not reconcile')
    bypass=terminal['BYPASS']
    eligible=eligibility['eligible'] if not eligibility['unrecorded'] else None
    return {'schema_version':'reuse-diagnosis-v1','trace_sha256':digest(trace),
        'evaluation_mode':trace['evaluation_mode'],'nature':trace['nature'],'requests':count,
        'terminal_outcomes':dict(terminal),'terminal_reasons':dict(reasons),
        'eligibility_observed':dict(eligibility),'eligible_requests':eligible,
        'bypass_fraction':bypass/count,'all_request_hit_fraction':terminal['HIT']/count,
        'eligible_hit_fraction':terminal['HIT']/eligible if eligible else None,
        'admission_counts':dict(admission),'by_source':{k:dict(v) for k,v in sorted(sources.items())},
        'semantic_terminal_stages':dict(stages),'instrumentation_coverage':dict(coverage),
        'semantic_expirations_observed':expirations if coverage['semantic_decision_recorded'] else None,
        'semantic_evictions_observed':evictions if coverage['semantic_decision_recorded'] else None,
        'prior_identical_request_occurrences_by_outcome':dict(prior_same),
        'provider_work':{'cache_generation_attempts':report['cache_arm']['provider_attempts'],
            'verifier_attempts':report['verifier_work']['provider_attempts'],
            'total_cache_attempts':report['total_cache_provider_attempts'],
            'fresh_attempts':report['fresh_arm']['provider_attempts'],
            'net_attempts_avoided':report['provider_attempts_avoided']},
        'accounting_reconciled':True,'safety_certified':False,
        'limitations':[
            'Eligibility is a recorded heuristic decision, not an independent safe/unsafe label.',
            'Prior identical request occurrence ignores TTL, eviction and answer correctness; it is NOT a safe opportunity count.',
            'No pair label is used as an answer correctness label. Candidate recall and correct reuse require independent audits.',
            'Missing diagnostic fields stay unrecorded. Exact expiry, invalidation and eviction causes are not inferred from a miss.',
            'A semantic candidate count of zero means none in the active complete-request scope, not zero semantic duplicates in the world.',
            'No fixture latency or token work is presented as live cost/latency improvement.']}
