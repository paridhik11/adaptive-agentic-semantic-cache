"""Policy-agnostic designated-pair metrics. No model calls or final split reads."""
from collections import Counter
from scipy.stats import beta


def cp(successes,n):
    if not n:return None
    return {'n':n,'successes':successes,'one_sided_confidence':.95,
        'safety_lower':float(beta.ppf(.05,successes,n-successes+1)) if successes else 0.,
        'error_upper':float(beta.ppf(.95,n-successes+1,successes)) if successes else 1.,
        'assumption':'independent Bernoulli group/lineage outcomes; purposive template data do not establish this population-sampling assumption',
        'population_guarantee_established':False}


def metrics(rows,decisions):
    if len(rows)!=len(decisions):raise ValueError('one decision per new row required')
    included=[(r,bool(d)) for r,d in zip(rows,decisions) if not r['is_exact']]
    accepted=[r for r,d in included if d]
    safe=[r for r in accepted if r['relation_to_cached']=='same_intent']
    unsafe=[r for r in accepted if r['relation_to_cached'] not in ('same_intent',None)]
    unknown=[r for r in accepted if r['relation_to_cached'] is None]
    positive=sum(r['relation_to_cached']=='same_intent' for r,d in included)
    def cluster(field):
        groups={}
        for r in accepted:groups.setdefault(r[field],[]).append(r)
        good=sum(all(r['relation_to_cached']=='same_intent' for r in rs) for rs in groups.values())
        failed=sum(any(r['relation_to_cached'] not in ('same_intent',None) for r in rs) for rs in groups.values())
        missing=len(groups)-good-failed
        return {'accepted':len(groups),'accepted_safe':good,'failed':failed,
            'unknown':missing,'conditional_cp':None if missing else cp(good,len(groups))}
    return {'evaluation_view':'designated_pair','evidence_type':'development_set_authored_template_generated',
        'accepted_total_including_exact':sum(decisions),'exact_accepts':sum(bool(d) and r['is_exact'] for r,d in zip(rows,decisions)),
        'nonexact_opportunities':len(included),'accepted_nonexact':len(accepted),'safe_accepts':len(safe),
        'unsafe_accepts':len(unsafe),'accepted_unknown':len(unknown),
        'precision_among_accepted':len(safe)/len(accepted) if accepted and not unknown else None,
        'labeled_precision_excluding_unknown':len(safe)/(len(safe)+len(unsafe)) if safe or unsafe else None,
        'positive_opportunities':positive,'positive_coverage':len(safe)/positive if positive else None,
        'abstentions_total':len(rows)-sum(decisions),
        'unsafe_by_family':dict(Counter(r['case_family'] for r in unsafe)),
        'unsafe_case_ids':[r['case_id'] for r in unsafe],
        'missed_positive_case_ids':[r['case_id'] for r,d in included if not d and r['relation_to_cached']=='same_intent'],
        'groups':cluster('group_id'),'lineages':cluster('source_lineage_id'),
        'answer_level_binomial_interval':None,
        'interval_note':'Correlated paraphrases are not independent trials; cluster CP values are conditional arithmetic, not population certification.'}


def evaluate_policies(rows,policies):
    """Any policy supplies decisions over the same immutable designated pairs."""
    return {name:metrics(rows,[bool(policy(r)) for r in rows]) for name,policy in policies.items()}
