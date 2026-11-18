"""Offline research policy over scored candidates. No production integration."""
from collections import Counter
from scipy.stats import beta


def choose(candidates, threshold, margin=.05, use_veto=True):
    """Only numeric classifier outputs and veto results enter this function.

    Labels/signatures/answers are deliberately not accepted as policy inputs.
    Candidates carry IDs, same-class probability, argmax class, and rejection.
    """
    if threshold is None:
        return None, 'no_calibrated_threshold'
    above = [c for c in candidates if c['predicted_class']=='same_intent' and c['same_probability']>=threshold]
    if not above:
        return None, 'classifier_abstention'
    ranked = sorted(above, key=lambda c:(-c['same_probability'],c['cached_case_id']))
    if len(ranked)>1 and ranked[0]['same_probability']-ranked[1]['same_probability']<=margin:
        return None, 'competing_candidates'
    # Veto cannot resolve ambiguity, promote another candidate, or create a hit.
    # Choose by classifier first; rejection ends lookup without fallback.
    if use_veto and ranked[0]['veto_reject']:
        return None, 'veto'
    return ranked[0]['cached_case_id'], 'semantic'


def decision_view(record, seed, threshold, margin=.05, use_veto=True):
    if record['exact_candidate'] is not None:
        return record['exact_candidate'], 'exact'
    scored = [{'cached_case_id':c['cached_case_id'],
               'same_probability':c['classifier'][str(seed)]['probabilities'][0],
               'predicted_class':c['classifier'][str(seed)]['predicted_class'],
               'veto_reject':c['veto']['reject']} for c in record['candidates']]
    return choose(scored,threshold,margin,use_veto)


def cp_error(errors, n, alpha=.05):
    if not n:
        return None
    return float(beta.ppf(1-alpha,errors+1,n-errors)) if errors<n else 1.


def summarize(records, decisions, alpha=.05):
    """Actual selected-candidate relations, never designated-pair attribution."""
    cases=[]
    for r,(cid,cause) in zip(records,decisions):
        # Positive opportunity means at least one suitable cache answer contract
        # exists in scope, including negatives that match a DIFFERENT seed.
        positive=bool(r['suitable_candidate_ids']) and r['exact_candidate'] is None
        selected=next((c for c in r['candidates'] if c['cached_case_id']==cid),None)
        relation=selected['actual_relation'] if selected else ('same_intent' if cause=='exact' else None)
        exact_query=r['exact_candidate'] is not None
        accepted=cid is not None and not exact_query
        cases.append({'case_id':r['case_id'],'group_id':r['group_id'],'lineage':r['source_lineage_id'],
            'template':r['template_id'],'family':r['case_family'],'positive_opportunity':positive,
            'selected_candidate':cid,'selected_relation':relation,'cause':cause,'is_exact_query':exact_query,
            'accepted_nonexact':accepted,'safe':accepted and relation=='same_intent',
            'unsafe':accepted and relation not in ('same_intent',None),
            'accepted_unknown':accepted and relation is None,
            'retrievable_positive':any(c['actual_relation']=='same_intent' for c in r['candidates'])})
    nonexact=[c for c in cases if not c['is_exact_query']]
    accepted=[c for c in nonexact if c['accepted_nonexact']]
    safe=sum(c['safe'] for c in accepted);unsafe=sum(c['unsafe'] for c in accepted);unknown=sum(c['accepted_unknown'] for c in accepted)
    positive=sum(c['positive_opportunity'] for c in nonexact)
    retrieved=sum(c['positive_opportunity'] and c['retrievable_positive'] for c in nonexact)

    def clusters(field):
        groups={}
        for c in accepted:groups.setdefault(c[field],[]).append(c)
        failed=sum(any(c['unsafe'] for c in cs) for cs in groups.values())
        missing=sum(not any(c['unsafe'] for c in cs) and any(c['accepted_unknown'] for c in cs) for cs in groups.values())
        good=len(groups)-failed-missing
        return {'accepted':len(groups),'safe':good,'failed':failed,'unknown':missing,
            'conditional_error_upper':None if missing else cp_error(failed,len(groups),alpha),
            'conditional_safety_lower':None if missing or not groups else 1-cp_error(failed,len(groups),alpha),
            'alpha':alpha,'population_guarantee_established':False}
    return {'evidence_type':'synthetic_template_held_out_actual_candidate_diagnostic_not_human_truth',
        'total_queries':len(cases),'exact_queries':sum(c['is_exact_query'] for c in cases),
        'exact_accepts':sum(c['is_exact_query'] and c['selected_candidate'] is not None for c in cases),
        'nonexact_opportunities':len(nonexact),'accepted_nonexact':len(accepted),'safe_accepts':safe,
        'unsafe_accepts':unsafe,'accepted_unknown':unknown,'positive_opportunities':positive,
        'precision':safe/len(accepted) if accepted and not unknown else None,
        'labeled_precision_excluding_unknown':safe/(safe+unsafe) if safe+unsafe else None,
        'safe_coverage':safe/positive if positive else None,
        'unsafe_accept_rate':unsafe/len(nonexact) if nonexact else None,
        'retrieval_recall_at_k':retrieved/positive if positive else None,
        'retrieval_missed_positives':positive-retrieved,
        'decision_lost_positives':sum(c['positive_opportunity'] and c['retrievable_positive'] and not c['safe'] for c in nonexact),
        'missed_positive_causes':dict(Counter(('retrieval_miss' if not c['retrievable_positive'] else c['cause']) for c in nonexact if c['positive_opportunity'] and not c['safe'])),
        'unsafe_by_family':dict(Counter(c['family'] for c in accepted if c['unsafe'])),
        'groups':clusters('group_id'),'lineages':clusters('lineage'),
        'bound_assumption':'independent Bernoulli cluster outcomes; purposive templates and shared lineages do not establish this assumption',
        'cases':cases}
