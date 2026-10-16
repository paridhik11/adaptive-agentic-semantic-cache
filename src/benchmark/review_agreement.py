"""Reconcile two separately supplied human review files without hiding disputes."""
from src.benchmark.report import validate_reviews, review_template
from src.benchmark.schema import digest


def reconcile(trace,first,second):
    left,right=validate_reviews(trace,first),validate_reviews(trace,second)
    fields=('safe_reuse','answer_correct','fresh_answer_correct')
    def reviewer(rows):
        active=[r for r in rows.values() if any(r.get(k) is not None for k in fields)]
        names={r['reviewer'].strip() for r in active}
        if not active or len(names)!=1 or any(r['method']!='human' for r in active):
            raise ValueError('each input needs one identified human reviewer with labels')
        return next(iter(names))
    a,b=reviewer(left),reviewer(right)
    if a.casefold()==b.casefold():raise ValueError('reviewers must be distinct')
    output=review_template(trace);stats={};conflicts=[]
    for field in fields:
        pairs=[]
        for row in output['reviews']:
            identity=row['event_id'];l=left.get(identity,{}).get(field);r=right.get(identity,{}).get(field)
            row['reviewer']='consensus:'+a+'|'+b
            if l is None or r is None:continue
            pairs.append((l,r))
            if l==r:row[field]=l
            else:conflicts.append({'event_id':identity,'field':field,'first':l,'second':r})
        n=len(pairs);observed=sum(l==r for l,r in pairs)/n if n else None
        if n:
            pa=sum(l for l,_ in pairs)/n;pb=sum(r for _,r in pairs)/n
            chance=pa*pb+(1-pa)*(1-pb)
            kappa=(observed-chance)/(1-chance) if chance<1 else None
        else:kappa=None
        stats[field]={'double_reviewed':n,'agreement':observed,'cohen_kappa':kappa}
    return {'schema_version':'review-agreement-v1','trace_sha256':digest(trace),
        'input_review_sha256':[digest(first),digest(second)],'reviewers':[a,b],
        'agreement':stats,'conflicts':conflicts,
        'limitations':['Reviewer identities are declared, not authenticated. Distinct names do not prove independence.',
                       'Conflicts and singly reviewed labels remain unknown in consensus. Adjudication requires new explicit review.'],
        'consensus_reviews':output}
