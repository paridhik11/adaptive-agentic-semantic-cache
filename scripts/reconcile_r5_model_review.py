"""Validate model-assisted labels against the original private R5 review packet."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import read_json, digest, write_new, _constant, _unique


def reconcile(archive,review):
    with zipfile.ZipFile(archive) as z:
        template=json.loads(z.read('reviews.template.json'),parse_constant=_constant,object_pairs_hook=_unique)
    returned=read_json(review)
    if set(returned)!=set(template) or returned.get('schema_version')!=template['schema_version'] or returned.get('independent_review_required') is not True:
        raise ValueError('Review schema changed')
    if not isinstance(returned.get('items'),list) or len(returned['items'])!=len(template['items']):raise ValueError('Item count changed')
    counts=Counter();reviewers=set()
    for index,(original,item) in enumerate(zip(template['items'],returned['items']),1):
        if set(item)!=set(original) or any(item[k]!=original[k] for k in ('review_id','question','reference','answer')):
            raise ValueError(f'Immutable content/order changed at item {index}')
        if not isinstance(item['reviewer'],str) or 'model-assisted' not in item['reviewer'].lower():raise ValueError('Model-assisted attribution required')
        if not isinstance(item['notes'],str) or not item['notes'].strip():raise ValueError('Rationale required')
        c,s=item['correct'],item['supported']
        if not ((c is None and s is None) or (type(c) is bool and type(s) is bool)):raise ValueError('Both labels must be booleans or both null')
        counts['unknown' if c is None else f'correct_{str(c).lower()}_supported_{str(s).lower()}']+=1
        reviewers.add(item['reviewer'])
    return {'evidence_kind':'third_party_model_assisted_review','source_sha256':hashlib.sha256(Path(review).read_bytes()).hexdigest(),
            'review_canonical_sha256':digest(returned),'source_archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'immutable_bindings_verified':True,'items':len(returned['items']),'reviewers_as_declared':sorted(reviewers),
            'label_counts_as_submitted':dict(counts),'labels_modified':False,'human_reviews':0,
            'quality_gate':'INCONCLUSIVE','population_accuracy':None,'external_model_calls_during_reconciliation':0,
            'limitations':['Reviewer model identity is self-declared, not authenticated.',
                          'Review judgments are disputed in the accompanying reconciliation note.',
                          'Small dependent development sample; no population accuracy or safety interval inferred.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=reconcile(a.archive,a.review);write_new(a.output,result);print(json.dumps(result,indent=2))
