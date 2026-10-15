"""Convert existing development pairs into a provenance-preserving request stream.

Pair labels are retained as provenance only, never reused as answer labels.
No download or new held-out/test claim is made by this conversion.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.schema import digest, read_json, validate_workload, write_new


def convert_pairs(pairs, *, source_digest, model, license_note):
    parent = {}
    def text_key(text):
        if not isinstance(text,str) or not text.strip():
            raise ValueError('nonempty query text required')
        return digest(' '.join(text.casefold().split()))
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for pair in pairs:
        a,b = (text_key(pair[k]) for k in ('query_a','query_b'))
        ra,rb=find(a),find(b)
        parent[max(ra,rb)]=min(ra,rb)
    rows=[]
    for i,pair in enumerate(pairs):
        source=pair.get('source',pair.get('dataset'))
        if not isinstance(source,str) or not source.strip():
            raise ValueError('each pair requires a source')
        for side in ('a','b'):
            text=pair['query_'+side]
            rows.append({'event_id':f'pair-{i}-{side}','tenant_id':'development-benchmark',
                'source':source,'group_id':find(text_key(text)),'split':'dev',
                'origin_pair_index':i,'original_pair_label_not_answer_gold':pair.get('label',pair.get('is_equivalent')),
                'request':{'model':model,'messages':[{'role':'user','content':text}], 'temperature':0,'max_tokens':256}})
    workload={'schema_version':'workload-v1','name':'converted-development-pairs',
        'nature':'constructed_natural_queries',
        'provenance':'Existing exposed development/training pairs, interleaved query_a then query_b. This ordering induces reuse opportunities; it is not organic traffic. Pair labels are NOT answer-reuse labels.',
        'license':license_note,'source_sha256':source_digest,
        'grouping':'connected components of normalized query text across pairs; no semantic/thread grouping guarantee',
        'requests':rows}
    return validate_workload(workload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='fixture-model')
    p.add_argument('--license-note',required=True,help='source terms or explicit unverified status; not inferred')
    args=p.parse_args()
    pairs=read_json(args.pairs)
    if not isinstance(pairs,list) or not pairs:p.error('nonempty pair array required')
    result=convert_pairs(pairs,source_digest=digest(pairs),model=args.model,license_note=args.license_note)
    write_new(args.output,result)
    print('Prepared development requests:',len(result['requests']))
    print('Sources:',dict(Counter(r['source'] for r in result['requests'])))
    print('No answer labels or sealed-test claims were created.')


if __name__=='__main__':main()
