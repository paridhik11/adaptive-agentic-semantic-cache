"""Validate returned review data; report provenance counts, never infer accuracy."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(packet, responses):
    manifest=json.loads((packet/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('packet integrity mismatch: '+name)
    originals=[json.loads(line) for line in (packet/'requests.jsonl').read_text().splitlines()]
    expected={row['item_id']:row for row in originals}
    rows=[json.loads(line) for line in responses.read_text().splitlines() if line.strip()]
    seen=set();counts={kind:{'questions':0,'labeled':0,'declared_distinct_author_reviewer':0}
                       for kind in ['human_authored','model_assisted']}
    for row in rows:
        identifier=row.get('item_id')
        if identifier not in expected or identifier in seen:raise ValueError('unknown/duplicate item')
        seen.add(identifier)
        if row.get('seed_id')!=expected[identifier]['seed_id']:raise ValueError('seed changed')
        question=row.get('new_question')
        if not isinstance(question,str) or not question.strip() or len(question)>4000:raise ValueError('missing/bounded question required')
        kind=row.get('evidence_type')
        if kind not in counts:raise ValueError('declare human_authored or model_assisted')
        if not isinstance(row.get('author_id'),str) or not row['author_id'].strip():raise ValueError('author declaration required')
        label=row.get('answer_sufficient')
        if label is not None and type(label) is not bool:raise ValueError('label must be boolean or null')
        counts[kind]['questions']+=1
        if label is not None:
            if row.get('relation') not in {'same_intent','more_specific','more_general','different_intent','ambiguous'}:
                raise ValueError('valid relation required for labeled item')
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ['reviewer_id','rationale']):
                raise ValueError('reviewer and rationale required for labels')
            counts[kind]['labeled']+=1
            counts[kind]['declared_distinct_author_reviewer']+=int(row['reviewer_id']!=row['author_id'])
    return {'evidence_type':'review-schema and provenance validation; NOT quality measurement',
            'received':len(rows),'missing_ids':sorted(set(expected)-seen),'by_provenance':counts,
            'identities_independently_verified':False,'accuracy':None,'precision':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--responses',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('refusing to overwrite review receipt')
    result=validate(a.packet,a.responses)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
