"""Keep declared human judgments separate from model judgments and predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def ingest(packet, labels, reviewer):
    manifest=json.loads((packet/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=h:raise ValueError('human packet changed')
    templates={r['human_id']:r for r in map(json.loads,(packet/'human-responses.template.jsonl').read_text().splitlines())}
    mapping={r['human_id']:r for r in json.loads((packet/'mapping.json').read_text())}
    if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('declared reviewer required')
    output=[]
    for human_id,label in labels.items():
        if human_id not in templates or label.get('judgment') not in {'Y','N','?'}:raise ValueError('invalid human item/label')
        row=dict(templates[human_id]);bound=mapping[human_id]
        if row['item_id']!=bound['item_id'] or hashlib.sha256(row['new_question'].encode()).hexdigest()!=bound['question_sha256']:
            raise ValueError('question binding mismatch')
        value=label['judgment'];reason=label.get('rationale')
        row.update(answer_sufficient={'Y':True,'N':False,'?':None}[value],reviewer_id=reviewer,
                   rationale=reason or f'User marked {value}; no additional rationale supplied.',
                   human_rationale_provided=bool(reason),review_source='user_chat_complete_review',
                   relation=None,evidence_type='human_review_of_model_authored')
        output.append(row)
    return sorted(output,key=lambda r:r['human_id'])


def compare(human, model):
    indexed={r['item_id']:r for r in model}
    counts={'human_yes':0,'human_no':0,'human_uncertain':0,'binary_agreements':0,'opposite_binary_judgments':0}
    differences=[]
    for row in human:
        other=indexed[row['item_id']]
        if any(row[k]!=other[k] for k in ['seed_id','new_question']):raise ValueError('review items differ')
        h,m=row['answer_sufficient'],other['answer_sufficient']
        counts['human_uncertain' if h is None else 'human_yes' if h else 'human_no']+=1
        if h is not None and m is not None:
            counts['binary_agreements' if h==m else 'opposite_binary_judgments']+=1
        if h is None or m is None or h!=m:
            differences.append({'human_id':row['human_id'],'item_id':row['item_id'],
                'human_judgment':h,'model_judgment':m,'human_rationale':row['rationale'],'model_rationale':other['rationale']})
    return {'evidence_type':'declared human review of model-authored source-excerpt questions; model diagnostics separate',
            'received':len(human),'counts':counts,'differences':differences,
            'identity_or_blinding_independently_verified':False,'system_predictions_evaluated':False,
            'accuracy':None,'reuse_precision':None,'savings':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--reviewer',required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('refusing to overwrite review evidence')
    human=ingest(a.packet,json.loads(a.labels.read_text()),a.reviewer)
    model=[json.loads(r) for r in a.model.read_text().splitlines()]
    report=compare(human,model)
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'human-responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in human))
    (a.output_dir/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
