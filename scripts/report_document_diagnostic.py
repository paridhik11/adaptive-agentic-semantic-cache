"""Compare sealed similarity recommendations with the actual selected-answer labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--human',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('refusing to overwrite evidence')
    manifest=json.loads((a.predictions.parent/'prediction-manifest.json').read_text())
    assert hashlib.sha256(a.predictions.read_bytes()).hexdigest()==manifest['predictions_sha256']
    predictions=[json.loads(row) for row in a.predictions.read_text().splitlines()]
    human={r['item_id']:r for r in map(json.loads,a.human.read_text().splitlines())}
    result={'evidence_type':'exposed model-authored/human-reviewed diagnostic; not fresh final or serving evaluation',
            'rows':len(predictions),'provider_calls':0,'serving_enabled':False,'methods':{},
            'labels_sha256':hashlib.sha256(a.human.read_bytes()).hexdigest(),
            'predictions_sha256':manifest['predictions_sha256']}
    for method in predictions[0]['recommendations']:
        counts={'recommended':0,'known_sufficient':0,'known_insufficient':0,'unknown_selected_answer':0,'abstained':0}
        cases=[]
        for row in predictions:
            h=human[row['item_id']]
            assert row['question']==h['new_question']
            selected=row['recommendations'][method]
            if selected is None:counts['abstained']+=1;continue
            counts['recommended']+=1
            label=h['answer_sufficient'] if selected==h['seed_id'] else None
            counts['unknown_selected_answer' if label is None else 'known_sufficient' if label else 'known_insufficient']+=1
            cases.append({'human_id':row['human_id'],'selected_seed':selected,'human_answer_label':label,
                          'candidate':next(c for c in row['candidates'] if c['seed_id']==selected)})
        result['methods'][method]={'counts':counts,'recommended_cases':cases}
    result['limitations']=['Three correlated, exposed FAQ sources; questions model-authored.',
        'Similarity is not actual-answer sufficiency. No serving promotion or threshold selection.',
        'No representative coverage, latency or cost claim; only diagnostic counts reported.']
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({name:info['counts'] for name,info in result['methods'].items()}))


if __name__=='__main__':main()
