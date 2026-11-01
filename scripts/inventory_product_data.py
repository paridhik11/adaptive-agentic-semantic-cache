"""Describe current data without opening any locked final records."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refusing to overwrite inventory')
    root=Path(__file__).resolve().parents[1]
    path=root/'data/intent_stage_b/v2/manifest.json'
    stage_b=json.loads(path.read_text())
    source=root/'src/api/demo_reference.json'
    demo=json.loads(source.read_text())
    result={'training_now':False,'locked_final_records_opened':False,
        'historical_training':{'path':'data/intent_stage_b/v2','manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'provenance':stage_b['evidence_type'],'public_data_used':stage_b['public_data_used'],
            'splits':{k:{'rows':v['rows'],'groups':v['groups']} for k,v in stage_b['splits'].items()},
            'note':'Pair rows are not independent groups. Null/ambiguous/exact pairs are not all training examples; see B1 receipt.',
            'status':'trained policies failed useful-coverage calibration; not promoted'},
        'product_tests':{'path':'tests/test_*document* and tests/test_*cache*',
            'provenance':'builder-authored deterministic fixtures; not independent natural-language labels'},
        'demo_sources':{'path':'src/api/demo_reference.json','source':demo['source'],
            'revision':demo['source_revision'],'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'documents':len(demo['documents']),'questions':[d['title'] for d in demo['documents']],
            'status':'exposed documentation excerpts; not a held-out model benchmark'},
        'independent_data':{'status':'not returned','human_labels':0,'request':'data/document_product_review/v1/requests.jsonl'},
        'public_pair_benchmarks':'No public pair dataset approved/downloaded for these current product milestones; historical studies separate.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
