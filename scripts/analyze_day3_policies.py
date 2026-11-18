"""Read preserved Day 3 diagnostics; no model loading, network, or policy tuning."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.policy.question_conditioned import QA_MARGINS, SUPPORT_THRESHOLDS, recommend


def summarize(result):
    rows=result['rows']
    def count(q=0., s=.95):
        accepted=[r for r in rows if recommend(r['assessment']['features'],q,s)]
        correct=sum(r['expected'] for r in accepted)
        return {'correct_accepts':correct,'incorrect_accepts':len(accepted)-correct,
                'positive_denominator':sum(r['expected'] for r in rows),
                'negative_denominator':sum(not r['expected'] for r in rows)}
    initial=count()
    assert initial['correct_accepts']==result['summary']['correct_accepts']
    assert initial['incorrect_accepts']==result['summary']['incorrect_accepts']
    return {'initial':initial,'cohorts':result['cohorts'],'cache':result['cache_summary'],
            'curve':[{'qa_margin':q,'support_threshold':s,**count(q,s)}
                     for q in QA_MARGINS for s in SUPPORT_THRESHOLDS]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'docs/audit/day3')
    args=parser.parse_args()
    output={}
    for name in ['qa-strict','qa-anchor','qa-function','qa-hybrid']:
        with gzip.open(args.input_dir/(name+'.json.gz'),'rt',encoding='utf-8') as handle:
            output[name]=summarize(json.load(handle))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
