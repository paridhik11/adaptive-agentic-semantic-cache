"""B3 actual-candidate dev diagnostics. Fixed thresholds; no policy selection."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import config,write_json,FLAG
from src.evaluation.intent_stage_b_policy import decision_view,summarize


def compact(value):
    return {k:v for k,v in value.items() if k!='cases'}


def baseline_specs():
    specs=[('always_miss',None),('exact_only',None)]
    for name in ['minilm_cosine','redis_embedding_cosine']:
        specs += [(name,t) for t in [.5,.6,.7,.8,.9,.95]]
    for name in ['quora','redis_reranker','existing_guard_stack']:
        specs += [(name,t) for t in [.5,.7,.9,.95,.99]]
    specs += [('redis_reranker_same_set_diagnostic',t) for t in [0,.01,.025,.05,.07275390625,.075,.0888671875,.1,.2,.3,.4,.5]]
    return specs


def baseline_decision(record,name,threshold):
    if record['exact_candidate'] is not None:return record['exact_candidate'],'exact'
    if name in ['always_miss','exact_only']:return None,'baseline_abstention'
    field={'minilm_cosine':'minilm_cosine','redis_embedding_cosine':'redis_cosine',
           'existing_guard_stack':'quora_score','redis_reranker_same_set_diagnostic':'redis_reranker_score'}.get(name,name+'_score')
    eligible=[c for c in record['candidates'] if c[field]>=threshold and
              (name!='existing_guard_stack' or c['existing_guards_passed'])]
    if not eligible:return None,'baseline_abstention'
    candidate=sorted(eligible,key=lambda c:(-c[field],c['cached_case_id']))[0]
    return candidate['cached_case_id'],'semantic'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','calibration'],default='dev')
    args=parser.parse_args();cfg=config()
    directory=ROOT/'artifacts/stage_b'/args.split
    records=[json.loads(x) for x in gzip.decompress((directory/'scored.jsonl.gz').read_bytes()).decode().splitlines()]
    output=[];details=[]
    for name,threshold in baseline_specs():
        result=summarize(records,[baseline_decision(r,name,threshold) for r in records])
        output.append({'baseline':name,'threshold':threshold,
            'provenance_flags':[FLAG] if name not in ['always_miss','exact_only','minilm_cosine'] else [],
            'same_set_diagnostic':name=='redis_reranker_same_set_diagnostic',**compact(result)})
        for case in result['cases']:
            if case['unsafe'] or case['accepted_unknown'] or (case['positive_opportunity'] and not case['safe']):
                details.append({'baseline':name,'threshold':threshold,**case})
    write_json(directory/'baseline-comparison.json',output)
    (directory/'baseline-errors.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r,sort_keys=True)+'\n' for r in details).encode(),mtime=0))
    if args.split=='dev':
        diagnostics=[]
        for seed in cfg['seeds']:
            with_veto=summarize(records,[decision_view(r,seed,.5,cfg['competition_margin'],True) for r in records])
            without=summarize(records,[decision_view(r,seed,.5,cfg['competition_margin'],False) for r in records])
            additional=[]
            for a,b in zip(with_veto['cases'],without['cases']):
                if a['accepted_nonexact']:assert b['accepted_nonexact'] and a['selected_candidate']==b['selected_candidate']
                if b['accepted_nonexact'] and not a['accepted_nonexact']:
                    additional.append({'case_id':a['case_id'],'family':a['family'],'wrongly_killed_true_positive':b['safe'],'veto_cause':a['cause']})
            diagnostics.append({'seed':seed,'threshold':.5,'selection':'fixed preregistered diagnostic, not selected',
                'with_veto':compact(with_veto),'without_veto':compact(without),
                'additional_rejections_by_family':dict(Counter(r['family'] for r in additional)),
                'wrongly_killed_true_positives':sum(r['wrongly_killed_true_positive'] for r in additional),
                'additional_rejections':additional})
        write_json(directory/'classifier-veto-diagnostic.json',diagnostics)
    print(json.dumps({'split':args.split,'baseline_points':len(output),'evidence_type':'synthetic_template_held_out_actual_candidate_diagnostic','no_threshold_selected':True}))


if __name__=='__main__':main()
