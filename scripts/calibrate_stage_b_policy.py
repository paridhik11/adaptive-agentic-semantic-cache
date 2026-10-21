"""B4 fixed-grid calibration only; never changes models, veto, or threshold grid."""
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import config,identity,write_json,FLAG
from src.evaluation.intent_stage_b_policy import decision_view,summarize


def load_scored(split):
    if split not in {'dev','calibration'}:raise ValueError('final forbidden')
    path=ROOT/'artifacts/stage_b'/split/'scored.jsonl.gz'
    identity(path)
    return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]


def compact(result):
    return {k:v for k,v in result.items() if k!='cases'}


def main():
    cfg=config();path=ROOT/'configs/intent_stage_b_calibration_v1.json';identity(path)
    contract=json.loads(path.read_text())
    output=ROOT/'artifacts/stage_b/calibration-v1'
    if output.exists():raise SystemExit('Refusing to replace selected policies')
    output.mkdir(parents=True)
    cal=load_scored('calibration')
    alpha=(1-contract['one_sided_confidence'])/(len(contract['thresholds'])*len(contract['seeds']))
    selections={};grid=[]
    # Crucially, dev is NOT opened until every seed threshold is selected.
    for seed in contract['seeds']:
        candidates=[]
        for threshold in contract['thresholds']:
            metrics=summarize(cal,[decision_view(r,seed,threshold,cfg['competition_margin']) for r in cal],alpha)
            bound=metrics['groups']['conditional_error_upper']
            qualifies=bound is not None and bound<=contract['target_group_error_rate'] and metrics['accepted_unknown']==0
            grid.append({'seed':seed,'threshold':threshold,'qualifies_conditionally':qualifies,**compact(metrics)})
            if qualifies:candidates.append((metrics['safe_coverage'],threshold))
        threshold=max(candidates)[1] if candidates else None
        selections[str(seed)]=threshold
        adapter=ROOT/'artifacts/stage_b/models'/str(seed)/'adapter.safetensors'
        policy={'schema':'stage-b-policy-v1','seed':seed,'threshold':threshold,
            'threshold_status':'conditional_calibration_gate_passed' if threshold is not None else 'no_qualifying_threshold_always_abstain',
            'trained_model':identity(adapter,False),'base_repo':cfg['base_repo'],'base_revision':cfg['base_revision'],
            'embedding_repo':'sentence-transformers/all-MiniLM-L6-v2','embedding_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
            'embedding_weight_sha256':'53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db',
            'base_weight_sha256':'73fd14ad7d08f3ef30eb25841c8f4ba89e91230f48159279233b37015ccb33fb',
            'veto_version':cfg['veto_version'],'veto_sha256':identity(ROOT/'src/evaluation/intent_stage_b_veto.py',False)['sha256'],
            'top_k':cfg['top_k'],'competition_margin':cfg['competition_margin'],
            'training_config_sha256':identity(ROOT/'configs/intent_stage_b_v1.json',False)['sha256'],
            'calibration_config_sha256':identity(path,False)['sha256'],
            'calibration_scored_sha256':identity(ROOT/'artifacts/stage_b/calibration/scored.jsonl.gz',False)['sha256'],
            'alpha_per_threshold':alpha,'confidence_contract':contract,'public_data_used':False,
            'provenance_flags':[FLAG],'serving_enabled':False,'population_guarantee_established':False,
            'seed_selection':'first listed seed 11 is primary, prespecified; other seeds report spread only'}
        write_json(output/f'policy-{seed}.json',policy)
    write_json(output/'calibration-grid.json',grid)
    write_json(output/'selection.json',{'evidence_type':'synthetic_calibration_only_conditional_CP','selected':selections,
        'dev_opened_during_selection':False,'final_read':False,'alpha_per_threshold':alpha})
    dev=load_scored('dev')
    frozen=[];dev_grid=[];failure_rows=[];detailed_failures=[]
    for split,records in [('dev',dev),('calibration',cal)]:
        for seed in contract['seeds']:
            threshold=selections[str(seed)]
            result=summarize(records,[decision_view(r,seed,threshold,cfg['competition_margin']) for r in records],alpha)
            frozen.append({'split':split,'seed':seed,'threshold':threshold,**compact(result)})
            for c in result['cases']:
                if c['unsafe'] or c['accepted_unknown'] or (c['positive_opportunity'] and not c['safe']):
                    failure_rows.append({'split':split,'seed':seed,'policy':'selected',**c})
                    source=next(r for r in records if r['case_id']==c['case_id'])
                    detailed_failures.append({'split':split,'seed':seed,'policy':'selected',**c,
                        'question':source['question'],'suitable_candidate_ids':source['suitable_candidate_ids'],
                        'candidates':[{'cached_case_id':candidate['cached_case_id'],
                            'cached_question':candidate['cached_question'],'actual_relation':candidate['actual_relation'],
                            'classifier':candidate['classifier'][str(seed)],'veto':candidate['veto'],
                            'minilm_cosine':candidate['minilm_cosine']} for candidate in source['candidates']],
                        'label_evidence':'Codex-authored programmatic synthetic oracle, not human review'})
            if split=='dev':
                for t in contract['thresholds']:
                    m=summarize(records,[decision_view(r,seed,t,cfg['competition_margin']) for r in records],alpha)
                    dev_grid.append({'seed':seed,'threshold':t,**compact(m)})
    write_json(output/'selected-metrics.json',frozen)
    write_json(output/'dev-grid-diagnostic.json',{'evidence_type':'dev_diagnostic_only_not_selection','points':dev_grid})
    raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in failure_rows).encode()
    (output/'all-selected-failures.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in detailed_failures).encode()
    (output/'selected-failures-detailed.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    print(json.dumps({'selected_thresholds':selections,'metrics':frozen},indent=2))


if __name__=='__main__':main()
