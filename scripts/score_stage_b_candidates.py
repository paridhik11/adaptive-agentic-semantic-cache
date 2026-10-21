"""Score actual scoped top-k candidates locally; final is rejected by argparse."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import (config,read_split,identity,snapshot,offline,hardware,write_json,SPECS,FLAG)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','calibration'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--model-key',choices=['quora','redis_reranker','11','29','47'])
    args=parser.parse_args()
    if args.output_dir.exists() and not args.resume:parser.error('Refusing to overwrite scored evidence; use --resume for incomplete checkpoints only')
    if (args.output_dir/'receipt.json').exists():parser.error('Completed scored evidence is immutable; choose a fresh directory')
    worker_lock=None
    if args.model_key:
        # A surviving worker holds this across a parent/session interruption.
        # A resumed invocation waits, then consumes the saved complete prefix.
        import fcntl
        lockpath=args.output_dir/'checkpoints'/('worker-'+args.model_key+'.lock')
        lockpath.parent.mkdir(parents=True,exist_ok=True)
        worker_lock=lockpath.open('a')
        fcntl.flock(worker_lock,fcntl.LOCK_EX)
    cfg=config()
    identity(ROOT/'configs/intent_stage_b_calibration_v1.json')  # frozen before any calibration work
    for name in ['src/evaluation/intent_stage_b_model.py','src/evaluation/intent_stage_b_veto.py',
                 'scripts/build_stage_b_intents.py','src/decision/local_equivalence.py']:
        identity(ROOT/name)
    data=read_split(args.split)
    bank=[r for r in data if r['role']=='cached']
    queries=[r for r in data if r['role']=='new']
    paths={key:snapshot(key) for key in SPECS}
    offline()
    import torch
    import numpy as np
    from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
    from src.evaluation.intent_stage_b_model import load_adapter,tokenize_pairs
    from src.evaluation.intent_stage_b_veto import veto
    from scripts.build_stage_b_intents import relation
    from src.decision.local_equivalence import evaluate_rule_guards
    from scripts.stage_b_checkpoint import load as load_checkpoint,save as save_checkpoint
    torch.set_num_threads(cfg['torch_threads'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    texts=list(dict.fromkeys(r['question'] for r in data)); textid={t:i for i,t in enumerate(texts)}
    timings={}
    def binding(key,inputs,kind):
        modelkey='quora' if key.isdigit() else key
        extra=identity(ROOT/'artifacts/stage_b/models'/key/'adapter.safetensors',False)['sha256'] if key.isdigit() else None
        files={str(p.relative_to(paths[modelkey])):identity(p,False)['sha256']
               for p in sorted(paths[modelkey].rglob('*')) if p.is_file() and p.name!='model.safetensors'}
        return {'schema':'stage-b-numeric-checkpoint-v1','kind':kind,'key':key,'split':args.split,
            'ordered_inputs_sha256':hashlib.sha256(json.dumps(inputs,ensure_ascii=False).encode()).hexdigest(),
            'model_weight_sha256':SPECS[modelkey][2],'adapter_sha256':extra,'snapshot_config_tokenizer_hashes':files,
            'config_sha256':identity(ROOT/'configs/intent_stage_b_v1.json',False)['sha256'],
            'model_code_sha256':identity(ROOT/'src/evaluation/intent_stage_b_model.py',False)['sha256'],
            'torch_version':torch.__version__,'dtype':'float32','attention':'eager'}

    def timing_view(state,n,kind):
        return {'load_seconds':sum(state['load_seconds']),
                'total_seconds':sum(state['load_seconds'])+sum(state['batch_ms'])/1000,
                'batch_size':32,'batch_ms':state['batch_ms'],'load_segments_seconds':state['load_seconds'],
                'n_unique_'+kind:n,'resumable':True,
                'note':'accumulated setup/load (including checkpoint reads) and batch compute; checkpoint writes, idle gaps and discarded interrupted work excluded; concurrent worker contention possible'}

    def encode(key):
        begin=time.perf_counter()
        path=paths[key]
        checkpoint=args.output_dir/'checkpoints'/('embedding-'+key+'.npz')
        bound=binding(key,texts,'embedding')
        result,state=load_checkpoint(checkpoint,bound)
        if len(result)>len(texts):raise ValueError('embedding checkpoint too long')
        if len(result)==len(texts):
            timings[key]=timing_view(state,len(texts),'texts')
            return np.asarray(result,dtype=np.float32)
        tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
            use_safetensors=True,attn_implementation='eager').float().eval()
        state['load_seconds'].append(time.perf_counter()-begin)
        pool=json.loads((path/'1_Pooling/config.json').read_text())
        for start in range(len(result),len(texts),32):
            t=time.perf_counter()
            tokens=tok(texts[start:start+32],padding=True,truncation=False,return_tensors='pt')
            if tokens['input_ids'].shape[-1]>cfg['max_tokens']:raise ValueError('overlength embedding')
            with torch.inference_mode():
                hidden=model(**tokens).last_hidden_state
                if pool.get('pooling_mode_mean_tokens'):
                    mask=tokens['attention_mask'].unsqueeze(-1);vector=(hidden*mask).sum(1)/mask.sum(1)
                elif pool.get('pooling_mode_cls_token'):vector=hidden[:,0]
                else:raise ValueError('unknown pooling')
                result.extend(torch.nn.functional.normalize(vector,dim=1).tolist())
            state['batch_ms'].append((time.perf_counter()-t)*1000)
            save_checkpoint(checkpoint,bound,result,state)
            if start%512==0:print(json.dumps({'split':args.split,'embedding':key,'done':start,'total':len(texts)}),flush=True)
        timings[key]=timing_view(state,len(texts),'texts')
        del model;gc.collect()
        return np.asarray(result,dtype=np.float32)
    dense=encode('minilm')
    bankvectors=np.asarray([dense[textid[r['question']]] for r in bank])
    records=[]
    for r in queries:
        begin=time.perf_counter()
        eligible=[i for i,c in enumerate(bank) if c['scope_id']==r['scope_id']]
        similarities=bankvectors@dense[textid[r['question']]]
        ranked=sorted(eligible,key=lambda i:(-float(similarities[i]),bank[i]['cached_case_id']))[:cfg['top_k']]
        search_ms=(time.perf_counter()-begin)*1000
        candidates=[]
        for i in ranked:
            c=bank[i];begin=time.perf_counter();rejection=veto(c['question'],r['question']);veto_ms=(time.perf_counter()-begin)*1000
            guard,reason=evaluate_rule_guards(c['question'],r['question'])
            candidates.append({'cached_case_id':c['cached_case_id'],'cached_question':c['question'],
                'actual_relation':relation(r['intent_signature'],c['intent_signature']),
                'minilm_cosine':float(similarities[i]),'veto':rejection,'veto_ms':veto_ms,
                'existing_guards_passed':guard,'existing_guard_reason':reason,'classifier':{}})
        exact=next((bank[i]['cached_case_id'] for i in eligible if bank[i]['question']==r['question']),None)
        suitable=[bank[i]['cached_case_id'] for i in eligible if relation(r['intent_signature'],bank[i]['intent_signature'])=='same_intent']
        designated=next(c for c in bank if c['cached_case_id']==r['cached_case_id'])
        records.append({k:r[k] for k in ['case_id','group_id','source_lineage_id','template_id','case_family','scope_id','question','relation_to_cached']} | {
            'exact_candidate':exact,'suitable_candidate_ids':suitable,'search_ms':search_ms,
            'candidates':candidates,'designated_cached_question':designated['question']})
    # Score every retrieved non-exact candidate, including unsafe and unknown ones.
    # Deduplication shares identical real scores; no predictions are invented.
    pairs=list(dict.fromkeys((c['cached_question'],r['question']) for r in records
                             if r['exact_candidate'] is None for c in r['candidates']))
    # Designated pairs also provide a matched train-vs-heldout classification diagnostic.
    designated_pairs=[(r['designated_cached_question'],r['question']) for r in records if r['exact_candidate'] is None]
    pairs=list(dict.fromkeys(pairs+designated_pairs));pairid={p:i for i,p in enumerate(pairs)}
    redis=encode('redis_embedding')
    for r in records:
        for c in r['candidates']:
            c['redis_cosine']=float(redis[textid[c['cached_question']]]@redis[textid[r['question']]])
    del dense,redis,bankvectors;gc.collect()
    if args.prepare_only:
        write_json(args.output_dir/'preparation.json',{'status':'embedding checkpoints complete','unique_pairs':len(pairs),'unique_texts':len(texts),'input_splits':[args.split],'final_read':False})
        print('PREPARATION COMPLETE',flush=True)
        return
    designated={str(seed):[] for seed in cfg['seeds']}
    for key in ['quora','redis_reranker']+[str(seed) for seed in cfg['seeds']]:
        if args.model_key and key!=args.model_key:continue
        begin=time.perf_counter()
        checkpoint=args.output_dir/'checkpoints'/('pairs-'+key+'.npz')
        bound=binding(key,pairs,'pairs')
        scores,state=load_checkpoint(checkpoint,bound)
        if len(scores)>len(pairs):raise ValueError('pair checkpoint too long')
        model=None
        if len(scores)<len(pairs):
            if key.isdigit():
                seed=int(key);adapter=ROOT/'artifacts/stage_b/models'/key/'adapter.safetensors';identity(adapter)
                receipt=json.loads(adapter.with_name('receipt.json').read_text())
                if identity(adapter,False)['sha256']!=receipt['export']['sha256']:raise ValueError('adapter hash mismatch')
                model,tok=load_adapter(paths['quora'],cfg,seed,adapter)
            else:
                tok=AutoTokenizer.from_pretrained(str(paths[key]),local_files_only=True,trust_remote_code=False)
                model=AutoModelForSequenceClassification.from_pretrained(str(paths[key]),local_files_only=True,
                    trust_remote_code=False,use_safetensors=True,attn_implementation='eager').float().eval()
            state['load_seconds'].append(time.perf_counter()-begin)
        for start in range(len(scores),len(pairs),32):
            t=time.perf_counter();ps=pairs[start:start+32]
            batch=tokenize_pairs(tok,[x[0] for x in ps],[x[1] for x in ps],cfg['max_tokens'])
            with torch.inference_mode():
                logits=model(**batch).logits
                probabilities=torch.softmax(logits,dim=-1) if key.isdigit() else torch.sigmoid(logits)
            scores.extend(probabilities.tolist());state['batch_ms'].append((time.perf_counter()-t)*1000)
            save_checkpoint(checkpoint,bound,scores,state)
            if start%512==0:print(json.dumps({'split':args.split,'pair_model':key,'done':start,'total':len(pairs)}),flush=True)
        timings[key]=timing_view(state,len(pairs),'pairs')
        # Preserve COMPLETE raw model outputs independently of transient caches,
        # including designated diagnostic pairs not in the retrieved pool.
        rawpath=args.output_dir/('raw-model-'+key+'.json.gz')
        payload=gzip.compress(json.dumps({'evidence_type':'measured_local_cpu_scores_on_synthetic_inputs_not_human_judgment',
            'binding':bound,'ordered_pairs':pairs,'probabilities':scores},sort_keys=True).encode(),mtime=0)
        if rawpath.exists() and rawpath.read_bytes()!=payload:
            raise ValueError('STOP: complete raw model output differs; preserve prior evidence')
        if not rawpath.exists():rawpath.write_bytes(payload)
        for r in records:
            for c in r['candidates']:
                if r['exact_candidate'] is not None:continue
                score=scores[pairid[(c['cached_question'],r['question'])]]
                if key.isdigit():
                    predicted=cfg['classes'][int(np.argmax(score))]
                    c['classifier'][key]={'probabilities':score,'predicted_class':predicted}
                else:c[key+'_score']=score[0]
            if key.isdigit() and r['exact_candidate'] is None:
                score=scores[pairid[(r['designated_cached_question'],r['question'])]]
                designated[key].append({'case_id':r['case_id'],'label':r['relation_to_cached'],
                    'predicted':cfg['classes'][int(np.argmax(score))],'probabilities':score})
        # Actual unbatched latency probe: first 5 nonexact requests, cold first
        # forward after load versus 2 warm repeats. Embedding/search measured in
        # a separate dedicated pipeline probe at B5; these are classifier-only.
        if key==str(cfg['seeds'][0]) and model is not None:
            samples=[r for r in records if r['exact_candidate'] is None][:5]
            probes=[]
            for repeat in range(2):
                for r in samples:
                    t=time.perf_counter();cs=r['candidates']
                    batch=tokenize_pairs(tok,[c['cached_question'] for c in cs],[r['question']]*len(cs),cfg['max_tokens'])
                    with torch.inference_mode():model(**batch)
                    probes.append({'case_id':r['case_id'],'repeat':repeat,'top_k':len(cs),'classifier_ms':(time.perf_counter()-t)*1000})
            timings[key]['warm_unbatched_topk_probes']=probes
        del model;gc.collect()
        print(json.dumps({'finished_model':key,'seconds':timings[key]['total_seconds']}),flush=True)
    if args.model_key:
        write_json(args.output_dir/('worker-'+args.model_key+'.json'),{'evidence_type':'local_raw_inference_completeness_not_accuracy',
            'key':args.model_key,'pairs':len(pairs),'timing':timings[args.model_key],
            'checkpoint':identity(checkpoint,False),'final_read':False,'remote_inference_calls':0})
        return
    # Exact records bypass model evaluation and are not used in semantic metrics.
    raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records).encode()
    path=args.output_dir/'scored.jsonl.gz';path.write_bytes(gzip.compress(raw,mtime=0))
    write_json(args.output_dir/'designated-classification.json',{'evidence_type':'synthetic_template_held_out_diagnostic',
        'predictions':designated,'no_threshold_selected':True})
    write_json(args.output_dir/'receipt.json',{'evidence_type':'local_cpu_actual_retrieved_candidate_measurement_synthetic',
        'split':args.split,'config_identity':identity(ROOT/'configs/intent_stage_b_v1.json',False),
        'hardware':hardware(),'timings':timings,'scored_identity':identity(path,False),
        'execution':'resumable atomic numeric checkpoints; per-model workers may run concurrently; standalone latency probe is separate',
        'complete_raw_model_outputs':[identity(args.output_dir/('raw-model-'+key+'.json.gz'),False)
            for key in ['quora','redis_reranker']+[str(seed) for seed in cfg['seeds']]],
        'unique_scored_pairs':len(pairs),'queries':len(records),'cache_entries':len(bank),
        'scope':'identical trusted scope only','public_pairs_used':False,'provenance_flags':[FLAG],
        'remote_inference_calls':0,'socket_connections_blocked':True,'final_read':False,
        'serving_enabled':False,'actual_relations':'evaluation-only synthetic contract oracle; not policy input',
        'baseline_candidate_pool':'same MiniLM top-3 pool for all policies; Redis cosine is a threshold comparator, not Redis full-bank retrieval'})
    print(json.dumps({'saved':str(path),'queries':len(records),'unique_pairs':len(pairs)}),flush=True)


if __name__=='__main__':main()
