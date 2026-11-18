"""B5 local stage timings, explicitly excluding application/provider claims."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import config,read_split,snapshot,offline,identity,hardware,write_json


def main():
    cfg=config();data=read_split('dev')
    bank=[r for r in data if r['role']=='cached']
    queries=[r for r in data if r['role']=='new' and not r['is_exact']][:5]
    mini=snapshot('minilm');base=snapshot('quora')
    offline()
    import torch
    from transformers import AutoTokenizer,AutoModel
    from src.evaluation.intent_stage_b_model import load_adapter,tokenize_pairs
    from src.evaluation.intent_stage_b_veto import veto
    torch.set_num_threads(cfg['torch_threads'])
    begin=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(str(mini),local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_pretrained(str(mini),local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager').float().eval()
    embedding_load_ms=(time.perf_counter()-begin)*1000

    def encode(texts):
        batch=tok(texts,padding=True,truncation=False,return_tensors='pt')
        if batch['input_ids'].shape[-1]>cfg['max_tokens']:raise ValueError('overlength')
        with torch.inference_mode():
            h=encoder(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            return torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=1)
    begin=time.perf_counter();bankvectors=encode([r['question'] for r in bank]);bank_build_ms=(time.perf_counter()-begin)*1000
    seed=cfg['seeds'][0];adapter=ROOT/'artifacts/stage_b/models'/str(seed)/'adapter.safetensors';identity(adapter)
    begin=time.perf_counter();model,pairtok=load_adapter(base,cfg,seed,adapter);classifier_load_ms=(time.perf_counter()-begin)*1000
    observations=[]
    for repeat in range(3):
        for index,r in enumerate(queries):
            t=time.perf_counter();v=encode([r['question']]);embedding_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();sims=(bankvectors@v[0]).tolist()
            ranked=sorted([i for i,c in enumerate(bank) if c['scope_id']==r['scope_id']],key=lambda i:(-sims[i],bank[i]['cached_case_id']))[:cfg['top_k']]
            search_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();batch=tokenize_pairs(pairtok,[bank[i]['question'] for i in ranked],[r['question']]*len(ranked),cfg['max_tokens'])
            with torch.inference_mode():model(**batch)
            classifier_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            for i in ranked:veto(bank[i]['question'],r['question'])
            veto_ms=(time.perf_counter()-t)*1000
            observations.append({'case_id':r['case_id'],'repeat':repeat,
                'state':'first_classifier_forward' if repeat==0 and index==0 else 'warm',
                'embedding_ms':embedding_ms,'search_ms':search_ms,'classifier_ms':classifier_ms,'veto_ms':veto_ms,
                'top_k':len(ranked),'cache_entries':len(bank)})
    write_json(ROOT/'artifacts/stage_b/local-latency-v1.json',{
        'evidence_type':'local_cpu_component_timing_on_synthetic_dev_not_application_latency',
        'hardware':hardware(),'seed':seed,'model_loading_cold':{'embedding_load_ms':embedding_load_ms,
            'classifier_load_ms':classifier_load_ms,'bank_build_ms':bank_build_ms},
        'cold_definition':'fresh model objects in a new process; OS disk/page cache not controlled, tokenizer/library imports may be warm; bank embedding precedes first query',
        'observations':observations,'samples':'first five nonexact dev records, three repetitions; dependent diagnostic timing observations',
        'remote_calls':0,'final_read':False,'serving_enabled':False})
    print(json.dumps({'saved':'artifacts/stage_b/local-latency-v1.json','observations':len(observations)}))


if __name__=='__main__':main()
