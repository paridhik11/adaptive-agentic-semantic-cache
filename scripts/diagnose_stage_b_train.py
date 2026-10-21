"""Final train-only classification diagnostics for all fixed seeds (no tuning)."""
import gc
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import config,read_split,snapshot,offline,identity,write_json


def main():
    cfg=config();data=read_split('train')
    cached={r['cached_case_id']:r for r in data if r['role']=='cached'}
    rows=[r for r in data if r['role']=='new' and not r['is_exact'] and r['relation_to_cached'] is not None]
    base=snapshot('quora');offline()
    import torch
    from src.evaluation.intent_stage_b_model import load_adapter,tokenize_pairs
    torch.set_num_threads(cfg['torch_threads'])
    out={}
    for seed in cfg['seeds']:
        path=ROOT/'artifacts/stage_b/models'/str(seed)/'adapter.safetensors';identity(path)
        model,tok=load_adapter(base,cfg,seed,path);predictions=[];start=time.perf_counter()
        for begin in range(0,len(rows),32):
            rs=rows[begin:begin+32]
            batch=tokenize_pairs(tok,[cached[r['cached_case_id']]['question'] for r in rs],[r['question'] for r in rs],cfg['max_tokens'])
            with torch.inference_mode():probs=model(**batch).logits.softmax(-1).tolist()
            for r,p in zip(rs,probs):
                predictions.append({'case_id':r['case_id'],'label':r['relation_to_cached'],'predicted':cfg['classes'][max(range(4),key=lambda i:p[i])],'probabilities':p})
        out[str(seed)]={'n':len(rows),'accuracy':sum(r['label']==r['predicted'] for r in predictions)/len(rows),'predictions':predictions,'seconds':time.perf_counter()-start}
        print(seed,out[str(seed)]['accuracy'],flush=True)
        del model;gc.collect()
    write_json(ROOT/'artifacts/stage_b/train-classification-v1.json',{'evidence_type':'train_set_fit_diagnostic_not_generalization','seeds':out,'final_read':False})


if __name__=='__main__':main()
