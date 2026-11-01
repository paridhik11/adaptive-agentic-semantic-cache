"""Check exact approved local checkpoints. No downloads, inference or fallback."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def inspect(cache):
    specs=[('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41'),
           ('cross-encoder/quora-distilroberta-base','f62e7a4b20b97195c2868e53ec59126df5eac743')]
    models=[]
    for model,revision in specs:
        snapshot=cache/('models--'+model.replace('/','--'))/'snapshots'/revision
        files={name:(snapshot/name).is_file() for name in ['config.json','model.safetensors','tokenizer.json']}
        tokenizer=files['tokenizer.json'] or all((snapshot/name).is_file() for name in ['vocab.json','merges.txt']) or (snapshot/'vocab.txt').is_file()
        models.append({'model':model,'revision':revision,'snapshot':str(snapshot),'files_present':files,
                       'ready':files['config.json'] and files['model.safetensors'] and tokenizer,
                       'revision_origin':'approved user pins, checked against src/policy/local_models.py'})
    return {'status':'READY_FOR_LOCAL_LOAD_CHECK' if all(m['ready'] for m in models) else 'BLOCKED_MISSING_APPROVED_CHECKPOINTS',
            'models':models,'downloads':0,'inference_calls':0,'pickle_loaded':False,
            'meaning':'File-presence preflight only; does not establish model loadability or semantic quality.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('refusing to overwrite evidence')
    source=Path(__file__).resolve().parents[1]/'src/policy/local_models.py'
    values=[n.value.value for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)]
    for pin in ['1110a243fdf4706b3f48f1d95db1a4f5529b4d41','f62e7a4b20b97195c2868e53ec59126df5eac743']:
        assert pin in values,'approved pin differs from repository source'
    if a.cache is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        a.cache=Path(HF_HUB_CACHE)
    result=inspect(a.cache);result['pin_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='READY_FOR_LOCAL_LOAD_CHECK' else 2


if __name__=='__main__':sys.exit(main())
