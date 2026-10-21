"""Download only the two explicitly approved Stage A revisions; no model loading."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
APPROVED = (
    ('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41','ENCODER_MODEL','ENCODER_REVISION'),
    ('cross-encoder/quora-distilroberta-base','f62e7a4b20b97195c2868e53ec59126df5eac743','PAIR_MODEL','PAIR_REVISION'),
)
FILES = {'README.md','LICENSE','LICENSE.txt','LICENSE.md','NOTICE','config.json',
    'config_sentence_transformers.json','sentence_bert_config.json','modules.json',
    '1_Pooling/config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
    'added_tokens.json','vocab.txt','vocab.json','merges.txt','model.safetensors'}


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('receipt exists; preserve it and choose a fresh output path')
    source=ROOT/'src/policy/local_models.py'
    if not source.is_file():
        raise SystemExit('STOP: missing '+str(source))
    source_hash=sha256(source)
    print(json.dumps({'path':str(source),'bytes':source.stat().st_size,'sha256':source_hash}),flush=True)
    constants={}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
            for target in node.targets:
                if isinstance(target,ast.Name):constants[target.id]=node.value.value
    from huggingface_hub import HfApi,snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    cache=Path(HF_HUB_CACHE)
    api=HfApi()
    plans=[]
    # Validate BOTH approved revisions before any artifact download. No fallback.
    for repo,revision,model_key,revision_key in APPROVED:
        if constants.get(model_key)!=repo or constants.get(revision_key)!=revision:
            raise SystemExit('STOP: approved identity differs from repository constants')
        info=api.model_info(repo,revision=revision)
        if info.sha!=revision:
            raise SystemExit('STOP: revision did not resolve exactly: '+repo)
        available={x.rfilename for x in info.siblings}
        if not {'model.safetensors','README.md'} <= available:
            raise SystemExit('STOP: safetensors or model card missing; no pickle fallback')
        plans.append((repo,revision,sorted(FILES & available),revision_key))
    receipt={'evidence_type':'approved_checkpoint_download_not_inference',
        'cache_root':str(cache),'trust_remote_code':False,'model_code_executed':False,
        'runtime':{'workspace':str(ROOT),'python':sys.executable,'python_version':platform.python_version(),
            'machine':platform.machine(),'hostname_sha256':hashlib.sha256(platform.node().encode()).hexdigest()},
        'models':[]}
    for repo,revision,names,revision_key in plans:
        snapshot=Path(snapshot_download(repo_id=repo,revision=revision,cache_dir=str(cache),
            allow_patterns=names,max_workers=4))
        if snapshot.name!=revision:
            raise SystemExit('STOP: unexpected snapshot directory')
        rows=[]
        for name in names:
            path=snapshot/name
            if not path.is_file():raise SystemExit('STOP: missing downloaded file '+str(path))
            row={'relative_path':name,'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path)}
            rows.append(row);print(json.dumps(row),flush=True)
        card=(snapshot/'README.md').read_text()
        frontmatter=card.split('---',2)[1] if card.startswith('---') else ''
        match=re.search(r'^license:\s*[\'\"]?([A-Za-z0-9_.-]+)',frontmatter,re.M)
        if not match:raise SystemExit('STOP: cannot identify license from pinned model card')
        license_id=match.group(1)
        for row in rows:
            row['model_card_license']=license_id
        receipt['models'].append({'repo_id':repo,'revision':revision,'snapshot':str(snapshot),
            'revision_source':{'kind':'repository_file','path':str(source),'constant':revision_key,'sha256':source_hash},
            'license':license_id,'license_source':str(snapshot/'README.md'),
            'license_source_sha256':sha256(snapshot/'README.md'),'files':rows,
            'pickle_weights_downloaded':False,'inference':'not_run'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'downloaded_revisions':[m['revision'] for m in receipt['models']],
                      'receipt':str(args.output),'cache_root':str(cache)},indent=2))


if __name__=='__main__':
    main()
