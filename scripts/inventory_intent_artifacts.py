"""Stage A1 local inventory. Standard library only; never downloads or loads models."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('AGENTS.md','docs/CURRENT_STATUS.md','docs/FACTS.md',
    'docs/DAY3_RESEARCH_AND_PROTOCOL.md','docs/SEMANTIC_EVALUATION_CONTRACT.md',
    'src/policy/encoders.py','src/policy/semantic_runtime.py',
    'src/policy/local_models.py','src/decision/local_equivalence.py',
    'scripts/inventory_intent_artifacts.py')
WEIGHT_SUFFIXES = {'.safetensors','.bin','.pt','.pth','.onnx','.joblib','.pkl'}


def inspect(path):
    path = Path(path)
    if not path.is_file():
        row = {'path':str(path),'status':'missing'}
    else:
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                sha.update(block)
        row = {'path':str(path),'status':'present','bytes':path.stat().st_size,'sha256':sha.hexdigest()}
    print(json.dumps(row),flush=True)
    return row


def constants(path):
    # Called only after inspect() printed the source identity. No module import.
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and isinstance(node.value,ast.Constant):
                    result[target.id] = node.value.value
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and isinstance(node.value,ast.Constant):
            result[node.target.id] = node.value.value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--download-receipt',type=Path)
    parser.add_argument('--extra-receipt',type=Path)
    parser.add_argument('--replace-existing',action='store_true',
                        help='Explicitly update inventory while retaining its previous SHA-256')
    args = parser.parse_args()
    previous_sha = None
    if args.output and args.output.exists() and not args.replace_existing:
        parser.error('output exists; use a fresh path to preserve inventory evidence')
    if args.output and args.output.exists():
        previous_sha = inspect(args.output)['sha256']
    source = [inspect(ROOT/name) for name in SOURCE_FILES]
    missing_files = [r['path'] for r in source if r['status']=='missing']
    if missing_files:
        report = {'stage':'A1','status':'blocked','missing_source_files':missing_files,'sources':source}
    else:
        local = constants(ROOT/'src/policy/local_models.py')
        equivalence = constants(ROOT/'src/decision/local_equivalence.py')
        specs = [
            ('pinned_dense_encoder',local['ENCODER_MODEL'],local['ENCODER_REVISION']),
            ('off_the_shelf_quora',local['PAIR_MODEL'],local['PAIR_REVISION']),
            ('local_equivalence_default',equivalence['DEFAULT_CROSS_ENCODER_MODEL'],equivalence['DEFAULT_CROSS_ENCODER_REVISION']),
            ('local_equivalence_ms_marco_option',equivalence['MS_MARCO_CROSS_ENCODER_MODEL'],equivalence['MS_MARCO_CROSS_ENCODER_REVISION']),
        ]
        hf_home = Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))
        roots = {hf_home/'hub'}
        for key in ('HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','SENTENCE_TRANSFORMERS_HOME'):
            if os.environ.get(key):
                roots.add(Path(os.environ[key]))
        models = []
        for role,repo,revision in specs:
            snapshots = [root/('models--'+repo.replace('/','--'))/'snapshots'/revision for root in sorted(roots)]
            files = [p for snapshot in snapshots if snapshot.is_dir() for p in sorted(snapshot.rglob('*'))
                     if p.is_file()]
            rows = [inspect(p) for p in files]
            card = next((p for p in files if p.name=='README.md' and p.parent.name==revision),None)
            license_id = None
            if card:
                text = card.read_text()
                frontmatter = text.split('---',2)[1] if text.startswith('---') else ''
                match = re.search(r'^license:\s*[\'\"]?([A-Za-z0-9_.-]+)',frontmatter,re.M)
                if match:
                    license_id = match.group(1)
            for row in rows:
                row['model_card_license'] = license_id
            weights = [r for r in rows if Path(r['path']).suffix in WEIGHT_SUFFIXES]
            models.append({'role':role,'repo_id':repo,'revision':revision,
                'status':'weight_files_found_not_loaded' if weights else 'missing_weights',
                'model_card_license':license_id,'license_source':str(card) if card else None,
                'searched_snapshots':[str(p) for p in snapshots],'files':rows})
        project = [inspect(p) for p in sorted((ROOT/'models').rglob('*'))
                   if p.is_file() and p.suffix in WEIGHT_SUFFIXES] if (ROOT/'models').is_dir() else []
        other = []
        for root in sorted(roots):
            for model in sorted(root.glob('models--*')):
                files = [inspect(p) for p in sorted(model.glob('snapshots/*/*'))
                         if p.is_file() and p.suffix in WEIGHT_SUFFIXES]
                if files:
                    other.append({'directory':str(model),'files':files})
        redis = [r for r in other if 'langcache' in r['directory'].lower()]
        report = {'stage':'A1','evidence_type':'local_artifact_inventory_not_inference',
            'status':'blocked_missing_required_dense_weights' if models[0]['status']=='missing_weights' else 'inventory_only',
            'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'sources':source,'cache_roots':[str(p) for p in sorted(roots)],'models':models,
            'project_weight_files':project,'other_cached_model_weights':other,
            'redis_langcache':{'status':'local_artifacts_found_identity_not_validated' if redis else 'not_found_no_local_pinned_identity','artifacts':redis},
            'lexical_substitution_used':False,'model_inference':'not_run','downloads':0,
            'dataset_construction':'not_run','baseline_evaluation':'not_run','git_mutations':False}
    report['runtime'] = {'workspace':str(ROOT),'python':sys.executable,
        'python_version':platform.python_version(),'machine':platform.machine(),
        'hostname_sha256':hashlib.sha256(platform.node().encode()).hexdigest()}
    report['previous_inventory_sha256'] = previous_sha
    if args.download_receipt:
        receipt_identity = inspect(args.download_receipt)
        if receipt_identity['status']!='present':
            raise SystemExit('STOP: missing download receipt')
        receipt = json.loads(args.download_receipt.read_text())
        report['download_receipt'] = {'identity':receipt_identity,'record':receipt}
        report['downloads'] = len(receipt['models'])
        report['same_download_inventory_environment'] = receipt['runtime']==report['runtime']
    flag='possible pretraining overlap with public pair data'
    for model in report.get('models',[]):
        if model['repo_id']=='cross-encoder/quora-distilroberta-base':
            model['provenance_flags']=[flag]
    if args.extra_receipt:
        identity=inspect(args.extra_receipt)
        if identity['status']!='present':raise SystemExit('STOP: missing extra receipt')
        extra=json.loads(args.extra_receipt.read_text())
        report['extra_download_receipt']={'identity':identity,'record':extra}
        for model in extra['models']:
            for row in model['files']:
                actual=inspect(Path(row['path']))
                if actual['status']!='present' or actual['sha256']!=row['sha256']:
                    raise SystemExit('STOP: Redis receipt/artifact mismatch')
            report['models'].append({**model,'status':'weight_files_verified_not_loaded',
                'provenance_flags':[flag]})
        report['redis_langcache']['status']='approved_pinned_artifacts_verified_not_loaded'
        report['redis_langcache']['provenance_flags']=[flag]
        report['downloads']+=len(extra['models'])
    worktree = ROOT.resolve()
    roots = [Path(p).resolve() for p in report.get('cache_roots',[])]
    outside = all(not path.is_relative_to(worktree) for path in roots)
    ignored = subprocess.run(['git','check-ignore','--no-index','--verbose',
        'models/_stage_a_ignore_probe.safetensors'],cwd=ROOT,capture_output=True,text=True)
    report['git_storage'] = {'cache_roots_outside_worktree':outside,
        'cache_directory_gitignore_status':'not_applicable_outside_worktree' if outside else 'requires_verification',
        'repository_safetensors_ignore_rule':ignored.stdout.strip(),
        'repository_safetensors_rule_verified':ignored.returncode==0}
    print(json.dumps(report,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('w' if args.replace_existing else 'x',encoding='utf-8') as stream:
            stream.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
