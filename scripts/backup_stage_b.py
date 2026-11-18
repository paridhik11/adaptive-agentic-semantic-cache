"""Retrievable local bundle: commits, working evidence, final stores and learned tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite a backup')
    paths=set()
    for name in ['artifacts/stage_a','artifacts/stage_b','data/intent_stage_a','data/intent_stage_b']:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and p.suffix not in {'.zip','.bundle','.pyc'} and '__pycache__' not in p.parts and 'checkpoints' not in p.parts:
                paths.add(p)
    for pattern in ['*stage_a*.py','*stage_b*.py','inventory_intent_artifacts.py']:
        paths.update((ROOT/'scripts').glob(pattern))
    for pattern in ['STAGE_A*','STAGE_B*']:
        paths.update((ROOT/'docs/audit').glob(pattern))
    for directory,pattern in [('src/evaluation','intent_stage_*.py'),('tests','test_intent_stage_*.py'),('configs','intent_stage_*.json')]:
        paths.update((ROOT/directory).glob(pattern))
    paths.update(ROOT/p for p in ['.gitattributes','.gitignore','docs/FACTS.md','docs/CURRENT_STATUS.md',
        'docs/STAGE_A_PROTOCOL.md','docs/STAGE_B_PROTOCOL.md','docs/STAGE_B_NATURAL_REVIEW.md','docs/PUBLIC_DATA_APPROVAL_STAGE_B.md'])
    for path in paths:
        if path.suffix in {'.bin','.pkl','.pickle'}:raise ValueError('pickle is not a valid backup input')
        if path.suffix=='.safetensors' and ROOT/'artifacts/stage_b/models' not in path.parents:
            raise ValueError('Only learned Stage B exports may be included, never pretrained caches')
    manifest={'schema':'stage-b-complete-backup-v1','evidence_type':'backup_hashes_not_model_evaluation',
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'files':[],'includes_pretrained_weights':False,'includes_python_environment':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='stage-b-backup-') as temporary:
        bundle=Path(temporary)/'stage-b-local.bundle'
        subprocess.run(['git','bundle','create',str(bundle),'stage-a/harness','stage-a/data','stage-a/results','stage-b/classifier'],cwd=ROOT,check=True)
        subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,capture_output=True)
        with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for path in sorted(paths):
                # Archive binary final bytes without parsing, displaying or evaluating.
                data=path.read_bytes();name=str(path.relative_to(ROOT))
                archive.writestr(name,data)
                manifest['files'].append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            data=bundle.read_bytes();archive.writestr('stage-b-local.bundle',data)
            manifest['files'].append({'path':'stage-b-local.bundle','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            archive.writestr('BACKUP_FILES.json',json.dumps(manifest,indent=2)+'\n')
            archive.writestr('RESTORE.md', '''# Restore\n\n1. Extract this ZIP into a holding directory.\n2. Run `git clone -b stage-b/classifier stage-b-local.bundle restored-repo`.\n3. Copy the archive's artifacts/, data/, scripts/, src/, tests/, configs/, docs/\n   into the same relative paths in restored-repo. The Git bundle preserves all\n   committed files; this overlay also restores ignored learned tensors, raw\n   decision logs and both excluded Stage A final stores. Do not inspect/evaluate\n   final records. Restore .gitattributes/.gitignore too if not cloning the bundle.\n4. Verify every file using BACKUP_FILES.json (SHA-256, byte count).\n5. Restore the exact approved pretrained snapshots with the pinned download\n   scripts, and install the pinned runtime versions in runtime-lock.json.\n   Pretrained cache/environment are intentionally not in this archive.\n6. Load each adapter with src.evaluation.intent_stage_b_model.load_adapter using\n   its receipt's exact base pin, config, seed and safetensors path.\n\nNo push, main merge, final evaluation or serving enablement is authorized.\n''')
    with zipfile.ZipFile(args.output) as archive:
        for entry in manifest['files']:
            data=archive.read(entry['path'])
            assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],entry['path']
    print(json.dumps({'path':str(args.output.resolve()),'bytes':args.output.stat().st_size,
        'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'verified_files':len(manifest['files']),
        'git_head':manifest['head'],'includes_all_three_learned_exports':True},indent=2))


if __name__=='__main__':main()
