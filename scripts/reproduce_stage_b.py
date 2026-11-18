"""Prepare an isolated LOCAL clone; optionally rerun B0-B5 offline, never final."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    target=args.workspace.resolve()
    if target.exists() or ROOT in target.parents:
        parser.error('Choose a NEW directory outside this repository')
    subprocess.run(['git','clone','--no-hardlinks','--branch','stage-b/classifier',str(ROOT),str(target)],check=True)
    # Delete generated output ONLY inside the freshly created isolated clone.
    generated=target/'artifacts/stage_b'
    generated.resolve().relative_to(target)
    shutil.rmtree(generated)
    generated.mkdir(parents=True)
    environment=dict(os.environ)
    dependency_path=str(ROOT/'.venv/lib/python3.12/site-packages')
    environment['PYTHONPATH']=dependency_path
    commands=[
        ['scripts/build_stage_b_intents.py','--output-dir','scratch/rebuilt-stage-b'],
        ['scripts/train_stage_b_classifier.py'],
        ['scripts/diagnose_stage_b_train.py'],
        ['scripts/review_stage_b_veto.py'],
        ['scripts/score_stage_b_candidates.py','--split','dev','--output-dir','artifacts/stage_b/dev'],
        ['scripts/report_stage_b_dev.py','--split','dev'],
        ['scripts/run_stage_b_scoring.py','--split','calibration','--output-dir','artifacts/stage_b/calibration'],
        ['scripts/calibrate_stage_b_policy.py'],
        ['scripts/report_stage_b_dev.py','--split','calibration'],
        ['scripts/benchmark_stage_b_local.py'],
        ['scripts/report_stage_b_final.py'],
        ['-m','pytest','-q','tests/test_intent_stage_a.py','tests/test_intent_stage_b.py'],
    ]
    # Non-model protocol metadata remains unchanged; copying is not a download.
    for name in ['runtime-lock.json','public-data-proposal.json','training-v1.json']:
        if name=='training-v1.json':continue  # must be produced by real training
        shutil.copyfile(ROOT/'artifacts/stage_b'/name,generated/name)
    (target/'REPRODUCTION_COMMANDS.json').write_text(json.dumps({
        'python':sys.executable,'dependency_path':dependency_path,'commands':commands,
        'pretrained_cache':'already-approved exact snapshots required; no automatic download',
        'final_evaluation':False,'serving_enabled':False},indent=2)+'\n')
    print(json.dumps({'clone':str(target),'execute':args.execute,'commands':commands},indent=2))
    if not args.execute:return
    for index,command in enumerate(commands):
        subprocess.run([sys.executable]+command,cwd=target,env=environment,check=True)
        if index==0:
            expected=target/'data/intent_stage_b/v2'
            actual=target/'scratch/rebuilt-stage-b'
            for path in sorted(expected.glob('*')):
                if hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256((actual/path.name).read_bytes()).digest():
                    raise ValueError('STOP: B0 regeneration mismatch '+path.name)
            print('B0 v2 byte hashes reproduce; no final data read.',flush=True)


if __name__=='__main__':main()
