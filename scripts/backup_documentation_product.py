"""Package committed product history and the opaque learned-model backup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    if args.output.exists():parser.error('refusing to overwrite a backup')
    if git('status','--porcelain'):parser.error('commit the coherent milestone first; working tree must be clean')
    stage_b=root/'artifacts/stage_b/stage-b-complete-backup.zip'
    if not stage_b.is_file():parser.error('opaque Stage B backup missing; learned exports would not be preserved')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        bundle=Path(temporary)/'project.bundle'
        subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=root,check=True)
        subprocess.run(['git','bundle','verify',str(bundle)],cwd=root,check=True,capture_output=True)
        files={'project.bundle':bundle,'stage-b-complete-backup.zip':stage_b}
        manifest={'git_head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
            'files':{name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in files.items()},
            'excludes':['credentials','Python environment','pretrained checkpoints','application databases'],
            'final_records':'opaque preserved Stage B archive; not opened or evaluated'}
        with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name,p in files.items():z.write(p,name)
            z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
            z.writestr('RESTORE.md','Verify MANIFEST.json hashes and git bundle verify project.bundle first.\n'
                'Restore into a new directory: git clone -b '+manifest['branch']+' project.bundle restored-project\n'
                'The Git bundle preserves source, tests, evidence, reports and prior local branch history.\n'
                'The opaque Stage B archive preserves ignored learned exports and excluded final records.\n'
                'Do not evaluate final records. No pretrained weights or environment are included.\n'
                'Use the active report for reproduction commands; no performance claims beyond its evidence.\n')
        with zipfile.ZipFile(args.output) as z:
            assert z.testzip() is None
            for name,info in manifest['files'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==info['sha256']
    print(json.dumps({'file':str(args.output.resolve()),'bytes':args.output.stat().st_size,
                      'sha256':sha(args.output),'git_head':manifest['git_head']},indent=2))


if __name__=='__main__':main()
