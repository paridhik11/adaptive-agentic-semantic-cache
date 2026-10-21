"""Package all uncommitted Stage A work and data; no secrets, weights or Git writes."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 archive=ROOT/'artifacts/stage_a/stage-a-backup.zip'
 if archive.exists():raise SystemExit('STOP: backup exists; preserve it')
 changed=subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
 files={ROOT/p for p in changed if p}
 # Include explicitly saved logs even when a global *.log ignore rule applies.
 files.update(p for p in (ROOT/'artifacts/stage_a').rglob('*') if p.is_file())
 files.update(p for p in (ROOT/'data/intent_stage_a').rglob('*') if p.is_file())
 files={p for p in files if p.is_file() and p!=archive and p.name!='BACKUP_MANIFEST.json'}
 for p in files:
  if p.suffix in {'.safetensors','.bin','.joblib','.pkl','.pt','.pth'} or p.name.startswith('.env'):
   raise SystemExit('STOP: unexpected weight/secret candidate '+str(p))
 entries=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 manifest={'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'restore':'Clone repository at base_git_commit, extract this archive over its root. Keep final locked; do not evaluate it. Install runtime-summary versions. Re-download only approved pins using included scripts if weights are absent.',
  'model_weights_included':False,'trained_models_exist':False,
  'pretrained_weight_restore_scripts':['scripts/download_stage_a_checkpoints.py','scripts/download_redis_stage_a.py'],
  'files':entries}
 mp=ROOT/'artifacts/stage_a/BACKUP_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for e in entries:z.write(ROOT/e['path'],e['path'])
  z.write(mp,str(mp.relative_to(ROOT)))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for e in entries:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(entries),'manifest':str(mp)},indent=2))
if __name__=='__main__':main()
