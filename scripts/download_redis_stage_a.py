"""Approved Redis baseline snapshots only. No pickle, model loading or inference."""
import argparse,hashlib,json,platform,sys,urllib.request
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download,hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE

MODELS=[('redis/langcache-embed-v1','2f6ffda359ab94a9c222947bfd87ea8f85ea7e0a'),('redis/langcache-reranker-v2-bce','f59539f4208f9005e10e45160a379441bdb42e9b')]
ALLOWED={'model.safetensors','config.json','config_sentence_transformers.json','sentence_bert_config.json','modules.json','1_Pooling/config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','vocab.txt','merges.txt'}
FLAG='possible pretraining overlap with public pair data'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():ap.error('receipt exists; choose a fresh path')
 api=HfApi();plans=[]
 for repo,rev in MODELS:
  info=api.model_info(repo,revision=rev);current=api.model_info(repo)
  assert info.sha==rev,'STOP: revision mismatch'
  names={s.rfilename for s in info.siblings}
  assert 'model.safetensors' in names,'STOP: no safetensors; no pickle fallback'
  # Model card read as metadata only; README is not downloaded into the snapshot.
  card=urllib.request.urlopen(hf_hub_url(repo,'README.md',revision=rev)).read()
  license_id=getattr(info.card_data,'license',None)
  plans.append((repo,rev,sorted(names&ALLOWED),current.sha,license_id,hashlib.sha256(card).hexdigest()))
 related=[]
 for m in api.list_models(author='redis',search='langcache',full=True):
  related.append({'repo_id':m.id,'main_revision':m.sha,'created_at':str(m.created_at),'last_modified':str(m.last_modified)})
 result={'evidence_type':'approved_artifact_download_not_inference','trust_remote_code':False,'pickle_downloaded':False,'model_loading':'not_run','cache_root':str(HF_HUB_CACHE),'related_hub_models_metadata_only':related,'models':[]}
 for repo,rev,names,current,license_id,card_sha in plans:
  snap=Path(snapshot_download(repo,revision=rev,cache_dir=HF_HUB_CACHE,allow_patterns=names,max_workers=4));assert snap.name==rev
  files=[]
  for name in names:
   p=snap/name;assert p.is_file()
   row={'path':str(p),'relative_path':name,'bytes':p.stat().st_size,'sha256':sha(p),'model_card_license':license_id,'license_status':'declared' if license_id else 'not_declared_in_pinned_card_metadata'}
   files.append(row);print(json.dumps(row),flush=True)
  result['models'].append({'repo_id':repo,'revision':rev,'current_main':current,'revision_is_current_main':rev==current,'snapshot':str(snap),'model_card_license':license_id,'license_status':'declared' if license_id else 'not_declared_in_pinned_card_metadata','model_card_sha256':card_sha,'model_card_url':hf_hub_url(repo,'README.md',revision=rev),'provenance_flags':[FLAG],'files':files})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'receipt':str(a.output),'models':[m['repo_id'] for m in result['models']]}))
if __name__=='__main__':main()
