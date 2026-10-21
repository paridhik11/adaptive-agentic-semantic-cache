"""Real local A4 designated-pair baselines. Refuses final before any file/model use."""
import argparse,gc,hashlib,json,math,os,platform,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
FLAG='possible pretraining overlap with public pair data'
SPECS={
 'minilm':('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41'),
 'quora':('cross-encoder/quora-distilroberta-base','f62e7a4b20b97195c2868e53ec59126df5eac743'),
 'redis_embedding':('redis/langcache-embed-v1','2f6ffda359ab94a9c222947bfd87ea8f85ea7e0a'),
 'redis_reranker':('redis/langcache-reranker-v2-bce','f59539f4208f9005e10e45160a379441bdb42e9b')}
def audit(p):
 p=Path(p)
 if not p.is_file():raise FileNotFoundError('STOP: '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 item={'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest()};print(json.dumps(item),flush=True);return item

def readlines(p):audit(p);return [json.loads(l) for l in p.read_text().splitlines()]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--split',choices=['train','dev','calibration'],default='dev');ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 if a.output_dir.exists():ap.error('output exists; preserve evidence and choose a fresh directory')
 cfgpath=ROOT/'configs/intent_stage_a.json';cfgid=audit(cfgpath);cfg=json.loads(cfgpath.read_text())
 if a.split!=cfg['split']:ap.error('Stage A config authorizes dev only; final is never accepted')
 for n in ['src/policy/encoders.py','src/decision/local_equivalence.py','src/decision/judge_call.py','src/cache/normalizer.py','src/evaluation/intent_stage_a.py']:audit(ROOT/n)
 dp=ROOT/'data/intent_stage_a';data=readlines(dp/(a.split+'.jsonl'));answers=readlines(dp/(a.split+'.answers.jsonl'))
 cached={r['cached_case_id']:r for r in data if r['role']=='cached'}
 rows=[]
 for r in data:
  if r['role']!='new':continue
  c=cached[r['cached_case_id']];rows.append({**r,'cached_question':c['question'],'is_exact':r['question']==c['question']})
 # Verify every model snapshot before importing/loading any weights.
 from huggingface_hub.constants import HF_HUB_CACHE
 snapshots={};weight_ids={}
 for key,(repo,rev) in SPECS.items():
  p=Path(HF_HUB_CACHE)/('models--'+repo.replace('/','--'))/'snapshots'/rev;snapshots[key]=p
  files=[audit(x) for x in sorted(p.rglob('*')) if x.is_file()]
  if not files or not (p/'model.safetensors').is_file():raise FileNotFoundError('STOP: missing pinned '+repo+'@'+rev)
  weight_ids[key]={'repo_id':repo,'revision':rev,'files':files,'provenance_flags':[FLAG] if key!='minilm' else [],'public_pairs_used':False}
 def forbidden(*args,**kwargs):raise RuntimeError('remote connections blocked in Stage A')
 socket.socket.connect=forbidden;socket.create_connection=forbidden
 import torch,numpy as np
 from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
 from src.policy.encoders import DenseEncoder
 from src.decision.local_equivalence import LocalEquivalenceDecisionStep
 from src.evaluation.intent_stage_a import metrics
 torch.manual_seed(cfg['seed']);torch.set_num_threads(cfg['torch_threads'])
 timings={};a.output_dir.mkdir(parents=True)
 text=list(dict.fromkeys([r['question'] for r in data]));tidx={t:i for i,t in enumerate(text)}
 def encode_direct(key):
  t=time.perf_counter();p=snapshots[key]
  tok=AutoTokenizer.from_pretrained(str(p),local_files_only=True,trust_remote_code=False)
  model=AutoModel.from_pretrained(str(p),local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager').float().eval()
  poolpath=p/'1_Pooling/config.json';pool=json.loads(poolpath.read_text());out=[]
  for start in range(0,len(text),cfg['batch_size']):
   batch=tok(text[start:start+cfg['batch_size']],padding=True,truncation=False,return_tensors='pt')
   if batch['input_ids'].shape[-1]>cfg['pair_max_tokens']:raise ValueError('STOP: embedding input window exceeded')
   with torch.inference_mode():
    h=model(**batch).last_hidden_state
    if pool.get('pooling_mode_cls_token'):v=h[:,0]
    elif pool.get('pooling_mode_mean_tokens'):
     mask=batch['attention_mask'].unsqueeze(-1);v=(h*mask).sum(1)/mask.sum(1)
    else:raise ValueError('unsupported pooling; no substitute')
    out.extend(torch.nn.functional.normalize(v.float(),dim=1).tolist())
  del model;gc.collect();timings[key+'_encoding_ms']=(time.perf_counter()-t)*1000;return np.asarray(out)
 t=time.perf_counter();encoder=DenseEncoder(*SPECS['minilm']);vectors=np.asarray([encoder.encode(t) for t in text]);del encoder;gc.collect();timings['minilm_encoding_ms']=(time.perf_counter()-t)*1000
 # Identical bank and candidate lists, independent of pair decisions.
 bank=list(cached.values());bankv=np.asarray([vectors[tidx[c['question']]] for c in bank]);retrieval=[];positive=[]
 for r in rows:
  sims=bankv@vectors[tidx[r['question']]];rank=sorted(range(len(bank)),key=lambda i:(-float(sims[i]),bank[i]['case_id']))[:max(cfg['retrieval_k'])]
  hits={str(k):any(bank[i]['group_id']==r['group_id'] for i in rank[:k]) for k in cfg['retrieval_k']}
  retrieval.append({'case_id':r['case_id'],'candidates':[{'cached_case_id':bank[i]['cached_case_id'],'cosine':float(sims[i])} for i in rank],'designated_group_in_top_k':hits})
  if r['relation_to_cached']=='same_intent' and not r['is_exact']:positive.append(hits)
 for r in rows:r['minilm_cosine']=float(vectors[tidx[r['question']]]@vectors[tidx[r['cached_question']]])
 del vectors,bankv;gc.collect();print('MiniLM retrieval complete',flush=True)
 redv=encode_direct('redis_embedding')
 for r in rows:r['redis_cosine']=float(redv[tidx[r['question']]]@redv[tidx[r['cached_question']]])
 del redv;gc.collect();print('Redis embedding complete',flush=True)
 for key in ['quora','redis_reranker']:
  t=time.perf_counter();p=snapshots[key];tok=AutoTokenizer.from_pretrained(str(p),local_files_only=True,trust_remote_code=False)
  model=AutoModelForSequenceClassification.from_pretrained(str(p),local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager').float().eval()
  for start in range(0,len(rows),cfg['batch_size']):
   batchrows=rows[start:start+cfg['batch_size']];batch=tok([r['cached_question'] for r in batchrows],[r['question'] for r in batchrows],padding=True,truncation=False,return_tensors='pt')
   if batch['input_ids'].shape[-1]>cfg['pair_max_tokens']:raise ValueError('STOP: pair input window exceeded')
   with torch.inference_mode():logits=model(**batch).logits
   if logits.shape[-1]!=1:raise ValueError('unexpected classifier head')
   for r,logit in zip(batchrows,logits[:,0].tolist()):
    if not math.isfinite(logit):raise ValueError('nonfinite model output')
    r[key+'_logit']=logit;r[key+'_score']=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
  del model;gc.collect();timings[key+'_pair_ms']=(time.perf_counter()-t)*1000;print(key+' inference complete',flush=True)
 # Inject precomputed REAL pinned model probabilities into the existing stack.
 # This preserves guard/threshold behavior, avoids its unpinned lazy loader,
 # and does not invent or substitute an inference result.
 class RecordedModel:
  def __init__(self,row):self.row=row
  def predict(self,pairs):
   assert pairs==[(self.row['cached_question'],self.row['question'])]
   return [self.row['quora_score']]
 for r in rows:
  result=LocalEquivalenceDecisionStep(model_instance=RecordedModel(r),threshold=.70,fallback_to_llm=False).decide(r['cached_question'],r['question'])
  r['existing_guards_passed']=result.guards_passed;r['existing_guard_reason']=result.guard_reason;r['existing_stack_default_accept']=result.decision=='REUSE'
 results=[]
 def record(name,threshold,accept,overlap=False):
  ds=[bool(accept(r)) for r in rows]
  results.append({'baseline':name,'threshold':threshold,'provenance_flags':[FLAG] if overlap else [],'public_pairs_used':False,**metrics(rows,ds)})
 record('always_miss',None,lambda r:False);record('exact_only',None,lambda r:r['is_exact'])
 for threshold in cfg['cosine_thresholds']:
  record('minilm_cosine',threshold,lambda r,t=threshold:r['minilm_cosine']>=t)
  record('redis_embedding_cosine',threshold,lambda r,t=threshold:r['redis_cosine']>=t,True)
 for threshold in cfg['pair_thresholds']:
  record('quora',threshold,lambda r,t=threshold:r['quora_score']>=t,True)
  record('redis_reranker',threshold,lambda r,t=threshold:r['redis_reranker_score']>=t,True)
  record('existing_local_equivalence',threshold,lambda r,t=threshold:r['existing_guards_passed'] and r['quora_score']>=t,True)
 parsed={'baseline':'historical_parsed_call_policy','status':'not_run','reason':'Requires deepset/minilm-uncased-squad2@934656cdda79824eabf503ed56e15c01ddbdbe3f and MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@6f5cf0a2b59cabb106aca4c287eed12e357e90eb; neither is approved for download in this task. No substitute used.'}
 report={'evidence_type':'development_set_authored_template_generated','same_set_status':'dev exposed for baseline comparison; no training/calibration/threshold selection','config':cfg,'config_identity':cfgid,'models':weight_ids,'timings_diagnostic_only':timings,'hardware':{'platform':platform.platform(),'cpu_count':os.cpu_count(),'torch_threads':torch.get_num_threads(),'torch_version':torch.__version__},'pairs':len(rows),'retrieval':{'denominator_nonexact_same_intent':len(positive),'recall_at_k':{str(k):sum(h[str(k)] for h in positive)/len(positive) if positive else None for k in cfg['retrieval_k']},'note':'Retrieval against dev bank, separate from designated-pair decision metrics.'},'baselines':results,'unavailable_baselines':[parsed],'final_read':False,'final_evaluated':False,'remote_calls':0,'socket_connections_blocked':True,'serving_enabled':False}
 for name,objects in [('scores',rows),('retrieval',retrieval),('answers',answers)]:
  (a.output_dir/(name+'.jsonl')).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in objects))
 (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'pairs':len(rows),'retrieval':report['retrieval'],'output':str(a.output_dir)}),flush=True)
if __name__=='__main__':main()
