"""Build authored/template Stage A data and lock final without evaluating it."""
import argparse,collections,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def audit(p):
 p=Path(p)
 if not p.is_file():raise FileNotFoundError('STOP: '+str(p))
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();print(json.dumps({'path':str(p.resolve()),'bytes':len(raw),'sha256':h}),flush=True);return h
for name in ['scripts/probe_question_conditioned.py','scripts/probe_decomposed_policy.py','scripts/probe_answer_applicability.py','src/policy/question_conditioned.py','src/policy/decomposed_applicability.py','LICENSE']:
 audit(ROOT/name)
from scripts.probe_question_conditioned import cases as old_cases
# Distinct task contracts, not variable-name variants. Shared lineage keeps all
# container/response-requirement variations of an operation in one split.
OPS=[
 ('chunk','divide {input} into consecutive blocks of 3 elements','list of blocks','seq=list(values)\nresult=[seq[i:i+3] for i in range(0,len(seq),3)]'),
 ('window','produce overlapping windows of 3 elements from {input}','list of windows','seq=list(values)\nresult=[seq[i:i+3] for i in range(max(0,len(seq)-2))]'),
 ('interleave','alternate elements from {input} and another iterable until either ends','list','result=[x for pair in zip(values,other) for x in pair]'),
 ('cartesian','form the Cartesian product of {input} with another iterable','list of pairs','import itertools\nresult=list(itertools.product(values,other))'),
 ('permutations','enumerate ordered arrangements of length 2 from {input}','list of tuples','import itertools\nresult=list(itertools.permutations(values,2))'),
 ('combinations','enumerate unordered selections of length 2 from {input}','list of tuples','import itertools\nresult=list(itertools.combinations(values,2))'),
 ('frequencies','count how often each integer occurs in {input}','mapping from integer to count','from collections import Counter\nresult=Counter(values)'),
 ('prefix-sum','compute successive cumulative sums of {input}','list of prefix sums','import itertools\nresult=list(itertools.accumulate(values))'),
 ('adjacent-group','group consecutive equal values in {input}','list of groups','import itertools\nresult=[list(g) for _,g in itertools.groupby(values)]'),
 ('take-prefix','take values from {input} until the first nonpositive value','list','import itertools\nresult=list(itertools.takewhile(lambda x:x>0,values))'),
 ('drop-prefix','discard the initial positive values from {input}','list','import itertools\nresult=list(itertools.dropwhile(lambda x:x>0,values))'),
 ('even-index','select every second element from {input} starting at index 0','list','import itertools\nresult=list(itertools.islice(values,0,None,2))'),
 ('partition','separate {input} into even and odd values','pair of lists','seq=list(values)\nresult=([x for x in seq if x%2==0],[x for x in seq if x%2!=0])'),
 ('pairwise','form consecutive adjacent pairs from {input}','list of pairs','import itertools\nresult=list(itertools.pairwise(values))'),
 ('smallest','find the 3 smallest values in {input}','list','import heapq\nresult=heapq.nsmallest(3,values)'),
 ('insertion','find where a new integer can be inserted in sorted {input} while keeping order','integer index','import bisect\nresult=bisect.bisect_left(list(values),target)'),
]
PROFILES=[('code','Return Python code only.'),('example','Include a worked example.'),('complexity','Explain worst-case time complexity.'),('version','Use Python 3.12.')]
TYPES=['a list of integers','a tuple of integers','an iterator of integers']
SPLITS=['train']*8+['dev']*3+['calibration']*3+['final']*2
RELATIONS={'same_intent','new_more_specific','new_more_general','different_intent'}
def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def line(x):return json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=ROOT/'data/intent_stage_a');a=ap.parse_args()
 if a.output_dir.exists():ap.error('STOP: dataset directory exists; final lock must not be overwritten')
 rows={s:[] for s in ['train','dev','calibration','final']};answers={};groups={};licenses={'authored':'project MIT license; builder authored, not human-reviewed','template-generated':'project MIT license; generated from authored task contracts'}
 def add_group(gid,lineage,split,cached,answer,signature,provenance,source):
  cid=gid+':cached';aid='answer:'+key([gid,answer])[:20]
  answers[aid]={'cached_answer_id':aid,'text':answer,'sha256':hashlib.sha256(answer.encode()).hexdigest(),'provenance':provenance,'generation_quality':'not_independently_reviewed','source':source}
  groups[gid]={'group_id':gid,'source_lineage_id':lineage,'split':split,'cached_case_id':cid,'question':cached,'cached_answer_id':aid,'signature':signature,'provenance':provenance,'source':source}
  add(gid,cached,'same_intent','cached_control',signature,role='cached')
 def add(gid,q,relation,family,sig,role='new',notes='',aid=None,provenance=None):
  g=groups[gid];r={'case_id':gid+':'+str(sum(x['group_id']==gid for x in rows[g['split']])), 'group_id':gid,'source_lineage_id':g['source_lineage_id'],'split':g['split'],'question':q,'role':role,'relation_to_cached':relation,'case_family':family,'cached_answer_id':aid or g['cached_answer_id'],'intent_signature':sig,'label_provenance':provenance or g['provenance'],'notes':notes,'cached_case_id':g['cached_case_id'],'scope_id':'authored-python-reference-v1','template_lineage_id':'stage-a-formatters-v1' if (provenance or g['provenance'])=='template-generated' else 'day3-spent','label_status':'ambiguous' if relation is None else 'unreviewed_authored','source':g['source'],'license':licenses[provenance or g['provenance']]}
  rows[g['split']].append(r)
 # Keep all historical developer-authored records in dev. Answer mistakes NEVER
 # become different-intent labels. Complete actual answer variants are retained.
 original=old_cases()
 for c in original:
  gid='spent:'+c['group']
  if gid not in groups:
   sig={'operation':c['group'].removeprefix('legacy-'),'input_type':'as specified in cached question','output_type':'as specified in cached question','constraints':[],'verification_status':'partial_legacy_signature_not_used_for_policy'}
   add_group(gid,'spent-source:'+key(c['context'])[:16],'dev',c['original'],c['answer'],sig,'authored','scripts/probe_question_conditioned.py; spent developer fixtures')
  same=c['expected'] or c['incoming']==c['original'] or any(x in c['id'] for x in ['wrong-answer','actual-answer'])
  relation='same_intent' if same else ('new_more_specific' if any(x in c['id'] for x in ['recursion','constraint']) else 'different_intent')
  family='natural_paraphrase' if same else ('added_requirement' if relation=='new_more_specific' else 'direction_role_inversion')
  aid='answer:'+key([c['id'],c['answer']])[:20]
  answers[aid]={'cached_answer_id':aid,'text':c['answer'],'sha256':hashlib.sha256(c['answer'].encode()).hexdigest(),'provenance':'authored','generation_quality':'historical_wrong_or_unsupported_variant' if not c['expected'] and same else 'not_independently_reviewed','source':c['id']}
  sig=dict(groups[gid]['signature']);sig['constraints']=[] if same else ['unverified changed requirement; see question']
  add(gid,c['incoming'],relation,family,sig,aid=aid,notes='Question-only reinterpretation of spent case '+c['id']+'; historical answer applicability is separate.')
 for gid,g in list(groups.items()):
  phrase=re.sub(r'^How (?:do|can|should) I\s+','',g['question'],flags=re.I).rstrip('?')
  for q in ['I need to '+phrase+'. How should I proceed?','Please explain how to '+phrase+'.','What approach can I use to '+phrase+'?']:
   add(gid,q,'same_intent','natural_paraphrase',g['signature'],provenance='template-generated',notes='Template paraphrase of spent question; not independently labeled.')
  add(gid,g['question'],'same_intent','exact_control',g['signature'])
  for q,rel,fam in [('Do not perform this task: '+g['question'],'different_intent','negation_polarity'),(g['question']+' Also include a proof of correctness.','new_more_specific','added_requirement'),('Instead, explain XML namespace parsing.','different_intent','entity_function_swap')]:
   sig=dict(g['signature']);sig['constraints']=[q];add(gid,q,rel,fam,sig,provenance='template-generated')
 for index,(op,description,out,code) in enumerate(OPS):
  for itype in TYPES:
   for profile,requirement in PROFILES:
    gid='generated:'+op+':'+str(TYPES.index(itype))+':'+profile;task=description.format(input=itype)
    sig={'operation':op,'input_type':itype,'output_type':out,'constraints':[requirement]}
    cached='How do I '+task+'? '+requirement
    add_group(gid,'operation-source:'+op,SPLITS[index],cached,code+'\n# Authored example; extra requested presentation/version compliance is not independently reviewed.',sig,'template-generated','authored OPS definition '+op+'; not downloaded public documentation')
    for q in ['I need to '+task+'. '+requirement,'What is a way to '+task+'? '+requirement,'Please show how to '+task+'. '+requirement]:add(gid,q,'same_intent','natural_paraphrase',sig)
    add(gid,cached,'same_intent','exact_control',sig)
    more={**sig,'constraints':sig['constraints']+['Also prove correctness.']};add(gid,cached+' Also prove correctness.','new_more_specific','added_requirement',more)
    general={**sig,'constraints':[]};add(gid,'How do I '+task+'?','new_more_general','removed_requirement',general)
    diff={**sig,'operation':'negated:'+op};add(gid,'Why should I avoid performing this task: '+task+'? '+requirement,'different_intent','negation_polarity',diff)
    swap={**sig,'operation':'XML-namespace-parse','input_type':'XML string','output_type':'namespace mapping'};add(gid,'How do I parse XML namespaces? '+requirement,'different_intent','entity_function_swap',swap)
    inverted={**sig,'operation':'inverse:'+op,'input_type':out,'output_type':itype};add(gid,'How do I reconstruct '+itype+' from the '+out+' produced when I '+task+'? '+requirement,'different_intent','direction_role_inversion',inverted)
    quantity={**sig,'constraints':sig['constraints']+['Interpret numeric values as milliseconds instead of seconds.']};add(gid,cached+' Interpret numeric values as milliseconds instead of seconds.','new_more_specific','quantity_version_unit',quantity)
    control={**sig,'operation':'configure-network-router','input_type':'router','output_type':'network configuration'};add(gid,'How can I configure a network router VLAN?','different_intent','negative_control',control)
    add(gid,'How do I do that?',None,'ambiguity',{'operation':None,'input_type':None,'output_type':None,'constraints':[]},notes='No antecedent in fixed scope; do not infer a definite relation.')
 # Assert disjoint group AND source-lineage assignment before writing any data.
 assigned={}
 for g in groups.values():
  lineage=g['source_lineage_id'];assert assigned.setdefault(lineage,g['split'])==g['split']
 assert len(groups)>=200
 for gid,g in groups.items():
  rs=[x for x in rows[g['split']] if x['group_id']==gid and x['role']=='new']
  assert sum(x['relation_to_cached']=='same_intent' and x['question']!=g['question'] for x in rs)>=3
  assert sum(x['relation_to_cached'] in {'new_more_specific','new_more_general','different_intent'} for x in rs)>=3
 a.output_dir.mkdir(parents=True)
 # Final is serialized once; no evaluation or per-case output. Hash goes to an
 # ALREADY TRACKED file immediately, without git add/commit/push.
 final_bytes=''.join(line(x) for x in rows['final']).encode();final_path=a.output_dir/'final.jsonl';final_path.write_bytes(final_bytes)
 manifest={'schema':'intent-stage-a-v1','split':'final','locked':True,'data_sha256':hashlib.sha256(final_bytes).hexdigest(),'groups':len({x['group_id'] for x in rows['final']}),'rows':len(rows['final']),'source_lineages':sorted({x['source_lineage_id'] for x in rows['final']}),'generator_sha256':audit(Path(__file__)),'label_provenance':'template-generated; builder authored; not independent human labels','evaluation':'NOT RUN'}
 manifest_path=a.output_dir/'final.manifest.json';mb=(json.dumps(manifest,indent=2)+'\n').encode();manifest_path.write_bytes(mb);mh=hashlib.sha256(mb).hexdigest()
 facts=ROOT/'docs/FACTS.md';audit(facts)
 with facts.open('a') as f:f.write('\n\n## Stage A final-split lock (created before any Stage A evaluation)\n\nFinal manifest SHA-256: `'+mh+'`. Data SHA-256: `'+manifest['data_sha256']+'`. Generator-created, template-authored labels; no final inference or per-case review. See `data/intent_stage_a/final.manifest.json`.\n')
 for split in ['train','dev','calibration']:(a.output_dir/(split+'.jsonl')).write_text(''.join(line(x) for x in rows[split]))
 # Answer stores are split too: baseline runner cannot read final answer records.
 for split in rows:
  ids={x['cached_answer_id'] for x in rows[split]};(a.output_dir/(split+'.answers.jsonl')).write_text(''.join(line(answers[i]) for i in sorted(ids)))
 summary={'schema':'intent-stage-a-v1','evidence_type':'authored_and_template_generated_not_human_reviewed','public_pairs_used':False,'public_pair_exclusion':'No vetted license-checked local public corpus selected; historical sealed/pilot/FAQ sets excluded.','final_manifest_sha256':mh,'splits':{}}
 for split,rs in rows.items():
  summary['splits'][split]={'groups':len({x['group_id'] for x in rs}),'source_lineages':len({x['source_lineage_id'] for x in rs}),'rows':len(rs),'new_rows':sum(x['role']=='new' for x in rs),'families':dict(collections.Counter(x['case_family'] for x in rs)),'provenance':dict(collections.Counter(x['label_provenance'] for x in rs)),'relations':dict(collections.Counter(str(x['relation_to_cached']) for x in rs))}
 (a.output_dir/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2));print('FINAL_MANIFEST_SHA256',mh)
if __name__=='__main__':main()
