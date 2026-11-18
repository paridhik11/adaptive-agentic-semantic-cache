"""Complete legacy dev metadata before A4; never reads or alters final data."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE={
'legacy-list-tuple':('convert','Python list','Python tuple',[]),'legacy-temperature':('convert','Celsius','Fahrenheit',[]),'legacy-sort':('sort','Python list','new ascending list',['leave input unchanged']),
'length':('length','Python list','integer',[]),'integer-string':('convert','integer','string',[]),'string-integer':('parse','decimal integer string','integer',[]),'dictionary-default':('lookup','dictionary and key','value or default',['default for missing key']),
'append':('append','list and list value','list with nested element',['append as one element']),'descending':('sort','iterable','new descending list',['descending']),'basename':('basename','filesystem path','final component',[]),'json-parse':('parse','JSON string','Python object',[]),'deduplication':('deduplicate','iterable','set',[]),'sum':('sum','numeric iterable','number',[]),
'lowercase':('lowercase','string','string',[]),'uppercase':('uppercase','string','string',[]),'strip':('strip','string','string',['leading and trailing whitespace']),'split':('split','string','list of words',['whitespace separator']),'join':('join','strings and delimiter','string',[]),'replace':('replace','string','string',['all occurrences']),'absolute':('absolute','number','nonnegative magnitude',[]),'rounding':('round','number','nearest integer',['ties to even']),'reverse-copy':('reverse','list','new reversed list',['leave input unchanged']),'all-predicate':('all','iterable','boolean',[])}
def sig(parts):return dict(zip(['operation','input_type','output_type','constraints'],parts))
def main():
 p=ROOT/'data/intent_stage_a/dev.jsonl';raw=p.read_bytes();print(p,len(raw),hashlib.sha256(raw).hexdigest());rows=[json.loads(l) for l in raw.splitlines()]
 before=hashlib.sha256(raw).hexdigest();changes=[]
 for r in rows:
  if not r['group_id'].startswith('spent:'):continue
  base=sig(BASE[r['group_id'][6:]]);base['constraints']=list(base['constraints']);q=r['question'].lower()
  if r['relation_to_cached']=='same_intent':r['intent_signature']=base;continue
  s={**base,'constraints':list(base['constraints'])}
  if 'proof of correctness' in q:s['constraints'].append('prove correctness');r['relation_to_cached']='new_more_specific'
  elif 'recursiv' in q:s['constraints'].append('recursive conversion');r['relation_to_cached']='new_more_specific'
  elif 'without using int' in q:s['constraints'].append('do not use int');r['relation_to_cached']='new_more_specific';r['case_family']='added_requirement'
  elif 'preserving' in q or 'preserve' in q:s['constraints'].append('preserve original order');r['relation_to_cached']='new_more_specific';r['case_family']='added_requirement'
  elif q.startswith('do not') or 'why should i not' in q:s['operation']='avoid:'+s['operation'];r['case_family']='negation_polarity'
  elif 'xml namespace' in q:s=sig(('parse','XML string','namespace mapping',[]));r['case_family']='entity_function_swap'
  elif 'largest element' in q:s=sig(('maximum','Python list','maximum element',[]));r['case_family']='entity_function_swap'
  elif 'average' in q:s=sig(('average','numeric iterable','number',[]));r['case_family']='entity_function_swap'
  elif 'parent directory' in q:s=sig(('dirname','filesystem path','parent path',[]));r['case_family']='entity_function_swap'
  elif 'remove a dictionary' in q:s=sig(('pop','dictionary and key','removed value',['mutate dictionary']));r['case_family']='entity_function_swap'
  elif 'each element' in q:s=sig(('extend','list and iterable','extended list',[]));r['case_family']='entity_function_swap'
  elif 'kelvin' in q:s=sig(('convert','Celsius','Kelvin',[]));r['case_family']='quantity_version_unit'
  elif 'fahrenheit to celsius' in q:s=sig(('convert','Fahrenheit','Celsius',[]))
  elif 'tuple into a list' in q or 'tuple to a list' in q:s=sig(('convert','Python tuple','Python list',[]))
  elif 'string to an integer' in q:s=sig(('parse','string','integer',[]))
  elif 'python object into a json' in q:s=sig(('serialize','Python object','JSON string',[]))
  elif 'descending' in q:s=sig(('sort','Python list','new descending list',['descending']))
  elif 'ascending' in q:s=sig(('sort','iterable','new ascending list',['ascending']))
  elif 'uppercase' in q:s=sig(('uppercase','string','string',[]))
  elif 'lowercase' in q:s=sig(('lowercase','string','string',[]))
  elif 'all spaces inside' in q:s=sig(('remove spaces','string','string',['all internal spaces']))
  elif 'join a list' in q:s=sig(('join','list of words','string',[]))
  elif 'split a string' in q:s=sig(('split','string','list',[]))
  elif 'only the first occurrence' in q:s=sig(('replace','string','string',['first occurrence only']))
  elif 'change the sign' in q:s=sig(('negate','number','number',[]))
  elif 'upward' in q:s=sig(('ceil','number','integer',[]))
  elif 'in place' in q:s=sig(('reverse','list','same list',['mutate input']))
  elif 'at least one' in q:s=sig(('any','iterable','boolean',[]))
  elif 'without creating' in q:s['constraints'].append('no new object');r['relation_to_cached']='new_more_specific';r['case_family']='added_requirement'
  else:s['constraints'].append('see explicit changed requirement in question');s['verification_status']='needs developer metadata review'
  if s!=r['intent_signature']:changes.append(r['case_id'])
  r['intent_signature']=s
 p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 report={'evidence_type':'pre_baseline_dev_metadata_completion','before_sha256':before,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changed_case_ids':changes,'final_data_read':False,'final_data_modified':False}
 out=ROOT/'data/intent_stage_a/dev_metadata_completion.json'
 index=1
 while out.exists():
  out=ROOT/('data/intent_stage_a/dev_metadata_completion.'+str(index)+'.json');index+=1
 out.write_text(json.dumps(report,indent=2)+'\n')
 # Refresh dev counts only; never read final per-case rows.
 mp=ROOT/'data/intent_stage_a/manifest.json';m=json.loads(mp.read_text());from collections import Counter
 m['splits']['dev']['relations']=dict(Counter(str(r['relation_to_cached']) for r in rows));m['splits']['dev']['families']=dict(Counter(r['case_family'] for r in rows));mp.write_text(json.dumps(m,indent=2)+'\n')
 print('Dev metadata completed; no final records read.')
if __name__=='__main__':main()
