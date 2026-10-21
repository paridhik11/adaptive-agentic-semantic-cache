"""Schema, locks, denominators and cluster accounting; not precision evidence."""
import hashlib,json,subprocess,sys
from pathlib import Path

import pytest
from src.evaluation.intent_stage_a import cp,metrics

ROOT=Path(__file__).resolve().parents[1]
def row(rel='same_intent',family='natural_paraphrase',group='g',lineage='s',exact=False):
 return {'case_id':group+str(rel)+family,'group_id':group,'source_lineage_id':lineage,'relation_to_cached':rel,'case_family':family,'is_exact':exact}
def test_exact_controls_do_not_inflate_semantic_precision_or_coverage():
 r=metrics([row(exact=True),row(),row('different_intent')],[True,False,False])
 assert r['accepted_total_including_exact']==1 and r['accepted_nonexact']==0
 assert r['precision_among_accepted'] is None and r['positive_opportunities']==1 and r['positive_coverage']==0

def test_group_failure_includes_any_unsafe_accept_even_with_ambiguous_accept():
 r=metrics([row(),row('different_intent'),row(None)],[True]*3)
 assert r['safe_accepts']==1 and r['unsafe_accepts']==1 and r['accepted_unknown']==1
 assert r['precision_among_accepted'] is None
 assert r['groups']['failed']==1 and r['groups']['unknown']==0 and r['groups']['accepted_safe']==0

def test_paraphrases_do_not_become_independent_group_trials():
 r=metrics([row(),row(),row(group='g2')],[True]*3)
 assert r['groups']['accepted']==2 and r['lineages']['accepted']==1
 assert r['answer_level_binomial_interval'] is None
 assert r['groups']['conditional_cp']['population_guarantee_established'] is False

def test_clopper_pearson_zero_error_upper_bound_and_no_accepts():
 assert cp(0,0) is None
 bound=cp(60,60)
 assert bound['safety_lower']==pytest.approx(.05**(1/60))
 assert bound['error_upper']==pytest.approx(1-.05**(1/60))

def test_final_split_rejected_before_loading_data_or_models():
 r=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_a_baselines.py'),'--split','final','--output-dir','unused'],capture_output=True,text=True)
 assert r.returncode==2 and 'invalid choice' in r.stderr and not r.stdout

def test_dev_pair_references_and_answer_references_resolve():
 root=ROOT/'data/intent_stage_a'
 rows=[json.loads(l) for l in (root/'dev.jsonl').read_text().splitlines()]
 cached={r['cached_case_id']:r for r in rows if r['role']=='cached'}
 answers={r['cached_answer_id']:r for r in map(json.loads,(root/'dev.answers.jsonl').read_text().splitlines())}
 for r in rows:
  assert r['cached_case_id'] in cached and r['cached_answer_id'] in answers
  assert r['relation_to_cached'] in {'same_intent','new_more_specific','new_more_general','different_intent',None}
  assert set(r['intent_signature'])>={'operation','input_type','output_type','constraints'}
  assert r['label_provenance'] in {'authored','template-generated','public-dataset','model-assisted'}
 assert all(r['split']=='dev' for r in rows)

def test_dataset_size_and_final_manifest_hash_recorded_in_existing_tracked_file():
 root=ROOT/'data/intent_stage_a'
 manifest=json.loads((root/'manifest.json').read_text())
 assert sum(s['groups'] for s in manifest['splits'].values())>=200
 digest=hashlib.sha256((root/'final.manifest.json').read_bytes()).hexdigest()
 assert digest==manifest['final_manifest_sha256'] and digest in (ROOT/'docs/FACTS.md').read_text()
 # Final data are not opened or parsed by this test.

def test_no_group_or_source_lineage_overlap_for_visible_splits():
 root=ROOT/'data/intent_stage_a';seen_g={};seen_s={}
 for split in ['train','dev','calibration']:
  for line in (root/(split+'.jsonl')).read_text().splitlines():
   r=json.loads(line);assert seen_g.setdefault(r['group_id'],split)==split;assert seen_s.setdefault(r['source_lineage_id'],split)==split
 final=json.loads((root/'final.manifest.json').read_text())
 assert not (set(final['source_lineages'])&set(seen_s))

def test_historical_bad_answer_does_not_create_negative_pair_label():
 rows=[json.loads(l) for l in (ROOT/'data/intent_stage_a/dev.jsonl').read_text().splitlines()]
 variants=[r for r in rows if 'Question-only reinterpretation' in r['notes'] and 'wrong-answer' in r['notes']]
 assert variants and all(r['relation_to_cached']=='same_intent' for r in variants)
