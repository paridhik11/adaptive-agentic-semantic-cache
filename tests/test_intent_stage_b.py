"""Targeted checks for split isolation, nulls, and later veto/policy invariants."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(split):
    if split not in {'train', 'dev', 'calibration'}:
        raise ValueError('final is forbidden')
    return [json.loads(x) for x in (ROOT/f'data/intent_stage_b/v2/{split}.jsonl').read_text().splitlines()]


def test_formatter_and_lineage_holdout():
    sets = [records(s) for s in ['train', 'dev', 'calibration']]
    for field in ['formatter_id', 'template_id', 'source_lineage_id', 'group_id', 'question']:
        # Literal ambiguous and exact question duplicates occur within a split,
        # never across splits; held-out wording is checked as well as IDs.
        values = [{r[field] for r in rs} for rs in sets]
        for i in range(3):
            for j in range(i):
                assert not values[i] & values[j], field
    assert len({r['source_lineage_id'] for r in sets[0]}) == 24


def test_every_train_lineage_has_all_four_relations_and_null():
    rs = records('train')
    for lineage in {r['source_lineage_id'] for r in rs}:
        assert {r['relation_to_cached'] for r in rs if r['source_lineage_id']==lineage} == {
            None, 'same_intent', 'new_more_specific', 'new_more_general', 'different_intent'}


def test_review_request_is_blank():
    rows = [json.loads(x) for x in (ROOT/'data/intent_stage_b/v2/natural-review-request.jsonl').read_text().splitlines()]
    assert len(rows) == 50
    assert all(not r['cached_question'] and not r['new_question'] and r['relation'] is None for r in rows)


def test_actual_candidate_relation_is_not_designated_label():
    from scripts.build_stage_b_intents import relation
    rows = records('dev')
    cached = [r for r in rows if r['role']=='cached']
    changed = next(r for r in rows if r['case_family']=='entity_swap')
    counterpart = next(c for c in cached if c['intent_signature']==changed['intent_signature'])
    assert changed['relation_to_cached'] == 'different_intent'
    assert relation(changed['intent_signature'], counterpart['intent_signature']) == 'same_intent'


def test_veto_rejects_direction_and_preserves_known_paraphrases():
    from src.evaluation.intent_stage_b_veto import veto
    rows=records('dev')
    cached={r['cached_case_id']:r for r in rows if r['role']=='cached'}
    for r in rows:
        if r['case_family']=='direction_inversion':
            assert veto(cached[r['cached_case_id']]['question'],r['question'])['reject']
        if r['case_family']=='paraphrase':
            assert not veto(cached[r['cached_case_id']]['question'],r['question'])['reject']
    assert veto('How do I convert A to B?', 'How do I convert A to B?')['reject']  # unsupported domain


def test_classifier_required_even_when_veto_does_not_reject():
    from src.evaluation.intent_stage_b_policy import choose
    candidate={'cached_case_id':'c','predicted_class':'same_intent','same_probability':.2,'veto_reject':False}
    assert choose([candidate],.9)[0] is None
    candidate['same_probability']=.99
    candidate['veto_reject']=True
    assert choose([candidate],.9)[0] is None


def test_competing_candidates_and_no_threshold_abstain():
    from src.evaluation.intent_stage_b_policy import choose
    cs=[{'cached_case_id':str(i),'predicted_class':'same_intent','same_probability':.99-i*.01,'veto_reject':False} for i in range(2)]
    assert choose(cs,.9)[1]=='competing_candidates'
    assert choose(cs,None)[0] is None
    cs[0]['veto_reject']=True
    assert choose(cs,.9)[1]=='competing_candidates'  # veto cannot promote runner-up
    cs[1]['same_probability']=.8
    assert choose(cs,.7)[1]=='veto'  # no fallback after top candidate rejection


def test_unknown_accept_cannot_produce_precision_bound():
    from src.evaluation.intent_stage_b_policy import summarize
    record={'case_id':'x','group_id':'g','source_lineage_id':'l','template_id':'t','case_family':'ambiguity',
            'suitable_candidate_ids':[],'exact_candidate':None,'candidates':[{'cached_case_id':'c','actual_relation':None}]}
    result=summarize([record],[('c','semantic')])
    assert result['precision'] is None
    assert result['accepted_unknown']==1
    assert result['groups']['conditional_error_upper'] is None


def test_final_rejected_before_scoring_or_output_creation(tmp_path):
    import subprocess
    import sys
    output=tmp_path/'forbidden'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/score_stage_b_candidates.py'),
                           '--split','final','--output-dir',str(output)],capture_output=True,text=True)
    assert result.returncode==2
    assert 'invalid choice' in result.stderr
    assert not output.exists()


def test_always_miss_and_exact_only_have_same_semantic_denominator():
    from scripts.report_stage_b_dev import baseline_decision
    from src.evaluation.intent_stage_b_policy import summarize
    record={'case_id':'x','group_id':'g','source_lineage_id':'l','template_id':'t','case_family':'exact_control',
            'suitable_candidate_ids':['c'],'exact_candidate':'c','candidates':[{'cached_case_id':'c','actual_relation':'same_intent'}]}
    miss=summarize([record],[baseline_decision(record,'always_miss',None)])
    exact=summarize([record],[baseline_decision(record,'exact_only',None)])
    assert miss['exact_accepts']==0 and exact['exact_accepts']==1
    assert miss['nonexact_opportunities']==exact['nonexact_opportunities']==0


def test_gold_relation_cannot_change_policy_decision():
    from src.evaluation.intent_stage_b_policy import decision_view
    candidate={'cached_case_id':'c','actual_relation':'same_intent',
               'classifier':{'11':{'probabilities':[.99,.003,.003,.004],'predicted_class':'same_intent'}},
               'veto':{'reject':False}}
    record={'exact_candidate':None,'candidates':[candidate],'suitable_candidate_ids':['c']}
    before=decision_view(record,11,.9)
    candidate['actual_relation']='different_intent'
    record['suitable_candidate_ids']=[]
    assert decision_view(record,11,.9)==before==('c','semantic')


def test_numeric_checkpoint_preserves_prefix_and_ignores_unfinished_temp(tmp_path):
    import numpy as np
    from scripts.stage_b_checkpoint import load,save
    path=tmp_path/'scores.npz';binding={'inputs':'abc','model':'def'}
    values=[[.1,.9],[.2,.8]]
    timings={'batch_ms':[1.2],'load_seconds':[.3]}
    save(path,binding,values,timings)
    path.with_suffix('.npz.tmp').write_bytes(b'interrupted incomplete write')
    actual,state=load(path,binding)
    assert np.array_equal(np.asarray(actual,dtype=np.float32),np.asarray(values,dtype=np.float32))
    assert state==timings


def test_numeric_checkpoint_rejects_different_model_or_inputs(tmp_path):
    import pytest
    from scripts.stage_b_checkpoint import load,save
    path=tmp_path/'scores.npz'
    save(path,{'inputs':'a','model':'b'},[[.9]],{'batch_ms':[1.],'load_seconds':[.1]})
    with pytest.raises(ValueError,match='differ'):
        load(path,{'inputs':'changed','model':'b'})


def test_checkpoint_contains_no_pickle_arrays(tmp_path):
    import numpy as np
    from scripts.stage_b_checkpoint import save
    path=tmp_path/'scores.npz'
    save(path,{'inputs':'a'},[[.9]],{'batch_ms':[],'load_seconds':[]})
    with np.load(path,allow_pickle=False) as archive:
        assert archive['data'].dtype==np.float32
        assert archive['metadata'].dtype.kind=='U'
