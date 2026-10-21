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
