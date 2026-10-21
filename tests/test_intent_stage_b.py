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
