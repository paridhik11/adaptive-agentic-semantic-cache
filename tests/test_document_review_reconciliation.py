"""Review ingestion must preserve uncertainty and original question bindings."""
from pathlib import Path
import pytest
from scripts.reconcile_document_reviews import ingest,compare

PACKET=Path(__file__).resolve().parents[1]/'data/document_product_review/human_v1'


def test_uncertain_human_judgment_is_not_replaced_by_model_yes():
    human=ingest(PACKET,{'H05':{'judgment':'?','rationale':'Ambiguous wording'}},'declared-reviewer')
    model=[{**human[0],'answer_sufficient':True,'rationale':'Model yes'}]
    report=compare(human,model)
    assert human[0]['answer_sufficient'] is None and human[0]['relation'] is None
    assert report['counts']['human_uncertain']==1 and report['counts']['binary_agreements']==0
    assert report['differences'][0]['model_judgment'] is True
    assert report['accuracy'] is None and report['reuse_precision'] is None


def test_missing_reason_is_marked_as_not_provided_not_invented():
    row=ingest(PACKET,{'H03':{'judgment':'Y'}},'declared-reviewer')[0]
    assert row['answer_sufficient'] is True and row['human_rationale_provided'] is False
    assert row['evidence_type']=='human_review_of_model_authored'


@pytest.mark.parametrize('identifier,label',[('H99','Y'),('H01','maybe'),('H01',True)])
def test_unknown_ids_or_implicit_label_conversion_are_rejected(identifier,label):
    with pytest.raises(ValueError):ingest(PACKET,{identifier:{'judgment':label}},'declared-reviewer')


def test_model_question_mismatch_cannot_be_compared_as_agreement():
    human=ingest(PACKET,{'H03':{'judgment':'Y'}},'declared-reviewer')
    model=[{**human[0],'new_question':'Changed question'}]
    with pytest.raises(ValueError):compare(human,model)
