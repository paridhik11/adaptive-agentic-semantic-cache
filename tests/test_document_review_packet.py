"""Schema tests use fabricated reviewer identities; not actual review labels."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.validate_document_review import validate


@pytest.fixture
def packet(tmp_path):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'packet'
    subprocess.run([sys.executable,str(root/'scripts/prepare_document_review.py'),'--output-dir',str(out)],check=True,capture_output=True)
    return out


def test_packet_contains_only_blank_questions_and_labels(packet):
    rows=[json.loads(row) for row in (packet/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==30 and len({r['seed_id'] for r in rows})==3
    assert all(r['new_question'] is None and r['answer_sufficient'] is None for r in rows)


def response(packet,tmp_path,**changes):
    row=json.loads((packet/'requests.jsonl').read_text().splitlines()[0])
    row.update(new_question='A fabricated schema test question',answer_sufficient=True,relation='same_intent',
        author_id='fabricated-author',reviewer_id='fabricated-reviewer',rationale='Schema test only',evidence_type='model_assisted')
    row.update(changes);path=tmp_path/'responses.jsonl';path.write_text(json.dumps(row)+'\n');return path


def test_model_assistance_is_separate_and_no_accuracy_is_invented(packet,tmp_path):
    result=validate(packet,response(packet,tmp_path))
    assert result['by_provenance']['model_assisted']['labeled']==1
    assert result['by_provenance']['human_authored']['labeled']==0
    assert len(result['missing_ids'])==29 and result['precision'] is None


@pytest.mark.parametrize('changes',[{'answer_sufficient':'yes'},{'evidence_type':'verified_human'},
    {'seed_id':'changed'},{'reviewer_id':None},{'new_question':''}])
def test_invalid_review_cannot_be_counted(packet,tmp_path,changes):
    with pytest.raises(ValueError):validate(packet,response(packet,tmp_path,**changes))


def test_source_tampering_is_rejected(packet,tmp_path):
    path=response(packet,tmp_path)
    (packet/'sources.json').write_text('[]')
    with pytest.raises(ValueError,match='integrity'):validate(packet,path)


def test_human_review_of_model_questions_is_separate_from_human_authorship(packet,tmp_path):
    path=response(packet,tmp_path,evidence_type='human_review_of_model_authored',question_origin='model_authored',relation=None)
    result=validate(packet,path)
    assert result['by_provenance']['human_review_of_model_authored']['labeled']==1
    assert result['by_provenance']['human_authored']['questions']==0


def test_human_form_omits_model_judgments_and_rationales():
    from scripts.prepare_blinded_human_review import prepare
    source={'seed_id':'source','original_question':'Original','cached_answer':'Reference answer'}
    rows=[{'item_id':'001','seed_id':'source','new_question':'New wording?',
           'author_id':'fixture-model','answer_sufficient':False,'rationale':'SECRET MODEL RATIONALE'}]
    text,mapping,template=prepare([source],rows)
    assert 'SECRET MODEL RATIONALE' not in text and 'answer_sufficient' not in text
    assert template[0]['answer_sufficient'] is None and template[0]['rationale'] is None
    assert mapping[0]['human_id']=='H01'
