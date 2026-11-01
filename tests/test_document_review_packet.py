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
