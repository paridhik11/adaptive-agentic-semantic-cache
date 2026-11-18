"""Preflight cannot turn absent or pickle-only weights into a readiness claim."""
from scripts.preflight_document_models import inspect


def test_empty_cache_is_explicitly_blocked(tmp_path):
    result=inspect(tmp_path)
    assert result['status']=='BLOCKED_MISSING_APPROVED_CHECKPOINTS'
    assert result['downloads']==result['inference_calls']==0
    assert not any(m['ready'] for m in result['models'])


def test_pickle_only_snapshot_is_not_accepted(tmp_path):
    result=inspect(tmp_path)
    for m in result['models']:
        from pathlib import Path
        path=Path(m['snapshot']);path.mkdir(parents=True)
        for name in ['config.json','tokenizer.json','pytorch_model.bin']:(path/name).write_text('fixture')
    assert inspect(tmp_path)['status']=='BLOCKED_MISSING_APPROVED_CHECKPOINTS'
