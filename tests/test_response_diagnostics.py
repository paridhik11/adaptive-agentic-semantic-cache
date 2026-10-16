import hashlib
import json
from pathlib import Path
import pytest
from tests.test_response_cache import response,request
from src.benchmark.diagnostics import response_diagnostics
from src.benchmark.free_pilot import PilotJournal
from src.benchmark.runner import run_workload
from src.benchmark.report import summarize
from src.benchmark.schema import read_json
from src.api.providers import FixtureChatProvider
from scripts.inspect_pilot_journal import inspect


def test_token_limit_and_reasoning_are_visible_without_text():
    value=response('');value['choices'][0]['finish_reason']='length'
    value['choices'][0]['message']['reasoning']='private reasoning text'
    value['usage']['completion_tokens_details']={'reasoning_tokens':128}
    d=response_diagnostics(value,request())
    assert d['rejection_reason']=='output_token_limit'
    assert d['choices'][0]['assistant_text_characters']==0
    assert d['usage']['reasoning_tokens']==128
    assert 'private reasoning text' not in json.dumps(d)


@pytest.mark.parametrize('value,reason',[(None,'invalid_response_shape'),({},'invalid_response_shape'),(response(''),'empty_assistant_answer'),(response(),'valid')])
def test_diagnostic_reasons(value,reason):
    d=response_diagnostics(value,request())
    assert d['rejection_reason']==(None if reason=='valid' else reason)


def test_journal_inspection_is_read_only(tmp_path):
    path=tmp_path/'calls.sqlite'
    journal=PilotJournal(path,run_id='pilot',fingerprint='test')
    journal.reserve(0,request());value=response('partial');value['choices'][0]['finish_reason']='length'
    journal.finish(0,'complete',value)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    result=inspect(path,'pilot')
    assert result['calls'][0]['response_diagnostics']['rejection_reason']=='output_token_limit'
    assert result['model_calls_made']==0
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    with pytest.raises(ValueError):inspect(path,'unknown')


def test_live_rejection_keeps_metadata_and_does_not_claim_simulated_tokens():
    class Truncated(FixtureChatProvider):
        mode='live'
        def __call__(self,body):
            r=super().__call__(body);r['choices'][0]['finish_reason']='length';return r
    w=read_json(Path(__file__).parents[1]/'examples/benchmark/workload.fixture.json');w['requests']=w['requests'][:1]
    trace=run_workload(w,Truncated());event=trace['events'][0]
    assert event['cache']['error']=='output_token_limit'
    assert event['cache']['response'] is None
    assert event['cache']['response_diagnostics']['choices'][0]['finish_reason']=='length'
    report=summarize(trace)
    assert report['cache_arm']['successful_answers']==0
    assert not any('Fixture calls' in text for text in report['limitations'])
