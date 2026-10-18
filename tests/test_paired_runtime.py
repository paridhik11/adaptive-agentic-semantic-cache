import copy
import pytest
from src.benchmark.paired_runtime import build_events,run_paired,summarize,review_packet,ReferenceReplay,run_agent_pairs
from src.benchmark.control import RunPaused

def corpus():
    return {'schema_version':'versioned-faq-corpus-v1','source_revision':'pinned','items':[
        {'faq_id':'tuple','group_id':'sequences','split':'dev','question':'Are tuples immutable?','reference_answer':'Tuples are immutable sequences.'},
        {'faq_id':'integer','group_id':'numbers','split':'dev','question':'Are integers mutable?','reference_answer':'Integers are immutable.'},
        {'faq_id':'hidden','group_id':'test','split':'test','question':'Hidden question','reference_answer':'Not in development.'}]}

def test_runtime_scopes_composition_accounting():
    events=build_events(corpus());trace=run_paired(events,ReferenceReplay(),include_unsafe=True);result=summarize(trace)
    assert len(events)==12 and all(e['faq_id']!='hidden' for e in events)
    assert result['arms']['off']['provider_attempts']==12
    for mode in ('exact','guarded'):
        assert result['arms'][mode]['provider_attempts']==10
        assert result['arms'][mode]['tiers'].get('SEMANTIC',0)==0
        assert result['arms'][mode]['statuses']['HIT']==2
    assert result['semantic_precision'] is None and result['answer_correctness'] is None
    assert result['arms']['off']['provider_tokens'] is None and not result['live_latency_valid']
    for row in trace['records']:
        if row['event']['kind'] in ('different_tenant','different_revision'):
            assert row['arms']['guarded']['status']=='MISS'

def test_unsafe_ablation_forbidden_live():
    class Live(ReferenceReplay):mode='live'
    with pytest.raises(ValueError):run_paired(build_events(corpus()),Live(),include_unsafe=True)

def test_quota_pause_propagates():
    class Paused(ReferenceReplay):
        def __call__(self,body):raise RunPaused('project_minute_quota')
    with pytest.raises(RunPaused):run_paired(build_events(corpus()),Paused())

def test_errors_count_attempts_not_successes():
    class Broken(ReferenceReplay):
        def __call__(self,body):raise RuntimeError('failed')
    summary=summarize(run_paired(build_events(corpus()),Broken()))
    assert summary['arms']['exact']['provider_attempts']==12
    assert summary['arms']['exact']['statuses']=={'ERROR':12}

def test_review_ids_bound_and_blinded():
    trace=run_paired(build_events(corpus()),ReferenceReplay());packet,mapping=review_packet(trace)
    assert packet['items'] and all(i['correct'] is None for i in packet['items'])
    assert all('arm' not in item for item in packet['items'])
    assert len(mapping)==36
    changed=copy.deepcopy(trace);changed['records'][0]['arms']['off']['response']['choices'][0]['message']['content']='Changed'
    assert review_packet(changed)[0]!=packet

def test_agent_memory_replay_records_total_work(tmp_path):
    trace=run_agent_pairs(corpus(),tmp_path)
    assert trace['arms']['off']['provider_calls']==12
    assert trace['arms']['exact']['provider_calls']==12
    assert trace['arms']['exact']['tool_calls']==8
    assert not trace['quality_measured']

def test_reconciliation_rejects_tampering_and_missing_judgments():
    from src.benchmark.paired_runtime import reconcile_reviews
    trace=run_paired(build_events(corpus()),ReferenceReplay())
    packet,_=review_packet(trace)
    report=reconcile_reviews(trace,packet)
    assert report['unique_reviewed_answers']==0 and report['quality_gate']=='INCONCLUSIVE'
    packet['items'][0].update(correct=True,supported=True,reviewer='declared test reviewer')
    assert reconcile_reviews(trace,packet)['unique_reviewed_answers']==1
    packet['items'][0]['answer']='tampered'
    with pytest.raises(ValueError):reconcile_reviews(trace,packet)
