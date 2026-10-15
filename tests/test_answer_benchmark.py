import copy
import json
from pathlib import Path

import pytest

from src.api.providers import FixtureChatProvider
from src.benchmark.runner import run_workload
from src.benchmark.schema import digest, read_json, validate_workload, write_new
from src.benchmark.report import interval, review_template, summarize, validate_reviews, validate_trace
from src.benchmark.views import review_html


@pytest.fixture
def workload():
    return read_json(Path(__file__).parents[1]/'examples/benchmark/workload.fixture.json')


@pytest.fixture
def trace(workload):
    return run_workload(workload, FixtureChatProvider())


def test_actual_candidate_lineage_and_missing_labels(trace):
    result = summarize(trace)
    assert result['requests'] == 24
    assert result['status_counts'] == {'MISS':4, 'HIT':2, 'BYPASS':18}
    assert result['provider_attempts_avoided'] == 2
    assert result['overall_reuse']['reviewed_incorrect_rate'] is None
    assert result['overall_reuse']['all_accepted_error_identification_bounds'] == [0,1]
    assert result['nonexact_reuse']['reviewed_clopper_pearson_95'] is None
    assert result['paired_answer_quality']['mean_cache_minus_fresh_accuracy'] is None
    assert result['safety_certified'] is False
    assert result['cache_arm']['provider_attempts'] == 22
    assert result['fresh_arm']['provider_attempts'] == 24


def test_partial_labels_are_not_assumed_representative(trace):
    reviews = review_template(trace)
    hit = next(r for r in reviews['reviews'] if r['candidate_event_id'])
    hit.update(safe_reuse=False, reviewer='fixture author', method='fixture')
    result = summarize(trace,reviews)['overall_reuse']
    assert result['reviewed'] == 1 and result['unreviewed'] == 1
    assert result['reviewed_incorrect_rate'] == 1
    assert result['all_accepted_error_identification_bounds'] == [.5,1]
    assert result['reviewed_clopper_pearson_95'][1] == 1


def test_intervals_and_undefined_denominators():
    assert interval(0,0) is None
    assert interval(0,36)[1] == pytest.approx(.0973937559)
    assert interval(36,36)[0] == pytest.approx(1-.0973937559)


@pytest.mark.parametrize('field', ['request_sha256','response_sha256','fresh_response_sha256','candidate_event_id'])
def test_reject_wrong_answer_or_candidate_labels(trace,field):
    reviews = review_template(trace)
    reviews['reviews'][0][field] = 'wrong'
    with pytest.raises(ValueError,match='bound'):
        validate_reviews(trace,reviews)


def test_reject_stale_trace_and_duplicate_review(trace):
    reviews=review_template(trace)
    reviews['trace_sha256']='wrong'
    with pytest.raises(ValueError,match='different trace'):
        summarize(trace,reviews)
    reviews=review_template(trace)
    reviews['reviews'].append(copy.deepcopy(reviews['reviews'][0]))
    with pytest.raises(ValueError,match='duplicate'):
        summarize(trace,reviews)


@pytest.mark.parametrize('mutation', ['future_candidate','different_answer','different_tenant','nan_time','negative_calls'])
def test_corrupt_telemetry_cannot_create_valid_evidence(trace,mutation):
    hit=next(e for e in trace['events'] if e['status']=='HIT')
    if mutation=='future_candidate': hit['candidate_event_id']=trace['events'][-1]['event_id']
    if mutation=='different_answer':
        hit['cache']['response']['id']='wrong'
        hit['response_sha256']=digest(hit['cache']['response'])
    if mutation=='different_tenant': hit['tenant_id']='another-tenant'
    if mutation=='nan_time': hit['cache']['latency_ms']=float('nan')
    if mutation=='negative_calls': hit['cache']['provider_calls']=-1
    with pytest.raises(ValueError): validate_trace(trace)


@pytest.mark.parametrize('kind', ['group','text'])
def test_split_leakage_rejected(workload,kind):
    row=copy.deepcopy(workload['requests'][0]);row['event_id']='new';row['split']='test'
    if kind=='text': row['group_id']='different';row['request']['messages'][0]['content']='  WHAT is PHOTOSYNTHESIS?  '
    workload['requests'].append(row)
    with pytest.raises(ValueError,match='crosses splits'): validate_workload(workload)


def test_entire_workload_validated_before_any_provider_call(workload):
    class Counting(FixtureChatProvider):
        calls=0
        def __call__(self,body):
            self.calls+=1
            return super().__call__(body)
    p=Counting();workload['requests'][-1]['request']['stream']=True
    with pytest.raises(ValueError): run_workload(workload,p)
    assert p.calls==0


def test_provider_failure_accounted_without_zero_cost_claim(workload):
    class Broken:
        mode='live'
        def __call__(self,body): raise RuntimeError('sensitive')
    trace=run_workload(workload,Broken());report=summarize(trace)
    assert report['cache_arm']['errors']==24
    assert report['fresh_arm']['errors']==24
    assert report['token_difference_fresh_minus_cache'] is None
    assert report['paired_latency']['mean_cache_minus_fresh_ms'] is None
    assert 'sensitive' not in json.dumps(trace)


def test_missing_usage_invalid_response_and_provider_retry_counts(workload):
    class Missing(FixtureChatProvider):
        def __call__(self,body):
            response=super().__call__(body);response.pop('usage');return response
    report=summarize(run_workload(workload,Missing()))
    assert report['token_difference_fresh_minus_cache'] is None
    assert report['cache_arm']['attempts_without_usage']==22


def test_paired_quality_and_group_interval(trace):
    reviews=review_template(trace)
    for row in reviews['reviews']:
        row.update(reviewer='test',method='fixture',answer_correct=False,fresh_answer_correct=True)
    result=summarize(trace,reviews)['paired_answer_quality']
    assert result['reviewed_pairs']==24
    assert result['mean_cache_minus_fresh_accuracy']==-1
    assert result['group_bootstrap_95']==[-1,-1]


def test_strict_json_exclusive_write_and_html_escape(trace,tmp_path):
    p=tmp_path/'evidence.json';write_new(p,trace)
    with pytest.raises(FileExistsError):write_new(p,trace)
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):read_json(p)
    p.write_text('{"a":NaN}')
    with pytest.raises(ValueError):read_json(p)
    trace['provenance']='</script><script>alert(1)</script>'
    page=review_html(trace,review_template(trace),summarize(trace))
    assert '</script><script>alert(1)' not in page
    assert 'textContent' in page


def test_test_split_cannot_be_run_as_development(workload):
    for row in workload['requests']:row['split']='test'
    with pytest.raises(ValueError,match='development only'):run_workload(workload,FixtureChatProvider(),split='test')


def test_pair_import_groups_shared_queries_without_transferring_labels():
    from scripts.prepare_answer_workload import convert_pairs
    pairs=[{'query_a':'A','query_b':'B','label':True,'source':'first'},
           {'query_a':' b ','query_b':'C','label':False,'source':'second'}]
    result=convert_pairs(pairs,source_digest=digest(pairs),model='fixture-model',license_note='fixture')
    assert len({r['group_id'] for r in result['requests']})==1
    assert all(r['split']=='dev' and 'safe_reuse' not in r for r in result['requests'])
    assert result['nature']=='constructed_natural_queries'


def test_forged_zero_call_miss_and_error_status_rejected(trace):
    trace['events'][0]['cache']['provider_calls']=0
    trace['events'][0]['cache']['usage']=[]
    with pytest.raises(ValueError,match='non-hit'):validate_trace(trace)


def test_live_calls_require_explicit_budget(workload,tmp_path,monkeypatch):
    from scripts.benchmark_answers import main
    path=tmp_path/'workload.json';write_new(path,workload)
    monkeypatch.delenv('UPSTREAM_CHAT_URL',raising=False)
    with pytest.raises(SystemExit):
        main(['run','--workload',str(path),'--output-dir',str(tmp_path/'run'),'--provider','http'])
    assert not (tmp_path/'run').exists()


def test_candidate_must_have_been_admitted(trace):
    hit=next(e for e in trace['events'] if e['status']=='HIT')
    source=next(e for e in trace['events'] if e['event_id']==hit['candidate_event_id'])
    source['stored']=False
    with pytest.raises(ValueError,match='earlier generated candidate'):validate_trace(trace)
