import copy
import pytest
from tests.test_answer_benchmark import workload,trace
from src.benchmark.report import review_template, summarize
from src.benchmark.review_agreement import reconcile


def reviews(trace,name):
    r=review_template(trace)
    for row in r['reviews']:
        row['reviewer']=name;row['answer_correct']=True;row['fresh_answer_correct']=True
        if row['candidate_event_id']:row['safe_reuse']=True
    return r


def test_consensus_does_not_hide_conflicts(trace):
    a,b=reviews(trace,'first'),reviews(trace,'second')
    b['reviews'][0]['answer_correct']=False
    report=reconcile(trace,a,b)
    assert len(report['conflicts'])==1
    assert report['consensus_reviews']['reviews'][0]['answer_correct'] is None
    assert report['agreement']['answer_correct']['agreement']==23/24
    assert summarize(trace,report['consensus_reviews'])['paired_answer_quality']['reviewed_pairs']==23


def test_single_review_does_not_become_consensus(trace):
    a,b=reviews(trace,'first'),reviews(trace,'second')
    b['reviews'][0]['answer_correct']=None
    report=reconcile(trace,a,b)
    assert report['consensus_reviews']['reviews'][0]['answer_correct'] is None
    assert report['conflicts']==[]


def test_distinct_human_reviewers_required(trace):
    a,b=reviews(trace,'same'),reviews(trace,' SAME ')
    with pytest.raises(ValueError,match='distinct'):reconcile(trace,a,b)
    b=reviews(trace,'other');b['reviews'][0]['method']='independent_judge'
    with pytest.raises(ValueError,match='human'):reconcile(trace,a,b)


def test_wrong_trace_binding_still_rejected(trace):
    a,b=reviews(trace,'first'),reviews(trace,'second');b['reviews'][0]['response_sha256']='wrong'
    with pytest.raises(ValueError):reconcile(trace,a,b)
