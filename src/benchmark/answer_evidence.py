"""Answer-bound development review and observed work; no inferred correctness."""
from __future__ import annotations

from dataclasses import asdict
import copy
import math
import time
from src.policy.semantic_runtime import digest


def review_binding(request, response, tenant, revision):
    return dict(incoming_request_sha256=digest(request), returned_response_sha256=digest(response),
                tenant=tenant, source_revision=revision)


class AnswerReviews:
    """Review declarations, not authenticated/independent human ground truth."""
    def __init__(self, rows=()):
        self.rows = {}
        for row in rows:
            binding = row['binding']
            if set(binding) != {'incoming_request_sha256', 'returned_response_sha256', 'tenant', 'source_revision'}:
                raise ValueError('incomplete review binding')
            if row['label'] not in {'sufficient', 'insufficient', 'unknown'}:
                raise ValueError('invalid review label')
            if not row.get('reviewer') or not row.get('rationale'):
                raise ValueError('reviewer declaration and rationale required')
            key = digest(binding)
            if key in self.rows:
                raise ValueError('duplicate review binding')
            self.rows[key] = dict(row)

    def lookup(self, binding):
        return self.rows.get(digest(binding), {'binding': binding, 'label': 'unknown',
                                             'reviewer': None, 'rationale': 'not reviewed'})


class ObservedProvider:
    """Count at the actual invocation boundary, including failures; never cached usage."""
    def __init__(self, provider, *, usage_kind):
        self.provider, self.usage_kind = provider, usage_kind
        self.events = []

    def __call__(self, request):
        event = {'request_sha256': digest(request), 'usage_kind': self.usage_kind,
                 'usage': None, 'error': None}
        self.events.append(event)
        started = time.perf_counter()
        try:
            response = self.provider(request)
            usage = response.get('usage') if isinstance(response, dict) else None
            if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                    for k in ('prompt_tokens', 'completion_tokens')):
                event['usage'] = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens')}
            return response
        except Exception as exc:
            event['error'] = type(exc).__name__
            raise
        finally:
            event['elapsed_ms'] = (time.perf_counter() - started) * 1000


def percentile(values, fraction):
    """Nearest-rank empirical quantile; null for no observations."""
    values = sorted(values)
    return values[max(0, math.ceil(len(values)*fraction)-1)] if values else None


def summarize(records):
    sem = [r for r in records if r['is_hit'] and r['tier'] == 'SEMANTIC']
    counts = {label: sum(r['answer_review']['label'] == label for r in sem)
              for label in ('sufficient', 'insufficient', 'unknown')}
    labeled = counts['sufficient'] + counts['insufficient']
    attempts = [e for r in records for e in r['provider_events']]
    complete = [r['elapsed_ms'] for r in records if r.get('error') is None]
    usages = [e['usage'] for e in attempts if e['usage'] is not None]
    # Missing usage never silently becomes zero; fixture counts never become billable tokens.
    measured = all(e['usage_kind'] == 'provider_reported' for e in attempts)
    return {
        'total_requests': len(records), 'completed_requests': len(complete),
        'errors': len(records)-len(complete),
        'exact_hits': sum(r['is_hit'] and r['tier'] == 'EXACT' for r in records),
        'semantic_hits': len(sem), 'semantic_review_counts': counts,
        'semantic_precision_labeled': counts['sufficient']/labeled if labeled else None,
        'semantic_precision_all_accepted': counts['sufficient']/len(sem) if sem and not counts['unknown'] else None,
        'precision_interval': None,
        'interval_reason': 'No population inference from correlated development fixtures; independent grouped review required.',
        'provider_attempts': len(attempts),
        'provider_errors': sum(e['error'] is not None for e in attempts),
        'usage_missing_attempts': len(attempts)-len(usages),
        'observed_usage_kind': sorted(set(e['usage_kind'] for e in attempts)),
        'billable_prompt_tokens': sum(u['prompt_tokens'] for u in usages) if measured and len(usages)==len(attempts) else None,
        'billable_completion_tokens': sum(u['completion_tokens'] for u in usages) if measured and len(usages)==len(attempts) else None,
        'monetary_cost': None, 'cost_reason': 'No live usage and dated rate table established.',
        'latency_p50_ms': percentile(complete, .5), 'latency_p95_ms': percentile(complete, .95),
        'latency_mean_ms': sum(complete)/len(complete) if complete else None,
        'latency_by_path': {path: {'n': len(v), 'p50_ms': percentile(v,.5), 'p95_ms': percentile(v,.95)}
                            for path in sorted({r['tier'] for r in records})
                            for v in [[r['elapsed_ms'] for r in records if r['tier']==path and r.get('error') is None]]},
    }


class ObservedApprover:
    """Capture the actual selected candidate and decision without changing the policy."""
    def __init__(self, approver):
        self.approver = approver
        self.identity = approver.identity
        self.events = []

    def __call__(self, request, candidate_request, answer, *, binding, provider=None):
        event = {'candidate_request': copy.deepcopy(candidate_request),
                 'candidate_response': copy.deepcopy(answer), 'binding': copy.deepcopy(binding)}
        self.events.append(event)
        try:
            verdict = self.approver(request, candidate_request, answer, binding=binding, provider=provider)
            event['decision'] = asdict(verdict)
            return verdict
        except Exception as exc:
            event['error'] = type(exc).__name__
            raise
