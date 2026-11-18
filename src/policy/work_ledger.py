"""Request-owned stage observations. Never infer provider work from cache misses."""
from contextlib import contextmanager
import copy
import time


class WorkLedger:
    STAGES = frozenset({'planning', 'generation', 'review', 'embedding', 'search',
                        'applicability', 'tool', 'admission', 'memory_embedding', 'memory_search'})

    def __init__(self):
        self.events = []

    @contextmanager
    def measure(self, stage):
        if stage not in self.STAGES:
            raise ValueError('unknown work stage')
        event = {'stage': stage, 'completed': False, 'error': None}
        self.events.append(event)
        started = time.perf_counter()
        try:
            yield event
            event['completed'] = True
        except BaseException as exc:
            event['error'] = type(exc).__name__
            raise
        finally:
            event['elapsed_ms'] = (time.perf_counter()-started)*1000

    def snapshot(self):
        return copy.deepcopy(self.events)
