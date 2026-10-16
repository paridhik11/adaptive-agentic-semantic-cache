"""Bounded SSE passthrough with explicit completion and cleanup.

Streaming deliberately bypasses cache reads and admission. Partial generations
are never reusable entries. Service slots remain held until completion/close.
"""
from __future__ import annotations
import json
import anyio
import threading
import time
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from src.api.providers import ProviderError


def _reject(_):raise ValueError('non-finite JSON')


def encode_event(event):
    return ('data: '+(event if event=='[DONE]' else json.dumps(event,allow_nan=False,separators=(',',':')))+'\n\n').encode()


class SSEReader:
    def __init__(self,response,*,max_bytes=4194304,max_event_bytes=65536,
                 max_seconds=180,clock=time.monotonic):
        self.response=response;self.max_bytes=max_bytes;self.max_event_bytes=max_event_bytes
        self.closed=False;self.clock=clock;self.deadline=clock()+max_seconds
    def _lines(self):
        total=0;buffer=bytearray()
        read=getattr(self.response,'read1',self.response.read)
        while not self.closed:
            if self.clock()>self.deadline:raise ProviderError('upstream_stream_deadline')
            raw=read(min(4096,self.max_bytes-total+1))
            if self.clock()>self.deadline:raise ProviderError('upstream_stream_deadline')
            if not raw:raise ProviderError('upstream_stream_incomplete')
            total+=len(raw)
            if total>self.max_bytes:raise ProviderError('upstream_stream_too_large')
            buffer.extend(raw)
            while b'\n' in buffer:
                index=buffer.index(b'\n')+1
                line=bytes(buffer[:index]);del buffer[:index]
                yield line
            if len(buffer)>self.max_event_bytes:raise ProviderError('upstream_stream_too_large')
    def __iter__(self):
        lines=[];event_bytes=0
        for raw in self._lines():
            event_bytes+=len(raw)
            if event_bytes>self.max_event_bytes:raise ProviderError('upstream_stream_too_large')
            line=raw.decode('utf-8').rstrip('\r\n')
            if not line:
                if lines:
                    text='\n'.join(lines)
                    if text=='[DONE]':yield text;return
                    yield json.loads(text,parse_constant=_reject)
                lines=[];event_bytes=0
            elif line.startswith('data:'):
                lines.append(line[5:].lstrip(' '))
    def close(self):
        if not self.closed:
            self.closed=True;self.response.close()


class StreamSession:
    def __init__(self,source,metrics,release,started):
        self.source=source;self.iterator=iter(source);self.metrics=metrics;self.release=release
        self.started=started;self.closed=False;self.done=False;self.finished=False;self.has_text=False
        self.usage_seen=False;self.failed=False;self.first=False;self.lock=threading.Lock()
    def __iter__(self):return self
    def __next__(self):
        if self.closed:raise StopIteration
        try:
            event=next(self.iterator)
            if event=='[DONE]':
                if not self.finished or not self.has_text:raise ValueError('incomplete stream')
                self.done=True;self.metrics.increment('stream_completed_total');self.close()
                return encode_event('[DONE]')
            if not isinstance(event,dict) or 'error' in event or not isinstance(event.get('choices'),list):
                raise ValueError('invalid event')
            choices=event['choices']
            if len(choices)>1:raise ValueError('multiple choices unsupported')
            for choice in choices:
                if self.finished or not isinstance(choice,dict) or choice.get('index',0)!=0:raise ValueError('invalid choice sequence')
                delta=choice.get('delta',{})
                if not isinstance(delta,dict) or delta.get('tool_calls') or delta.get('function_call') or delta.get('refusal'):
                    raise ValueError('unsupported delta')
                content=delta.get('content')
                if content is not None and not isinstance(content,str):raise ValueError('invalid content')
                if content:
                    self.has_text=True
                    if not self.first:
                        self.first=True
                        self.metrics.increment('stream_first_text_microseconds_total',int((time.perf_counter()-self.started)*1_000_000))
                        self.metrics.increment('stream_first_text_count')
                finish=choice.get('finish_reason')
                if finish is not None:
                    if finish!='stop':raise ValueError('stream not fully completed')
                    self.finished=True
            if event.get('usage') is not None:
                if self.usage_seen or not self.finished:raise ValueError('usage must occur once after finish')
                # Validate once; absence/invalid usage remains explicit in metrics.
                self.metrics.usage(event);self.usage_seen=True
            return encode_event(event)
        except Exception:
            self.failed=True;self.metrics.increment('upstream_errors_total')
            self.metrics.increment('stream_failed_total');self.close()
            return encode_event({'error':{'type':'upstream_stream_failed','message':'Stream ended without a complete answer.'}})
    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            try:
                if hasattr(self.source,'close'):self.source.close()
            except Exception:
                self.metrics.increment('stream_cleanup_errors_total')
            finally:
                if not self.usage_seen:self.metrics.increment('upstream_usage_missing_total')
                if not self.done and not self.failed:self.metrics.increment('stream_cancelled_total')
                self.metrics.finish('BYPASS' if self.done else 'ERROR',time.perf_counter()-self.started)
                self.release()


class ManagedStreamingResponse(StreamingResponse):
    def __init__(self,session,**kwargs):
        self.session=session
        super().__init__(session,media_type='text/event-stream',**kwargs)
    async def __call__(self,scope,receive,send):
        try:
            await super().__call__(scope,receive,send)
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self.session.close)
