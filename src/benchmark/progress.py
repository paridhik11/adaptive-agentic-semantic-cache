"""Local, auto-refreshing experiment status page. No credentials or remote assets."""
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path
import tempfile


class ProgressPage:
    def __init__(self,path,*,run_id,model,total_requests):
        self.path=Path(path);self.arms={}
        self.state={'run_id':run_id,'model':model,'requests':total_requests,'status':'ready',
            'new_calls':0,'replayed_calls':0,'current':'Not started','diagnostics':None,'report':None}
        self._write()
    def __call__(self,event):
        kind=event['kind']
        if kind=='arm_started':
            self.state.update(status='running',current=f"{event['event_id']} / {event['arm']}: {event['question']}")
            print('RUN',event['event_id'],event['arm'],flush=True)
        elif kind=='arm_finished':
            self.arms[(event['event_id'],event['arm'])]=event
            print('RESULT',event['event_id'],event['arm'],event['status'],f"provider calls={event['provider_calls']}",flush=True)
        elif kind.startswith('provider_'):
            self.state['new_calls']=event['new_calls'];self.state['replayed_calls']=event['replayed_calls']
            if kind=='provider_started':self.state['status']='waiting for model'
        elif kind=='paused':self.state.update(status='paused',current=event['reason'],diagnostics=event.get('response_diagnostics'))
        elif kind=='completed':self.state.update(status='completed',current='Evidence saved to '+event['output_dir'],report=event['report'])
        elif kind=='waiting':self.state.update(status='waiting for minute quota',current=event['message'])
        self._write()
    def _write(self):
        esc=lambda value:html.escape(str(value))
        s=self.state;cards=''
        for label,value in [('Status',s['status']),('New API attempts this invocation',s['new_calls']),('Replayed calls',s['replayed_calls']),('Completed arms',len(self.arms))]:
            cards+=f'<div class="card"><small>{esc(label)}</small><strong>{esc(value)}</strong></div>'
        rows='';answers=''
        for (identity,arm),event in self.arms.items():
            rows+=f"<tr><td>{esc(identity)}</td><td>{esc(arm)}</td><td>{esc(event['status'])}</td><td>{event['provider_calls']}</td><td>{event['latency_ms']:.1f}</td></tr>"
            if event.get('answer'):answers+=f"<details><summary>{esc(identity)} / {esc(arm)} — answer preview, unreviewed</summary><pre>{esc(event['answer'])}</pre></details>"
        report=''
        if s['report']:
            r=s['report'];report=f"<h2>Outcome</h2><pre>{esc(json.dumps({'status_counts':r['status_counts'],'accepted_nonexact':r['nonexact_reuse']['accepted'],'provider_attempts_avoided':r['provider_attempts_avoided'],'latency_valid':r['latency_valid'],'safety_certified':False},indent=2))}</pre>"
        diagnostics=f"<h2>Why it paused</h2><pre>{esc(json.dumps(s['diagnostics'],indent=2))}</pre>" if s['diagnostics'] else ''
        refresh='' if s['status']=='completed' else '<meta http-equiv="refresh" content="2">'
        page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">{refresh}<title>Live cache experiment</title>
<style>body{{font:16px system-ui;background:#101823;color:#edf4fc;margin:30px auto;max-width:1100px;padding:0 18px}}p{{line-height:1.6;color:#bfd0e4}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}.card{{background:#1e3046;padding:18px;border-radius:10px}}strong{{display:block;font-size:24px;margin-top:8px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #3c526f}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1e3046;padding:18px}}details{{margin:16px 0}}.notice{{color:#ffd28d}}</style>
<h1>Live cache experiment</h1><p class="notice">Engineering pilot with authored repeat cases. Answers are unreviewed; this does not certify accuracy or real-traffic savings.</p>
<p>Run: {esc(s['run_id'])}<br>Model: {esc(s['model'])}<br>Updated: {esc(datetime.now(timezone.utc).isoformat())}</p><div class="grid">{cards}</div>
<h2>Current step</h2><p>{esc(s['current'])}</p><h2>Fresh generation vs cache</h2><table><thead><tr><th>Request</th><th>Arm</th><th>Status</th><th>Model calls</th><th>Processing ms</th></tr></thead><tbody>{rows}</tbody></table>
{diagnostics}{report}<h2>Returned answer previews</h2>{answers}<p>This local page refreshes while the process runs. It contains no API key and makes no third-party requests. Replayed-call timings are not live latency measurements.</p></html>'''
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='status-',suffix='.html',dir=self.path.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(page)
            os.replace(name,self.path)
        finally:
            if os.path.exists(name):os.unlink(name)
