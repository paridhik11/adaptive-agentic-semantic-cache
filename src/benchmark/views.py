"""Offline answer review and report UI. All untrusted content uses textContent."""
import json


def review_html(trace, template, report):
    payload = json.dumps({'trace': trace, 'reviews': template, 'report': report}, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Answer review · Semantic Cache</title>
<style>body{font:16px system-ui;background:#101823;color:#eaf0f8;margin:0;padding:28px;max-width:1150px;margin:auto}h1{margin-bottom:8px}p{color:#b6c7dc;line-height:1.6}.cards,.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}.card,article{background:#1b293b;border:1px solid #344a64;border-radius:12px;padding:20px;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}button,select,input{font:inherit;padding:10px;border-radius:6px;border:1px solid #6a839f;margin:6px;background:#f4f8fc;color:#101823}label{display:inline-block}strong{font-size:22px}small{color:#a9c3df}.warn{color:#ffce80}</style>
<h1>Answer-bound review</h1><p>This view compares the actual served answer with fresh generation. Review correctness against the request and trustworthy references; a fresh answer is not ground truth. Unknown labels remain unknown.</p>
<p id="mode" class="warn"></p><div id="summary" class="cards"></div>
<label>Your reviewer identifier <input id="reviewer" placeholder="Reviewer name or ID"></label>
<button id="download">Download review JSON</button><p id="notice" role="status"></p>
<p>Local review only. No network calls or browser storage. Download your labels before closing. Review files contain answer hashes; this page itself contains request and response text.</p><div id="events"></div>
<script id="payload" type="application/json">'''+payload+'''</script><script>
'use strict';const data=JSON.parse(document.getElementById('payload').textContent);
const el=(tag,text,parent)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(parent)parent.append(n);return n;};
document.getElementById('mode').textContent='Mode: '+data.trace.evaluation_mode+' · Split: '+data.trace.split+' · Safety certification: NOT ESTABLISHED';
for(const [title,value] of [['Requests',data.report.requests],['Accepted reuse',data.report.overall_reuse.accepted],['Unreviewed hits',data.report.overall_reuse.unreviewed],['Non-exact hits',data.report.nonexact_reuse.accepted]]){const card=el('div',undefined,document.getElementById('summary'));card.className='card';el('p',title,card);el('strong',String(value),card);}
data.trace.events.forEach((event,i)=>{const r=data.reviews.reviews[i],article=el('article',undefined,document.getElementById('events'));el('h2',event.event_id+' · '+event.status,article);el('small','Source: '+event.source+' · Group: '+event.group_id+' · Candidate: '+(event.candidate_event_id||'none'),article);el('h3','Request',article);el('pre',JSON.stringify(event.request,null,2),article);const answers=el('div',undefined,article);answers.className='answers';for(const [arm,label] of [['cache','Served answer'],['fresh','Fresh baseline']]){const box=el('div',undefined,answers);el('h3',label,box);el('pre',JSON.stringify(event[arm].response,null,2),box);if(event[arm].error)el('p','ERROR: '+event[arm].error,box);}
for(const [key,title,enabled] of [['safe_reuse','Was reuse safe?',event.status==='HIT'],['answer_correct','Is the served answer correct?',!event.cache.error],['fresh_answer_correct','Is the fresh answer correct?',!event.fresh.error]]){if(!enabled)continue;const label=el('label',title,article),select=el('select',undefined,label);for(const [v,t] of [['','Unknown'],['true','Yes'],['false','No']]){const o=el('option',t,select);o.value=v;}select.onchange=()=>r[key]=select.value===''?null:select.value==='true';}
const note=el('input',undefined,article);note.placeholder='Reference / reasoning / uncertainty';note.setAttribute('aria-label','Review notes');note.oninput=()=>r.notes=note.value;
});
document.getElementById('download').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){document.getElementById('notice').textContent='Enter a reviewer identifier first.';return;}for(const row of data.reviews.reviews){row.reviewer=reviewer;row.method='human';}const blob=new Blob([JSON.stringify(data.reviews,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='answer-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('notice').textContent='Downloaded. Run the report command with these labels to recompute metrics.';};
</script></html>'''
