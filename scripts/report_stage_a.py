"""Recompute metrics from preserved real-model scores; no models or final data."""
import hashlib,json,sys,collections,platform,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load(p,jsonl=False):
 raw=p.read_bytes();print(p.resolve(),len(raw),hashlib.sha256(raw).hexdigest())
 return [json.loads(l) for l in raw.splitlines()] if jsonl else json.loads(raw)
def main():
 from src.evaluation.intent_stage_a import metrics
 base=ROOT/'artifacts/stage_a/baselines-v1';old=load(base/'metrics.json');rows=load(base/'scores.jsonl',True);supp=load(ROOT/'configs/intent_stage_a_redis_supplement.json')
 out=ROOT/'artifacts/stage_a/verified'
 if out.exists():raise SystemExit('STOP: verified outputs exist; preserve them')
 out.mkdir()
 def decisions(b):
  name,t=b['baseline'],b['threshold']
  if name=='always_miss':return [False]*len(rows)
  if name=='exact_only':return [r['is_exact'] for r in rows]
  field={'minilm_cosine':'minilm_cosine','redis_embedding_cosine':'redis_cosine','quora':'quora_score','redis_reranker':'redis_reranker_score','existing_local_equivalence':'quora_score'}[name]
  return [r[field]>=t and (name!='existing_local_equivalence' or r['existing_guards_passed']) for r in rows]
 results=[]
 for b in old['baselines']:
  d=decisions(b)
  results.append({'baseline':b['baseline'],'threshold':b['threshold'],'provenance_flags':b['provenance_flags'],'public_pairs_used':False,**metrics(rows,d)})
 supplements=[]
 for t in supp['thresholds']:
  supplements.append({'baseline':'redis_reranker','threshold':t,'evidence_type':'same-set development diagnostic','provenance_flags':['possible pretraining overlap with public pair data'],'public_pairs_used':False,**metrics(rows,[r['redis_reranker_score']>=t for r in rows])})
 verified={**old,'baselines':results,'redis_supplementary_same_set':supplements,'supplement_config':supp,
  'metrics_note':'Recomputed with corrected group classification: a known unsafe accept fails its group even if another accepted label is ambiguous. Original metrics.json retained. No score/model/threshold changes to primary grid.',
  'runtime_versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','sentence-transformers','numpy','scipy','safetensors']}}
 (out/'metrics.json').write_text(json.dumps(verified,indent=2)+'\n')
 # Compact all-case decisions/failure reasons for every baseline/threshold.
 events=[]
 for b in [*results,*supplements]:
  for r,d in zip(rows,decisions(b)):
   events.append({'case_id':r['case_id'],'cached_case_id':r['cached_case_id'],'cached_answer_id':r['cached_answer_id'],
    'baseline':b['baseline'],'threshold':b['threshold'],'accept':bool(d),'relation':r['relation_to_cached'],'family':r['case_family'],
    'cause':('exact_control' if r['is_exact'] else 'accepted' if d else r['existing_guard_reason'] if b['baseline']=='existing_local_equivalence' and not r['existing_guards_passed'] else 'below_threshold_or_always_miss'),
    'provenance_flags':b['provenance_flags']})
 (out/'decisions.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in events))
 manifest=load(ROOT/'data/intent_stage_a/manifest.json');receipt=load(ROOT/'docs/audit/STAGE_A_REDIS_RECEIPT.json')
 lines=['# Stage A report — stop after A4','',
 'Evidence: developer-authored and template-generated development set. Not independent human review, calibration or final evaluation. No threshold selected; no trained model. Same-set supplementary Redis diagnostics are labeled below.',
 '', '## Files and preservation','',
 '`data/intent_stage_a/` contains the dataset, split answer stores, manifests and dev metadata-completion provenance. `configs/intent_stage_a*.json` freeze the primary and supplementary grids. `scripts/build_stage_a_intents.py`, `scripts/complete_stage_a_dev_signatures.py`, `scripts/run_stage_a_baselines.py`, `scripts/report_stage_a.py`, and `src/evaluation/intent_stage_a.py` implement generation, inference and policy-agnostic metrics. `artifacts/stage_a/` preserves original scores, candidates, metrics, logs, corrected metrics and all decisions. Nothing was committed, pushed or branched.',
 '', '## Dataset counts','', '| Split | Groups | Source lineages | Rows | New-question rows |','| --- | ---: | ---: | ---: | ---: |']
 for split,s in manifest['splits'].items():lines.append(f"| {split} | {s['groups']} | {s['source_lineages']} | {s['rows']} | {s['new_rows']} |")
 lines+=['', 'Groups are task contracts, not independent observations. Generated groups vary operation, input container and a material response requirement. Shared operation lineage is kept within one split. Formatter templates are shared across splits, so split isolation does not establish broad linguistic generalization.',
 '', 'All 75 old cases are spent dev data. Historical wrong answers remain available for audit and can have same-intent pair labels. Generated answers are authored code examples, not measured LLM generations; presentation/version compliance is not independently reviewed. No public corpus was imported: existing historical sealed/FAQ/pilot material was excluded, and no vetted license-checked local public-pair corpus was selected.',
 '', 'Final manifest SHA-256: `'+manifest['final_manifest_sha256']+'`. Written into already tracked `docs/FACTS.md` at creation. Final questions and answers were not evaluated; the harness rejects `--split final` before data/model loading. Only final manifest metadata/hashes are checked. Train/calibration received no model evaluation.',
 '', '### Counts per split, family and provenance','', '```json',json.dumps({s:{'families':v['families'],'provenance':v['provenance'],'relations':v['relations']} for s,v in manifest['splits'].items()},indent=2),'```',
 '', '## A4 primary fixed-grid baseline table','',
 'DESIGNATED PAIRS: every policy sees the same cached/new question pair. These are not serving hit-rate measurements. Exact controls are excluded from semantic precision/coverage. Precision is null when no non-exact accepts exist or accepted labels are unknown. Unknowns are never imputed safe.',
 '', '| Baseline | Threshold | Non-exact accepts | Pair precision | Positive coverage | Unsafe | Unknown | Overlap flag* |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
 def table(b):
  precision='null' if b['precision_among_accepted'] is None else f"{b['precision_among_accepted']:.4f}"
  coverage='null' if b['positive_coverage'] is None else f"{b['positive_coverage']:.4f}"
  return f"| {b['baseline']} | {b['threshold']} | {b['accepted_nonexact']} | {precision} | {coverage} | {b['unsafe_accepts']} | {b['accepted_unknown']} | {'yes' if b['provenance_flags'] else 'not asserted'} |"
 lines += [table(b) for b in results]
 lines+=['', '*Flag text: **possible pretraining overlap with public pair data**. Applied to Quora, both Redis models, and the Quora-based existing guard stack; no public pairs are used in this dataset. The flag must stay in future public-pair evaluation tables. Unflagged does not mean proven contamination-free.',
 '', '### Redis supplementary grid — SAME-SET diagnostic, no new inference','',
 'The primary 0.5–0.99 grid rejected everything. A broader low-score grid and publisher-provided thresholds are applied to the saved logits. This is exposed-dev descriptive analysis, not an independently selected operating point or calibration result.',
 '', '| Baseline | Threshold | Non-exact accepts | Pair precision | Positive coverage | Unsafe | Unknown | Overlap flag* |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
 lines += [table(b) for b in supplements]
 lines+=['', '### Retrieval, separately','', '```json',json.dumps(old['retrieval'],indent=2),'```',
 '', 'Every dev query has an immutable saved MiniLM top-5 candidate list. Decision baselines are compared on designated pairs; they do not rescore the retrieved bank in this Stage A study. A later integrated policy must be evaluated on actual retrieved candidates and their actual relations. Positive recall alone does not establish negative rejection.',
 '', '## Group/lineage endpoints and uncertainty','',
 'For each table row, verified `metrics.json` includes accepted-group and accepted-lineage counts, all-accepted-safe outcomes, failed/unknown outcomes and exact one-sided 95% Clopper–Pearson bounds. A single definitive unsafe accept fails its group even when another label is unknown. Answer-level binomial intervals are intentionally absent because paraphrases are correlated. Group and lineage bounds are explicitly conditional on independent Bernoulli sampling, which these purposive templates do not establish. None is a population-safety guarantee.',
 '', '## Failures and unavailable work','',
 '- Historical parsed-call policy: **NOT RUN**. Requires `deepset/minilm-uncased-squad2@934656cdda79824eabf503ed56e15c01ddbdbe3f` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli@6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. They are absent and not approved for download here. No helper was substituted as that baseline.',
 '- Baselines show substantial unsafe accepts or coverage collapse. No policy is selected or enabled from these results. Every unsafe accept and missed positive is listed per baseline in verified metrics and decisions JSONL, with case family and cause.',
 '- Existing local_equivalence has an unpinned lazy loader and a heuristic logit/probability conversion. The harness injects actual pinned Quora probabilities into the unchanged stack and disables its remote fallback. Model and guard timings here are diagnostic, not a serving-cost comparison.',
 '- The initial harness preflight joined on event `case_id` instead of logical `cached_case_id`; fixed before inference. Original error log preserved. These are separate schema identifiers, not duplicate final records.',
 '- Legacy signatures were completed before inference, and two metadata-completion receipts preserve that process. Any remaining unverified signature metadata is explicitly marked and is not a policy input; Stage B veto construction must review it.',
 '- Generated labels are builder-authored assumptions; ambiguous relation is null. The locked template-generated final is not independent human ground truth.',
 '', '## Redis artifact identities and licenses','']
 for m in receipt['models']:
  lines.append('- `'+m['repo_id']+'@'+m['revision']+'`: '+('current main at metadata check' if m['revision_is_current_main'] else 'older than current main')+'; card license: '+str(m['model_card_license'])+' ('+m['license_status']+').')
 lines+=['', 'Newer embedding versions `redis/langcache-embed-v2` and `redis/langcache-embed-v3-small` exist. No later numbered BCE reranker was listed in the checked Redis repositories; `redis/langcache-reranker-v2-mnrl` is a later-created alternative loss variant, not a newer BCE revision. All metadata IDs and timestamps are in the receipt. No newer model files were downloaded. The embed-v1 card metadata does not declare a license; it is recorded as unknown rather than inferred from its base model.',
 '', '## Exact commands','', 'Run from the repository root. Dependencies are already installed under the local runtime path here. Use a fresh output directory when repeating inference; never overwrite a baseline run.',
 '', '```bash',
 'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/download_redis_stage_a.py --output docs/audit/STAGE_A_REDIS_RECEIPT.json',
 'python scripts/inventory_intent_artifacts.py --download-receipt docs/audit/STAGE_A_DOWNLOAD_RECEIPT.json --extra-receipt docs/audit/STAGE_A_REDIS_RECEIPT.json --output docs/audit/STAGE_A_INVENTORY.json --replace-existing',
 'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/build_stage_a_intents.py',
 'python scripts/complete_stage_a_dev_signatures.py',
 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/run_stage_a_baselines.py --split dev --output-dir artifacts/stage_a/baselines-v1',
 'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_a.py',
 'PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py',
 '```',
 '', 'Generation refuses an existing dataset directory; do not regenerate the locked final. Restore the archived dataset/manifests for exact replication. Download/report commands also preserve existing receipts/output directories. Original output paths above identify the actual run; use fresh paths or a clean clone for recreation.',
 '', '## Stop / next approval','',
 'Stage B training, veto construction, calibration and Stage C integration were not performed. Final evaluation, live/provider calls, UI/agent changes, serving enablement, Git commits/pushes/branches were not performed. No trained model exists to export. If Stage B is authorized later, its learned artifact must be exported as safetensors with hash, base revision, seed and exact training command, and added to the retrievable backup.',
 '', 'Minimum user action: back up the supplied archive, review this Stage A report, then give the explicit Stage B go-ahead if desired. Completing the missing historical parsed-call comparison additionally needs separate approval for its two pinned checkpoints; otherwise that comparator remains unavailable and B5 cannot claim to beat it.']
 (ROOT/'docs/audit/STAGE_A_REPORT.md').write_text('\n'.join(lines)+'\n')
 print('Report and verified metrics written; no inference or final data read.')
if __name__=='__main__':main()
