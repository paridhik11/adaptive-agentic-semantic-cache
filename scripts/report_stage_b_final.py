"""B5 honest closeout from immutable scored evidence; does not run any model."""
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_b_common import config,write_json,identity,FLAG
from src.evaluation.intent_stage_b_policy import decision_view,summarize,cp_error


def read(path):return json.loads((ROOT/path).read_text())
def pct(value):return 'null' if value is None else f'{value*100:.2f}%'
def compact(result):return {k:v for k,v in result.items() if k!='cases'}


def main():
    cfg=config();selection=read('artifacts/stage_b/calibration-v1/selection.json')['selected']
    frozen=read('artifacts/stage_b/calibration-v1/selected-metrics.json')
    training=read('artifacts/stage_b/training-v1.json')
    trainfit=read('artifacts/stage_b/train-classification-v1.json')['seeds']
    latency=read('artifacts/stage_b/local-latency-v1.json')
    records={s:[json.loads(x) for x in gzip.decompress((ROOT/f'artifacts/stage_b/{s}/scored.jsonl.gz').read_bytes()).decode().splitlines()] for s in ['dev','calibration']}
    classification=[]
    for split in ['dev','calibration']:
        predictions=read(f'artifacts/stage_b/{split}/designated-classification.json')['predictions']
        mapping={r['case_id']:r for r in records[split]}
        for seed,rows in predictions.items():
            known=[r for r in rows if r['label'] is not None]
            item={'split':split,'seed':int(seed),'n':len(known),'accuracy':sum(r['label']==r['predicted'] for r in known)/len(known),
                'confusion':dict(Counter(r['label']+' -> '+r['predicted'] for r in known)),
                'per_lineage':{},'per_template':{}}
            for field,key in [('source_lineage_id','per_lineage'),('template_id','per_template')]:
                for value in sorted({mapping[r['case_id']][field] for r in known}):
                    sub=[r for r in known if mapping[r['case_id']][field]==value]
                    item[key][value]={'n':len(sub),'correct':sum(r['label']==r['predicted'] for r in sub)}
            classification.append(item)
    comparisons=[];subgroups=[];family=[];negative_suites=[];baseline_tables={}
    for split in ['dev','calibration']:
        baselines=read(f'artifacts/stage_b/{split}/baseline-comparison.json')
        baseline_tables[split]=baselines
        for selected in [r for r in frozen if r['split']==split]:
            seed=selected['seed'];threshold=selected['threshold'];rs=records[split]
            decisions=[decision_view(r,seed,threshold,cfg['competition_margin']) for r in rs]
            caseview=summarize(rs,decisions)['cases']
            for name in sorted({b['baseline'] for b in baselines}):
                points=[b for b in baselines if b['baseline']==name and b['unsafe_accepts']<=selected['unsafe_accepts'] and b['accepted_unknown']<=selected['accepted_unknown']]
                best=max(points,key=lambda x:x['safe_coverage']) if points else None
                comparisons.append({'split':split,'seed':seed,'baseline':name,
                    'policy_safe_coverage':selected['safe_coverage'],'policy_unsafe_accepts':selected['unsafe_accepts'],
                    'baseline_max_coverage_at_no_greater_observed_unsafe_and_unknown_count':best['safe_coverage'] if best else None,
                    'baseline_threshold':best['threshold'] if best else None,
                    'strict_coverage_improvement':bool(best and selected['safe_coverage']>best['safe_coverage']),
                    'caveat':'observed same-cohort point comparison only; not independent or causal evidence',
                    'provenance_flags':[] if name in ['always_miss','exact_only','minilm_cosine'] else [FLAG]})
            for field in ['template_id','source_lineage_id']:
                for value in sorted({r[field] for r in rs}):
                    indices=[i for i,r in enumerate(rs) if r[field]==value]
                    result=summarize([rs[i] for i in indices],[decisions[i] for i in indices])
                    subgroups.append({'split':split,'seed':seed,'field':field,'value':value,**compact(result)})
            for value in sorted({r['case_family'] for r in rs}):
                indices=[i for i,r in enumerate(rs) if r['case_family']==value]
                result=summarize([rs[i] for i in indices],[decisions[i] for i in indices])
                family.append({'split':split,'seed':seed,'family':value,**compact(result)})
                negatives=[i for i in indices if not rs[i]['suitable_candidate_ids'] and
                           rs[i]['exact_candidate'] is None and rs[i]['relation_to_cached'] is not None]
                endpoints={}
                for field in ['group_id','source_lineage_id']:
                    clusters={rs[i][field] for i in negatives}
                    failed={rs[i][field] for i in negatives if caseview[i]['unsafe']}
                    endpoints[field]={'tested_clusters':len(clusters),'clusters_with_unsafe_accept':len(failed),
                                      'conditional_error_upper_95':cp_error(len(failed),len(clusters)),
                                      'population_guarantee_established':False}
                negative_suites.append({'split':split,'seed':seed,'family':value,'negative_queries':len(negatives),
                    'definition':'no suitable cached contract anywhere in identical scope; definite question label only',
                    'evidence_type':'synthetic_negative_suite_conditional_cluster_arithmetic',**endpoints})
    write_json(ROOT/'artifacts/stage_b/classification-generalization.json',{
        'evidence_type':'synthetic_designated_pair_classification_diagnostic_not_reuse_precision',
        'train':{s:{k:v for k,v in r.items() if k!='predictions'} for s,r in trainfit.items()},'held_out':classification})
    write_json(ROOT/'artifacts/stage_b/comparisons.json',comparisons)
    write_json(ROOT/'artifacts/stage_b/per-template-lineage.json',subgroups)
    write_json(ROOT/'artifacts/stage_b/per-family.json',family)
    write_json(ROOT/'artifacts/stage_b/hard-negative-safety.json',negative_suites)
    manifest=identity(ROOT/'data/intent_stage_a/final.manifest.json',False)
    assert manifest['sha256']=='1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af'
    primary=cfg['seeds'][0]
    dev=next(x for x in classification if x['split']=='dev' and x['seed']==primary)
    cal=next(x for x in classification if x['split']=='calibration' and x['seed']==primary)
    lede=f"Primary-seed train four-class accuracy is {pct(trainfit[str(primary)]['accuracy'])}; held-out dev is {pct(dev['accuracy'])}, calibration {pct(cal['accuracy'])}. These are synthetic designated-pair diagnostics, not reuse precision or independent accuracy."
    if trainfit[str(primary)]['accuracy']-min(dev['accuracy'],cal['accuracy'])>.1:
        lede='**GENERALIZATION FAILURE / TEMPLATE-MEMORIZATION WARNING.** '+lede
    guard=next(b for b in baseline_tables['dev'] if b['baseline']=='existing_guard_stack' and b['threshold']==.7)
    lede+=f" The unchanged guarded Quora baseline at 0.7 achieves {guard['safe_accepts']}/{guard['positive_opportunities']} safe dev reuses with {guard['unsafe_accepts']} observed unsafe accepts under the synthetic oracle."
    report=['# Stage B closeout — STOP after B5','',lede,'',
        'Labels are programmatic synthetic contracts authored by Codex (model-assisted construction), not human review. General semantic serving remains **DISABLED**. No human-written natural set has been returned. Nothing here is independent human ground truth. Models, generated contracts and the bounded veto vocabulary were authored by the same builder; template holdout alone does not remove that dependence.','',
        '## 1. Local branches and commits','',
        'Stage A: `6cea9ec` harness → `ec77788` data → `699223a` results. Stage B: `stage-b/classifier` from results. B0–B4 commits below; B5 is the report commit immediately following this generation. No push, merge, main rebase or history rewrite.','', '```text',
        subprocess.check_output(['git','log','--oneline','stage-a/results..HEAD'],cwd=ROOT,text=True).strip(),'```','',
        'After B5: `git log --oneline stage-a/results..stage-b/classifier` gives the complete six-commit Stage B series. The backup includes the final Git bundle.','',
        '## 2. Files and experimental scope','',
        'New versioned datasets: data/intent_stage_b/v2. Unmeasured v1 construction draft and its generator are preserved. Train 96 groups/24 operation lineages, dev 64/8, calibration 256/8; whole question formatters and requirement/negative wrappers are disjoint across splits. All families and provenance counts are in each manifest. More numeric/version calibration variants increase group count, not independent lineage diversity.','',
        'Harness: build_stage_b_intents.py, stage_b_common.py, train_stage_b_classifier.py, diagnose_stage_b_train.py, review_stage_b_veto.py, score_stage_b_candidates.py, report_stage_b_dev.py, calibrate_stage_b_policy.py, benchmark_stage_b_local.py, report_stage_b_final.py, backup_stage_b.py. Model/veto/policy are isolated in src/evaluation/; production code is unchanged. Configs and protocol record all settings. tests/test_intent_stage_b.py covers isolation, nulls, actual-candidate attribution and veto/competition invariants.','',
        'Scored candidate files, predictions, model receipts, baseline grids, selected policies, comparisons, every selected-policy failure, per-family/template/lineage tables and timing observations are under artifacts/stage_b/. Raw large logs use deterministic lossless gzip. Model updates remain ignored by Git.','',
        '## 3. Exact commands','', 'Run in an isolated Linux/WSL reproduction checkout with the pinned runtime and empty generated output directories. Native Windows parallel-runner support is not validated (worker locks use fcntl). Byte-identical original Stage A generator replay was verified on Linux; platform newline behavior elsewhere is not guaranteed. Existing evidence is deliberately not overwritten.','', '```bash',
        'python scripts/build_stage_b_intents.py --output-dir scratch/stage-b-rebuild',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/train_stage_b_classifier.py',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/diagnose_stage_b_train.py',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/review_stage_b_veto.py',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/score_stage_b_candidates.py --split dev --output-dir artifacts/stage_b/dev',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_dev.py --split dev',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/run_stage_b_scoring.py --split calibration --output-dir artifacts/stage_b/calibration',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/calibrate_stage_b_policy.py',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_dev.py --split calibration',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/benchmark_stage_b_local.py',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_final.py',
        'PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py tests/test_intent_stage_b.py', '```','',
        'Pretrained cache restoration uses only scripts/download_stage_a_checkpoints.py and scripts/download_redis_stage_a.py with their already-approved exact revisions. The restored inventory and receipts are preserved. Stage B restores and experiments share the same current environment. The original Stage A evidence came from an earlier host; all baseline methods were rerun within Stage B on its current cohorts. See environment-provenance.json; no cross-machine latency claim. No pickle or remote code was loaded.','',
        '## 4. Results — synthetic template-held-out actual-candidate evidence','',
        '| Split | Seed | Threshold | Safe / accepted | Unsafe / unknown | Precision | Safe coverage | Retrieval recall@3 |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in frozen:
        report.append(f"| {r['split']} | {r['seed']} | {r['threshold']} | {r['safe_accepts']}/{r['accepted_nonexact']} | {r['unsafe_accepts']} / {r['accepted_unknown']} | {pct(r['precision'])} | {pct(r['safe_coverage'])} ({r['safe_accepts']}/{r['positive_opportunities']}) | {pct(r['retrieval_recall_at_k'])} |")
    report += ['', 'Safe coverage counts actual cache-bank matches, including a designated negative that safely matches another cached question. A retrieval hit does not imply classifier acceptance. Exact hits are excluded from semantic precision and coverage. Precision with no accepts is null. Quora-derived policies and both Redis baselines carry **possible pretraining overlap with public pair data**; public pairs were not used.', '',
        'Calibration used only calibration labels. Threshold/config hashes, adapter/base/embedding hashes, veto version, seed and abstention settings are frozen per policy. Exact one-sided group bounds use alpha=0.05/30 for the prespecified threshold/seed family. Pointwise group/lineage bounds are also available in baseline/subgroup tables. These are conditional arithmetic under independent Bernoulli cluster assumptions not established by purposive shared-template/lineage sets. No population guarantee follows.','',
        '| Split | Seed | Accepted groups / failures | Adjusted group error upper | Accepted lineages / failures | Adjusted lineage error upper |','| --- | --- | --- | --- | --- | --- |']
    for r in frozen:
        report.append(f"| {r['split']} | {r['seed']} | {r['groups']['accepted']} / {r['groups']['failed']} | {pct(r['groups']['conditional_error_upper'])} | {r['lineages']['accepted']} / {r['lineages']['failed']} | {pct(r['lineages']['conditional_error_upper'])} |")
    grid=read('artifacts/stage_b/calibration-v1/calibration-grid.json')
    low=[r for r in grid if r['threshold']==.5]
    report += ['', 'Why calibration failed: at the fixed 0.5 diagnostic threshold, '+
        '; '.join(f"seed {r['seed']} accepts {r['safe_accepts']} safe cases across {r['groups']['accepted']} groups" for r in low)+
        '. No fixed-grid point meets the adjusted group-bound requirement. This is chiefly lack of acceptance evidence, not a discovered zero-error guarantee. Raising thresholds cannot repair missing useful coverage.', '']
    report += ['', '### Fixed Stage A baseline settings on the new cohorts', '',
        'Evidence: actual retrieved-candidate synthetic development/calibration diagnostics, not human truth. These are the previously fixed baseline settings, not newly selected operating points.', '',
        '| Split | Baseline / threshold | Safe / accepted | Unsafe / unknown | Safe coverage | Accepted groups / failures | Pointwise group error upper 95% |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    fixed={('minilm_cosine',.8),('redis_embedding_cosine',.8),('quora',.7),('existing_guard_stack',.7),('redis_reranker_same_set_diagnostic',.0888671875)}
    for split,baselines in baseline_tables.items():
        for b in baselines:
            if (b['baseline'],b['threshold']) in fixed:
                report.append(f"| {split} | {b['baseline']} / {b['threshold']} | {b['safe_accepts']}/{b['accepted_nonexact']} | {b['unsafe_accepts']} / {b['accepted_unknown']} | {pct(b['safe_coverage'])} | {b['groups']['accepted']} / {b['groups']['failed']} | {pct(b['groups']['conditional_error_upper'])} |")
    report += ['', 'These baseline bounds are pointwise and conditional on the unestablished independent-cluster assumptions; they are not the multiplicity-adjusted trained-policy gate. Lineage bounds and every fixed grid point are in each baseline-comparison.json. No baseline is promoted or substituted for the prespecified primary seed.']
    report += ['', '### Baseline comparison at matched observed unsafe/unknown budget', '',
        '| Split, primary seed | Comparator | Policy safe coverage | Best comparator coverage within budget | Strict improvement? |','| --- | --- | --- | --- | --- |']
    for c in comparisons:
        if c['seed']==primary:
            report.append(f"| {c['split']}, {primary} | {c['baseline']} | {pct(c['policy_safe_coverage'])} | {pct(c['baseline_max_coverage_at_no_greater_observed_unsafe_and_unknown_count'])} | {c['strict_coverage_improvement']} |")
    report += ['', 'All seed comparisons and fixed baseline points are saved, including negative results. Redis embedding thresholds use the common MiniLM candidate pool, not an independently optimized Redis retrieval system. The Redis low-threshold sweep is explicitly a same-set diagnostic. Baseline selectors retain their original candidate guard/ranking behavior. The new policy has additional competition abstention and a top-choice veto without fallback, so these are whole-policy comparisons, not an isolated causal effect of fine-tuning. Historical parsed-call comparator: **UNAVAILABLE / NOT RUN**, as requested.','',
        '### Unsafe accepts by requested family (primary seed)', '',
        '| Split | Direction inversion | Negation | Entity swap | Added requirement | Quantity | Version | Unit |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in frozen:
        if r['seed']==primary:
            counts=r['unsafe_by_family']
            report.append('| '+r['split']+' | '+' | '.join(str(counts.get(x,0)) for x in ['direction_inversion','negation','entity_swap','added_requirement','quantity','version','unit'])+' |')
    report += ['', 'These are observed unsafe selected-candidate counts, not independent error-rate estimates. Every unsafe acceptance and every missed positive for ALL selected seed policies is listed in calibration-v1/all-selected-failures.jsonl.gz (full questions, retrieved candidates, probabilities and veto reasons in selected-failures-detailed.jsonl.gz), with actual selected relation, group, lineage, template, family and cause. Baseline errors are likewise saved per split. Per-family conditional cluster bounds and denominators are in per-family.json. Hard-negative suite group/lineage denominators and exact conditional zero-error bounds are separately saved in hard-negative-safety.json; they include all tested negative clusters, not only accepted clusters.','',
        '### Abstention and retrieval cost', '', '| Split | Seed | Retrieval-missed positives | Decision-lost retrievable positives | Miss causes |', '| --- | --- | --- | --- | --- |']
    for r in frozen:report.append(f"| {r['split']} | {r['seed']} | {r['retrieval_missed_positives']} | {r['decision_lost_positives']} | {json.dumps(r['missed_positive_causes'],sort_keys=True)} |")
    report += ['', 'Veto-dev-summary.json reports designated-pair potential rejections per family; dev/classifier-veto-diagnostic.json reports additional rejections and wrongly killed true positives conditional on the fixed 0.5 classifier diagnostic. The veto never resolves a tie or promotes a runner-up. Unverified Stage A metadata is quarantined, reviewed in legacy-signature-review.json, and never used as policy input.','',
        '### Seed spread and template memorization', '', '| Seed | Post-fit train four-class accuracy | Dev held-out accuracy | Calibration held-out accuracy | Training wall seconds |', '| --- | --- | --- | --- | --- |']
    for s in training['all_seeds']:
        seed=s['seed'];d=next(x for x in classification if x['split']=='dev' and x['seed']==seed);c=next(x for x in classification if x['split']=='calibration' and x['seed']==seed)
        report.append(f"| {seed} | {pct(trainfit[str(seed)]['accuracy'])} | {pct(d['accuracy'])} | {pct(c['accuracy'])} | {s['wall_seconds']:.3f} |")
    report += ['', 'Classification-generalization.json preserves all confusion counts and per-held-out-template/per-lineage classification counts. Per-template-lineage.json preserves selected-policy reuse metrics and bounds for every template and lineage, including zero-accept rows. No best seed is substituted: seed 11 was fixed as primary before training. Natural-wording performance: **NOT RUN**, blank request pending.','',
        '### Local latency — component measurements only', '', 'Hardware: `'+json.dumps(latency['hardware'])+'`.', '', 'Cold definition: '+latency['cold_definition']+'.', '',
        'Model loading / bank preparation milliseconds: `'+json.dumps(latency['model_loading_cold'])+'`.', '', '| Stage | Warm median ms | Warm p95 ms | Observations |','| --- | --- | --- | --- |']
    warm=[r for r in latency['observations'] if r['state']=='warm']
    for stage in ['embedding_ms','search_ms','classifier_ms','veto_ms']:
        values=sorted(r[stage] for r in warm)
        report.append(f"| {stage} | {statistics.median(values):.3f} | {values[min(len(values)-1,int(.95*len(values)))]:.3f} | {len(values)} dependent local probes |")
    report += ['', 'Calibration scoring had one interrupted process with no traceback; its log and failure metadata are preserved. Recovery added atomic identity-bound numeric checkpoints and bounded concurrent CPU workers, without changing models, data, veto or acceptance logic. Scoring batch times can include worker contention and are not fair application-latency comparisons. The standalone local probe ran separately after scoring.', '',
        'Full first-forward and warm observations are saved in local-latency-v1.json. Per-model batch/load times for every seed and baseline are in each score receipt; batch throughput is not singleton query latency. No latency percentage, monetary savings or application benefit is claimed.','',
        'Validation: `artifacts/stage_b/tests-final.txt` records the targeted harness tests; closeout-integrity.json verifies frozen data, models and decision functions plus the unchanged final-manifest hash. Tests verify implementation invariants, not model accuracy.','',
        '## 5. Failures and unknowns','',
        '- An always-miss exact-control bookkeeping bug was corrected; original B3 aggregates are retained and every prior semantic metric is identical across all 41 points. See exact-accounting-correction.md and exact-accounting-invariance.json.',
        '- One calibration process stopped without a traceback before raw results were written; the cause is unknown. Its lost scores were recomputed under the frozen contract with resumable checkpoints. See calibration-interruption-v1.json.',
        '- Synthetic data, shared structured task language, builder-authored labels and domain-aligned rules can inflate apparent safety. Whole-template holdout does not establish natural-language robustness.',
        '- The synthetic four-class oracle treats ASCII and Unicode input kinds as incomparable. ASCII is a subset of Unicode, so specificity ordering for that contrast is a known unverified simplification; four-class accuracy is oracle agreement, not independently established relation accuracy.',
        '- Training operations are sequence-focused; held-out operations are string-focused. Lineage holdout is confounded with input-domain shift, so a gap cannot be attributed to template memorization alone.',
        '- Operation matching is a bounded lexical vocabulary, not a general semantic intent parser. Legitimate paraphrases using unsupported operation wording will abstain; useful natural-wording coverage is unmeasured.',
        '- The veto output type is domain-derived, not a general parser for arbitrary requested output representations; generic signature safety remains unproven.',
        '- Few independent operation lineages limit evidence regardless of group count. Conditional group/lineage bounds do not certify a real user workload.',
        '- A threshold with no qualifying calibration bound becomes always-abstain; this is a failed useful-coverage result, not 100% precision.',
        '- Answer correctness/sufficiency is NOT measured. Question equivalence alone cannot make an incorrect cached answer correct.',
        '- The full original 125-accepted-group final-evidence contract remains unresolved; none of these development results replaces final or human review.',
        '- PAWS approval is pending; Stack Exchange duplicate licensing/Python filtering is unresolved. No public data has been used.', '',
        '## 6. Blockers and minimum owner actions','',
        'Return data/intent_stage_b/v2/natural-review-request.jsonl after 50 pairs have been independently written and labeled. Declare human or model-assisted provenance and reviewer independence. Until then, no natural-wording claim. Optional PAWS approval details are in docs/PUBLIC_DATA_APPROVAL_STAGE_B.md; no download occurs without approval. The concrete follow-up data correction is a NEW versioned cohort that mixes sequence, text and conversion domains in training while holding out wording templates and operation lineages separately. Review specificity labels such as ASCII versus Unicode before training. Preserve this spent study and do not tune on the future natural set. No follow-up study or Stage C has been executed; no promotion decision is made for the owner.','',
        '## 7. Not done','',
        'The locked Stage A final records were NOT opened, trained on, tuned on or evaluated during Stage B. Only its manifest hash was checked; Part 1 performed the separately authorized isolated regeneration/hash test. No final inference or per-case review occurred. No live/provider calls, historical parsed-call comparator, public-pair download, independent human ground truth, UI/agent changes, Stage C integration, semantic serving enablement, remote judge, push, merge, rebase or history rewrite.','',
        '## 8. Files to back up and learned model retrieval','',
        'Download artifacts/stage_b/stage-b-complete-backup.zip. It includes the local Git bundle, working-tree datasets/manifests/configs/harness/reports, all raw baseline and policy outputs, the excluded Stage A final stores, and all three ignored learned safetensors adapters. BACKUP_FILES.json gives every file size and SHA-256; the ZIP contains restore instructions. No pretrained weights, Python environment, transient inference caches or secrets are included.', '',
        'The three adapters are at artifacts/stage_b/models/{11,29,47}/adapter.safetensors. All learned tensors are exported; reconstruct with the pinned Quora safetensors base via load_adapter. Exact commands, seeds, base pin, learned/base SHA-256 and hardware are in training-v1.json and model-receipts/. The frozen policies identify each adapter hash. Preserve the archive before environment reset; a commit does not preserve ignored learned exports.','']
    (ROOT/'docs/audit/STAGE_B_REPORT.md').write_text('\n'.join(report))
    write_json(ROOT/'artifacts/stage_b/closeout-summary.json',{'evidence_type':'synthetic_template_held_out_diagnostic_not_human_truth',
        'primary_seed':primary,'selected':selection,'metrics':frozen,'final_manifest':manifest,
        'natural_eval_status':'NOT RUN; blank reviewer request pending','general_semantic_serving':'DISABLED','stopped_after':'B5'})
    print(lede)
    print(json.dumps({'selected':selection,'report':'docs/audit/STAGE_B_REPORT.md'}))


if __name__=='__main__':main()
