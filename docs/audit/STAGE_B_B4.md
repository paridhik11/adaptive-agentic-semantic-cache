# B4: frozen calibration and operational recovery

Evidence type: synthetic calibration-only threshold selection, not human ground truth.

The original B0 calibration contract was used unchanged: target accepted-group
error 0.05, confidence 0.95, ten thresholds and three prespecified seeds, exact
one-sided Clopper–Pearson with Bonferroni alpha 0.05/30. Independence assumptions
are not established by these purposive, shared-lineage synthetic cohorts.

No threshold qualifies for any seed (11, 29, 47). Each selected threshold is null:
always abstain on semantic reuse. Precision is undefined with no semantic accepts;
this does not meet useful coverage or demonstrate real-world safety. Dev was
opened only after all calibration choices were written. Models, data, veto and
acceptance logic stayed frozen. See calibration-v1/selection.json,
calibration-grid.json, selected-metrics.json and the three versioned policy files.
All selected-policy missed positives and their retrieved candidates are preserved.

A first scoring process stopped without traceback; the cause is unknown. Its log
and interruption receipt are retained. Recovery added identity-bound atomic,
pickle-free checkpoints, process locks, bounded workers and per-model raw score
exports. Repeated computation uses the same observations and does not increase
sample size. All five pair models completed 7,631 actual calibration candidates.
Checkpoint caches are excluded from Git and backup; raw scores are retained.

An always-miss exact-control accounting defect was corrected. Original B3 outputs
are retained. All existing semantic metrics across all 41 baseline points and all
14,468 original error-row fields are unchanged; see exact-accounting-invariance.json.
The correction does not change policy acceptance or improve held-out results.

Commands (Linux/WSL; pinned runtime; existing evidence is not overwritten):

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/run_stage_b_scoring.py --split calibration --output-dir artifacts/stage_b/calibration
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/calibrate_stage_b_policy.py
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/report_stage_b_dev.py --split calibration
```

No final records were opened/evaluated. No live/provider calls, public data,
serving promotion, UI changes, push or merge occurred.
