# Semantic preparation milestone

October 20, 2026. This milestone executes the corrected semantic-first direction.
No final evaluation, live provider call or new answer-quality claim.

## Implemented and measured

- A pinned local Quora pair-score adapter with explicit logit conversion,
  rejection of oversized input rather than silent truncation, and no implicit
  downloads. It is a diagnostic component, not a serving approver.
- An inventory/explicit-download/local-probe runner. Default execution does not
  download. Weight hashes and installed library versions are recorded.
- Actual pretrained MiniLM dense and Quora pair inference, including a repeat
  with offline flags and Python socket connections blocked. The missing
  historical fine-tuned checkpoint was not reconstructed or substituted silently.
- Five authored seeds and eight exposed development probes. Dense retrieval
  selects the intended seed for the three authored paraphrases. Quora gives
  very low pair scores to two of those positives. Do not lower a serving
  threshold to turn this probe into a safety claim.
- Prospective sample-planning arithmetic and an evaluation contract: distinct
  group/answer/query denominators, tolerant gate, hard negatives, written
  sufficiency rubric, missing-label rules and early review-capacity checkpoint.
- Focused offline regressions: 45 passed, including the existing semantic
  runtime checks. Full suite was not rerun; no existing serving behavior changed.

Raw aggregate diagnostics, model hashes, source hashes and captured focused
test output are in [SEMANTIC_PREPARATION_RESULT.json](SEMANTIC_PREPARATION_RESULT.json).
Examples are authored public development material, not uploaded private answers.
Changed source/docs received a pattern scan for OpenRouter-style keys, GitHub
tokens and private-key headers; scope and limitations are recorded in the JSON.

## Limits and next implementation

There is no generalizing answer-applicability policy yet. Existing explicit
answer approvals remain unchanged; this milestone does not turn on new semantic
serving. No actual-answer precision, semantic hit rate, live latency or monetary
savings has been established. Independent human reviewers are unconfirmed and
the final cohort is not frozen. A small final cohort cannot substantiate the
target bound merely by containing many correlated paraphrases.

Next: implement and test an answer-aware local applicability policy on exposed
development material, with explicit abstention. Investigate the pair model's
positive-coverage failure before choosing its role. Build source/reference and
answer-bound development records; do not label Codex's own policy's final test.
Then calibrate, persist semantic metadata and integrate lookup before planning.
Keep UI expansion and another live pilot out of this work.

## Reproduce

Use the project's installed Python environment. On this host, prepend
`PYTHONPATH=.venv/lib/python3.12/site-packages` to Python commands because the
environment's launcher is unavailable.

```bash
python scripts/run_semantic_probe.py
python scripts/run_semantic_probe.py --prepare --probe
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_semantic_probe.py --probe --output runs/semantic-dev/offline-probe.json
python scripts/plan_semantic_review.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_semantic_preparation.py tests/test_semantic_runtime.py -q
```

The blocked-network check additionally replaced `socket.socket.connect` and
`socket.create_connection` with functions raising `RuntimeError` before running
the probe through `runpy.run_path`. This is an in-process check, not a host-wide
network sandbox. Public artifact preparation used network access; inference did
not. A missing SOCKS dependency was installed locally for this environment's
proxy, not added as a general product dependency.
