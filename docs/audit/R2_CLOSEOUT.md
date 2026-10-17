# R2 closeout: workload fit and public-source preparation

## Decision

R2 is complete as a workload investigation, with the broad cache utility case
narrowed and unproven. The source and split checks pass. OASST1 shows few
literal or canonical repeat opportunities in its crowdsourced first-turn
prompt corpus; its source-user grouping shows none. This does not support a
general promise of high cache hit rate or cost/latency savings. Continue into
R3 only as a bounded development test for versioned technical Q&A and shared
agent tasks. Keep semantic reuse out of the serving default until answer labels
and net provider work support it.

## Evidence

`R2_OASST_NATURAL_REPETITION.json` binds the source manifest, pinned dataset
revision, license and Parquet checksum. It records aggregate counts only; no
prompt text or user ID is committed. The analysis processes all eligible
first-turn roots in the selected shard chronologically, with message-tree
deduplication. Of the repeated prompts, 49 exact and 57 normalized repeat
occurrences had an assistant reply record in the earlier thread timestamped
before the later repeat. This checks record availability only; answer content
was not reviewed. The observed counts are in `docs/FACTS.md`; they describe
this source shard and are not a representative traffic estimate or safe-hit
count.

`R2_PREPARATION_SUMMARY.json` binds the pinned CPython FAQ and license hashes,
the answer-reference corpus hash and constructed workload hash. Topic sections
stay within one train/dev/reserved-test split. Repeated workload events are
explicitly constructed and are never pooled into natural prevalence. The
adversarial cases are author-written test intentions, not independent labels.

## Failures and decisions

- Two WildChat viewer collection attempts returned HTTP 429 before a complete
  export. The current README describes an ODC-BY license change, while a
  retained LICENSE.md still contains AI2 ImpACT terms. Neither incomplete
  output was used.
- OASST1 was selected as a limited alternative after verifying its pinned
  Apache-2.0 source and timestamp/tree schema. It is crowdsourced assistant
  traffic, not production API traffic, and its source user IDs are not cache
  tenant identities.
- FAQ answers are documentation references, not correctness labels for model
  outputs. Pretraining exposure is unknown. No OpenRouter calls were made.
- No raw OASST prompts, conversation IDs or user IDs are in the repository.
  The source shard stays under ignored `runs/`; the reproducible exporter writes
  only aggregate counts.

## Exit assessment

Source rights/provenance and group-split checks pass for the selected data.
Natural repeat prevalence is low in the analyzed corpus, and answer
substitutability, semantic precision, opportunity capture, production
representativeness and net savings remain unknown. The phase therefore narrows
the engineering hypothesis; it does not pass a semantic quality or utility
gate. The next phase must test candidate retrieval on grouped development
reference questions and adversarial scope changes while retaining exact-only
serving behavior.

## Reproduction and checks

```bash
pip install '.[data]'
python scripts/prepare_oasst_natural.py \
  --source-dir runs/r2-oasst-source-recompute \
  --output-dir runs/r2-oasst-recompute \
  --limit 9845 \
  --download
python scripts/prepare_python_faq.py \
  --source-dir runs/r2-python-source-recompute \
  --output-dir runs/r2-python-recompute \
  --download
PYTHONPATH=<review-venv-site-packages>:<runtime-deps> \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/test_natural_repetition.py tests/test_faq_corpus.py -q
```

Final validation:

```text
Full default offline suite after reply-ordering edge-case check: 556 passed, 23 deselected
Focused FAQ/natural-repeat tests: 11 passed
Strict numeric-token check: PASS (0 unledgered numeric tokens)
Pre-commit Gitleaks working-tree and history scans: no leaks found (117 prior commits scanned)
OpenRouter/model API calls during R2: 0
```

These checks validate preparation and accounting code. They do not establish
answer quality, safe semantic reuse or live latency/cost savings.
