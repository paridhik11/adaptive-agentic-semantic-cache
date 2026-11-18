# Answer-aware diagnostic policy milestone

October 21, 2026. Implementation and failed development baselines, not a passed
semantic-quality milestone. See [development protocol](../ANSWER_POLICY_DEVELOPMENT.md)
and [aggregate raw results](ANSWER_POLICY_RESULT.json).

## Delivered

`src/policy/answer_applicability.py` now binds checks to the actual full cached
answer, incoming/candidate requests and trusted context. It checks sufficiency
and source support separately, rejects scope changes and incomplete answers,
refuses silent truncation and fails closed on invalid scores or inference errors.
Pinned CPU FLAN and Qwen adapters load local safetensors only. The scorer's
yes/no token scores are not correctness probabilities.

`scripts/probe_answer_applicability.py` explicitly prepares public model artifacts
when requested, then blocks Python socket connections during inference. The
fifteen authored cases include paraphrases, changed requirements and wrong,
unsupported, irrelevant and incomplete answers. They are exposed development
material in three correlated groups, not fresh independent evaluation.

`ReuseDecision` and gateway metrics now distinguish local check attempts/time
from remote verifier work. Existing callbacks retain zero-valued defaults.
Invalid work metadata cannot authorize reuse. Exact hits skip the local checks.
Stage counters cover returned requests, not every possible later failure.

The new policy's serving interface always abstains, with no enable flag. A real
dense-cache integration probe confirmed that a positive diagnostic does not
become a hit. Default gateway configuration and existing approvals are unchanged.

## Findings and verification

| Local model | Positive recommendations / 4 authored positives | Unsafe recommendations / 11 authored negatives |
| --- | --- | --- |
| FLAN-T5-base | 0 | 0 |
| Qwen3-0.6B | 4 | 8 |

Both are unusable baselines. Qwen approved changed units/directions and incorrect
actual answers. Do not turn its four positive recommendations into a success
claim. Neither model was enabled for serving and thresholds were not lowered.
Both results, per-check scores, immutable bindings, model hashes and environment
versions remain in the aggregate artifact. No independent labels were created.

Full offline regressions: **726 passed, 23 deselected**, 119.43 seconds. Focused
checks passed 62 tests; a final validation-guard adjustment passed its 17 focused
tests. Syntax/diff and changed-file credential pattern checks passed. These are
engineering checks, not accuracy observations. No external generation calls or
live quotas were used.

## Remaining work

Safe general semantic serving is unfinished. The next verifier revision needs
better separation of request meaning/requirements and actual-answer evidence,
with explicit abstention. These baselines do not justify merely calibrating a
more permissive threshold. Retain the failed cases as development regressions;
use separate groups for subsequent calibration and fresh evaluation. Independent
development red-teaming and final human review remain unconfirmed. Persistence,
pre-agent semantic integration and live utility measurement remain later work.

Reproduction commands are in the development protocol. On this hosted workspace,
prepend `PYTHONPATH=.venv/lib/python3.12/site-packages` to use its installed ML/test
dependencies. Weight files are excluded from Git; reproduce them with the pinned
explicit preparation commands.
