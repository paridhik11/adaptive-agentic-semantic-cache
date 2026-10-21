# Stage A: four-class intent-pair baselines

User-authorized A2–A4 only. No training, calibration selection, final evaluation,
UI/agent changes, live calls, commits, pushes or branches. General semantic
serving remains disabled. All judgments below concern pair intent; actual cached
answer quality is recorded separately, never converted into pair labels.

## Dataset frozen before baseline inference

Use all 75 previous authored development rows first, with question-only relation
labels and separate historical answer-quality notes. All remain dev/spent.
Generate 192 distinct task contracts: 16 Python operations × 3 input container
kinds × 4 material response requirements. These are template-generated authored
fixtures, not 192 independent natural-language observations. Group IDs identify
contracts; source-lineage IDs identify operation families. Related container and
response-requirement variants never cross splits. Formatter templates are shared
across splits and explicitly limit generalization claims. No already-vetted,
licensed public pair corpus is available for inclusion; historical sealed files,
FAQ reserves and pilot questions are excluded.

Generated groups contain three non-exact paraphrases, an exact control, and at
least three hard negatives, including specificity/generalization, polarity,
operation/function changes, parameter/version changes and no-match controls.
Ambiguity is represented with relation=null and label_status=ambiguous, because
none of the four definite relations is defensible. Accepted unknowns remain
unreviewed, never imputed safe. Added schema fields include case_id, cached_case_id,
scope_id, template_lineage_id, label_status, historical_answer_variant_id and
source/license details. A separate answer store carries complete authored answers
and answer-quality provenance. Intent signatures are metadata, not policy inputs
or an acceptance allowlist.

Split by BOTH group and source lineage. All old lineages are dev. New operation
lineages are assigned deterministically before inference: eight train, three dev,
three calibration, two final. Final data are generated, hashed and locked; the
manifest SHA is appended immediately to the already tracked docs/FACTS.md.
Do not open final per-case data or evaluate final in Stage A. The harness refuses
final before model loading. Train/calibration are not evaluated in Stage A.
This locked generated final set is not independent human ground truth, nor an
adequate replacement for the previous final quality-evidence contract.

## Comparable baseline experiment

Primary baseline table is a DESIGNATED PAIR experiment: each new question is
compared with its designated cached question, identically for every policy.
It is not an end-to-end serving hit-rate experiment. Separately measure MiniLM
retrieval recall@1/3/5 over dev cached questions on labeled same-intent opportunities;
report exact controls separately and distinguish retrieval misses from decision
abstentions. Fixed candidate retrieval lists are preserved. No threshold is chosen.

Baselines: always miss; exact only; MiniLM cosine grid [0.5,0.6,0.7,0.8,0.9,0.95];
Quora sigmoid and Redis reranker sigmoid grids [0.5,0.7,0.9,0.95,0.99]; Redis
embedding cosine on the same designated pairs as an additional approved diagnostic;
existing local_equivalence guards plus its pinned Quora model at the same grid
(default 0.70 reported). For the existing stack, inject the pinned model instead
of using its unpinned lazy loader. Record the original heuristic behavior.

The historical parsed-call policy also needs unapproved absent QA/NLI checkpoints;
report it as NOT RUN if unavailable. Do not replace it with an unrelated helper
and call it the historical baseline. No download beyond the four approved models.

Use actual local inference, CPU, fixed max input window, refuse truncation and
block socket connections during experiments. trust_remote_code=False and
use_safetensors=True for direct Transformers loading. No pickle files loaded.
Save raw logits/cosines, pair bindings, decisions, failures, timing/hardware/model
identities, fixed candidate lists and per-family counts in the working tree.

## Metrics and uncertainty

Precision = safe accepted non-exact labeled pairs / accepted non-exact pairs.
Positive coverage = safe non-exact accepts / labeled non-exact same-intent
opportunities. Exact controls are excluded from these denominators. Report
accepted unknown labels separately; any such acceptance withholds a complete
precision bound. Unsafe accepts are definitive non-same-intent labels accepted.
Group accepted-safe means every accepted non-exact query in that group is safe;
lineage accepted-safe uses the stricter corresponding lineage condition.

Provide one-sided 95% Clopper–Pearson group and lineage bounds, explicitly
conditional on independent Bernoulli sampling. Templates and purposive authored
cases violate a population sampling interpretation; these arithmetic bounds do
not establish population safety. Do not compute answer-level binomial intervals
as though correlated paraphrases were independent. Precision with no accepts is
null, not 100%. Missing/ambiguous labels stay null.

Quora and both Redis baselines carry the provenance flag
"possible pretraining overlap with public pair data" in all tables and raw model
metadata. No public pairs are used now; future public-pair evaluations must retain
this flag. No result here is a fresh/sealed performance claim.

## Local publication preparation (2026-10-21)

Evidence type: deterministic regeneration and hash checks, not model evaluation.
`python scripts/verify_stage_a_regeneration.py --receipt artifacts/stage_a/final-regeneration-v1.json`
regenerated the original generator in an isolated temporary source tree. The
manifest, final question store and final answer store matched byte-for-byte by
size and SHA-256. Final manifest SHA-256 remains
`1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af`.
No final records were inspected or evaluated. The generator uses deterministic
ordered enumeration, not a random seed. The final stores are excluded from Git
and retained in the backup. The generator remains public and reproducible; this
is an evaluation access restriction, not cryptographic secrecy.

`.gitattributes` disables text conversion for Stage A data and artifacts and
requires LF for `docs/FACTS.md`. These additions did not change the final hash.
The current authorization permits stacked LOCAL commits only; it supersedes
older commit/push instructions. No push, merge, main rebase or force operation
is authorized.
