# Stage B protocol — frozen before training

Scope: offline question-relation research through B5 only. Final is forbidden;
no provider calls, production/agent/UI changes, serving, push or merge. Local
commits B0 through B5 are authorized. Evidence is synthetic template-held-out
development/calibration; no independent human ground truth.

B0: Version v1 was an unmeasured construction draft. Its generator is preserved
in `artifacts/stage_b/b0-draft-generator.py`. Before any training or inference,
v2 expands calibration quantity/version variants to give the prespecified
confidence arithmetic enough groups. Stage A is unchanged. Train: 96 groups,
24 operation lineages; dev: 64 groups, 8 held-out lineages; calibration: 256
groups, 8 further held-out lineages. Each split has four distinct question
formatters and split-specific requirement and negative wrappers. Manifest gives
counts per family, lineage, formatter/template and provenance. Structured task
wording remains synthetic; disjoint wrappers do not prove natural generalization.

Calibration sizing before training: a zero-error group bound at 95% confidence
needs 59 accepted groups pointwise, or 125 with Bonferroni over 10 thresholds
and 3 seeds. With 256 candidate groups this needs about half the groups to have
accepted reuse. Correlation within only 8 calibration lineages remains a serious
limitation; group counts cannot manufacture independent source diversity.

The 50-slot natural request has blank questions and labels. Humans or a blind
independent reviewer must write both questions and declare identity/evidence
type. Until returned, natural evaluation is NOT RUN. Public metadata proposals
are approval-only. No public records will be downloaded without approval;
proceed with synthetic data only meanwhile. Quora and Redis retain the possible
pretraining-overlap flag.

B1: train only, excluding exact controls and null ambiguity, using the pinned
Quora base. CPU adaptation is explicitly LoRA on all query/value projections
plus a four-class output head; all other pretrained tensors are frozen. This
is parameter-efficient fine-tuning, not full-weight fine-tuning. No new base,
no dependency on a missing historical fine-tuned model. All settings and three
seeds are fixed in `configs/intent_stage_b_v1.json` before training. Two epochs
per seed, no dev early stopping, no best-seed selection. Export every learned
tensor as safetensors with exact base pin and reconstruction loader; these
adapters are not standalone full-model checkpoints. Record hardware and time.

B2: parse question text only. Signature rules can veto, never approve. Unknown
operations/requirements abstain. Review Stage A unverified metadata but do not
use it as trusted policy input. Preserve that dataset and document exclusions.

B3: trusted scope eligibility, exact question lookup, MiniLM top-3, trained pair
classifier, question-side veto, ambiguity and competing-candidate abstention.
Classification is evaluated on every actual retrieved candidate with its actual
synthetic contract relation. The synthetic metadata oracle is evaluation-only;
it is not passed to the decision policy. Track retrieval recall and decision
abstention independently. Baselines use their existing Stage A grids on these
same actual MiniLM candidates; Redis reranker remains a same-set diagnostic.
Also report Redis's own embedding retrieval as a separate retrieval diagnostic
if measured. No historical parsed-call comparator is available.

B4: `configs/intent_stage_b_calibration_v1.json` is written before calibration.
Exact one-sided Clopper–Pearson group error bounds, Bonferroni across fixed
thresholds and seeds. Unknown accepts prevent qualification. Select threshold
on calibration only; no qualifying threshold means always abstain. Preserve
all seed policies and raw results. Conditional bounds assume independent
Bernoulli groups, which purposive shared-template/lineage data do not establish;
no population certification. Selected threshold applies to held-out dev without
further changes. Lineage bounds reported separately.

B5: report every unsafe reuse and missed positive by cause, per-family, per-
formatter and per-lineage performance, seed spread, train-vs-heldout behavior,
local cold/load and warm stage latency. No app/live latency or cost claim.
Natural results remain unavailable absent returned reviews. Stop after report.
