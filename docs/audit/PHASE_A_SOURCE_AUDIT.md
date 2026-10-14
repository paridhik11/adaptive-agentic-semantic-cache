# Source-level Phase A audit

Source: user-supplied `phase-a-handoff.bundle`, verified and imported on
2026-10-15. Original source revisions: `d1f9d9772513b20249242fd70cc531dacca625f7`,
`bd3dfe9fc235c0d1ce17f8f56af8d853272db3ee`,
`f159ee060e5c7f1299b6f8a651a10d2d2aa6b10d`, and
`a373e7e9b0b6b51617661f44192a1b16c3e8e731`. The local merge preserves their
original history. Connector publication may use new commit metadata; original
source revisions are provenance, not proof of the published commit identity.

## What can now be verified

Run `python scripts/audit_phase_a_artifacts.py`. This uses only the standard
library and saved artifacts, never models, APIs, generation, or sealed inference.
The separate `PHASE_A_ARTIFACT_RECONCILIATION.json` records the results. Original
sealed telemetry, summary, and stream remain unchanged.

- All 2,200 saved events reconcile: 1,200 seed events and 1,000 probes.
- Historical probe outcomes: 60 TP, 26 FP, 476 FN, 438 TN. Historical non-exact
  hits: 75, comprising 54 TP and 21 FP (28%). These reproduce the runner's
  **designated-pair labels**, not actual retrieved-answer correctness.
- Canonical path count 30 comprises 19 seed events and 11 accepted probe hits.
  The 11 probes carry six TP and five FP labels. Every one of those five FP
  records retrieved text exactly identical to the incoming query from another
  seed. The previous report-only interpretation as five unsafe exact matches
  was too strong: this evidence instead exposes a label-attribution bug.
- Among 86 accepted probe hits, 67 retrieve the designated seed, ten retrieve
  identical text from another candidate, and nine retrieve a different,
  nonidentical candidate without a label for that actual pair. Seven of those
  nine were counted as TP and two as FP using the unrelated designated seed.
  Do not relabel them automatically or publish a corrected safety headline.
- 27 seed events actually make REUSE decisions, then are forcibly indexed and
  excluded from accepted-hit metrics. The latency simulation nevertheless
  treats their REUSE decisions as saved generation. Seed requests are therefore
  neither uniformly cold nor handled consistently across utility metrics.
- WildChat's 200 events are all seeds and supply zero labeled probes. Zero
  accepted probe hits means IRR is undefined, not demonstrated 0% safety.
- A single empty-cache similarity is `-Infinity`, which is not standard JSON.
  The new audit flags it and emits strict JSON without modifying that artifact.

## Protocol and reproducibility gaps

1. **AUTO_REUSE verification differs from preregistration.** The protocol says
   all prospective semantic reuses pass both guards and pairwise equivalence.
   The runner only runs guards in AUTO_REUSE. Of its 43 AUTO_REUSE path events,
   six are seeds and 37 are accepted probes (21 TP and 16 FP by original labels).
   The report's attribution of all failures to the fine-tuned cross-encoder is
   unsupported because that model is not invoked on this path.
2. **Canonical eligibility precedes classification.** Stage 0 accepts before
   stability or rule guards; all seeds and probe misses are admitted regardless
   of eligibility. The recovery runtime prevents this by requiring policy
   eligibility before lookup and admission and using strict request equality.
3. **The frozen config is not an executable complete specification.** Some
   guard values are hard-coded elsewhere; the classifier is neither pinned nor
   frozen; model revisions are not passed to loaders. Missing classifier weights
   can trigger silent retraining. The runner uses a hard-coded checkpoint path.
4. **Checkpoint weights are missing.** The bundle includes config/tokenizer
   files, but no `.safetensors` or `pytorch_model*.bin` weight file. The exact
   trained model cannot be reconstructed from this snapshot alone. No inference
   reproduction is claimed.
5. **Seal bytes are platform-dependent.** Checked-out LF stream SHA-256 is
   `f685beccb75c5544dd8d786402e7e20c52cf83bb2520fe5c31ccf22811c17b44`.
   Restoring CRLF yields the declared
   `0e7aff256e4d8ee715a0b55f2007422cd99be9040765683799669f2f4861d991`.
   This explains the mismatch without implying fabrication. The frozen config's
   LF hash matches its documented hash. The original runner's raw-byte check
   fails on the LF checkout; do not silently edit the old seal.
6. **QQP split description disagrees with the builder.** Preregistration says
   validation pairs 500–1000; the builder downloads QQP train and starts at
   offset 10,000. Zero identical-pair overlap was found under strip/casefold
   normalization, but 13 stream events share text with fine-tuning examples;
   one shares text with validation. This is not proof of pair-level contamination,
   but it prevents a claim of zero text overlap or independently grouped splits.
7. **CQADupStack holdout is not guaranteed by offset.** Calibration randomly
   shuffles all query IDs; the sealed builder merely skips the first 50 IDs.
   These do not define disjoint splits. Calibration does not persist pair IDs,
   and Python's process-randomized `hash(subforum)` changes its sampling seed.
   Exact overlap cannot be certified from the supplied calibration summary.
   The sealed probes contain 18 pairs in each of 11 forums and only two in
   wordpress. Twelve represented forums does not mean held-out whole domains.
8. **Human audit is unfinished.** All 100 `human_label` cells in the supplied
   audit CSV are empty. It cannot establish independent human adjudication.
9. **Latency/savings remain projections.** Generation is never performed in
   this runner. Indexing occurs after decision timing. Simulated misses all get
   the same baseline median; repeat-share p50 is a weighted arithmetic formula.
   No full service p50/p95, actual answer quality, or net token savings is measured.

## Engineering decision

Preserve the failed historical result as spent evaluation evidence. Develop a
new evaluator that binds labels to the actual retrieved candidate, separates
seeds/probes/natural traffic, records completed answers and all work, and rejects
unlabeled pairs from safety claims. Freeze future executable configuration and
checkpoint hashes, group splits by question/thread lineage, and use a new sealed
set after development. Semantic reuse stays disabled in the response runtime.

If exact reproduction of the historical model is desired, obtain its actual
checkpoint weights and recorded SHA-256; retraining a replacement is a new model.
The gateway and engineering foundation can proceed independently.
