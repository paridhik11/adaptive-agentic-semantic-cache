# Decomposed verifier development contract

October 21, 2026. Defined before the new local experiment. This is development,
not a final evaluation or a modification of the final quality gate.

## Method and scope

Preserve the FLAN and Qwen failures. Replace generic chat yes/no voting in this
new diagnostic with separately auditable decisions:

1. Preserve full request/system/source scope and complete-answer admission.
2. Conservatively reject explicit changes in quantities/versions, conversion
   direction/entities, and added constraints such as recursion or negation.
   These development-informed guards are generic rules, not approved query pairs.
3. Select a verbatim trusted source span using the original question only, with
   pinned MiniLM embeddings. Neither the incoming question nor answer steers the
   source selection. Ambiguous/weak selection abstains.
4. Require selected-source numbers and API names to appear in the actual answer.
   This deliberately rejects some valid alternative expressions/methods.
5. Use pinned DeBERTa NLI for bidirectional request entailment, answer-to-selected
   evidence coverage, and source-to-each-answer-claim support. Every check must
   pass; neutral, contradiction, input overflow and missing checks abstain.

The fixed initial NLI entailment threshold is 0.95. Evidence selection uses a
0.50 cosine floor and 0.03 margin. These are development settings, not calibrated
accuracy probabilities or production thresholds. They are fixed before this run
and will not be changed to make its cases pass. The previous models' yes/no
thresholds do not have the same statistical meaning as NLI class scores.

The selected span can omit material requirements or be the wrong source.
Bidirectional entailment of task descriptions is not proven equivalence of
arbitrary requests. Source support alone does not prove sufficiency. These are
explicit remaining failure modes, not reasons to label the policy safe.

## Data and interpretation

Run the existing 15 exposed authored cases unchanged as a development replay.
Additionally run 30 builder-authored cases across ten further intent groups:
list length, integer/string conversions, dictionary fallback, list append,
descending sort, path basename, JSON parsing, set deduplication and numeric sum.
Each new group has one positive paraphrase, one changed-request negative and
one wrong-answer negative. No per-case thresholds or approved request registry.
All source text and expectations are authored fixtures, not independent human
review or authenticated public documentation. New groups are broader development
coverage, not a held-out final set or evidence of unseen pretrained-model data.

Report both cohorts separately, including every failed positive and unsafe
recommendation. Do not give population precision or binomial safety intervals
for these purposive, correlated fixtures. No final gate is evaluated.

## Serving and experiment boundary

The application-facing policy always abstains. Its recommendations are observable
diagnostics only. An isolated runner may translate those recommendations into
decisions for a fixture-only cache experiment with Python socket connections
blocked and a locally defined scripted provider. This is not production serving
or live model savings. No gateway switch enables this experimental adapter.

After the initial direct-assessment run, extend the fixture cache experiment to
all 45 unchanged cases and preserve the initial result as `decomposed-v1.json`.
The expanded engineering replay uses `decomposed-cache-v1.json`; thresholds and
policy remain unchanged. Report exact hits separately: complete-but-wrong seeded
answers can pass the existing exact-cache admission contract, which does not
certify factual correctness. An exact reuse of such a seed is not a semantic
verifier decision and must not be omitted or counted as a semantic success.

Record source/model hashes, source-span positions, per-check scores, embedding
and NLI attempts, setup work and elapsed local policy time. Initialization and
local inference costs must not be presented as zero. Timing is diagnostic, not
a paired application benchmark. Preserve prior failed results unchanged.

## Reproduce

```bash
python scripts/probe_decomposed_policy.py --prepare --output runs/semantic-dev/decomposed-v1.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/probe_decomposed_policy.py --output runs/semantic-dev/decomposed-offline.json
```

Model source: https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
Use the immutable revision recorded in `src/policy/decomposed_applicability.py`.
The model is trained for entailment/neutral/contradiction; that training does not
establish semantic-cache safety. Final calibration, independent labels, persistent
semantic metadata and pre-agent integration remain separate roadmap milestones.
