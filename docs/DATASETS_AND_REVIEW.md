# Dataset inventory, review request and Group 4 status

2026-11-01. No new training is running. No new public pair dataset or pretrained
checkpoint was downloaded. This continues the existing project and fixed plan.

## Historical training and tests

| Use | Data | Counts/provenance |
| --- | --- | --- |
| Stage B train | `data/intent_stage_b/v2/train.jsonl` | 1,344 pair rows / 96 groups; builder-authored synthetic templates |
| Stage B dev | `data/intent_stage_b/v2/dev.jsonl` | 896 pair rows / 64 groups; synthetic template holdout |
| Stage B calibration | `data/intent_stage_b/v2/calibration.jsonl` | 3,584 pair rows / 256 groups; synthetic |
| Current product regression | `tests/` cache, document-agent, gateway and memory fixtures | Engineering contracts, scripted model responses; not independent answer-quality observations |
| Demo source corpus | `src/api/demo_reference.json` | 3 bundled CPython 3.12 Programming FAQ excerpts |
| New independent request | `data/document_product_review/v1/requests.jsonl` | 30 blank authoring slots; zero returned questions or labels |

Pair rows are correlated and are not independent groups. Stage B training excluded
exact/null/ambiguous pairs; its B1 report records 1,056 fitted pairs per seed.
The pinned Quora DistilRoBERTa base was fine-tuned historically with three seeds.
Those policies failed useful-coverage calibration and remain unpromoted. Their
training fit is not generalization or answer sufficiency. See the preserved
`docs/audit/STAGE_B_REPORT.md`. Earlier Stage A and historical public-data studies
are separate; they are not new product validation. Missing follow-up artifacts
are not silently recreated or used for claims.

The demo topics are hexadecimal/octal literals, tuple/list conversion and negative
indexing. Exact source revision: `0fb18b02c8ad56299d6a2910be0bab8ad601ef24`, from
the existing bundled reference metadata. These exposed excerpts are not a held-out
natural-language benchmark. Cached answers in the new packet are verbatim excerpts,
not newly collected LLM outputs. Source license is included.

## Concrete independent input needed

The authoring packet and instructions are now in
`data/document_product_review/v1/`. A person who has not built or inspected the
policy should write natural questions and judge each actual excerpt answer's
sufficiency. Ideally a second person checks the judgments. Author and reviewer
identities/independence are declared, not externally verified. If a model is used,
the evidence must be marked `model_assisted` and counted separately from humans.

Each source has ten requested slots: five equivalent-intent requests and five
plausible mismatches/added requirements. These are authoring requests, not labels.
All questions and judgments remain null until returned. The builder must not fill
them and call them independent. The packet manifest hashes were written at its
creation. It is a small purposive diagnostic, NOT the locked final split, a final
safety certification, or a representative savings workload. Three correlated
topics cannot establish broad precision, even if every observed decision is right.

```bash
python scripts/prepare_document_review.py --output-dir NEW_PACKET_DIRECTORY
python scripts/validate_document_review.py --packet data/document_product_review/v1 --responses RETURNED_FILE.jsonl --output NEW_REVIEW_RECEIPT.json
python scripts/inventory_product_data.py --output NEW_INVENTORY.json
```

The tools refuse output overwrites. The validator checks original packet integrity,
item IDs, source binding, label types and provenance; partial returns remain
partial, and unknown judgments stay null. It does not invent accuracy/precision.

## General policy and evaluation boundary

The current answer-bound approval registry does not automatically generalize to
unseen question–answer bindings. The existing fixtures prove execution and work
avoidance after qualification, not general qualification itself. Returning this
packet is necessary evidence preparation, not a fix for that missing policy.
A general candidate decision policy must be documented and frozen before any
new held-out evaluation; policy development must not tune to these returned
independent judgments. No threshold or frozen model was changed here.

Any later actual-LLM-answer quality study needs complete collected responses and
separate blinded answer review. Any live utility comparison needs a concrete
approved provider budget and the existing durable quota journal. Neither is
executed or authorized by this offline preparation. Never reset spent runs.

## Product QA evidence and remaining work

New engineering/schema regression result: **301 passed in 4.97s**, raw
`artifacts/documentation_rebuild/group4/tests.txt`. The eight added schema cases
use fabricated reviewer identities only; they do not supply labels to the real
packet. Original Stage A/B source/data/results and locked final records are unchanged.

An offline wheel built and installed into a separate target. HTTP tests imported
the package from that target, loaded its assets, ran a scripted source task and
its approved exact repeat ([3,0] actual fixture calls), and exercised memory
management. No torch/faiss/sentence-transformers import or external inference.
Receipt: `artifacts/documentation_rebuild/group4/package-smoke.json` and build log.

Build/install reproduction, using empty temporary output locations:

```bash
python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir TEMP_WHEELS .
python -m pip install --no-deps --no-index --target TEMP_INSTALLED TEMP_WHEELS/adaptive_agentic_semantic_cache-0.1.0-py3-none-any.whl
```

The smoke reused existing FastAPI/httpx dependencies. It is NOT a clean dependency
installation; the local dependency wheel cache was empty and no network install
was attempted. Browser visual QA, a clean-machine dependency installation, final
UI/memory controls, hosting/deployment and live utility are still unfinished.
Group 4 is partial and the product is not performance-certified or ready for
unqualified savings/accuracy claims on a resume.

One coherent local milestone commit preserves this preparation. No push, merge,
new model training/download, live/provider inference or final-split evaluation.


## Authorized dual review update — 2026-11-01

The user volunteered human review and requested a separate model-assisted review.
`model_assisted_v1/responses.jsonl` contains 30 model-authored, self-reviewed rows;
method/provenance is preserved beside it. The reviewer agent was instructed to
read source excerpts only, not policy code or predictions. This is model-assisted
diagnostic judgment, not independent human ground truth.

`human_v1/Human_Answer_Review.md` withholds model labels, reasons and requested
positive/negative slot types. Its H01–H30 order is shuffled. The user can return
Y/N/? in chat; source and question hashes bind responses to this form. Human
labels are blank. They will be recorded as `human_review_of_model_authored`,
separate from both model judgments and genuinely human-authored questions. We
will not copy model labels into human fields or reveal predictions in advance.
Human review does not establish independence of question authorship. Blinding is
best effort through the delivered form, not externally verified. No model-assisted
accuracy score or serving acceptance rate is inferred from these self-judgments.

Reproduce form generation into a new output path:

```bash
python scripts/prepare_blinded_human_review.py --packet data/document_product_review/v1 --model-responses data/document_product_review/model_assisted_v1/responses.jsonl --output-dir NEW_HUMAN_PACKET
```
