# Optional public-data proposal — not downloaded

Metadata queried through the Hugging Face API and pinned model/dataset cards;
no dataset records or parquet files downloaded. B0–B5 proceed synthetic-only.
Every future public-pair table must carry possible pretraining overlap with
public pair data, including Quora and both Redis baselines.

| Dataset ID | Exact revision | Proposed subset / size | License status |
| --- | --- | --- | --- |
| google-research-datasets/paws | 161ece9501cf0a11f3e48bd356eaa82de46d6a09 | labeled_final: train 49,401, validation 8,000, test 8,000; files total 10,899,391 bytes | Card: other; Google permits use for any purpose, attribution appreciated; no warranty |
| sentence-transformers/stackexchange-duplicates | 1c9657aec12d9e101667bb9593efcc623c4a68ff | title-title-pair: 304,525 pairs; parquet 25,068,596 bytes; mixed Stack Exchange, Python-only filter not established | No license declared in pinned card; do not approve/download until upstream licensing and attribution are resolved |

The attempted BeIR/cqadupstack-python ID did not resolve. The actual
mteb/cqadupstack-retrieval revision 11452ff64abeb3a37db5e0cd75bfec4763c11d61
contains no Python subset in its listed files and no declared license, so it is
not a suitable substitute. No guessed revision or license is used.

Owner action: optionally approve the pinned PAWS labeled_final subset for a
separate future diagnostic, or leave public data unused. The Stack Exchange
candidate is blocked on licensing and provenance, not ready for download.
Exact metadata and byte counts: artifacts/stage_b/public-data-proposal.json.
