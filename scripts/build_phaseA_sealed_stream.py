"""Phase A4 — Sealed Test Stream Builder (N = 2,200 Queries).

Constructs the sealed test stream strictly from public held-out slices:
  1. QQP held-out partition (500 pairs = 1,000 queries from indices 10,000 to 10,500).
  2. PAWS test split (300 pairs = 600 queries: 150 paraphrases, 150 adversarial word scrambles).
  3. CQADupStack held-out threads (200 pairs = 400 queries across 12 subforums, indices 50+).
  4. WildChat organic stream (200 real conversational user queries, indices 5,000 to 5,200).

Composition:
  - Total queries: N = 2,200 queries.
  - Positive non-exact follow-ups (opportunities for safe reuse): ~660 queries (well above target >= 200).
  - Negative follow-ups (adversarial / non-equivalent pairs testing safety): ~340 queries.
  - Cold seeds: 1,100 queries.
  - Organic stream queries: 100 queries.

Outputs:
  - data/raw/phaseA_sealed_test_stream.json
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_STREAM_PATH = REPO_ROOT / "data" / "raw" / "phaseA_sealed_test_stream.json"
QQP_PARQUET_PATH = REPO_ROOT / "data" / "qqp_train.parquet"
WILDCHAT_PARQUET_PATH = REPO_ROOT / "data" / "wildchat" / "train-00000.parquet"
CQADUPSTACK_DIR = REPO_ROOT / "data" / "cqadupstack"


def load_paws_test_pairs(n_pos: int = 150, n_neg: int = 150) -> List[Dict[str, Any]]:
    """Download and extract held-out PAWS test pairs."""
    url = "https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/test-00000-of-00001.parquet"
    print(f"Downloading pristine PAWS test split from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        table = pq.read_table(io.BytesIO(resp.read()))

    pos, neg = [], []
    for i in range(len(table)):
        lbl = table["label"][i].as_py()
        q1 = table["sentence1"][i].as_py().strip()
        q2 = table["sentence2"][i].as_py().strip()
        if q1 and q2 and q1.lower() != q2.lower():
            item = {"query_a": q1, "query_b": q2, "is_safe": bool(lbl == 1), "source": "paws_test", "domain": "adversarial_paws"}
            if lbl == 1 and len(pos) < n_pos:
                pos.append(item)
            elif lbl == 0 and len(neg) < n_neg:
                neg.append(item)
            if len(pos) >= n_pos and len(neg) >= n_neg:
                break
    return pos + neg


def load_qqp_heldout_pairs(n_pairs: int = 500, start_offset: int = 10000) -> List[Dict[str, Any]]:
    """Extract held-out QQP pairs starting from offset 10,000."""
    print(f"Extracting {n_pairs} held-out QQP pairs from offset {start_offset}...")
    from huggingface_hub import hf_hub_download
    qqp_file = Path(hf_hub_download(
        repo_id="nyu-mll/glue",
        filename="qqp/train-00000-of-00001.parquet",
        repo_type="dataset",
    ))
    table = pq.read_table(qqp_file)
    pairs = []
    for i in range(start_offset, len(table)):
        lbl = table["label"][i].as_py()
        q1 = table["question1"][i].as_py()
        q2 = table["question2"][i].as_py()
        if q1 and q2 and isinstance(q1, str) and isinstance(q2, str):
            q1, q2 = q1.strip(), q2.strip()
            if q1 and q2 and q1.lower() != q2.lower():
                pairs.append({
                    "query_a": q1,
                    "query_b": q2,
                    "is_safe": bool(lbl == 1),
                    "source": "qqp_heldout",
                    "domain": "general_community",
                })
                if len(pairs) >= n_pairs:
                    break
    return pairs


def load_cqadupstack_heldout_pairs(n_pairs: int = 200, pairs_per_subforum: int = 18) -> List[Dict[str, Any]]:
    """Extract held-out CQADupStack duplicate threads across subforums."""
    print(f"Extracting {n_pairs} held-out CQADupStack duplicate threads...")
    subforum_files = sorted(CQADUPSTACK_DIR.glob("*.json"))
    pairs = []

    for sf_path in subforum_files:
        subforum = sf_path.stem
        with open(sf_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        corpus_map = dict(zip(data["corpus"]["ids"], data["corpus"]["sentences"]))
        query_map = dict(zip(data["queries"]["ids"], data["queries"]["sentences"]))
        rel_map = dict(zip(data["relevant_docs"]["ids"], data["relevant_docs"]["relevant_docs"]))

        sf_pairs = 0
        all_qids = list(query_map.keys())
        # Use held-out slice starting at index 50 to avoid overlap with A2 validation
        heldout_qids = all_qids[50:]

        for qid in heldout_qids:
            if qid in rel_map and rel_map[qid]:
                doc_id = rel_map[qid][0]
                if doc_id in corpus_map:
                    q_text = query_map[qid].strip()
                    doc_text = corpus_map[doc_id].strip()
                    if q_text and doc_text and q_text.lower() != doc_text.lower():
                        pairs.append({
                            "query_a": q_text,
                            "query_b": doc_text,
                            "is_safe": True,
                            "source": "cqadupstack_heldout",
                            "domain": subforum,
                        })
                        sf_pairs += 1
                        if sf_pairs >= pairs_per_subforum or len(pairs) >= n_pairs:
                            break
        if len(pairs) >= n_pairs:
            break

    return pairs[:n_pairs]


def load_wildchat_stream_queries(n_queries: int = 200, start_offset: int = 5000) -> List[Dict[str, Any]]:
    """Extract organic chat queries from WildChat from offset 5,000."""
    print(f"Extracting {n_queries} organic queries from WildChat from offset {start_offset}...")
    parquet_file = pq.ParquetFile(WILDCHAT_PARQUET_PATH)
    queries = []
    seen_count = 0

    for batch in parquet_file.iter_batches(batch_size=1000, columns=["conversation"]):
        conversations = batch["conversation"].to_pylist()
        for conv_list in conversations:
            if conv_list and len(conv_list) > 0:
                first_turn = conv_list[0]
                if isinstance(first_turn, dict) and first_turn.get("role") == "user":
                    content = first_turn.get("content", "").strip()
                    if content and 3 <= len(content) <= 1000:
                        seen_count += 1
                        if seen_count > start_offset:
                            queries.append({
                                "query": content,
                                "source": "wildchat_traffic",
                                "domain": "real_chat",
                            })
                            if len(queries) >= n_queries:
                                break
        if len(queries) >= n_queries:
            break

    return queries


def assemble_sealed_stream(
    qqp_pairs: List[Dict[str, Any]],
    paws_pairs: List[Dict[str, Any]],
    cqa_pairs: List[Dict[str, Any]],
    wc_queries: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Interleave seeds and follow-ups with controlled lags to create a realistic stream."""
    random.seed(seed)
    all_pairs = qqp_pairs + paws_pairs + cqa_pairs
    random.shuffle(all_pairs)

    events: List[Optional[Dict[str, Any]]] = [None] * (len(all_pairs) * 2 + len(wc_queries))
    occupied_slots = set()

    pair_id_counter = 1

    # Place pairs: seed first, then follow-up with lag in [2, 12]
    current_head = 0
    for pair in all_pairs:
        while current_head in occupied_slots:
            current_head += 1

        seed_slot = current_head
        occupied_slots.add(seed_slot)

        # Pick lag
        lag = random.randint(2, 10)
        followup_slot = seed_slot + lag
        while followup_slot in occupied_slots or followup_slot >= len(events):
            followup_slot += 1
            if followup_slot >= len(events):
                # Expand events list if needed
                events.extend([None] * 50)

        occupied_slots.add(followup_slot)

        pair_name = f"PAIR-{pair_id_counter:04d}"
        seed_event = {
            "query": pair["query_a"],
            "source": pair["source"],
            "pair_id": pair_name,
            "is_seed": True,
            "matched_seed_id": None,
            "is_reuse_safe": False,  # Cold seed must be indexed / miss
            "domain": pair["domain"],
        }
        followup_event = {
            "query": pair["query_b"],
            "source": pair["source"],
            "pair_id": pair_name,
            "is_seed": False,
            "matched_seed_id": None,  # Will be resolved with exact event ID
            "is_reuse_safe": pair["is_safe"],
            "domain": pair["domain"],
        }

        events[seed_slot] = seed_event
        events[followup_slot] = followup_event
        pair_id_counter += 1

    # Fill remaining empty slots with WildChat queries
    wc_idx = 0
    compacted_events: List[Dict[str, Any]] = []

    for slot_idx in range(len(events)):
        if events[slot_idx] is not None:
            compacted_events.append(events[slot_idx])
        elif wc_idx < len(wc_queries):
            wc_q = wc_queries[wc_idx]
            compacted_events.append({
                "query": wc_q["query"],
                "source": wc_q["source"],
                "pair_id": f"WC-{wc_idx + 1:04d}",
                "is_seed": True,
                "matched_seed_id": None,
                "is_reuse_safe": False,
                "domain": wc_q["domain"],
            })
            wc_idx += 1

    # Assign event IDs and link follow-ups to seed IDs
    pair_to_seed_id: Dict[str, str] = {}
    for idx, ev in enumerate(compacted_events, start=1):
        ev_id = f"PA-SEALED-{idx:04d}"
        ev["id"] = ev_id
        if ev["is_seed"]:
            pair_to_seed_id[ev["pair_id"]] = ev_id
        else:
            ev["matched_seed_id"] = pair_to_seed_id.get(ev["pair_id"])

    return compacted_events


def main() -> None:
    print("=" * 80)
    print("PHASE A4: ASSEMBLING SEALED TEST STREAM (N = 2,200 QUERIES)")
    print("=" * 80)

    # 1. Load public held-out slices
    qqp_pairs = load_qqp_heldout_pairs(n_pairs=500, start_offset=10000)
    paws_pairs = load_paws_test_pairs(n_pos=150, n_neg=150)
    cqa_pairs = load_cqadupstack_heldout_pairs(n_pairs=200, pairs_per_subforum=18)
    wc_queries = load_wildchat_stream_queries(n_queries=200, start_offset=5000)

    print(f"\nLoaded components:")
    print(f"  QQP held-out pairs:        {len(qqp_pairs)} ({sum(1 for p in qqp_pairs if p['is_safe'])} pos, {sum(1 for p in qqp_pairs if not p['is_safe'])} neg)")
    print(f"  PAWS test pairs:           {len(paws_pairs)} ({sum(1 for p in paws_pairs if p['is_safe'])} pos, {sum(1 for p in paws_pairs if not p['is_safe'])} neg)")
    print(f"  CQADupStack held-out:      {len(cqa_pairs)} ({sum(1 for p in cqa_pairs if p['is_safe'])} pos)")
    print(f"  WildChat organic queries:  {len(wc_queries)}")

    # 2. Assemble stream
    stream = assemble_sealed_stream(qqp_pairs, paws_pairs, cqa_pairs, wc_queries, seed=42)
    print(f"\nAssembled stream length: {len(stream)} queries")

    # Counts
    n_seeds = sum(1 for e in stream if e["is_seed"])
    n_followups = sum(1 for e in stream if not e["is_seed"])
    n_safe_followups = sum(1 for e in stream if not e["is_seed"] and e["is_reuse_safe"])
    n_unsafe_followups = sum(1 for e in stream if not e["is_seed"] and not e["is_reuse_safe"])

    print(f"  Cold seeds:                  {n_seeds}")
    print(f"  Follow-up queries:           {n_followups}")
    print(f"  Safe non-exact targets:      {n_safe_followups} (Potential hits)")
    print(f"  Unsafe adversarial targets:  {n_unsafe_followups} (Adversarial negatives)")

    OUTPUT_STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STREAM_PATH, "w", encoding="utf-8") as f:
        json.dump(stream, f, indent=2)

    # Compute SHA-256
    with open(OUTPUT_STREAM_PATH, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"\nPersisted sealed stream to: {OUTPUT_STREAM_PATH}")
    print(f"SHA-256 Checksum: {sha256_hash}")


if __name__ == "__main__":
    main()
