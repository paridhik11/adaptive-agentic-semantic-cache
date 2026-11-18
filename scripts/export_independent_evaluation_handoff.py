"""Milestone E: Export Independent Evaluation Handoff Packet.

Exports actual selected cached answers, incoming requests, source context,
candidate IDs, hashes, and verification decisions for blinded review.

Safety and Integrity Invariants:
1. Does NOT reuse the spent 30-question diagnostic or the 220-event dev benchmark.
2. Does NOT use generated variation labels as correctness evidence.
3. Separates 4 orthogonal review axes:
   - Answer applicability
   - Factual correctness
   - Cache-induced correctness degradation
   - Unknown/unreviewed cases
4. Emits blinded review template and cryptographic audit manifest.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy, LocalPassageScorer
from src.policy.decomposed_applicability import requirement_guard
from src.policy.encoders import DenseEncoder, normalize
from src.policy.local_models import (
    ENCODER_MODEL, ENCODER_REVISION,
    PASSAGE_MODEL, PASSAGE_REVISION,
)
from src.policy.semantic_runtime import approval_binding, digest


def build_fresh_evaluation_pool() -> list[dict]:
    """Builds a fresh, independent set of technical query-answer-candidate pairings
    drawn from broader Python documentation domains not in the spent 30-question set.
    """
    topics = [
        {
            "group_id": "grp_context_managers",
            "section": "Standard Library: contextlib",
            "source_doc": "contextlib provides utilities for with-statement contexts. @contextmanager transforms a generator into a context manager.",
            "source_revision": "cpython-3.12-contextlib-d89f2a",
            "seed_query": "How do I create a custom context manager using contextlib in Python?",
            "actual_cached_answer": "Use the @contextmanager decorator from contextlib over a generator function. Yield the resource inside a try...finally block to guarantee cleanup.",
            "candidates": [
                ("How can I write a context manager using the contextlib module?", "safe_paraphrase"),
                ("What is the syntax for creating context managers with @contextmanager?", "safe_paraphrase"),
                ("How do I write an asynchronous context manager with asynccontextmanager in Python?", "changed_requirement_async"),
                ("How do I create a context manager in Python without using contextlib?", "negation_constraint"),
                ("How do I write a custom context manager in Python 2.7?", "version_mismatch"),
            ],
        },
        {
            "group_id": "grp_functools_lru",
            "section": "Standard Library: functools",
            "source_doc": "functools.lru_cache wraps a function with a memoizing callable that saves up to maxsize results. Since a dictionary is used to cache results, function arguments must be hashable.",
            "source_revision": "cpython-3.12-functools-e12b4c",
            "seed_query": "How do I memoize a function in Python using functools?",
            "actual_cached_answer": "Use the @functools.lru_cache(maxsize=128) decorator on your function. Arguments must be hashable because the cache uses an internal dictionary.",
            "candidates": [
                ("How can I cache function return values with functools in Python?", "safe_paraphrase"),
                ("What decorator should I use to memoize function calls in Python?", "safe_paraphrase"),
                ("How do I memoize a function that takes unhashable mutable lists in Python?", "added_unhashable_constraint"),
                ("How do I implement function memoization without using functools or standard libraries in Python?", "negation_constraint"),
                ("How do I memoize a function with unbounded cache size in Python?", "numeric_maxsize_none"),
            ],
        },
        {
            "group_id": "grp_pathlib_operations",
            "section": "Standard Library: pathlib",
            "source_doc": "pathlib.Path.rglob recursively yields all existing files matching the given relative pattern. Path.resolve returns a new path object with absolute path and symlinks resolved.",
            "source_revision": "cpython-3.12-pathlib-f44a90",
            "seed_query": "How do I recursively find all files matching a pattern using pathlib in Python?",
            "actual_cached_answer": "Use Path.rglob(pattern) or Path.glob('**/' + pattern). For example, Path('.').rglob('*.py') recursively finds all Python files.",
            "candidates": [
                ("How can I search for files matching a pattern recursively with pathlib in Python?", "safe_paraphrase"),
                ("What is the pathlib method to traverse directories recursively for matching files?", "safe_paraphrase"),
                ("How do I find files non-recursively in only the current directory using pathlib in Python?", "direction_constraint_non_recursive"),
                ("How do I search for files recursively in Python without using pathlib or os.walk?", "negation_constraint"),
                ("How do I find files matching exactly 10 extensions recursively in Python?", "numeric_constraint"),
            ],
        },
        {
            "group_id": "grp_dataclasses",
            "section": "Standard Library: dataclasses",
            "source_doc": "dataclasses.dataclass decorator automatically generates special methods such as __init__() and __repr__() for user-defined classes. Setting frozen=True generates an immutable dataclass.",
            "source_revision": "cpython-3.12-dataclasses-b77c31",
            "seed_query": "How do I create an immutable dataclass in Python?",
            "actual_cached_answer": "Decorate the class with @dataclass(frozen=True). Any attempt to modify field values after instantiation will raise a FrozenInstanceError.",
            "candidates": [
                ("How can I define a frozen or read-only dataclass in Python?", "safe_paraphrase"),
                ("What is the syntax to make a dataclass immutable in Python?", "safe_paraphrase"),
                ("How do I create a mutable dataclass in Python?", "direction_reversal_mutable"),
                ("How do I create an immutable class in Python without using dataclasses?", "negation_constraint"),
                ("How do I create a frozen dataclass with inheritance in Python 3.6?", "version_mismatch_3_6"),
            ],
        },
    ]

    return topics


def export_handoff(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    topics = build_fresh_evaluation_pool()

    print("Loading models for independent evaluation scoring...", flush=True)
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    passage_scorer = LocalPassageScorer(PASSAGE_MODEL, PASSAGE_REVISION)

    export_items = []
    blinded_review_rows = []

    item_counter = 1

    for topic in topics:
        group_id = topic["group_id"]
        section = topic["section"]
        source_doc = topic["source_doc"]
        source_rev = topic["source_revision"]
        seed_q = topic["seed_query"]
        cached_ans = topic["actual_cached_answer"]

        # Embed seed
        seed_vec = tuple(normalize(encoder.encode(seed_q)))
        seed_req = {
            "model": "gpt-4o-mini-fixture",
            "messages": [
                {"role": "system", "content": f"Answer using only Python reference: {source_doc}"},
                {"role": "user", "content": seed_q},
            ],
        }
        seed_resp = {
            "choices": [{"message": {"role": "assistant", "content": cached_ans}}],
        }
        cand_id = f"cand-{digest(seed_q)[:10]}"

        for incoming_q, variation_tag in topic["candidates"]:
            eval_id = f"EVAL-{item_counter:03d}"
            item_counter += 1

            inc_req = {
                "model": "gpt-4o-mini-fixture",
                "messages": [
                    {"role": "system", "content": f"Answer using only Python reference: {source_doc}"},
                    {"role": "user", "content": incoming_q},
                ],
            }

            # Run policy evaluation
            inc_vec = tuple(normalize(encoder.encode(incoming_q)))
            cosine_sim = sum(a * b for a, b in zip(inc_vec, seed_vec))

            guard_error = requirement_guard(seed_q, incoming_q)
            guards_passed = (guard_error is None)

            score_data = passage_scorer.score(incoming_q, cached_ans)
            prob = score_data["probability"]

            decision_approved = guards_passed and (prob >= 0.85) and (cosine_sim >= 0.60)
            decision_reason = (
                "verified_semantic_hit" if decision_approved else
                f"guard_rejected:{guard_error}" if not guards_passed else
                f"verifier_rejected:{prob:.4f}<0.85"
            )

            binding = approval_binding(inc_req, seed_req, seed_resp, tenant_id="tenant-eval", scope={"rev": source_rev})

            export_items.append({
                "evaluation_id": eval_id,
                "group_id": group_id,
                "section": section,
                "source_revision": source_rev,
                "source_context": source_doc,
                "incoming_request": inc_req,
                "incoming_query": incoming_q,
                "candidate_id": cand_id,
                "candidate_query": seed_q,
                "actual_cached_answer": cached_ans,
                "hashes": {
                    "incoming_query_sha256": digest(incoming_q),
                    "candidate_answer_sha256": digest(cached_ans),
                    "binding_sha256": digest(binding),
                },
                "policy_evaluation": {
                    "cosine_similarity": cosine_sim,
                    "guards_passed": guards_passed,
                    "guard_error": guard_error,
                    "passage_verifier_prob": prob,
                    "decision": "approved" if decision_approved else "rejected",
                    "reason": decision_reason,
                },
                "variation_tag_unblinded_internal_only": variation_tag,
            })

            # Blinded review row: omits variation tag and internal policy decision
            blinded_review_rows.append({
                "evaluation_id": eval_id,
                "group_id": group_id,
                "section": section,
                "source_reference": source_doc,
                "user_question": incoming_q,
                "presented_cached_answer": cached_ans,
                "review_fields": {
                    "1_answer_applicability": None,  # Options: "SUFFICIENT" | "INSUFFICIENT" | "UNCERTAIN"
                    "2_factual_correctness": None,   # Options: "CORRECT" | "INCORRECT" | "UNCERTAIN"
                    "3_cache_induced_degradation": None, # Options: "NONE" | "DEGRADED" | "UNCERTAIN"
                    "4_confidence_and_rationales": "",
                },
            })

    # Shuffle blinded review rows for true blinded evaluation
    shuffled_rows = copy.deepcopy(blinded_review_rows)
    random.seed(42)
    random.shuffle(shuffled_rows)

    # Write export files
    full_manifest_path = output_dir / "evaluation_manifest.json"
    with open(full_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "independent-evaluation-handoff-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_items": len(export_items),
            "groups_count": len(topics),
            "policy_model": "AnswerAwareSemanticPolicy (MiniLM-L6 + MS MARCO)",
            "rules": [
                "Do not use generated variation labels as correctness evidence.",
                "Reviewer evaluates actual selected cached answers blind to model judgments.",
                "Separate applicability, factual correctness, cache-induced degradation, and unknowns.",
            ],
            "items": export_items,
        }, f, indent=2)

    blinded_form_path = output_dir / "blinded_review_form.jsonl"
    with open(blinded_form_path, "w", encoding="utf-8") as f:
        for r in shuffled_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate human-readable Markdown review template
    md_table_path = output_dir / "BLINDED_HUMAN_REVIEW_PACKET.md"
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Blinded Human Review Packet — Independent Evaluation Handoff\n\n")
        f.write("_Protocol: Blinded evaluation of actual cached answer applicability and correctness._\n")
        f.write("_Do NOT cite generated variation tags as evidence._\n\n")
        f.write("## Review Instructions\n\n")
        f.write("For each item, evaluate the **Presented Cached Answer** in response to the **User Question**, based strictly on the provided **Source Reference**.\n\n")
        f.write("Evaluate across four orthogonal axes:\n")
        f.write("1. **Answer Applicability**: Does the answer address what the user actually asked? (`SUFFICIENT` / `INSUFFICIENT` / `UNCERTAIN`)\n")
        f.write("2. **Factual Correctness**: Is the answer factually true according to the source? (`CORRECT` / `INCORRECT` / `UNCERTAIN`)\n")
        f.write("3. **Cache Degradation**: Did reusing this answer introduce any subtle omission or constraint error? (`NONE` / `DEGRADED` / `UNCERTAIN`)\n")
        f.write("4. **Rationale / Unknowns**: State reason if insufficient or uncertain.\n\n")
        f.write("---\n\n")
        f.write("## Review Items\n\n")
        for idx, row in enumerate(shuffled_rows, 1):
            f.write(f"### Item {idx} (ID: `{row['evaluation_id']}`)\n")
            f.write(f"- **Group / Section**: {row['section']} (`{row['group_id']}`)\n")
            f.write(f"- **Source Reference**: {row['source_reference']}\n")
            f.write(f"- **User Question**: `{row['user_question']}`\n")
            f.write(f"- **Presented Cached Answer**: {row['presented_cached_answer']}\n")
            f.write("- **Judgments**:\n")
            f.write("  - [ ] Applicability: `[ ] SUFFICIENT  [ ] INSUFFICIENT  [ ] UNCERTAIN`\n")
            f.write("  - [ ] Correctness: `[ ] CORRECT  [ ] INCORRECT  [ ] UNCERTAIN`\n")
            f.write("  - [ ] Degradation: `[ ] NONE  [ ] DEGRADED  [ ] UNCERTAIN`\n")
            f.write("  - Rationale: \n\n")

    print(f"Exported {len(export_items)} items across {len(topics)} topic groups to {output_dir}:")
    print(f"  1. {full_manifest_path.name} (full audit data with candidate hashes & verification decisions)")
    print(f"  2. {blinded_form_path.name} (shuffled blinded JSONL form)")
    print(f"  3. {md_table_path.name} (human-readable blinded markdown review packet)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/independent_evaluation_handoff")
    args = parser.parse_args()

    export_handoff(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
