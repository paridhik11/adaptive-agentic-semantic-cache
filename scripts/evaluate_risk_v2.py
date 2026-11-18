"""Measure three cache policies against frozen, answer-bound risk fixtures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_v2_policy import ReportingControlCache
from src.policy.decomposed_applicability import answer_satisfies_requirements
from src.policy.encoders import DenseEncoder
from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION, LocalPassageScorer
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope

FIXTURE = "artifacts/risk_v2/fixtures.json"
FROZEN = (
    FIXTURE, "docs/RISK_V2_PROTOCOL.md", "scripts/evaluate_risk_v2.py", "scripts/risk_v2_policy.py",
    "src/policy/hybrid_cache.py", "src/policy/decomposed_applicability.py", "src/policy/encoders.py",
    "src/policy/local_models.py", "src/policy/response_cache.py", "src/policy/semantic_runtime.py",
)
SEAL = ROOT / "artifacts/risk_v2/seal.json"
POLICIES = {
    "exact_only": (ConservativeHybridCache, "exact"),
    "current_guarded": (ConservativeHybridCache, "guarded"),
    "reporting_control": (ReportingControlCache, "guarded"),
}


def load_groups():
    groups = json.loads((ROOT / FIXTURE).read_text(encoding="utf-8"))
    identifiers = set()
    owners = {}
    for group in groups:
        identifier = group["id"]
        if identifier in identifiers:
            raise ValueError(f"Duplicate group: {identifier}")
        identifiers.add(identifier)
        probes = group["probes"]
        if (len(probes) != 5 or len({row["text"] for row in probes}) != 5
                or len({row["type"] for row in probes}) != 5
                or any(type(row["eligible"]) is not bool for row in probes)
                or sum(row["eligible"] for row in probes) != 3
                or probes[0] != {"type": "exact_repeat", "text": group["seed"], "eligible": True}):
            raise ValueError(f"Invalid probe design for {identifier}")
        for prompt in {row["text"] for row in probes} | {group["seed"]}:
            normalized = " ".join(re.findall(r"\w+", prompt.casefold()))
            if owners.setdefault(normalized, identifier) != identifier:
                raise ValueError(f"Prompt overlap across groups: {identifier}")
    if Counter(group["split"] for group in groups) != {"calibration": 6, "holdout": 24}:
        raise ValueError("Split sizes disagree with the protocol")
    return groups


def inputs():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in FROZEN}


def request(text):
    return {"model": "risk-fixture", "temperature": 0, "max_tokens": 128,
            "messages": [{"role": "system", "content": "Answer from the approved oversight return."},
                         {"role": "user", "content": text}]}


def response(content, finish="stop"):
    return {"model": "risk-fixture", "choices": [
        {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}},
    ]}


def run_group(group, model, encoder, scorer):
    cache_type, mode = POLICIES[model]
    scope = KnowledgeScope("risk-fixture", "v2", group["id"], "reporting")
    cache = cache_type(
        ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "frozen_fixture")),
        scope=scope, mode=mode, encoder=encoder, passage_scorer=scorer,
        retrieval_threshold=0.60, verification_threshold=0.85,
    )
    primed = response(group["answer"])
    if answer_satisfies_requirements(group["seed"], group["answer"]):
        raise ValueError(f"{group['id']}: invalid seed response")
    seed = cache.process(request(group["seed"]), provider=lambda _: primed)
    if not seed.stored:
        raise ValueError(f"{group['id']}: seed not stored")
    # The marker's finish reason keeps the probe from changing cache state.
    fallback = response("Controlled fallback marker", finish="length")
    for probe in group["probes"]:
        calls = 0

        def provider(_):
            nonlocal calls
            calls += 1
            return fallback

        result = cache.process(request(probe["text"]), provider=provider)
        if result.stored or calls != result.provider_calls:
            raise ValueError("Probe modified cache or logging disagrees with scripted transport")
        reused = result.status == "HIT"
        if result.response != (primed if reused else fallback) or calls != int(not reused):
            raise ValueError("Returned answer identity contradicts decision")
        yield {
            "group": group["id"], "type": probe["type"], "policy": model, "eligible": probe["eligible"],
            "served": reused, "served_tier": result.tier, "calls": calls,
            "elapsed_ms": round(result.elapsed_ms, 5),
            "answer_origin": "verified_seed_identity" if reused else "scripted_fallback_marker",
            "answer_sha256": hashlib.sha256(
                result.response["choices"][0]["message"]["content"].encode("utf-8")
            ).hexdigest(),
            "guard_reasons": result.diagnosis["guard_rejections"],
            "stage": result.diagnosis["stage"],
        }


def proportion(success, total):
    if not total:
        return {"count": success, "n": 0, "rate": None, "exact_95_ci": None}
    ci = binomtest(success, total).proportion_ci(method="exact")
    return {"count": success, "n": total, "rate": success / total, "exact_95_ci": [ci.low, ci.high]}


def summarize(rows):
    semantic = [row for row in rows if row["served_tier"] == "SEMANTIC"]
    eligible = [row for row in rows if row["eligible"] and row["type"] != "exact_repeat"]
    unsuitable = [row for row in rows if not row["eligible"]]
    return {
        "requests": len(rows), "groups": len({row["group"] for row in rows}),
        "semantic_incorrect_reuse": proportion(sum(not row["eligible"] for row in semantic), len(semantic)),
        "eligible_non_exact_capture": proportion(sum(row["served"] for row in eligible), len(eligible)),
        "challenge_accepts": proportion(sum(row["served"] for row in unsuitable), len(unsuitable)),
        "exact_hits": sum(row["served_tier"] == "EXACT" for row in rows),
        "semantic_hits": len(semantic),
        "scripted_calls_avoided": sum(1 - row["calls"] for row in rows),
        "calls_avoided_on_eligible_rows": sum(1 - row["calls"] for row in rows if row["eligible"]),
        "cache_path_p50_ms": statistics.median(row["elapsed_ms"] for row in rows),
        "exception_types": dict(Counter(row["type"] for row in unsuitable if row["served"])),
    }


def run(split, groups):
    import torch

    torch.set_num_threads(2)
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    scorer = LocalPassageScorer()
    rows = [
        row for group in groups if group["split"] == split
        for policy in POLICIES for row in run_group(group, policy, encoder, scorer)
    ]
    by_policy = {policy: [row for row in rows if row["policy"] == policy] for policy in POLICIES}
    baseline = by_policy["current_guarded"]
    candidate = by_policy["reporting_control"]
    changes = Counter()
    for current, proposed in zip(baseline, candidate, strict=True):
        for key in ("group", "type", "eligible"):
            if current[key] != proposed[key]:
                raise ValueError("Comparison is unpaired")
        if current["served"] != proposed["served"]:
            kind = "valid" if current["eligible"] else "incorrect"
            change = "removed" if current["served"] else "added"
            changes[f"{kind}_{change}"] += 1
    paired = {key: changes[key] for key in ("valid_removed", "valid_added", "incorrect_removed", "incorrect_added")}
    result = {
        "split": split, "evidence": "frozen authored answer fixtures with pinned local neural cache execution",
        "models": [encoder.identity, scorer.identity], "sealed_inputs": inputs(),
        "arms": {policy: summarize(records) for policy, records in by_policy.items()},
        "paired_changes": paired,
        "candidate_qualifies_for_followup": (
            paired["incorrect_removed"] > 0 and paired["incorrect_added"] == paired["valid_removed"] == 0
        ),
        "receipts": rows,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "calibration", "holdout"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = load_groups()
    if args.action == "freeze":
        with SEAL.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(inputs(), file, indent=2)
            file.write("\n")
        return
    if args.output is None:
        parser.error("pass a new --output path for the receipt")
    if json.loads(SEAL.read_text(encoding="utf-8")) != inputs():
        parser.error("frozen source or fixtures changed")
    # Exclusive create protects prior receipts, including the first holdout.
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        report = run(args.action, groups)
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "receipts"}, indent=2))


if __name__ == "__main__":
    main()
