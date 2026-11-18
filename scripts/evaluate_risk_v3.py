"""Replay the frozen four-arm reporting context comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from evaluate_risk_v2 import FROZEN as V2_SOURCES, ROOT, request, response, summarize, proportion
from risk_v2_policy import ReportingControlCache
from risk_v3_context import ReportingCoordinates, process_report
from src.policy.encoders import DenseEncoder
from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION, LocalPassageScorer
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope

DATA = ROOT / "artifacts/risk_v3"
FROZEN = (*V2_SOURCES[2:], "artifacts/risk_v3/fixtures.json", "docs/RISK_V3_PROTOCOL.md",
          "scripts/evaluate_risk_v3.py", "scripts/risk_v3_context.py")
ARMS = {
    "exact": (ConservativeHybridCache, "exact"),
    "guarded": (ConservativeHybridCache, "guarded"),
    "reporting": (ReportingControlCache, "guarded"),
    "context": (ReportingControlCache, "guarded"),
}
CHANGED = {"period": "month", "entity": "entity", "scope": "population"}
SUITABLE = ("exact_repeat", "paraphrase_a", "paraphrase_b", "unresolved")
SCENARIOS = (*SUITABLE, "missing_artifact", *CHANGED)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def load():
    groups = json.loads((DATA / "fixtures.json").read_text(encoding="utf-8"))
    if (Counter(g["split"] for g in groups) != {"calibration": 6, "holdout": 24}
            or len({g["id"] for g in groups}) != len(groups)):
        raise ValueError("group allocation disagrees with the protocol")
    owners = {}
    for group in groups:
        seed = group["context"]
        ReportingCoordinates(**seed)
        if tuple(p["type"] for p in group["probes"]) != SCENARIOS:
            raise ValueError(f"scenario mismatch in {group['id']}")
        within = set()
        for probe in group["probes"]:
            kind, ctx = probe["type"], probe["context"]
            norm = " ".join(re.findall(r"\w+", probe["text"].casefold()))
            if owners.setdefault(norm, group["id"]) != group["id"] or norm in within:
                raise ValueError("duplicate probe text")
            within.add(norm)
            if probe["eligible"] is not (kind in SUITABLE):
                raise ValueError("incorrect eligibility assignment")
            if kind == "exact_repeat" and probe["text"] != group["seed"]:
                raise ValueError("exact repeat does not match the priming prompt")
            if ctx is None:
                if kind != "unresolved":
                    raise ValueError("missing reporting coordinates")
            else:
                ReportingCoordinates(**ctx)
                fields = {key for key, val in seed.items() if val != ctx[key]}
                if kind == "unresolved" or fields != ({CHANGED[kind]} if kind in CHANGED else set()):
                    raise ValueError("wrong material coordinate changed")
    return groups


def replay(group, arm, encoder, scorer):
    engine, mode = ARMS[arm]
    cache = engine(
        ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "context_fixture")),
        scope=KnowledgeScope("risk-fixture", "v3", group["id"], "reporting"),
        mode=mode, encoder=encoder, passage_scorer=scorer, retrieval_threshold=.60, verification_threshold=.85,
    )
    cached_answer = response(group["answer"])
    fallback = response("Controlled fallback marker", finish="length")

    def execute(text, context, provider):
        body = request(text)
        if arm != "context":
            return cache.process(body, provider=provider)
        coords = ReportingCoordinates(**context) if context is not None else None
        return process_report(cache, body, provider=provider, coordinates=coords)

    if not execute(group["seed"], group["context"], lambda _: cached_answer).stored:
        raise ValueError("priming failed")
    for probe in group["probes"]:
        observed_calls = 0

        def provider(_):
            nonlocal observed_calls
            observed_calls += 1
            return fallback

        result = execute(probe["text"], probe["context"], provider)
        reused = result.status == "HIT"
        if (result.stored or observed_calls != result.provider_calls or observed_calls != int(not reused)
                or result.response != (cached_answer if reused else fallback)):
            raise ValueError("response identity or call count disagrees with the cache decision")
        yield {
            "group": group["id"], "type": probe["type"], "policy": arm, "eligible": probe["eligible"],
            "resolved": probe["context"] is not None, "served": reused, "served_tier": result.tier,
            "calls": observed_calls, "elapsed_ms": round(result.total_latency_ms, 5),
            "response_sha256": hashlib.sha256(
                result.response["choices"][0]["message"]["content"].encode()
            ).hexdigest(),
            "stage": result.diagnosis["stage"], "guard_rejections": result.diagnosis["guard_rejections"],
        }


def compare(base, candidate):
    delta = Counter()
    for left, right in zip(base, candidate, strict=True):
        if any(left[k] != right[k] for k in ("group", "type", "eligible", "resolved")):
            raise ValueError("unpaired comparison")
        if left["served"] == right["served"]:
            continue
        kind = "suitable" if left["eligible"] else "unsuitable"
        direction = "lost" if left["served"] else "gained"
        delta[f"{kind}_{direction}"] += 1
        if left["eligible"] and left["resolved"] and left["served"]:
            delta["resolved_suitable_lost"] += 1
        if left["type"] in CHANGED and left["served"]:
            delta["material_unsuitable_prevented"] += 1
    return {k: delta[k] for k in (
        "suitable_lost", "suitable_gained", "unsuitable_lost", "unsuitable_gained",
        "resolved_suitable_lost", "material_unsuitable_prevented",
    )}


def run(split, groups):
    import torch
    torch.set_num_threads(2)
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    scorer = LocalPassageScorer()
    receipts = {
        arm: [row for group in groups if group["split"] == split for row in replay(group, arm, encoder, scorer)]
        for arm in ARMS
    }
    metrics = {}
    for arm, rows in receipts.items():
        suitable_resolved = [r for r in rows if r["eligible"] and r["type"] != "exact_repeat" and r["resolved"]]
        unresolved = [r for r in rows if not r["resolved"]]
        material = [r for r in rows if r["type"] in CHANGED]
        metrics[arm] = summarize(rows) | {
            "resolved_nonexact_capture": proportion(sum(r["served"] for r in suitable_resolved), len(suitable_resolved)),
            "unresolved_capture": proportion(sum(r["served"] for r in unresolved), len(unresolved)),
            "changed_context_accepts": proportion(sum(r["served"] for r in material), len(material)),
        }
    paired = compare(receipts["reporting"], receipts["context"])
    return {
        "evidence": "pinned neural replay on fresh synthetic requests with caller-resolved context",
        "split": split, "models": [encoder.identity, scorer.identity], "sealed_inputs": hashes(),
        "arms": metrics, "paired_reporting_vs_context": paired,
        "meets_followup_criterion": bool(paired["material_unsuitable_prevented"] and
                                          not paired["unsuitable_gained"] and not paired["resolved_suitable_lost"]),
        "receipts": [row for rows in receipts.values() for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "calibration", "holdout"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = load()
    seal = DATA / "seal.json"
    if args.action == "freeze":
        with seal.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(hashes(), indent=2) + "\n")
        return
    if args.output is None or json.loads(seal.read_text(encoding="utf-8")) != hashes():
        parser.error("pass a new --output path and verify frozen sources and data")
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(run(args.action, groups), indent=2) + "\n")


if __name__ == "__main__":
    main()
