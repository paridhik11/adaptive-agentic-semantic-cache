"""Compare real local generation with gated cache reuse on a fixed mixed replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from evaluate_risk_v3 import DATA, hashes, load, request, response
from measure_risk_v2_exact import MAX_NEW_TOKENS, MODEL, REVISION
from risk_v2_policy import ReportingControlCache
from risk_v3_context import ReportingCoordinates, process_report
from src.policy.encoders import DenseEncoder
from src.policy.local_models import ENCODER_MODEL, ENCODER_REVISION, LocalPassageScorer
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope

SCENARIOS = ("exact_repeat", "paraphrase_a", "period", "unresolved")


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if json.loads((DATA / "seal.json").read_text(encoding="utf-8")) != hashes():
        raise ValueError("source or fixture differs from the frozen version")
    groups = load()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, use_safetensors=True, dtype=torch.float32,
    ).eval()
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    scorer = LocalPassageScorer()

    def generate(body):
        inputs = tokenizer.apply_chat_template(
            body["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        with torch.inference_mode():
            result = llm.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                  pad_token_id=tokenizer.eos_token_id)
        completion = result[0, inputs["input_ids"].shape[1]:]
        return response(tokenizer.decode(completion, skip_special_tokens=True), finish="length") | {
            "usage": {"completion_tokens": len(completion)}
        }

    generate(request(groups[0]["seed"]))
    rows, preparation = [], []
    for group in [g for g in groups if g["split"] == "holdout"][:6]:
        cache = ReportingControlCache(
            ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "pilot")),
            scope=KnowledgeScope("risk-fixture", "v3-timing", group["id"], "reporting"),
            mode="guarded", encoder=encoder, passage_scorer=scorer, retrieval_threshold=.60,
            verification_threshold=.85,
        )
        seed = response(group["answer"])
        start = time.perf_counter()
        prime = process_report(cache, request(group["seed"]), provider=lambda _: seed,
                               coordinates=ReportingCoordinates(**group["context"]))
        if not prime.stored:
            raise ValueError("seed was not stored")
        preparation.append(round((time.perf_counter() - start) * 1000, 5))
        probes = {p["type"]: p for p in group["probes"]}
        for scenario in SCENARIOS:
            probe = probes[scenario]
            coords = ReportingCoordinates(**probe["context"]) if probe["context"] else None
            body = request(probe["text"])
            trusted_body = coords.bind(body) if coords is not None else body
            timed = {}
            order = ("fresh", "cache") if len(rows) % 2 == 0 else ("cache", "fresh")
            for arm in order:
                start = time.perf_counter()
                if arm == "fresh":
                    result = cache.process(trusted_body, provider=generate, bypass=True)
                else:
                    result = process_report(cache, body, provider=generate, coordinates=coords)
                ms = round((time.perf_counter() - start) * 1000, 5)
                timed[arm] = (result, ms)
            fresh, fresh_ms = timed["fresh"]
            gated, gated_ms = timed["cache"]
            hit = gated.status == "HIT"
            if (fresh.provider_calls != 1 or gated.provider_calls != int(not hit) or gated.stored
                    or (hit and (not probe["eligible"] or gated.response != seed))
                    or (not hit and gated.response != fresh.response)):
                raise ValueError(f"invalid response or accounting for {group['id']}/{scenario}")
            rows.append({
                "group": group["id"], "scenario": scenario, "fresh_ms": fresh_ms, "gated_ms": gated_ms,
                "fresh_calls": fresh.provider_calls, "gated_calls": gated.provider_calls, "gated_tier": gated.tier,
                "fresh_tokens": fresh.response["usage"]["completion_tokens"],
                "gated_tokens": gated.response["usage"]["completion_tokens"] if not hit else 0,
                "returned_digest": hashlib.sha256(
                    gated.response["choices"][0]["message"]["content"].encode()
                ).hexdigest(),
            })
    return {
        "evidence": "measured paired local CPU mixed-request timing replay",
        "model": MODEL, "revision": REVISION, "platform": platform.platform(),
        "torch": torch.__version__, "threads": torch.get_num_threads(), "max_new_tokens": MAX_NEW_TOKENS,
        "fixture_seal": hashes(), "pairs": len(rows),
        "gated_hits_by_scenario": dict(Counter(row["scenario"] for row in rows if not row["gated_calls"])),
        "fresh_total_ms": sum(row["fresh_ms"] for row in rows),
        "gated_total_ms": sum(row["gated_ms"] for row in rows),
        "fresh_p50_ms": statistics.median(row["fresh_ms"] for row in rows),
        "gated_p50_ms": statistics.median(row["gated_ms"] for row in rows),
        "preparation_ms": preparation,
        **{field: sum(row[field] for row in rows)
           for field in ("fresh_calls", "gated_calls", "fresh_tokens", "gated_tokens")},
        "receipts": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(run(), indent=2) + "\n")


if __name__ == "__main__":
    main()
