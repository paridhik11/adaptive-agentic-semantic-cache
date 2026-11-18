"""Measure warm exact reuse against a paired local generation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from evaluate_risk_v2 import FIXTURE, ROOT, SEAL, load_groups, request, response
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_NEW_TOKENS = 32


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fixture_hash = hashlib.sha256((ROOT / FIXTURE).read_bytes()).hexdigest()
    if fixture_hash != json.loads(SEAL.read_text(encoding="utf-8"))[FIXTURE]:
        raise ValueError("Unexpected fixture bytes")
    groups = load_groups()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, use_safetensors=True, dtype=torch.float32,
    ).eval()
    cache = ScopedResponseCache(eligibility=lambda _: EligibilityDecision(True, "local_exact_pilot"))

    def generate(body):
        input_ids = tokenizer.apply_chat_template(
            body["messages"], add_generation_prompt=True, tokenize=True, return_tensors="pt",
            return_dict=True,
        )
        with torch.inference_mode():
            output = model.generate(
                **input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = output[0, input_ids["input_ids"].shape[1]:]
        text = tokenizer.decode(completion, skip_special_tokens=True)
        if not text.strip():
            raise ValueError("Local model yielded an empty completion")
        return response(text) | {"usage": {"completion_tokens": len(completion)}}

    generate(request(next(group["seed"] for group in groups if group["split"] == "calibration")))
    rows = []
    for index, group in enumerate(group for group in groups if group["split"] == "holdout"):
        if index == 8:
            break
        body = request(group["seed"])
        prime = cache.process(body, tenant_id="risk-pilot", provider=generate)
        if not prime.stored:
            raise ValueError(f"Could not prime {group['id']}")
        trials = {}
        for kind in (("fresh", "cached") if index % 2 == 0 else ("cached", "fresh")):
            started = time.perf_counter()
            outcome = cache.process(body, tenant_id="risk-pilot", provider=generate, bypass=(kind == "fresh"))
            trials[kind] = (outcome, (time.perf_counter() - started) * 1_000)
        fresh, fresh_ms = trials["fresh"]
        cached, cached_ms = trials["cached"]
        if (fresh.provider_calls != 1 or cached.status != "HIT" or cached.provider_calls != 0
                or fresh.response != cached.response):
            raise ValueError(f"Pair failed identity or call validation: {group['id']}")
        text = cached.response["choices"][0]["message"]["content"]
        rows.append({
            "group": group["id"], "fresh_ms": round(fresh_ms, 5), "cached_ms": round(cached_ms, 5),
            "prime_ms": round(prime.total_latency_ms, 5),
            "fresh_completion_tokens": fresh.response["usage"]["completion_tokens"],
            "cached_generated_tokens": 0, "fresh_calls": fresh.provider_calls, "cached_calls": cached.provider_calls,
            "answer_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return {
        "evidence": "measured local CPU component pilot on warm literal repeats", "model": MODEL,
        "revision": REVISION, "platform": platform.platform(), "machine": platform.machine(),
        "torch_version": torch.__version__, "threads": torch.get_num_threads(), "max_new_tokens": MAX_NEW_TOKENS,
        "fixture_sha256": fixture_hash, "pairs": len(rows),
        "fresh_p50_ms": statistics.median(row["fresh_ms"] for row in rows),
        "cached_p50_ms": statistics.median(row["cached_ms"] for row in rows),
        "paired_p50_ms_avoided": statistics.median(row["fresh_ms"] - row["cached_ms"] for row in rows),
        "fresh_completion_tokens": sum(row["fresh_completion_tokens"] for row in rows),
        "cached_completion_tokens": sum(row["cached_generated_tokens"] for row in rows),
        "receipts": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        report = run()
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "receipts"}, indent=2))


if __name__ == "__main__":
    main()
