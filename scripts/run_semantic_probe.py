"""Inventory, explicitly prepare, then probe pinned models on exposed dev examples.

No generation API, API key, final evaluation labels or serving approval. Default
execution inventories only; --prepare permits public model artifact downloads.
--probe loads local artifacts only and reports retrieval/pair diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policy.local_models import (ENCODER_MODEL, ENCODER_REVISION, PAIR_MODEL,
                                     PAIR_REVISION, LocalPairScorer)

MODELS = {"encoder": (ENCODER_MODEL, ENCODER_REVISION), "pair": (PAIR_MODEL, PAIR_REVISION)}
SEEDS = [
    "How do I convert a Python list into a tuple?",
    "How do I convert a Python tuple into a list?",
    "How do I convert Celsius to Fahrenheit?",
    "How do I convert Fahrenheit to Celsius?",
    "How can I sort a Python list in ascending order?",
]
PROBES = [
    ("paraphrase", "How can I make a tuple from an existing Python list?", 0),
    ("paraphrase", "What is the way to change a tuple into a list in Python?", 1),
    ("paraphrase", "Given a temperature in Celsius, how can I express it in Fahrenheit?", 2),
    ("direction_flip", "How do I convert a Python tuple into a list?", 1),
    ("constraint", "How do I convert a nested Python list into a tuple recursively?", None),
    ("negation", "Why should I not convert a Python list into a tuple?", None),
    ("unit_change", "How do I convert Celsius to Kelvin?", None),
    ("constraint", "How can I sort a Python list without changing its original order?", None),
]


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inventory():
    packages = {}
    for name in ("torch", "sentence-transformers", "transformers", "huggingface-hub", "socksio"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cache = Path(os.environ.get("HF_HUB_CACHE") or
                 Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub")
    models = {}
    for role, (name, revision) in MODELS.items():
        snapshot = cache / ("models--" + name.replace("/", "--")) / "snapshots" / revision
        weights = []
        if snapshot.exists():
            for p in sorted(snapshot.rglob("*")):
                if p.is_file() and p.suffix in {".safetensors", ".bin"}:
                    weights.append({"file": str(p.relative_to(snapshot)), "bytes": p.stat().st_size,
                                    "sha256": file_digest(p)})
        models[role] = {"model": name, "revision": revision, "snapshot_present": snapshot.exists(),
                        "weights": weights, "inference_verified": False}
    return {"python": sys.version.split()[0], "packages": packages, "models": models,
            "fine_tuned_checkpoint_available": any((ROOT / "models/phaseA_cross_encoder_finetuned").glob("*.safetensors"))
                or any((ROOT / "models/phaseA_cross_encoder_finetuned").glob("*.bin"))}


def probe():
    from src.policy.encoders import DenseEncoder
    started = time.perf_counter()
    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    scorer = LocalPairScorer()
    load_seconds = time.perf_counter() - started
    vectors = [encoder.encode(text) for text in SEEDS]
    rows = []
    for kind, text, expected_index in PROBES:
        started = time.perf_counter()
        query_tokens = len(encoder.model.tokenizer(text, truncation=False)["input_ids"])
        if query_tokens > encoder.model.max_seq_length:
            raise ValueError("query exceeds encoder input window")
        vector = encoder.encode(text)
        ranked = sorted(((sum(a*b for a,b in zip(vector, v)), i) for i,v in enumerate(vectors)), reverse=True)
        score, index = ranked[0]
        pair = scorer.score(SEEDS[index], text)
        # Always also measure against the original list->tuple seed: a direction
        # flip can look extremely similar despite requiring the opposite answer.
        rows.append({"kind": kind, "query": text, "nearest_seed": SEEDS[index], "cosine": score,
                     "authored_expected_seed": expected_index, "selected_seed": index,
                     "pair": pair, "against_first_seed": scorer.score(SEEDS[0], text),
                     "diagnostic_ms": (time.perf_counter()-started)*1000})
    return {"load_seconds": load_seconds, "seeds": SEEDS, "rows": rows,
            "claim": "exposed authored development diagnostics; no answer applicability labels or serving decisions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-dev/model-probe.json"))
    args = parser.parse_args()
    result = {"schema_version": "semantic-model-probe-v1", "external_generation_calls": 0,
              "evidence_type": "development_local_model_diagnostic", "before": inventory()}
    code = 0
    try:
        if args.prepare:
            from huggingface_hub import snapshot_download
            for name, revision in MODELS.values():
                snapshot_download(name, revision=revision,
                    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "README.md"],
                    ignore_patterns=["onnx/*", "openvino/*"], max_workers=4)
        if args.probe:
            result["probe"] = probe()
        result["after"] = inventory()
        if args.probe:
            for model in result["after"]["models"].values():
                model["inference_verified"] = True
        result["status"] = "complete"
    except Exception as error:
        result.update(status="blocked", error_type=type(error).__name__)
        # Exception strings can contain signed artifact URLs; keep logs private
        # and avoid publishing them in aggregate evidence.
        result["after"] = inventory()
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "error_type": result.get("error_type"), "external_generation_calls": 0}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
