"""Regenerate Stage A in isolation; inspect final files only through hashes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "1d83b84584b62fc1d5662a45e0db7d5753f5871fa273dadb62f1f3a7b01e39af"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Refusing to replace an existing receipt")
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="stage-a-reproduction-") as temporary:
        clone = Path(temporary) / "source"
        clone.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(clone, filter="data")
        # The original generator is not yet tracked. Preserve its exact bytes.
        for name in ["scripts/build_stage_a_intents.py", "docs/FACTS.md"]:
            shutil.copyfile(ROOT / name, clone / name)
        output = clone / "regenerated"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / ".venv/lib/python3.12/site-packages")
        command = [sys.executable, str(clone / "scripts/build_stage_a_intents.py"),
                   "--output-dir", str(output)]
        completed = subprocess.run(command, cwd=clone, env=environment,
                                   text=True, capture_output=True, check=True)
        comparisons = {}
        for name in ["final.manifest.json", "final.jsonl", "final.answers.jsonl"]:
            recorded = digest(ROOT / "data/intent_stage_a" / name)
            regenerated = digest(output / name)
            comparisons[name] = {"recorded": recorded, "regenerated": regenerated,
                                 "byte_hash_match": recorded == regenerated}
        success = (all(x["byte_hash_match"] for x in comparisons.values()) and
                   comparisons["final.manifest.json"]["recorded"]["sha256"] == EXPECTED)
        receipt = {
            "evidence_type": "deterministic_regeneration_hash_check_not_model_evaluation",
            "base_commit": base, "generator": digest(ROOT / "scripts/build_stage_a_intents.py"),
            "seed": "deterministic ordered enumeration; no random-number generator",
            "isolated_command": "python scripts/build_stage_a_intents.py --output-dir regenerated",
            "generator_exit_code": completed.returncode,
            "final_records_inspected_or_evaluated": False,
            "comparisons": comparisons, "reproducible": success,
        }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not success:
        raise SystemExit("Final reproduction mismatch; do not exclude final stores")


if __name__ == "__main__":
    main()
