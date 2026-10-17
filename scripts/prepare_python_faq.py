"""Download/verify and prepare a pinned CPython FAQ reference workload offline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.faq_corpus import build_constructed_workload, parse_programming_faq, validate_group_splits
from src.benchmark.schema import read_json, validate_workload, write_new, digest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/benchmark/python_faq.sources.json"


def verify_sources(source_dir: Path, *, download: bool) -> dict:
    manifest = read_json(MANIFEST)
    for item in manifest["files"]:
        target = source_dir / Path(item["path"]).name
        if not target.exists() and download:
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(item["url"], timeout=30) as response:
                target.write_bytes(response.read())
        if not target.is_file():
            raise FileNotFoundError(f"missing source file: {target}; pass --download to fetch the pinned revision")
        raw = target.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item["sha256"] or len(raw) != item["bytes"]:
            raise ValueError(f"source checksum/size mismatch for {item['path']}")
    return manifest


def prepare(source_dir: Path, output_dir: Path, *, download: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory exists; refusing to overwrite evidence: {output_dir}")
    manifest = verify_sources(source_dir, download=download)
    faq_path = source_dir / "programming.rst"
    license_path = source_dir / "LICENSE"
    corpus = parse_programming_faq(faq_path)
    split_check = validate_group_splits(corpus)
    workload = validate_workload(build_constructed_workload(corpus))
    output_dir.mkdir(parents=True)
    write_new(output_dir / "corpus.json", corpus)
    write_new(output_dir / "constructed_workload.json", workload)
    (output_dir / "LICENSE").write_bytes(license_path.read_bytes())
    split_counts = {}
    for row in corpus["items"]:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "schema_version": "python-faq-preparation-v1",
        "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_revision": manifest["revision"],
        "source_license": manifest["license"],
        "source_files": [{"path": x["path"], "sha256": x["sha256"], "bytes": x["bytes"]} for x in manifest["files"]],
        "corpus_sha256": digest(corpus),
        "corpus_file_sha256": hashlib.sha256((output_dir / "corpus.json").read_bytes()).hexdigest(),
        "workload_sha256": digest(workload),
        "workload_file_sha256": hashlib.sha256((output_dir / "constructed_workload.json").read_bytes()).hexdigest(),
        "group_split_check": split_check,
        "question_counts_by_split": split_counts,
        "request_count_by_split": {key: value * 2 for key, value in split_counts.items()},
        "answer_reuse_gold": None,
        "nature": "reference QA plus constructed exact repeats; natural prevalence unmeasured",
    }
    write_new(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="download missing files at the pinned commit")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source_dir, args.output_dir, download=args.download), indent=2))


if __name__ == "__main__":
    main()
