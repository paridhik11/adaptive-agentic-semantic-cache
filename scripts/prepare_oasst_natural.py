"""Download a pinned OASST1 shard and export only repeat-rate aggregates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.natural_repetition import summarize_first_turns
from src.benchmark.schema import read_json, write_new


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "examples/benchmark/oasst1.sources.json"


def verify_files(source_dir: Path, *, download: bool) -> tuple[dict, list[dict]]:
    manifest = read_json(MANIFEST_PATH)
    verified = []
    for item in manifest["files"]:
        target = source_dir / item["path"]
        if not target.exists() and download:
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{manifest['revision']}/{item['path']}"
            with urlopen(url, timeout=90) as response:
                target.write_bytes(response.read())
        if not target.is_file():
            raise FileNotFoundError(f"missing pinned source {target}; use --download to fetch it")
        raw = target.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item["sha256"] or len(raw) != item["bytes"]:
            raise ValueError(f"source checksum/size mismatch: {item['path']}")
        verified.append({"path": item["path"], "sha256": actual, "bytes": len(raw)})
    return manifest, verified


def prepare(source_dir: Path, output_dir: Path, *, limit: int = 5000, download: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory exists; refusing to overwrite: {output_dir}")
    manifest, verified = verify_files(source_dir, download=download)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Parquet reading requires pyarrow; install with `pip install '.[data]'`") from error
    parquet_path = source_dir / "data/train-00000-of-00001-b42a775f407cee45.parquet"
    table = parquet.read_table(parquet_path)
    rows = table.to_pylist()
    summary = summarize_first_turns(rows, limit=limit)
    summary.update({
        "dataset": manifest["dataset"],
        "source_revision": manifest["revision"],
        "source_license": manifest["license"],
        "source_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "verified_files": verified,
        "source_message_count": table.num_rows,
        "selection": {
            "root_role": "prompter",
            "root_parent_id": None,
            "excluded_deleted": True,
            "excluded_synthetic": True,
            "deduplicate_group": "message_tree_id",
            "sort": ["created_date", "message_id"],
        },
    })
    output_dir.mkdir(parents=True)
    write_new(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source_dir, args.output_dir, limit=args.limit, download=args.download), indent=2))


if __name__ == "__main__":
    main()
