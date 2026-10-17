"""Validate a local pilot bundle and write only its redacted aggregate audit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.pilot_package import summarize_pilot_package
from src.benchmark.schema import write_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="local pilot-result-package-v1 JSON; never copied to output")
    parser.add_argument("--output", type=Path, required=True,
                        help="new path for the answer-free aggregate audit JSON")
    args = parser.parse_args(argv)
    summary = summarize_pilot_package(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, summary)
    print(json.dumps({
        "audit_written": str(args.output),
        "source_package_sha256": summary["source_package_sha256"],
        "trace_sha256": summary["trace_sha256"],
        "status_counts": summary["status_counts"],
        "provider_attempts": summary["provider_attempts"],
        "answer_review": summary["answer_review"],
        "safety_certified": summary["answer_review"]["safety_certified"],
    }, indent=2))


if __name__ == "__main__":
    main()
