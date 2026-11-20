"""Pinned public inputs and group-isolated sampling for the independent audit."""

import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import unquote
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MPI = "https://qa.mpi-inf.mpg.de/comqa/"
HF = "https://huggingface.co/datasets/"
INPUTS = {
    "comqa.json": (
        MPI + "comqa_train.json",
        "aa25b2747221a6147737066fa7f8509414a6333e6eebb20d7970fc50166eba9c",
    ),
    "comqa_dev.json": (
        MPI + "comqa_dev.json",
        "937f898f996e9a7317d311553cf5b43c4a53e0bb841ae8eed4d8cf15936fbc84",
    ),
    "comqa_test.json": (
        MPI + "comqa_test.json",
        "b81e19c4108198781167ddb30d26172571e76cc8e684aabb181537a45e816867",
    ),
    "paws.parquet": (
        HF
        + "google-research-datasets/paws/resolve/161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final/test-00000-of-00001.parquet",
        "ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d",
    ),
    "qqp.parquet": (
        HF
        + "nyu-mll/glue/resolve/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/qqp/validation-00000-of-00001.parquet",
        "efd86a539c412d74874ee451573d7bd142f56c47fe36de033b9f367d8bb0fa71",
    ),
    "cpython.zip": (
        "https://codeload.github.com/python/cpython/zip/refs/tags/v3.12.10",
        "156640ce1c4a58de57dc84f5cd1f6d50020edbc26f1e42de95fe8a1b765a3105",
    ),
}


def pinned(directory, download):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in INPUTS.items():
        path = directory / name
        if not path.exists() and download:
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"Unexpected download for {name}")
            path.write_bytes(data)
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unrecognised input {name}")


def key(text):
    return sha256(("audit-v1/" + text).encode()).hexdigest()


def partition(group):
    return "holdout" if int(key(group)[0], 16) % 2 else "development"


def fold(text):
    return " ".join(text.casefold().split())


def answers(row):
    return {
        fold(
            unquote(a.removeprefix("https://en.wikipedia.org/wiki/")).replace("_", " ")
        )
        for a in row["answers"]
        if a.strip()
    }


def comqa(directory):
    used_answers, used_questions = set(), set()
    for name in ("comqa_dev.json", "comqa_test.json"):
        for row in json.loads((directory / name).read_bytes()):
            used_answers.update(answers(row))
            used_questions.update(
                map(fold, row.get("questions", [row.get("question", "")]))
            )
    groups = []
    raw = json.loads((directory / "comqa.json").read_bytes())
    for row in sorted(raw, key=lambda r: key(r["cluster_id"])):
        questions = list(
            {fold(q): q.strip() for q in row["questions"] if q.strip()}.values()
        )
        labels = answers(row)
        if (
            len(questions) < 2
            or not labels
            or used_answers & labels
            or any(fold(q) in used_questions for q in questions)
        ):
            continue
        used_answers.update(labels)
        used_questions.update(map(fold, questions))
        groups.append(
            (
                row["cluster_id"],
                partition(row["cluster_id"]),
                questions[0],
                questions[1],
                ", ".join(sorted(labels)),
            )
        )
    return groups


def python_docs(directory):
    # No evaluation of executable source. Accept only isolated literal operations.
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.operator,
        ast.unaryop,
        ast.boolop,
        ast.cmpop,
    )
    rows = []
    with ZipFile(directory / "cpython.zip") as archive:
        for path in sorted(archive.namelist()):
            if "/Doc/" not in path or not path.endswith(".rst"):
                continue
            lines = archive.read(path).decode().splitlines()
            for i, line in enumerate(lines[:-1]):
                match = re.fullmatch(r"\s*>>> (.{1,120})", line)
                if not match or "#" in line:
                    continue
                expr, output = match[1].strip(), lines[i + 1].strip()
                try:
                    tree = ast.parse(expr, mode="eval")
                    if not all(isinstance(node, allowed) for node in ast.walk(tree)):
                        continue
                    literal = ast.literal_eval(output)
                except (ValueError, SyntaxError):
                    continue
                if (
                    not output
                    or len(output) > 100
                    or "#" in output
                    or type(literal)
                    not in (str, bytes, int, bool, float, tuple, list, dict, type(None))
                ):
                    continue
                stem = "In Python 3.12, what is the displayed result of "
                rows.append(
                    (
                        path + "#" + str(i + 1),
                        partition(path),
                        stem + expr + "?",
                        stem + "(" + expr + ")?",
                        output,
                    )
                )
    # Exclude repeated questions, or outputs equivalent in the Python display.
    seen, outputs, selected = set(), set(), []
    for row in sorted(rows, key=lambda r: key(r[0])):
        value = repr(ast.literal_eval(row[-1]))
        if row[2] in seen or value in outputs:
            continue
        seen.add(row[2])
        outputs.add(value)
        selected.append(row)
    return selected


EXPOSED = [
    "data/candidate_b_train_pairs.json",
    "data/phase8_tuning_val.json",
    "data/raw/phase8_sealed_test_stream.json",
    "data/raw/phaseA_sealed_test_stream.json",
]


def pair_data(directory, dataset, split):
    import pyarrow.parquet as pq

    known = set()
    for path in EXPOSED:
        for row in json.loads((ROOT / path).read_bytes()):
            known.update(fold(x) for x in row.values() if isinstance(x, str))
    parent = {}

    def root(node):
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    rows = []
    for r in pq.read_table(directory / (dataset + ".parquet")).to_pylist():
        a, b, index = (
            (r["sentence1"], r["sentence2"], r["id"])
            if dataset == "paws"
            else (r["question1"], r["question2"], r["idx"])
        )
        if not a or not b:
            continue
        x, y = root(fold(a)), root(fold(b))
        parent[max(x, y)] = min(x, y)
        rows.append((str(index), a, b, int(r["label"])))
    contaminated = {root(x) for x in known if x in parent}
    chosen = defaultdict(list)
    seen = set()
    for index, a, b, label in sorted(rows, key=lambda row: key(dataset + row[0])):
        group = root(fold(a))
        if (
            group in contaminated
            or group in seen
            or partition(dataset + group) != split
            or len(chosen[label]) >= 128
        ):
            continue
        seen.add(group)
        chosen[label].append((index, a, b, label))
    return chosen[0] + chosen[1]


if __name__ == "__main__":
    import argparse
    import importlib.metadata as metadata
    import platform

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    pinned(args.directory, args.download)
    source_notes = {
        "paws.parquet": [
            "PAWS-Wiki labeled_final test",
            8000,
            "Sentence equivalence",
            "Google dataset-specific permission for any purpose with acknowledgement appreciated",
            "https://github.com/google-research-datasets/paws/blob/master/LICENSE",
        ],
        "qqp.parquet": [
            "GLUE QQP validation",
            40430,
            "Question equivalence",
            "GLUE defers to the original Quora terms. Publish IDs and derived scores.",
            "https://huggingface.co/datasets/nyu-mll/glue/blob/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/README.md",
        ],
        "comqa.json": [
            "ComQA train",
            3966,
            "Cluster membership and official normalized short answer sets",
            "Publisher offers research download and citation. No explicit open redistribution licence identified.",
            "https://qa.mpi-inf.mpg.de/comqa/",
        ],
        "comqa_dev.json": [
            "ComQA dev",
            966,
            "Prior exposure exclusion",
            "Same terms as train",
            "https://qa.mpi-inf.mpg.de/comqa/",
        ],
        "comqa_test.json": [
            "ComQA test",
            2243,
            "Prior exposure exclusion and spent reproduction",
            "Same terms as train",
            "https://qa.mpi-inf.mpg.de/comqa/",
        ],
        "cpython.zip": [
            "CPython v3.12.10 documentation",
            None,
            "Publisher-displayed Python output and FAQ answers",
            "PSF License. Documentation examples also available under the Zero-Clause BSD licence.",
            "https://docs.python.org/3.12/license.html",
        ],
    }
    inputs = {
        name: dict(
            url=url,
            sha256=digest,
            bytes=(args.directory / name).stat().st_size,
            source=source_notes[name][0],
            source_rows=source_notes[name][1],
            labels=source_notes[name][2],
            terms=source_notes[name][3],
            terms_url=source_notes[name][4],
        )
        for name, (url, digest) in INPUTS.items()
    }
    groups = {}
    for split in ["development", "holdout"]:
        groups[split] = {}
        for name, loader, cap in [("python", python_docs, 40), ("comqa", comqa, 128)]:
            rows = [r for r in loader(args.directory) if r[1] == split][:cap]
            groups[split][name] = [
                {"group": r[0], "shard": i % 2} for i, r in enumerate(rows)
            ]
        for name in ["paws", "qqp"]:
            groups[split][name] = [r[0] for r in pair_data(args.directory, name, split)]
    manifest = dict(
        inputs=inputs,
        baseline="b6b9a946f52aea4a723573ce19196a66edc9f6ac",
        frozen_selection_commit="ba3755b",
        grouping_protocol="docs/INDEPENDENT_SEMANTIC_PROTOCOL.md",
        grouping_implementation="scripts/independent_semantic_data.py",
        groups=groups,
        excluded_prior_sources={
            p: sha256((ROOT / p).read_bytes()).hexdigest() for p in EXPOSED
        },
        python=platform.python_version(),
        packages={
            pkg: metadata.version(pkg)
            for pkg in [
                "pyarrow",
                "numpy",
                "scipy",
                "scikit-learn",
                "torch",
                "transformers",
                "sentence-transformers",
                "fastapi",
            ]
        },
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
