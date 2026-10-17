"""Prepare a version-pinned, group-split technical Q&A reference corpus."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.benchmark.schema import digest


SPLIT_BY_SECTION = {
    "General Questions": "train",
    "Core Language": "train",
    "Performance": "train",
    "Numbers and strings": "dev",
    "Sequences (Tuples/Lists)": "dev",
    "Objects": "test",
    "Modules": "test",
}


def _underline(line: str) -> str | None:
    stripped = line.strip()
    if len(stripped) >= 3 and len(stripped) >= 1 and len(set(stripped)) == 1 and stripped[0] in "=-~^`:+*#\"'":
        return stripped[0]
    return None


def _plain_rst(text: str) -> str:
    text = re.sub(r"\.\. _[^:]+:\s*\n", "", text)
    text = re.sub(r":(?:func|mod|class|meth|attr|pep|ref|term|data|const|exc):`([^`<>]+)(?:\s*<[^>]+>)?`", r"\1", text)
    text = re.sub(r"`([^`<>]+)\s*<https?://[^>]+>`_", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_programming_faq(path: str | Path) -> dict:
    source = Path(path).read_bytes()
    text = source.decode("utf-8")
    lines = text.splitlines()
    headings: list[tuple[int, str, str]] = []
    section = None
    for index in range(len(lines) - 1):
        marker = _underline(lines[index + 1])
        title = lines[index].strip()
        if not title or marker is None or len(lines[index + 1].strip()) < len(title):
            continue
        if marker == "=":
            section = title
        elif marker == "-" and section is not None:
            headings.append((index, title, section))
    rows = []
    for position, (start, question, section) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        answer = _plain_rst("\n".join(lines[start + 2:end]))
        if not answer:
            raise ValueError(f"empty source answer at line {start + 1}")
        if section not in SPLIT_BY_SECTION:
            raise ValueError(f"unmapped FAQ section: {section}")
        slug = re.sub(r"[^a-z0-9]+", "-", question.casefold()).strip("-")[:72]
        rows.append({
            "faq_id": f"python312-{start + 1}-{slug}",
            "section": section,
            "group_id": f"python312:{section}",
            "split": SPLIT_BY_SECTION[section],
            "source_line": start + 1,
            "question": question,
            "reference_answer": answer,
            "answer_reuse_gold": None,
            "pretraining_contamination": "unknown",
        })
    if len({row["faq_id"] for row in rows}) != len(rows):
        raise ValueError("FAQ IDs are not unique")
    return {
        "schema_version": "versioned-faq-corpus-v1",
        "source_revision": "0fb18b02c8ad56299d6a2910be0bab8ad601ef24",
        "source_path": "Doc/faq/programming.rst",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_bytes": len(source),
        "source_license": "PSF-2.0",
        "nature": "reference_documentation_not_natural_traffic",
        "items": rows,
    }


def validate_group_splits(corpus: dict) -> dict:
    seen: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for row in corpus.get("items", []):
        group, split = row.get("group_id"), row.get("split")
        if not isinstance(group, str) or split not in ("train", "dev", "test"):
            raise ValueError("invalid group or split")
        if group in seen and seen[group] != split:
            raise ValueError("topic group crosses splits")
        seen[group] = split
        key = " ".join(row["question"].casefold().split())
        if key in normalized and normalized[key] != split:
            raise ValueError("normalized question crosses splits")
        normalized[key] = split
    return {"group_splits": seen, "question_count": len(corpus.get("items", [])),
            "corpus_sha256": digest(corpus)}


def build_constructed_workload(corpus: dict, *, model: str = "fixture-model") -> dict:
    """Create seed+exact-repeat events; never label these as natural repeats."""
    requests = []
    for row in corpus["items"]:
        group = row["faq_id"]
        request = {"model": model, "temperature": 0, "max_tokens": 512,
                   "messages": [{"role": "user", "content": row["question"]}]}
        for kind in ("seed", "constructed-exact-repeat"):
            requests.append({
                "event_id": f"{group}-{kind}",
                "tenant_id": "r2-python-faq",
                "source": "cpython-programming-faq-3.12.0",
                "group_id": group,
                "split": row["split"],
                "request": request,
                "constructed_repeat": kind == "constructed-exact-repeat",
                "reference_faq_id": row["faq_id"],
                "answer_reuse_gold": None,
                "provenance": "constructed_exact_repeat" if kind != "seed" else "source_question",
            })
    return {
        "schema_version": "workload-v1",
        "name": "cpython-programming-faq-v3.12.0-constructed-exact-repeats",
        "nature": "synthetic_fixture",
        "provenance": "Pinned official FAQ questions; each question is followed by a constructed exact repeat. This measures exact-cache mechanics only, not natural prevalence or answer correctness.",
        "license": "PSF-2.0",
        "requests": requests,
    }
