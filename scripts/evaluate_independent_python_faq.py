"""Positive-only Python FAQ answer transfer, supplemental to the frozen study."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from evaluate_independent_semantics import (
    ARMS,
    batch_scores,
    dump,
    embed,
    fixture_app,
    load_chat_models,
    reject,
    score_record,
)
from independent_semantic_data import key, partition, pinned
from src.api.chat_keywords import keyword_signature
from src.api.chat_reuse_checks import semantic_answer_reason
from src.policy.decomposed_applicability import (
    answer_satisfies_requirements,
    requirement_guard,
)


def faq_pairs(directory):
    records = []
    with ZipFile(directory / "cpython.zip") as z:
        for path in sorted(z.namelist()):
            if "/Doc/faq/" not in path or not path.endswith(".rst"):
                continue
            text = z.read(path).decode()
            sections = list(re.finditer(r"(?m)^([^\n]+)\n(?:-{3,}|={3,})[ \t]*$", text))
            for i, section in enumerate(sections):
                q = section[1].strip()
                match = re.fullmatch(r"How (?:can|do) I (.+\?)", q)
                if not match:
                    continue
                end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
                answer = re.sub(
                    r"(?m)^\.\. _[^\n]+:\s*\Z", "", text[section.end() : end]
                ).strip()
                if (
                    not answer
                    or len(answer) > 850
                    or answer.startswith("..")
                    or ".. include::" in answer
                ):
                    continue
                probe = (
                    "How do I " if q.startswith("How can I ") else "How can I "
                ) + match[1]
                records.append(
                    (
                        path + "#" + str(text[: section.start()].count("\n") + 1),
                        partition(path),
                        q,
                        probe,
                        answer,
                    )
                )
    return sorted(records, key=lambda r: key(r[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    pinned(args.directory, False)
    models = load_chat_models()
    rows = faq_pairs(args.directory)
    combo = json.loads((args.output / "selection.json").read_bytes())["combined"]
    encoder, pair, passage = models
    texts = [r[2] for r in rows] + [r[3] for r in rows]
    seeds, queries = np.split(embed(encoder, texts), 2)
    pairs = [(r[3], r[2]) for r in rows]
    _, p, _ = batch_scores(pair, pairs + [(s, q) for q, s in pairs])
    f, b = np.split(p, 2)
    logits, probs, _ = batch_scores(passage, [(r[3], r[4]) for r in rows])
    refs, _, _ = batch_scores(passage, [(r[2], r[4]) for r in rows])
    records = []
    for i, (idx, split, s, q, answer) in enumerate(rows):
        c = dict(
            cosine=float(queries[i] @ seeds[i]),
            guard=requirement_guard(s, q) or answer_satisfies_requirements(q, answer),
            extra_guard=semantic_answer_reason(s, q, answer, keyword_signature(q)),
            pair_forward=float(f[i]),
            pair_reverse=float(b[i]),
            passage_logit=float(logits[i]),
            source_passage_logit=float(refs[i]),
            passage=float(probs[i]),
            answer_words=len(answer.split()),
        )
        outcomes = {}
        for arm in [*ARMS, "combined"]:
            config = combo if arm == "combined" else [arm]
            outcomes[arm] = (
                reject(c, "relative5" in config, "typed5" in config) or "accepted"
            )
        for arm, mode in [("live_optional", "guarded"), ("live_exact", "exact")]:
            with (
                TemporaryDirectory() as tmp,
                fixture_app(
                    Path(tmp),
                    Path(tmp) / "fixture.db",
                    models,
                    mode,
                    {q: answer, s: answer},
                ) as ask,
            ):
                ask(s)
                r = ask(q)
            outcomes[arm] = "accepted" if r["cache_tier"] else r["cache_reason"]
            if r["cache_tier"]:
                assert r["answer"] == answer
        records.append(dict(id=idx, split=split, decisions=outcomes, **c))
    counts = {
        arm: score_record(
            sum(r["decisions"][arm] == "accepted" for r in records), len(records)
        )
        for arm in outcomes
    }
    dump(
        args.output / "python-faq.json", dict(positive_capture=counts, records=records)
    )
