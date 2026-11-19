"""Replay publicly annotated ComQA question pairs through the live-chat API offline."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from urllib.parse import unquote

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.live_chat import create_live_chat_app, DEFAULT_MODEL

DEV_SHA = "937f898f996e9a7317d311553cf5b43c4a53e0bb841ae8eed4d8cf15936fbc84"


def clusters(raw):
    if sha256(raw).hexdigest() != DEV_SHA:
        raise ValueError("Expected the published ComQA development JSON. See docs/LIVE_CHAT.md.")
    groups = []
    for row in json.loads(raw):
        questions = list(dict.fromkeys(q.strip() for q in row["questions"] if q.strip()))
        answers = frozenset(a.strip() for a in row["answers"] if a.strip())
        if len(questions) >= 2 and answers:
            groups.append((questions[0], questions[1], answers))
    return groups


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


def overlap(left, right):
    return len(left & right) / len(left | right)


def answer_text(answers):
    return ", ".join(unquote(a.removeprefix("https://en.wikipedia.org/wiki/")).replace("_", " ")
                     for a in sorted(answers))


def replay(raw, models=None):
    groups = clusters(raw)
    vocabulary = [words(seed) for seed, _, _ in groups]
    counts = dict(groups=len(groups), positive_semantic=0, positive_exact=0,
                  positive_clarifications=0, positive_wrong_answer=0, negative_hits=0,
                  provider_calls=0, requests=0)
    for i, (seed, paraphrase, source_answers) in enumerate(groups):
        # Use the most lexically similar seed with disjoint annotated answers.
        # File order resolves ties. Controls may be reused between groups.
        j = max((j for j in range(len(groups)) if source_answers.isdisjoint(groups[j][2])),
                key=lambda j: overlap(vocabulary[i], vocabulary[j]))
        control = groups[j][0]
        answers = {seed: answer_text(source_answers), paraphrase: answer_text(source_answers),
                   control: answer_text(groups[j][2])}

        def transport(body):
            return {"model": DEFAULT_MODEL, "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": answers[body["messages"][-1]["content"]]}}]}

        with TemporaryDirectory() as tmp:
            app = create_live_chat_app(tmp, journal_path=Path(tmp) / "quota.sqlite",
                                       api_key="offline-fixture", transport=transport, semantic_models=models)
            with TestClient(app) as client:
                html = client.get("/").text
                client.headers["x-demo-session"] = re.search(r'name="demo-session" content="([^"]+)"', html)[1]
                for index, question in enumerate((seed, paraphrase, control)):
                    response = client.post("/api/chat", json={"question": question})
                    response.raise_for_status()
                    result = response.json()
                    counts["requests"] += 1
                    counts["provider_calls"] += result["provider_calls"]
                    if index == 1:
                        counts["positive_semantic"] += result["cache_tier"] == "SEMANTIC"
                        counts["positive_exact"] += result["cache_tier"] == "EXACT"
                        counts["positive_clarifications"] += result["cache_status"] == "CLARIFY"
                        counts["positive_wrong_answer"] += (
                            result["cache_tier"] is not None and result["answer"] != answers[paraphrase])
                    if index == 2:
                        counts["negative_hits"] += result["cache_tier"] is not None
    return {"dataset": "ComQA development", "sha256": DEV_SHA, "local_models": bool(models), **counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--with-models", action="store_true")
    args = parser.parse_args()
    if args.with_models:
        from src.api.chat_semantic import load_chat_models
        models = load_chat_models()
    else:
        models = None
    print(json.dumps(replay(args.dataset.read_bytes(), models), indent=2))
