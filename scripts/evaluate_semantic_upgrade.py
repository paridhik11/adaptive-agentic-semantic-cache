"""Replay authored semantic cases against a checkout using the live chat API."""
import argparse
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
import re
from statistics import median
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "artifacts/semantic_upgrade/fixtures.json"
FIXTURE_SHA = "6bd58d482e5e2e62dc0131db812dc5875c10bf0bb5d1dbc88ecda1609d25a769"


@contextmanager
def legacy_mode(cache_class, mode, observed):
    """Apply exact and shadow routing to the old serving method for comparison."""
    normal = cache_class.process
    if mode == "guarded":
        yield
        return

    def replay(cache, body, *, tenant_id, provider, bypass=False):
        if mode == "exact":
            return cache.exact.process(body, tenant_id=tenant_id, provider=provider, bypass=bypass), None, None
        exact_process = cache.exact.process

        def exact_with_shadow(*args, **kwargs):
            resolver = kwargs.get("on_miss")
            if resolver:
                def shadow(request):
                    observed["shadow"] = resolver(request) is not None
                    return None
                kwargs["on_miss"] = shadow
            return exact_process(*args, **kwargs)

        with patch.object(cache.exact, "process", exact_with_shadow):
            result, _, decision = normal(cache, body, tenant_id=tenant_id, provider=provider, bypass=bypass)
        return result, None, decision

    with patch.object(cache_class, "process", replay):
        yield


def interval(count, total):
    if not total:
        return None
    ci = binomtest(count, total).proportion_ci()
    return [round(ci.low, 6), round(ci.high, 6)]


def latency(values):
    values = sorted(values)
    return {"p50_ms": round(median(values), 3),
            "p95_ms": round(values[(95 * len(values) + 99) // 100 - 1], 3)}


def evaluate(rows, revision):
    from src.api.chat_semantic import ChatSemanticReuse
    from src.api.live_chat import DEFAULT_MODEL, create_live_chat_app

    supports_modes = "semantic_mode" in signature(create_live_chat_app).parameters
    results = {}
    for arm in ("no_cache", "exact", "guarded", "shadow"):
        counts = Counter(exact_hits=0, correct_semantic=0, unsuitable_semantic=0, probe_fallbacks=0,
                         provider_calls=0, synthetic_completion_tokens=0, shadow_correct=0, shadow_unsuitable=0)
        reasons = Counter()
        probe_timings, comparable_timings, verification = [], [], []
        max_store_bytes = 0
        case_results = []
        for case_id, group, seed, answer, probe, suitable in rows:
            options = {"semantic_mode": arm if arm != "no_cache" else "exact"} if supports_modes else {}
            observed = {}
            calls = []

            def transport(body):
                question = body["messages"][-1]["content"]
                content = answer if question == seed or suitable else "Fresh answer to the changed request."
                tokens = len(content.split())
                calls.append(tokens)
                return {"model": DEFAULT_MODEL, "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": content}}], "usage": {"completion_tokens": tokens}}

            legacy = arm if not supports_modes else "guarded"
            with TemporaryDirectory() as tmp, legacy_mode(ChatSemanticReuse, legacy, observed):
                app = create_live_chat_app(tmp, journal_path=Path(tmp) / "quota.sqlite",
                                           api_key="offline-fixture", transport=transport, **options)
                with TestClient(app) as client:
                    client.headers["x-demo-session"] = re.search(
                        r'name="demo-session" content="([^"]+)"', client.get("/").text)[1]

                    def ask(question):
                        response = client.post("/api/chat", json={"question": question, "cache": arm != "no_cache"})
                        response.raise_for_status()
                        return response.json()

                    ask(seed)
                    calls.clear()
                    repeat = ask(seed)
                    result = ask(probe)
                max_store_bytes = max(max_store_bytes, (Path(tmp) / "chat.sqlite").stat().st_size)
            counts["provider_calls"] += len(calls)
            counts["synthetic_completion_tokens"] += sum(calls)
            counts["exact_hits"] += repeat["cache_tier"] == "EXACT"
            hit = result["cache_tier"] == "SEMANTIC"
            if hit and suitable:
                assert result["answer"] == answer
            counts["correct_semantic"] += hit and suitable
            counts["unsuitable_semantic"] += hit and not suitable
            counts["probe_fallbacks"] += not hit
            diag = result.get("semantic_verification", {})
            shadow = diag.get("shadow_recommendation", observed.get("shadow", False))
            counts["shadow_correct"] += bool(shadow and suitable)
            counts["shadow_unsuitable"] += bool(shadow and not suitable)
            reasons.update(diag.get("rejections", {}))
            probe_timings.append(result["latency_ms"])
            comparable_timings.extend((repeat["latency_ms"], result["latency_ms"]))
            if "verification_ms" in diag:
                verification.append(diag["verification_ms"])
            case_results.append({"case": case_id, "group": group, "probe_suitable": suitable,
                                 "probe_tier": result["cache_tier"], "reason": result["cache_reason"],
                                 "rejections": diag.get("rejections", {})})
        n = len(rows)
        hits = counts["correct_semantic"] + counts["unsuitable_semantic"]
        supported = sum(row[-1] for row in rows)
        results[arm] = {**counts, "requests_excluding_priming": n * 2, "probes": n,
                        "suitable_probes": supported,
                        "suitable_capture_interval": interval(counts["correct_semantic"], supported),
                        "precision_interval": interval(counts["correct_semantic"], hits),
                        "unsuitable_rate_interval": interval(counts["unsuitable_semantic"], n - supported),
                        "end_to_end": latency(comparable_timings), "probe_end_to_end": latency(probe_timings),
                        "total_probe_verification_ms": round(sum(verification), 3) if verification else None,
                        "maximum_cache_file_bytes": max_store_bytes, "rejections": dict(reasons),
                        "cases": case_results}
    for arm in results.values():
        arm["net_calls_avoided"] = results["no_cache"]["provider_calls"] - arm["provider_calls"]
        arm["synthetic_tokens_avoided"] = (
            results["no_cache"]["synthetic_completion_tokens"] - arm["synthetic_completion_tokens"])
    return {"revision": revision, "fixture_sha256": FIXTURE_SHA, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", choices=["development", "reserved"], required=True)
    arguments = parser.parse_args()
    raw = FIXTURES.read_bytes()
    if sha256(raw).hexdigest() != FIXTURE_SHA:
        raise ValueError("Semantic upgrade fixture does not match the pinned digest")
    sys.path.insert(0, str(arguments.repo.resolve()))
    result = evaluate(json.loads(raw)[arguments.split], arguments.revision)
    print(json.dumps({"split": arguments.split, **result}, indent=2))
