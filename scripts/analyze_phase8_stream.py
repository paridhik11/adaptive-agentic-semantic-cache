import json

stream = json.load(open('data/raw/phase8_sealed_test_stream.json', encoding='utf-8'))
telemetry = json.load(open('data/phase8_simulation_telemetry.json', encoding='utf-8'))
arm_d = telemetry['arm_d_new_pipeline']['telemetry']
stream_map = {e['id']: e for e in stream}

stats = {}
fps = []
for t in arm_d:
    e = stream_map[t['id']]
    qtype = e['query_type']
    s = stats.setdefault(qtype, {'n': 0, 'hits': 0, 'tp': 0, 'fp': 0, 'misses': 0})
    s['n'] += 1
    if t['decision'] == 'REUSE':
        s['hits'] += 1
        if t['outcome'] == 'FP':
            s['fp'] += 1
            fps.append((e, t))
        else:
            s['tp'] += 1
    else:
        s['misses'] += 1

print("CATEGORY BREAKDOWN (Arm d):")
for k, v in stats.items():
    print(f"  {k:22s}: n={v['n']:2d}, hits={v['hits']:2d}, tp={v['tp']:2d}, fp={v['fp']:2d}")

print("\nFALSE POSITIVES IN ARM D (FP=3):")
for e, t in fps:
    print(f"  ID: {e['id']}, Type: {e['query_type']}, Source: {e['source']}")
    print(f"    Incoming Query: \"{e['query']}\"")
    ref = stream_map.get(e.get('reference_seed_id'), {})
    ref_q = ref.get('query', 'N/A')
    print(f"    Reference Seed: \"{ref_q}\"")
    print(f"    Path: {t.get('path')}, Local score: {t.get('local_score')}, Guard: {t.get('guard_reason')}")
