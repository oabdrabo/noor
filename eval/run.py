import json
import os
import random
import urllib.request

BASE = os.environ.get("NOOR_URL", "https://noor.pyxis3.ai").rstrip("/")
HERE = os.path.dirname(__file__)


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body, timeout=30):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def search(q, limit=10):
    d = post("/api/search", {"query": q, "limit": limit, "similarity_threshold": 0.0})
    return [f"{x['surah']}:{x['ayah']}" for x in d.get("results", [])]


def score(cases):
    r1 = r5 = r10 = mrr = 0
    for q, expect in cases:
        got = search(q)
        exp = set(expect)
        ranks = [i for i, g in enumerate(got) if g in exp]
        r1 += bool(got) and got[0] in exp
        r5 += any(g in exp for g in got[:5])
        r10 += any(g in exp for g in got[:10])
        mrr += 1 / (ranks[0] + 1) if ranks else 0
        print(f"  {'OK ' if ranks else 'MISS'} {q[:44]:44} -> {got[:3]}  exp {expect}")
    n = len(cases)
    print(
        f"  Recall@1={r1 / n:.0%}  @5={r5 / n:.0%}  @10={r10 / n:.0%}  MRR={mrr / n:.3f}  (n={n})"
    )


print(f"=== Noor retrieval eval @ {BASE} ===")
print("\n[curated: query -> expected verse]")
gold = json.load(open(os.path.join(HERE, "gold.json")))
score([(g["q"], g["expect"]) for g in gold])

print("\n[self-retrieval: a verse's translation should retrieve itself]")
pool = []
for s in (2, 18, 36, 55, 67, 76):
    for v in get(f"/api/search/surah/{s}")["verses"]:
        if v.get("translation") and len(v["translation"]) > 50:
            pool.append((f"{s}:{v['ayah']}", v["translation"]))
random.seed(0)
random.shuffle(pool)
sample = pool[:40]
r1 = r10 = mrr = 0
for ref, tr in sample:
    got = search(tr[:300])
    ranks = [i for i, g in enumerate(got) if g == ref]
    r1 += got[:1] == [ref]
    r10 += ref in got
    mrr += 1 / (ranks[0] + 1) if ranks else 0
n = len(sample)
print(f"  Recall@1={r1 / n:.0%}  @10={r10 / n:.0%}  MRR={mrr / n:.3f}  (n={n})")
