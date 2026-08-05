# Retrieval eval

Measures search quality against a small gold set plus a self-retrieval probe.

```sh
NOOR_URL=https://noor.pyxis3.ai python3 eval/run.py   # or http://localhost:8000
```

- **curated** - known query → expected verse (Recall@1/5/10, MRR)
- **self-retrieval** - a verse's own translation should retrieve it (sanity probe; healthy ≈ 90%+ Recall@10)

Run before and after any embedding or indexing change to prove it helped, not just changed.

## Baseline (MiniLM, arabic+english concatenated into one vector)

| set                   | Recall@1 | Recall@10 | MRR  |
| --------------------- | -------- | --------- | ---- |
| curated (n=20)        | 45%      | 50%       | 0.46 |
| self-retrieval (n=40) | 38%      | 40%       | 0.39 |

Self-retrieval at 40% is the tell: English queries are matched against half-Arabic document vectors. Fix = embed Arabic and English separately + a stronger encoder, then re-run.

## After (e5-small, English-passage vectors, `query:`/`passage:` prefixes)

| set                   | Recall@1 | Recall@10 | MRR  |
| --------------------- | -------- | --------- | ---- |
| curated (n=20)        | 70%      | 90%       | 0.76 |
| self-retrieval (n=40) | 100%     | 100%      | 1.00 |

Self-retrieval is now perfect and curated Recall@10 went 50% → 90% - the Arabic+English concatenation was the dominant defect.
