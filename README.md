# Noor

AI-powered semantic search over the Quran and Hadith corpora. Multilingual (Arabic + English) embeddings with sub-second response on a single-node deployment.

نُور — *"light"* in Arabic.

## What it does

- **Semantic search** in either Arabic or English; returns top matches across the Quran + major Hadith collections
- **Arabic-aware normalisation** (diacritics, tashkeel, alef variants, hamza on/below) before embedding
- **Vector store** backed by `sqlite-vec` — embeddings live in a single SQLite file, no separate DB to operate
- **TF-IDF + KMeans clustering** for topic discovery and concept maps
- **Word-cloud generation** per topic / query
- FastAPI backend, single-page Vue 3 frontend, nginx in front

## Stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | Strong multilingual baseline; small enough to run on CPU |
| Vector store | `sqlite-vec` | Zero-ops, file-backed, deployable on any host |
| API | FastAPI + Pydantic v2 | Async + automatic schema |
| Frontend | Vue 3 via CDN (no build step) | One HTML file, fastest possible iteration |
| Edge | nginx | Static + reverse proxy to `/api/` |

## Run locally

```sh
pip install -r requirements.txt
python main.py
# API on http://localhost:8000
# Frontend served via nginx at http://localhost:3000 (see nginx.conf)
```

## Architecture notes

The encoder is loaded lazily on first request so cold-start cost is paid once. Heavy ML libraries (`sklearn`, `wordcloud`) are imported lazily through small accessor functions to keep import time low for the API process.

The corpus → embedding → store pipeline is idempotent: re-running ingest with the same source files is a no-op via row hashing.

## Why

Modern Quran/Hadith study tools are mostly keyword search. I wanted a tool that answers *intent* questions — *"verses about patience in adversity"*, *"hadith on neighbours' rights"* — and surfaces matches by meaning, not surface form. Multilingual embeddings make this practical; `sqlite-vec` makes it deployable on a Raspberry Pi.

## Author

[Omar A.](https://github.com/oabdrabo) · AI infrastructure engineer · [LinkedIn](https://linkedin.com/in/oabdrabo)

---

Built by [Omar A.](https://github.com/oabdrabo). Other work: **[PYXIS3](https://github.com/pyxis3-ai)** (vendor-neutral AI ops platform), [`vllm-bench`](https://github.com/pyxis3-ai/vllm-bench), [`lens`](https://github.com/oabdrabo/lens).
