#!/usr/bin/env python3
"""Noor - Islamic Knowledge Search API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import io
import base64
import sqlite3
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import Counter


# Heavy libs lazy-loaded on first use
def _kmeans():
    from sklearn.cluster import KMeans

    return KMeans


def _tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer


def _wordcloud():
    from wordcloud import WordCloud

    return WordCloud


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

# --------------- helpers ---------------


def serialize_f32(vector):
    return vector.astype(np.float32).tobytes()


def normalize_arabic(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    for old, new in {
        "\u0671": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0670": "",
        "\u06da": "",
        "\u06d6": "",
        "\u06d7": "",
        "\u06d9": "",
        "\u06db": "",
        "\u06dc": "",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
    }.items():
        text = text.replace(old, new)
    return text


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return names[0]


# --------------- sqlite-vec search ---------------


def search_quran_vec(db, query_embedding, limit):
    if db is None:
        return []
    vec_results = db.execute(
        "SELECT rowid, distance FROM quran_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        [serialize_f32(query_embedding), limit],
    ).fetchall()
    if not vec_results:
        return []
    rowids = [r[0] for r in vec_results]
    distances = {r[0]: r[1] for r in vec_results}
    ph = ",".join("?" * len(rowids))
    meta = {
        m[0]: m
        for m in db.execute(
            f"SELECT verse_id, surah, ayah, text, translation FROM quran_metadata WHERE verse_id IN ({ph})",
            rowids,
        ).fetchall()
    }
    return [
        {
            "surah": meta[rid][1],
            "ayah": meta[rid][2],
            "text": meta[rid][3],
            "translation": meta[rid][4],
            "score": max(0.0, 1.0 - distances[rid]),
        }
        for rid in rowids
        if rid in meta
    ]


def search_hadith_vec(db, query_embedding, limit):
    if db is None:
        return []
    vec_results = db.execute(
        "SELECT rowid, distance FROM hadith_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        [serialize_f32(query_embedding), limit],
    ).fetchall()
    if not vec_results:
        return []
    rowids = [r[0] for r in vec_results]
    distances = {r[0]: r[1] for r in vec_results}
    ph = ",".join("?" * len(rowids))
    meta = {
        m[0]: m
        for m in db.execute(
            f"SELECT hadith_id, collection_name, hadith_number, text, reference FROM hadith_metadata WHERE hadith_id IN ({ph})",
            rowids,
        ).fetchall()
    }
    return [
        {
            "collection": meta[rid][1],
            "hadith_number": meta[rid][2],
            "text": meta[rid][3],
            "reference": meta[rid][4],
            "score": max(0.0, 1.0 - distances[rid]),
        }
        for rid in rowids
        if rid in meta
    ]


# --------------- embedding ---------------

_embedding_cache = {}


def get_embedding(model, text: str) -> np.ndarray:
    """Cached single-text embedding (max 2000 entries)."""
    if not text or not str(text).strip():
        return np.zeros(EMBED_DIM, dtype=np.float32)
    text_str = str(text).strip()[:2000]
    if text_str in _embedding_cache:
        return _embedding_cache[text_str]
    try:
        result = np.array(model.encode(text_str), dtype=np.float32)
        if len(_embedding_cache) >= 2000:
            _embedding_cache.pop(next(iter(_embedding_cache)))
        _embedding_cache[text_str] = result
        return result
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return np.zeros(EMBED_DIM, dtype=np.float32)


# --------------- models ---------------


class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    surah_filter: Optional[int] = None
    similarity_threshold: float = 0.3


class CountQuery(BaseModel):
    word: str
    case_sensitive: bool = False


class QAQuery(BaseModel):
    question: str
    context_limit: int = 5


# --------------- lifespan ---------------


def _init_db():
    import sqlite_vec

    db = sqlite3.connect("/config/vectors.db", check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    for p in [
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA cache_size=10000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA mmap_size=268435456",
    ]:
        db.execute(p)
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("""CREATE TABLE IF NOT EXISTS quran_metadata (
        verse_id INTEGER PRIMARY KEY, surah INTEGER, ayah INTEGER,
        text TEXT, translation TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS hadith_metadata (
        hadith_id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT, hadith_number TEXT, text TEXT, reference TEXT)""")
    db.commit()
    return db


def _ensure_model_version(db, model_name):
    stored = db.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    if stored is None or stored[0] != model_name:
        print(
            f"[{datetime.now()}] Model changed ({stored[0] if stored else 'none'} -> {model_name}), rebuilding...",
            flush=True,
        )
        for t in ["quran_vec", "hadith_vec"]:
            db.execute(f"DROP TABLE IF EXISTS {t}")
        db.execute("DELETE FROM quran_metadata")
        db.execute("DELETE FROM hadith_metadata")
        db.commit()
    db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS quran_vec USING vec0(embedding float[{EMBED_DIM}] distance_metric=cosine)"
    )
    db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS hadith_vec USING vec0(embedding float[{EMBED_DIM}] distance_metric=cosine)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_quran_surah ON quran_metadata(surah, ayah)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_hadith_collection ON hadith_metadata(collection_name)"
    )
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('model', ?)", [model_name]
    )
    db.commit()


_index_stop = None


def _background_index(app):
    """Build vector indexes incrementally in background thread."""
    import threading

    global _index_stop
    _index_stop = threading.Event()
    s = app.state
    BATCH = 64
    stop = _index_stop

    def _run():
        try:
            # --- Quran vectors ---
            total = len(s.df_verses)
            existing = s.db.execute("SELECT COUNT(*) FROM quran_vec").fetchone()[0]
            if existing >= total:
                print(
                    f"[{datetime.now()}] Quran vectors cached ({existing}), skipping",
                    flush=True,
                )
                s.quran_ready = True
            else:
                ar_col = _col(s.df_verses, "ayah_ar", "text")
                en_col = _col(s.df_verses, "ayah_en", "translation")
                sc_col = _col(s.df_verses, "surah_no", "surah")
                ac_col = _col(s.df_verses, "ayah_no_surah", "ayah")
                texts = (
                    s.df_verses[ar_col].fillna("").astype(str)
                    + " "
                    + s.df_verses[en_col].fillna("").astype(str)
                ).tolist()

                action = "Resuming" if existing > 0 else "Building"
                print(
                    f"[{datetime.now()}] {action} quran index ({existing}/{total})",
                    flush=True,
                )

                for i in range(existing, total, BATCH):
                    if stop.is_set():
                        return
                    batch_emb = s.model.encode(texts[i : i + BATCH], batch_size=32)
                    meta_rows, vec_rows = [], []
                    for j, idx in enumerate(range(i, min(i + BATCH, total))):
                        row = s.df_verses.iloc[idx]
                        meta_rows.append(
                            (
                                idx,
                                int(row[sc_col]),
                                int(row[ac_col]),
                                str(row.get(ar_col, ""))[:5000],
                                str(row.get(en_col, ""))[:5000],
                            )
                        )
                        vec_rows.append(
                            (
                                idx,
                                serialize_f32(np.array(batch_emb[j], dtype=np.float32)),
                            )
                        )
                    s.db.executemany(
                        "INSERT OR REPLACE INTO quran_metadata VALUES (?,?,?,?,?)",
                        meta_rows,
                    )
                    s.db.executemany(
                        "INSERT INTO quran_vec (rowid, embedding) VALUES (?,?)",
                        vec_rows,
                    )
                    s.db.commit()
                    done = min(i + BATCH, total)
                    s.index_progress = f"Quran: {done}/{total}"
                    if done % 256 == 0 or done == total:
                        print(
                            f"[{datetime.now()}] Indexed {done}/{total} quran verses",
                            flush=True,
                        )

                s.quran_ready = True
                print(
                    f"[{datetime.now()}] Quran index complete ({total} verses)",
                    flush=True,
                )

            if stop.is_set():
                return

            # --- Hadith vectors ---
            hadith_count = s.db.execute("SELECT COUNT(*) FROM hadith_vec").fetchone()[0]
            if hadith_count > 0:
                print(
                    f"[{datetime.now()}] Hadith vectors cached ({hadith_count}), skipping",
                    flush=True,
                )
                s.hadith_ready = True
            else:
                print(
                    f"[{datetime.now()}] Downloading hadith collections...", flush=True
                )
                import urllib.request

                hadith_records = []
                for url, name in [
                    (
                        "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.json",
                        "Sahih Bukhari",
                    ),
                    (
                        "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.json",
                        "Sahih Muslim",
                    ),
                ]:
                    if stop.is_set():
                        return
                    with urllib.request.urlopen(url, timeout=30) as resp:
                        data = json.loads(resp.read())
                    print(
                        f"[{datetime.now()}] {name}: {len(data['hadiths'])} hadiths",
                        flush=True,
                    )
                    for h in data["hadiths"]:
                        hadith_records.append(
                            {
                                "collection": name,
                                "hadith_number": str(h.get("hadithNumber", "")),
                                "text": h.get("text", "")[:9900],
                                "reference": f"{name.split()[-1]} {h.get('hadithNumber', '')}",
                            }
                        )

                df_hadith = pd.DataFrame(hadith_records)
                total_h = len(df_hadith)
                print(f"[{datetime.now()}] Indexing {total_h} hadiths...", flush=True)
                hadith_texts = [
                    str(t).strip()[:2000] if t and str(t).strip() else "[No text]"
                    for t in df_hadith["text"]
                ]

                for i in range(0, total_h, BATCH):
                    if stop.is_set():
                        return
                    batch = hadith_texts[i : i + BATCH]
                    batch_emb = s.model.encode(batch, batch_size=32)
                    batch_df = df_hadith.iloc[i : i + len(batch)]
                    meta_rows = [
                        (
                            r["collection"],
                            str(r["hadith_number"]),
                            str(r["text"]),
                            r["reference"],
                        )
                        for _, r in batch_df.iterrows()
                    ]
                    s.db.executemany(
                        "INSERT INTO hadith_metadata (collection_name, hadith_number, text, reference) VALUES (?,?,?,?)",
                        meta_rows,
                    )
                    last_id = s.db.execute("SELECT last_insert_rowid()").fetchone()[0]
                    start_id = last_id - len(batch) + 1
                    vec_rows = [
                        (
                            start_id + j,
                            serialize_f32(np.array(batch_emb[j], dtype=np.float32)),
                        )
                        for j in range(len(batch))
                    ]
                    s.db.executemany(
                        "INSERT INTO hadith_vec (rowid, embedding) VALUES (?,?)",
                        vec_rows,
                    )
                    s.db.commit()
                    done = min(i + BATCH, total_h)
                    s.index_progress = f"Hadith: {done}/{total_h}"
                    if done % 256 == 0 or done == total_h:
                        print(
                            f"[{datetime.now()}] Indexed {done}/{total_h} hadiths",
                            flush=True,
                        )

                s.hadith_ready = True
                print(
                    f"[{datetime.now()}] Hadith index complete ({total_h} hadiths)",
                    flush=True,
                )

            s.index_progress = "Ready"
            print(f"[{datetime.now()}] INDEXING COMPLETE", flush=True)
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            pass  # DB closed during shutdown
        except Exception as e:
            if not stop.is_set():
                logger.error(f"Background indexing error: {e}")
                import traceback

                traceback.print_exc()
                s.index_progress = f"Error: {e}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{datetime.now()}] Starting Noor API...", flush=True)

    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/config/models")
    os.environ.setdefault("HF_HOME", "/config/models/huggingface")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    print(f"[{datetime.now()}] Model loaded ({EMBED_DIM}d)", flush=True)

    db = _init_db()
    _ensure_model_version(db, MODEL_NAME)
    df_verses = pd.read_csv("/config/quran-dataset.csv")
    print(f"[{datetime.now()}] {len(df_verses)} verses loaded", flush=True)

    app.state.model = model
    app.state.db = db
    app.state.df_verses = df_verses
    app.state.quran_ready = False
    app.state.hadith_ready = False
    app.state.index_progress = "Starting..."
    app.state.cache = {}

    index_thread = _background_index(app)
    print(f"[{datetime.now()}] API ready, indexing in background", flush=True)

    yield

    if _index_stop:
        _index_stop.set()
    if index_thread:
        index_thread.join(timeout=5)
    db.close()


# --------------- app ---------------

_domain = os.environ.get("BASE_DOMAIN", "jsr.bz")

app = FastAPI(title="Noor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"https://{_domain}", f"https://noor.{_domain}"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# --------------- routes ---------------


@app.get("/")
async def root(request: Request):
    s = request.app.state
    return {
        "message": "Noor API",
        "status": "ready" if s.quran_ready else "indexing",
        "verses": len(s.df_verses) if s.df_verses is not None else 0,
    }


@app.get("/api/health")
@app.get("/api/status")
async def health_status(request: Request):
    s = request.app.state
    return {
        "status": "ready" if s.quran_ready else "indexing",
        "quran_ready": s.quran_ready,
        "hadith_ready": s.hadith_ready,
        "index_progress": s.index_progress,
        "total_verses": len(s.df_verses) if s.df_verses is not None else 0,
        "sqlite_connected": s.db is not None,
    }


# ---- Search ----


def _indexing_resp(s, **extra):
    return {**extra, "indexing": True, "progress": s.index_progress}


@app.post("/api/search")
async def search_verses(query: SearchQuery, request: Request):
    s = request.app.state
    if not s.quran_ready:
        return _indexing_resp(s, results=[], total=0)

    try:
        if not query.query or not query.query.strip():
            return {"results": [], "total": 0}

        # Verse reference lookup (e.g. "2:255")
        verse_match = re.match(r"^(\d+)[:\s]+(\d+)$", query.query.strip())
        if verse_match:
            surah_num, ayah_num = int(verse_match.group(1)), int(verse_match.group(2))
            sc, ac = (
                _col(s.df_verses, "surah_no", "surah"),
                _col(s.df_verses, "ayah_no_surah", "ayah"),
            )
            row = s.df_verses[
                (s.df_verses[sc] == surah_num) & (s.df_verses[ac] == ayah_num)
            ]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "results": [
                        {
                            "surah": int(r[sc]),
                            "ayah": int(r[ac]),
                            "text": r.get("ayah_ar", ""),
                            "translation": r.get("ayah_en", ""),
                            "score": 1.0,
                        }
                    ],
                    "total": 1,
                }

        # Arabic keyword fallback
        has_arabic = any("\u0600" <= c <= "\u06ff" for c in query.query)
        keyword_results = []
        if has_arabic:
            nq = normalize_arabic(query.query)
            for _, row in s.df_verses.iterrows():
                ar = row.get("ayah_ar", "")
                if pd.notna(ar) and nq in normalize_arabic(ar):
                    keyword_results.append(
                        {
                            "surah": int(row.get("surah_no", 1)),
                            "ayah": int(row.get("ayah_no_surah", 1)),
                            "text": ar,
                            "translation": row.get("ayah_en", ""),
                            "score": 0.95,
                        }
                    )
                    if len(keyword_results) >= query.limit:
                        break

        # Semantic search
        qe = get_embedding(s.model, query.query)
        results = search_quran_vec(s.db, qe, query.limit * 2)
        filtered = [
            r
            for r in results
            if r["score"] >= query.similarity_threshold
            and (not query.surah_filter or r["surah"] == query.surah_filter)
        ]

        # Merge keyword + semantic, deduplicate
        if keyword_results:
            seen = set()
            unique = []
            for r in keyword_results + filtered:
                key = (r["surah"], r.get("ayah", 0))
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            filtered = unique

        filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)[
            : query.limit
        ]
        return {"results": filtered, "total": len(filtered)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/api/hadith/search")
async def search_hadith(query: SearchQuery, request: Request):
    s = request.app.state
    if not s.hadith_ready:
        return _indexing_resp(s, results=[], total=0)

    try:
        qe = get_embedding(s.model, query.query)
        results = search_hadith_vec(s.db, qe, query.limit)
        filtered = [r for r in results if r["score"] >= query.similarity_threshold]
        return {
            "results": filtered,
            "total": len(filtered),
            "query": query.query,
            "source": "hadith",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- QA ----


@app.post("/api/qa")
async def islamic_qa(query: QAQuery, request: Request):
    s = request.app.state
    if not s.quran_ready:
        return _indexing_resp(
            s, answer="System is indexing, please wait...", relevant_verses=[]
        )

    try:
        qe = get_embedding(s.model, query.question)
        results = search_quran_vec(s.db, qe, query.context_limit)
        verses = []
        total_rel = 0
        for r in results:
            total_rel += r["score"]
            verses.append(
                {
                    **r,
                    "confidence": "high"
                    if r["score"] > 0.7
                    else "medium"
                    if r["score"] > 0.5
                    else "low",
                }
            )

        qtype = (
            "guidance"
            if any(w in query.question.lower() for w in ["how", "should", "guide"])
            else "definition"
            if any(w in query.question.lower() for w in ["what", "who", "define"])
            else "general"
        )
        intros = {
            "guidance": "The Quran provides guidance on this matter through these verses:",
            "definition": "The Quran describes this concept in the following way:",
            "general": "Based on Quranic teachings, regarding your question:",
        }
        context = " ".join(v.get("translation", "") for v in verses[:3])

        return {
            "question": query.question,
            "answer": f"{intros[qtype]} {context[:300]}...",
            "question_type": qtype,
            "relevant_verses": verses,
            "confidence": round((total_rel / max(query.context_limit, 1)) * 100, 1),
            "sources_count": len(verses),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Count ----


@app.post("/api/count")
async def count_word_occurrences(query: CountQuery, request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        word = query.word.strip()
        if not word:
            raise HTTPException(status_code=400, detail="Word cannot be empty")

        ar_col = _col(df, "ayah_ar", "text")
        en_col = _col(df, "ayah_en", "translation")
        sc_col = _col(df, "surah_no", "surah")
        ac_col = _col(df, "ayah_no_surah", "ayah")

        combined = df[ar_col].fillna("") + " " + df[en_col].fillna("")
        flags = re.IGNORECASE if not query.case_sensitive else 0
        match_counts = combined.str.count(re.escape(word), flags=flags)
        mask = match_counts > 0
        matched = df[mask]
        counts = match_counts[mask]

        examples = []
        for idx in matched.index[:10]:
            text = str(df.at[idx, ar_col]) if pd.notna(df.at[idx, ar_col]) else ""
            examples.append(
                {
                    "surah": int(df.at[idx, sc_col]),
                    "ayah": int(df.at[idx, ac_col]),
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "matches": int(counts[idx]),
                }
            )

        return {
            "word": word,
            "count": int(counts.sum()),
            "examples": examples,
            "total_verses_with_word": int(mask.sum()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Similar / Tafsir ----


@app.post("/api/similar")
async def similar_verses(req: Dict[str, Any], request: Request):
    query_text = req.get("reference", req.get("query", ""))
    limit = req.get("limit", 10)
    if not query_text:
        raise HTTPException(status_code=400, detail="Query or reference required")

    sq = SearchQuery(query=query_text, limit=limit)
    result = await search_verses(sq, request)
    if "results" in result:
        result["similar_verses"] = result.pop("results")
    return result


@app.post("/api/tafsir")
async def tafsir_endpoint(req: Dict[str, Any], request: Request):
    s = request.app.state
    if not s.quran_ready:
        return _indexing_resp(s, related_verses=[])

    reference = req.get("reference")
    if reference:
        m = re.match(r"^(\d+)[:\s]+(\d+)$", reference)
        if not m:
            raise HTTPException(status_code=400, detail="Invalid reference format")
        surah, ayah = int(m.group(1)), int(m.group(2))
    else:
        surah, ayah = req.get("surah"), req.get("ayah")

    if not surah or not ayah:
        raise HTTPException(status_code=400, detail="Surah and ayah required")

    df = s.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="System loading")

    sc, ac = _col(df, "surah_no", "surah"), _col(df, "ayah_no_surah", "ayah")
    row = df[(df[sc] == surah) & (df[ac] == ayah)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Verse {surah}:{ayah} not found")

    verse_text = row.iloc[0].get("ayah_ar", "")
    verse_trans = row.iloc[0].get("ayah_en", "")

    qe = get_embedding(s.model, verse_trans or verse_text)
    similar = search_quran_vec(s.db, qe, 5)
    related = [
        {
            "reference": f"{r['surah']}:{r['ayah']}",
            "translation": r["translation"],
            "score": r["score"],
        }
        for r in similar
        if not (r["surah"] == surah and r["ayah"] == ayah)
    ][:3]

    return {
        "reference": f"{surah}:{ayah}",
        "surah": surah,
        "ayah": ayah,
        "verse": verse_text,
        "translation": verse_trans,
        "related_verses": related,
        "note": "Tafsir data not yet available. Showing semantically related verses.",
    }


# ---- Analytics ----


@app.get("/api/stats")
async def get_stats(request: Request):
    s = request.app.state
    quran_count = (
        s.db.execute("SELECT COUNT(*) FROM quran_vec").fetchone()[0] if s.db else 0
    )
    hadith_count = (
        s.db.execute("SELECT COUNT(*) FROM hadith_vec").fetchone()[0] if s.db else 0
    )
    return {
        "total_verses": len(s.df_verses) if s.df_verses is not None else 0,
        "status": "Ready" if s.quran_ready else "Loading",
        "indexed_verses": quran_count,
        "total_hadiths": hadith_count,
    }


@app.get("/api/analytics/surah")
async def surah_analytics(request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    sc = _col(df, "surah_no", "surah")
    stats = df.groupby(sc).size().reset_index(name="verse_count")
    longest = stats.loc[stats["verse_count"].idxmax()]
    return {
        "total_surahs": len(stats),
        "total_verses": len(df),
        "average_verses_per_surah": round(len(df) / len(stats), 1),
        "longest_surah": {
            "number": int(longest[sc]),
            "verses": int(longest["verse_count"]),
        },
    }


@app.get("/api/analytics/frequency")
async def word_frequency(request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    tc = _col(df, "ayah_ar", "text")
    all_text = " ".join(df[tc].fillna(""))
    words = re.findall(r"\b\w+\b", all_text.lower())
    return {
        "words": dict(Counter(words).most_common(20)),
        "total_unique_words": len(set(words)),
    }


@app.get("/api/analytics/distribution")
async def theme_distribution(request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    themes = {
        "Faith & Belief": ["believe", "faith", "trust", "Allah", "Lord"],
        "Prayer & Worship": ["pray", "worship", "prostrate", "bow", "glorify"],
        "Charity & Kindness": ["give", "charity", "poor", "needy", "kindness"],
        "Justice & Truth": ["just", "fair", "judge", "right", "truth"],
        "Mercy & Forgiveness": ["mercy", "forgive", "compassion", "grace", "pardon"],
        "Guidance & Wisdom": ["guide", "wisdom", "knowledge", "understand", "reflect"],
        "Paradise & Hell": ["paradise", "garden", "hell", "fire", "punishment"],
        "Prophets & Messengers": ["prophet", "messenger", "Moses", "Jesus", "Abraham"],
    }

    en_col = _col(df, "ayah_en", "translation")
    distribution = {}
    for theme, keywords in themes.items():
        count = sum(
            df[en_col].str.contains(kw, case=False, na=False).sum() for kw in keywords
        )
        distribution[theme] = int(count)

    return {
        "themes": distribution,
        "total_verses": len(df),
        "chart_data": {
            "labels": list(distribution.keys()),
            "values": list(distribution.values()),
        },
    }


@app.get("/api/analytics/themes")
async def get_theme_clusters(request: Request):
    s = request.app.state
    if s.df_verses is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    if "themes" not in s.cache:
        n = min(1000, len(s.df_verses))
        sample = s.df_verses.sample(n=n, random_state=42)
        ar, en = _col(sample, "ayah_ar", "text"), _col(sample, "ayah_en", "translation")
        sc, ac = (
            _col(sample, "surah_no", "surah"),
            _col(sample, "ayah_no_surah", "ayah"),
        )
        texts = (
            sample[ar].fillna("").astype(str) + " " + sample[en].fillna("").astype(str)
        ).tolist()
        embeddings = np.array(s.model.encode(texts, batch_size=32), dtype=np.float32)
        labels = _kmeans()(n_clusters=15, random_state=42, n_init=10).fit_predict(
            embeddings
        )

        Tfidf = _tfidf()
        themes = {}
        for cid in np.unique(labels):
            cluster = sample[labels == cid]
            try:
                vec = Tfidf(max_features=10, stop_words="english")
                vec.fit_transform(cluster[en].fillna(""))
                kw = list(vec.get_feature_names_out()[:5])
            except Exception:
                kw = ["mixed_theme"]
            themes[int(cid)] = {
                "keywords": kw,
                "size": len(cluster),
                "sample_verses": cluster.head(3)[[sc, ac, en]].to_dict("records"),
            }
        s.cache["themes"] = themes
        s.cache["_themes_n"] = len(embeddings)

    return {
        "themes": s.cache["themes"],
        "total_themes": len(s.cache["themes"]),
        "sample_size": s.cache.get("_themes_n", 0),
    }


@app.get("/api/analytics/wordcloud")
async def generate_wordcloud_endpoint(request: Request):
    s = request.app.state
    if s.df_verses is None:
        raise HTTPException(status_code=503, detail="Dataset not available")

    if "wordcloud" not in s.cache:
        all_text = " ".join(
            s.df_verses[_col(s.df_verses, "ayah_en", "translation")].fillna("")
        )
        wc = _wordcloud()(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate(all_text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        s.cache["wordcloud"] = {
            "image": f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
            "frequencies": dict(wc.words_),
        }

    return {
        "wordcloud_image": s.cache["wordcloud"]["image"],
        "word_frequencies": s.cache["wordcloud"]["frequencies"],
    }


# ---- Browse ----


@app.get("/api/search/surah/{surah_no}")
async def get_surah_verses(surah_no: int, request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    sc = _col(df, "surah_no", "surah")
    ac = _col(df, "ayah_no_surah", "ayah")
    sv = df[df[sc] == surah_no]
    if sv.empty:
        raise HTTPException(status_code=404, detail=f"Surah {surah_no} not found")

    return {
        "surah": surah_no,
        "surah_name": f"Surah {surah_no}",
        "total_verses": len(sv),
        "verses": [
            {
                "ayah": int(r[ac]),
                "text": r.get("ayah_ar", ""),
                "translation": r.get("ayah_en", ""),
            }
            for _, r in sv.iterrows()
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
