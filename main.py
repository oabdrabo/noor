#!/usr/bin/env python3
"""Noor - Islamic Knowledge Search API.

AI-powered Quran and Hadith semantic search using multilingual embeddings
and sqlite-vec for vector storage.
"""
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
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import re
from collections import Counter
from functools import lru_cache
import arabic_reshaper
from bidi.algorithm import get_display

# Heavy libs lazy-loaded on first use (saves ~300MB at startup)
def _kmeans():
    from sklearn.cluster import KMeans
    return KMeans

def _tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer

def _cosine_similarity():
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity

def _umap():
    import umap
    return umap

def _networkx():
    import networkx as nx
    return nx

def _wordcloud():
    from wordcloud import WordCloud
    return WordCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- helpers ---------------

def serialize_f32(vector):
    """Serialize a numpy array to bytes for sqlite-vec."""
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32).tobytes()
    import struct
    return struct.pack(f'{len(vector)}f', *vector)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for matching (remove diacritics, unify letters)."""
    import unicodedata
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    replacements = {
        '\u0671': '\u0627', '\u0623': '\u0627', '\u0625': '\u0627',
        '\u0622': '\u0627', '\u0670': '', '\u06DA': '', '\u06D6': '',
        '\u06D7': '', '\u06D9': '', '\u06DB': '', '\u06DC': '',
        '\u0649': '\u064A', '\u0629': '\u0647',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def prepare_arabic_text(text: str) -> str:
    """Reshape Arabic text for display."""
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _col(df, *names):
    """Return the first column name that exists in the dataframe."""
    for n in names:
        if n in df.columns:
            return n
    return names[0]


# --------------- sqlite-vec search ---------------

def search_quran_vec(db, query_embedding, limit):
    """Search quran vectors and return results with metadata."""
    if db is None:
        return []
    query_bytes = serialize_f32(query_embedding)
    vec_results = db.execute(
        'SELECT rowid, distance FROM quran_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?',
        [query_bytes, limit]
    ).fetchall()
    if not vec_results:
        return []
    rowids = [r[0] for r in vec_results]
    distances = {r[0]: r[1] for r in vec_results}
    placeholders = ','.join('?' * len(rowids))
    metadata = db.execute(
        f'SELECT verse_id, surah, ayah, text, translation FROM quran_metadata WHERE verse_id IN ({placeholders})',
        rowids
    ).fetchall()
    meta_map = {m[0]: m for m in metadata}
    results = []
    for rowid in rowids:
        m = meta_map.get(rowid)
        if m:
            results.append({
                'surah': m[1], 'ayah': m[2], 'text': m[3], 'translation': m[4],
                'score': max(0.0, 1.0 - distances[rowid])
            })
    return results


def search_hadith_vec(db, query_embedding, limit):
    """Search hadith vectors and return results with metadata."""
    if db is None:
        return []
    query_bytes = serialize_f32(query_embedding)
    vec_results = db.execute(
        'SELECT rowid, distance FROM hadith_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?',
        [query_bytes, limit]
    ).fetchall()
    if not vec_results:
        return []
    rowids = [r[0] for r in vec_results]
    distances = {r[0]: r[1] for r in vec_results}
    placeholders = ','.join('?' * len(rowids))
    metadata = db.execute(
        f'SELECT hadith_id, collection_name, hadith_number, text, reference FROM hadith_metadata WHERE hadith_id IN ({placeholders})',
        rowids
    ).fetchall()
    meta_map = {m[0]: m for m in metadata}
    results = []
    for rowid in rowids:
        m = meta_map.get(rowid)
        if m:
            results.append({
                'collection': m[1], 'hadith_number': m[2], 'text': m[3], 'reference': m[4],
                'score': max(0.0, 1.0 - distances[rowid])
            })
    return results


# --------------- embedding ---------------

_embedding_cache = {}

def get_embedding(model, text: str) -> np.ndarray:
    """Get embedding for a single text using the multilingual model. LRU cached."""
    if not text or not str(text).strip():
        return np.zeros(1024, dtype=np.float32)
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
        return np.zeros(1024, dtype=np.float32)


def get_batch_embeddings(model, texts: list) -> np.ndarray:
    """Get embeddings for multiple texts."""
    print(f"[{datetime.now()}] Generating embeddings for {len(texts)} texts...", flush=True)
    try:
        clean = [str(t).strip()[:2000] if t and str(t).strip() else "" for t in texts]
        embeddings = model.encode(clean, show_progress_bar=True, batch_size=8)
        print(f"[{datetime.now()}] Completed generating {len(embeddings)} embeddings", flush=True)
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return np.zeros((len(texts), 1024), dtype=np.float32)


# --------------- clustering ---------------

def generate_theme_clusters(embeddings, n_clusters=15):
    """Generate thematic clusters from embeddings."""
    print(f"[{datetime.now()}] Running KMeans clustering ({n_clusters} clusters)...", flush=True)
    try:
        KMeans = _kmeans()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        print(f"[{datetime.now()}] Clustering complete", flush=True)
        return kmeans, labels
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return None, np.zeros(len(embeddings))


def analyze_themes(df, cluster_labels):
    """Analyze and label thematic clusters using TF-IDF."""
    TfidfVectorizer = _tfidf()
    theme_analysis = {}
    for cid in np.unique(cluster_labels):
        cluster_verses = df[cluster_labels == cid]
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            vectorizer.fit_transform(cluster_verses['ayah_en'].fillna(''))
            keywords = list(vectorizer.get_feature_names_out()[:5])
        except Exception:
            keywords = ['mixed_theme']
        theme_analysis[int(cid)] = {
            'keywords': keywords,
            'size': len(cluster_verses),
            'sample_verses': cluster_verses.head(3)[['surah_no', 'ayah_no_surah', 'ayah_en']].to_dict('records')
        }
    return theme_analysis


# --------------- Pydantic models ---------------

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    language: str = "both"
    surah_filter: Optional[int] = None
    similarity_threshold: float = 0.3

class CountQuery(BaseModel):
    word: str
    case_sensitive: bool = False

class QAQuery(BaseModel):
    question: str
    context_limit: int = 5


# --------------- lifespan ---------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan: load models and data once at startup."""
    print("Starting Noor - Islamic Knowledge Search API...", flush=True)
    print(f"[{datetime.now()}] Starting lifespan startup", flush=True)

    # Model cache on PVC for persistence
    os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', '/config/models')
    os.environ.setdefault('HF_HOME', '/config/models/huggingface')
    os.environ.setdefault('TRANSFORMERS_CACHE', '/config/models/huggingface')

    # Load BGE-M3: best multilingual model (Arabic+English), 1024d
    print(f"[{datetime.now()}] Loading BAAI/bge-m3 model...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-m3')
    print(f"[{datetime.now()}] Model loaded (bge-m3, 1024d)", flush=True)

    # Initialize SQLite with sqlite-vec
    print(f"[{datetime.now()}] Initializing SQLite vector database...", flush=True)
    import sqlite_vec
    db = sqlite3.connect('/config/vectors.db', check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Performance PRAGMAs
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute("PRAGMA cache_size = 10000")
    db.execute("PRAGMA temp_store = MEMORY")
    db.execute("PRAGMA mmap_size = 268435456")

    # Create non-vec tables first (needed before model version check)
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('''CREATE TABLE IF NOT EXISTS quran_metadata (
        verse_id INTEGER PRIMARY KEY, surah INTEGER, ayah INTEGER,
        text TEXT, translation TEXT
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS hadith_metadata (
        hadith_id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT, hadith_number TEXT, text TEXT, reference TEXT
    )''')
    db.commit()

    # Model version tracking — drop and recreate vec tables if model changes
    CURRENT_MODEL = 'BAAI/bge-m3'
    stored_model = db.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    need_rebuild = (stored_model is None) or (stored_model[0] != CURRENT_MODEL)
    if need_rebuild:
        old = stored_model[0] if stored_model else 'unknown'
        print(f"[{datetime.now()}] Model changed ({old} -> {CURRENT_MODEL}), rebuilding vectors...", flush=True)
        db.execute('DROP TABLE IF EXISTS quran_vec')
        db.execute('DROP TABLE IF EXISTS hadith_vec')
        db.execute('DELETE FROM quran_metadata')
        db.execute('DELETE FROM hadith_metadata')
        db.commit()

    # Create vec tables with correct dimensions (1024d for bge-m3)
    db.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS quran_vec USING vec0(
        embedding float[1024] distance_metric=cosine
    )''')
    db.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS hadith_vec USING vec0(
        embedding float[1024] distance_metric=cosine
    )''')
    db.commit()
    print(f"[{datetime.now()}] SQLite tables ready", flush=True)

    # Indexes for faster metadata lookups
    db.execute('CREATE INDEX IF NOT EXISTS idx_quran_surah ON quran_metadata(surah, ayah)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_hadith_collection ON hadith_metadata(collection_name)')
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('model', ?)", [CURRENT_MODEL])
    db.commit()

    # Load dataset
    print(f"[{datetime.now()}] Loading Quran dataset...", flush=True)
    df_verses = pd.read_csv('/config/quran-dataset.csv')
    print(f"[{datetime.now()}] Loaded {len(df_verses)} verses", flush=True)

    # Index quran vectors (skip if already done)
    quran_count = db.execute('SELECT COUNT(*) FROM quran_vec').fetchone()[0]
    if quran_count == 0:
        print(f"[{datetime.now()}] Building quran vector index...", flush=True)
        ar_col = _col(df_verses, 'ayah_ar', 'text')
        en_col = _col(df_verses, 'ayah_en', 'translation')
        texts = (df_verses[ar_col].fillna('').astype(str) + ' ' + df_verses[en_col].fillna('').astype(str)).tolist()

        embeddings = get_batch_embeddings(model, texts)
        print(f"[{datetime.now()}] Batch inserting {len(texts)} verses...", flush=True)

        sc_col = _col(df_verses, 'surah_no', 'surah')
        ac_col = _col(df_verses, 'ayah_no_surah', 'ayah')
        meta_rows = [(i, int(r[sc_col]), int(r[ac_col]),
                       str(r.get(ar_col, ''))[:5000], str(r.get(en_col, ''))[:5000])
                      for i, (_, r) in enumerate(df_verses.iterrows())]
        vec_rows = [(i, serialize_f32(embeddings[i])) for i in range(len(texts))]

        db.executemany('INSERT OR REPLACE INTO quran_metadata (verse_id, surah, ayah, text, translation) VALUES (?, ?, ?, ?, ?)', meta_rows)
        db.executemany('INSERT INTO quran_vec (rowid, embedding) VALUES (?, ?)', vec_rows)
        db.commit()
        print(f"[{datetime.now()}] Quran index built ({len(texts)} verses)", flush=True)
    else:
        print(f"[{datetime.now()}] Quran vectors cached ({quran_count}), skipping", flush=True)

    print(f"[{datetime.now()}] Quran vector database ready!", flush=True)

    # Analytics deferred to first request (saves 15-20s startup)
    verse_embeddings = None
    cluster_model = None
    theme_labels = None

    # Index hadiths
    hadith_ready = False
    hadith_count = db.execute('SELECT COUNT(*) FROM hadith_vec').fetchone()[0]
    if hadith_count == 0:
        try:
            print(f"[{datetime.now()}] Downloading Hadith collections...", flush=True)
            import urllib.request
            bukhari_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.json"
            muslim_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.json"

            with urllib.request.urlopen(bukhari_url, timeout=30) as resp:
                bukhari_data = json.loads(resp.read())
            print(f"[{datetime.now()}] Bukhari: {len(bukhari_data['hadiths'])} hadiths", flush=True)

            with urllib.request.urlopen(muslim_url, timeout=30) as resp:
                muslim_data = json.loads(resp.read())
            print(f"[{datetime.now()}] Muslim: {len(muslim_data['hadiths'])} hadiths", flush=True)

            hadith_records = []
            for src, name in [(bukhari_data, 'Sahih Bukhari'), (muslim_data, 'Sahih Muslim')]:
                for h in src['hadiths']:
                    text = h.get('text', '')[:9900]
                    hadith_records.append({
                        'collection': name,
                        'hadith_number': str(h.get('hadithNumber', '')),
                        'text': text,
                        'reference': f"{name.split()[-1]} {h.get('hadithNumber', '')}"
                    })

            df_hadith = pd.DataFrame(hadith_records)
            print(f"[{datetime.now()}] Total hadiths: {len(df_hadith)}", flush=True)

            hadith_texts = [str(t).strip() if t and str(t).strip() else "[No text]" for t in df_hadith['text']]
            batch_size = 500
            hid_counter = 1
            for i in range(0, len(hadith_texts), batch_size):
                try:
                    batch = hadith_texts[i:i+batch_size]
                    batch_emb = get_batch_embeddings(model, batch)
                    batch_df = df_hadith.iloc[i:i+len(batch)]
                    meta_rows = [(r['collection'], str(r['hadith_number']), str(r['text']), r['reference'])
                                  for _, r in batch_df.iterrows()]
                    db.executemany('INSERT INTO hadith_metadata (collection_name, hadith_number, text, reference) VALUES (?, ?, ?, ?)', meta_rows)
                    # Get the rowid range for vec inserts
                    last_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
                    start_id = last_id - len(batch) + 1
                    vec_rows = [(start_id + j, serialize_f32(batch_emb[j])) for j in range(len(batch))]
                    db.executemany('INSERT INTO hadith_vec (rowid, embedding) VALUES (?, ?)', vec_rows)
                    if (i + batch_size) % 2000 == 0:
                        print(f"  Indexed {min(i + batch_size, len(hadith_texts))}/{len(hadith_texts)} hadiths...", flush=True)
                except Exception as batch_err:
                    logger.error(f"Hadith batch {i//batch_size} error: {batch_err}")
                    continue
            db.commit()
            hadith_count = db.execute('SELECT COUNT(*) FROM hadith_vec').fetchone()[0]
            print(f"[{datetime.now()}] Hadith index built ({hadith_count} hadiths)", flush=True)
            hadith_ready = True
        except Exception as e:
            logger.error(f"Hadith init failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[{datetime.now()}] Hadith vectors cached ({hadith_count}), skipping", flush=True)
        hadith_ready = True

    print(f"[{datetime.now()}] STARTUP COMPLETE - API READY!", flush=True)

    # Store everything on app.state
    app.state.model = model
    app.state.db = db
    app.state.df_verses = df_verses
    app.state.verse_embeddings = verse_embeddings
    app.state.cluster_model = cluster_model
    app.state.theme_labels = theme_labels
    app.state.hadith_ready = hadith_ready
    app.state.quran_ready = True
    # Cache for expensive operations
    app.state.cache = {}

    yield

    # Shutdown
    db.close()
    print("Noor shutdown complete.", flush=True)


# --------------- app ---------------

base_domain = os.environ.get("BASE_DOMAIN", "jsr.bz")

app = FastAPI(title="Noor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"https://{base_domain}", f"https://noor.{base_domain}"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# --------------- routes ---------------

@app.get("/")
async def root(request: Request):
    df = request.app.state.df_verses
    return {"message": "Noor API", "status": "ready", "verses": len(df) if df is not None else 0}

@app.get("/api")
async def api_root():
    return {"endpoints": [
        "/api/search", "/api/hadith/search", "/api/qa", "/api/count",
        "/api/similar", "/api/tafsir", "/api/export", "/api/analytics/themes"
    ]}

@app.get("/api/health")
@app.get("/api/status")
async def health_status(request: Request):
    s = request.app.state
    return {
        "status": "healthy",
        "quran_ready": getattr(s, 'quran_ready', False),
        "hadith_ready": getattr(s, 'hadith_ready', False),
        "total_verses": len(s.df_verses) if s.df_verses is not None else 0,
        "sqlite_connected": s.db is not None,
        "endpoints": ["/api/search", "/api/qa", "/api/similar", "/api/tafsir",
                       "/api/count", "/api/analytics/themes", "/api/export", "/api/hadith/search"]
    }


# ---- Search ----

@app.post("/api/search")
async def search_verses(query: SearchQuery, request: Request):
    s = request.app.state
    if not s.quran_ready or s.df_verses is None:
        raise HTTPException(status_code=503, detail="System loading")

    try:
        if not query.query or not query.query.strip():
            return {"results": [], "total": 0}

        # Verse reference lookup (e.g. "2:255")
        verse_match = re.match(r'^(\d+)[:\s]+(\d+)$', query.query.strip())
        if verse_match:
            surah_num, ayah_num = int(verse_match.group(1)), int(verse_match.group(2))
            sc, ac = _col(s.df_verses, 'surah_no', 'surah'), _col(s.df_verses, 'ayah_no_surah', 'ayah')
            row = s.df_verses[(s.df_verses[sc] == surah_num) & (s.df_verses[ac] == ayah_num)]
            if not row.empty:
                r = row.iloc[0]
                return {"results": [{
                    "surah": int(r[sc]), "ayah": int(r[ac]),
                    "text": r.get('ayah_ar', ''), "translation": r.get('ayah_en', ''),
                    "score": 1.0
                }], "total": 1}

        # Arabic keyword fallback
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in query.query)
        keyword_results = []
        if has_arabic:
            nq = normalize_arabic(query.query)
            for _, row in s.df_verses.iterrows():
                ar = row.get('ayah_ar', '')
                if pd.notna(ar) and nq in normalize_arabic(ar):
                    keyword_results.append({
                        "surah": int(row.get('surah_no', 1)), "ayah": int(row.get('ayah_no_surah', 1)),
                        "text": ar, "translation": row.get('ayah_en', ''), "score": 0.95
                    })
                    if len(keyword_results) >= query.limit:
                        break

        # Semantic search
        qe = get_embedding(s.model, query.query)
        results = search_quran_vec(s.db, qe, query.limit * 2)
        filtered = [r for r in results
                     if r['score'] >= query.similarity_threshold
                     and (not query.surah_filter or r['surah'] == query.surah_filter)]

        # Merge keyword + semantic, deduplicate
        if keyword_results:
            all_results = keyword_results + filtered
            seen = set()
            unique = []
            for r in all_results:
                key = (r['surah'], r.get('ayah', 0))
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            filtered = unique

        filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:query.limit]
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
        raise HTTPException(status_code=503, detail="Hadith system loading")

    try:
        qe = get_embedding(s.model, query.query)
        results = search_hadith_vec(s.db, qe, query.limit)
        filtered = [r for r in results if r['score'] > query.similarity_threshold]
        return {"results": filtered, "total": len(filtered), "query": query.query, "source": "hadith"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/advanced")
async def advanced_search(query: SearchQuery, request: Request):
    s = request.app.state
    if not s.quran_ready or s.df_verses is None:
        raise HTTPException(status_code=503, detail="System loading")

    try:
        qe = get_embedding(s.model, query.query)
        raw = search_quran_vec(s.db, qe, query.limit * 3)
        results = []
        scores = []
        for r in raw:
            if r['score'] >= query.similarity_threshold:
                if query.surah_filter and r['surah'] != query.surah_filter:
                    continue
                r['confidence'] = "high" if r['score'] > 0.8 else "medium" if r['score'] > 0.6 else "low"
                r['relevance_percentage'] = round(r['score'] * 100, 1)
                results.append(r)
                scores.append(r['score'])

        results = sorted(results, key=lambda x: x['score'], reverse=True)[:query.limit]
        return {
            "results": results,
            "analytics": {
                "total_matches": len(results),
                "avg_confidence": round(np.mean(scores) * 100, 1) if scores else 0,
                "high_confidence_matches": sum(1 for s in scores if s > 0.8),
                "search_method": "semantic_vector"
            }
        }
    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/multi-vector")
async def multi_vector_search(query: SearchQuery, request: Request):
    s = request.app.state
    if not s.quran_ready or s.df_verses is None:
        raise HTTPException(status_code=503, detail="System loading")

    try:
        semantic_weight, keyword_weight = 0.7, 0.3

        # Semantic search
        qe = get_embedding(s.model, query.query)
        semantic_results = search_quran_vec(s.db, qe, query.limit * 2)

        # Keyword search (vectorized)
        query_words = query.query.lower().split()
        combined = {}

        for r in semantic_results:
            key = f"{r['surah']}:{r['ayah']}"
            combined[key] = {"semantic": r['score'], "keyword": 0, "verse": r}

        # Vectorized keyword matching
        ar_col = _col(s.df_verses, 'ayah_ar', 'text')
        en_col = _col(s.df_verses, 'ayah_en', 'translation')
        sc_col = _col(s.df_verses, 'surah_no', 'surah')
        ac_col = _col(s.df_verses, 'ayah_no_surah', 'ayah')

        text_series = (s.df_verses[ar_col].fillna('') + ' ' + s.df_verses[en_col].fillna('')).str.lower()
        for word in query_words:
            mask = text_series.str.contains(re.escape(word), na=False)
            for idx in s.df_verses[mask].index:
                row = s.df_verses.loc[idx]
                key = f"{int(row[sc_col])}:{int(row[ac_col])}"
                score = sum(1 for w in query_words if w in text_series[idx]) / len(query_words)
                if key in combined:
                    combined[key]["keyword"] = max(combined[key]["keyword"], score)
                else:
                    combined[key] = {
                        "semantic": 0, "keyword": score,
                        "verse": {"surah": int(row[sc_col]), "ayah": int(row[ac_col]),
                                  "text": row.get(ar_col, ''), "translation": row.get(en_col, '')}
                    }

        results = []
        for data in combined.values():
            final = semantic_weight * data["semantic"] + keyword_weight * data["keyword"]
            if final > 0.1:
                v = data["verse"]
                results.append({
                    "surah": int(v.get('surah', 1)), "ayah": int(v.get('ayah', 1)),
                    "text": v.get('text', ''), "translation": v.get('translation', ''),
                    "final_score": round(final, 3),
                    "semantic_score": round(data["semantic"], 3),
                    "keyword_score": round(data["keyword"], 3),
                })

        results = sorted(results, key=lambda x: x['final_score'], reverse=True)[:query.limit]
        return {"results": results, "total": len(results)}
    except Exception as e:
        logger.error(f"Multi-vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- QA ----

@app.post("/api/qa")
@app.post("/api/qa/islamic")
async def islamic_qa(query: QAQuery, request: Request):
    s = request.app.state
    if not s.quran_ready or s.df_verses is None:
        raise HTTPException(status_code=503, detail="System loading")

    try:
        qe = get_embedding(s.model, query.question)
        results = search_quran_vec(s.db, qe, query.context_limit)
        verses = []
        total_rel = 0
        for r in results:
            total_rel += r['score']
            verses.append({**r, "confidence": "high" if r['score'] > 0.7 else "medium" if r['score'] > 0.5 else "low"})

        qtype = "guidance" if any(w in query.question.lower() for w in ["how", "should", "guide"]) \
            else "definition" if any(w in query.question.lower() for w in ["what", "who", "define"]) \
            else "general"
        intros = {
            "guidance": "The Quran provides guidance on this matter through these verses:",
            "definition": "The Quran describes this concept in the following way:",
            "general": "Based on Quranic teachings, regarding your question:"
        }
        context = " ".join(v.get('translation', '') for v in verses[:3])
        answer = f"{intros[qtype]} {context[:300]}..."

        return {
            "question": query.question, "answer": answer, "question_type": qtype,
            "relevant_verses": verses,
            "confidence": round((total_rel / max(query.context_limit, 1)) * 100, 1),
            "sources_count": len(verses)
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

        ar_col = _col(df, 'ayah_ar', 'text')
        en_col = _col(df, 'ayah_en', 'translation')
        sc_col = _col(df, 'surah_no', 'surah')
        ac_col = _col(df, 'ayah_no_surah', 'ayah')

        case = not query.case_sensitive
        combined = df[ar_col].fillna('') + ' ' + df[en_col].fillna('')
        match_counts = combined.str.count(re.escape(word), flags=re.IGNORECASE if case else 0)
        mask = match_counts > 0
        matched = df[mask]
        counts = match_counts[mask]

        count = int(counts.sum())
        examples = []
        for idx in matched.index[:10]:
            text = str(df.at[idx, ar_col]) if pd.notna(df.at[idx, ar_col]) else ""
            examples.append({
                "surah": int(df.at[idx, sc_col]), "ayah": int(df.at[idx, ac_col]),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "matches": int(counts[idx])
            })

        return {"word": word, "count": count, "examples": examples, "total_verses_with_word": int(mask.sum())}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Arabic Analysis ----

@app.post("/api/arabic/analyze")
async def analyze_arabic_text(text_data: dict):
    try:
        text = text_data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        diacritics = sum(1 for c in text if '\u064B' <= c <= '\u065F')
        words = text.split()
        word_count = len(words)

        return {
            "analysis": {
                "character_analysis": {
                    "total_chars": len(text), "arabic_chars": arabic_chars,
                    "diacritics": diacritics,
                    "diacritic_ratio": round(diacritics / max(arabic_chars, 1), 3)
                },
                "word_analysis": {
                    "word_count": word_count,
                    "avg_word_length": round(sum(len(w) for w in words) / max(word_count, 1), 2),
                    "unique_words": len(set(words))
                },
                "text_type": "classical" if diacritics > arabic_chars * 0.1 else "modern"
            },
            "summary": f"Arabic text: {arabic_chars} chars, {word_count} words, {diacritics} diacritics"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Analytics ----

@app.get("/api/analytics/surah")
async def surah_analytics(request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    sc = _col(df, 'surah_no', 'surah')
    stats = df.groupby(sc).size().reset_index(name='verse_count')
    longest = stats.loc[stats['verse_count'].idxmax()]
    return {
        "total_surahs": len(stats), "total_verses": len(df),
        "average_verses_per_surah": round(len(df) / len(stats), 1),
        "longest_surah": {"number": int(longest[sc]), "verses": int(longest['verse_count'])}
    }

@app.get("/api/analytics/frequency")
async def word_frequency(request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    tc = _col(df, 'ayah_ar', 'text')
    all_text = " ".join(df[tc].fillna(''))
    words = re.findall(r'\b\w+\b', all_text.lower())
    return {"words": dict(Counter(words).most_common(20)), "total_unique_words": len(set(words))}


@app.get("/api/stats")
async def get_stats(request: Request):
    s = request.app.state
    quran_count = s.db.execute('SELECT COUNT(*) FROM quran_vec').fetchone()[0] if s.db else 0
    hadith_count = s.db.execute('SELECT COUNT(*) FROM hadith_vec').fetchone()[0] if s.db else 0
    return {
        "total_verses": len(s.df_verses) if s.df_verses is not None else 0,
        "status": "Ready" if s.quran_ready else "Loading",
        "indexed_verses": quran_count,
        "total_hadiths": hadith_count,
    }


@app.get("/api/surah/{surah_no}")
async def get_surah_info(surah_no: int, request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    if surah_no < 1 or surah_no > 114:
        raise HTTPException(status_code=400, detail="Invalid surah number (1-114)")

    sv = df[df[_col(df, 'surah_no', 'surah')] == surah_no]
    if sv.empty:
        raise HTTPException(status_code=404, detail=f"Surah {surah_no} not found")

    first = sv.iloc[0]
    return {
        "surah_no": surah_no,
        "name_en": first.get('surah_name_en', f'Surah {surah_no}'),
        "total_verses": len(sv),
        "juz_no": int(first.get('juz_no', 0)) if pd.notna(first.get('juz_no')) else None
    }


@app.get("/api/juz/{juz_no}")
async def get_juz_info(juz_no: int, request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    if juz_no < 1 or juz_no > 30:
        raise HTTPException(status_code=400, detail="Invalid juz number (1-30)")

    jv = df[df['juz_no'] == juz_no]
    if jv.empty:
        raise HTTPException(status_code=404, detail=f"Juz {juz_no} not found")

    return {
        "juz_no": juz_no, "total_verses": len(jv),
        "verses": [{"reference": f"{int(r['surah_no'])}:{int(r['ayah_no_surah'])}",
                     "arabic": r.get('ayah_ar', ''), "translation": r.get('ayah_en', '')}
                    for _, r in jv.head(10).iterrows()],
        "surahs_included": sorted(jv['surah_no'].unique().tolist())
    }


@app.get("/api/search/surah/{surah_no}")
async def get_surah_verses(surah_no: int, request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    sc = _col(df, 'surah_no', 'surah')
    sv = df[df[sc] == surah_no]
    if sv.empty:
        raise HTTPException(status_code=404, detail=f"Surah {surah_no} not found")

    return {
        "surah": surah_no, "surah_name": sv.iloc[0].get('surah_name_en', f'Surah {surah_no}'),
        "total_verses": len(sv),
        "verses": [{"ayah": int(r[_col(df, 'ayah_no_surah', 'ayah')]),
                     "text": r.get('ayah_ar', ''), "translation": r.get('ayah_en', '')}
                    for _, r in sv.iterrows()]
    }


# ---- Advanced Analytics ----

def _ensure_analytics(s):
    """Lazy-compute analytics on first request."""
    if s.verse_embeddings is None:
        print(f"[{datetime.now()}] Computing analytics (deferred)...", flush=True)
        sample_size = min(1000, len(s.df_verses))
        sample_df = s.df_verses.sample(n=sample_size, random_state=42)
        ar_col = _col(s.df_verses, 'ayah_ar', 'text')
        en_col = _col(s.df_verses, 'ayah_en', 'translation')
        combined_texts = (sample_df[ar_col].fillna('').astype(str) + ' ' + sample_df[en_col].fillna('').astype(str)).tolist()
        s.verse_embeddings = get_batch_embeddings(s.model, combined_texts)
        s.cluster_model, s.theme_labels = generate_theme_clusters(s.verse_embeddings)
        print(f"[{datetime.now()}] Analytics ready!", flush=True)

@app.get("/api/analytics/themes")
async def get_theme_clusters(request: Request):
    s = request.app.state
    if s.df_verses is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    _ensure_analytics(s)

    if 'themes' not in s.cache:
        sample_df = s.df_verses.sample(n=min(1000, len(s.df_verses)), random_state=42)
        s.cache['themes'] = analyze_themes(sample_df, s.theme_labels)

    return {"themes": s.cache['themes'], "total_themes": len(s.cache['themes']),
            "sample_size": len(s.verse_embeddings)}


@app.get("/api/analytics/embeddings/visualization")
async def get_embeddings_visualization(request: Request):
    s = request.app.state
    if s.verse_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not available")

    _ensure_analytics(s)
    # Cache UMAP projection
    if 'umap_2d' not in s.cache:
        reducer = _umap().UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        s.cache['umap_2d'] = reducer.fit_transform(s.verse_embeddings)

    return {
        "embeddings_2d": s.cache['umap_2d'].tolist(),
        "cluster_labels": s.theme_labels.tolist() if s.theme_labels is not None else None,
        "method": "UMAP", "dimensions": list(s.cache['umap_2d'].shape)
    }


@app.post("/api/analytics/similarity/network")
async def create_similarity_network(request: Request, threshold: float = 0.8):
    s = request.app.state
    if s.verse_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not available")
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    _ensure_analytics(s)
    nx = _networkx()
    sim = _cosine_similarity()(s.verse_embeddings)
    G = nx.Graph()
    for i in range(len(s.verse_embeddings)):
        G.add_node(i)
        for j in range(i + 1, len(s.verse_embeddings)):
            if sim[i][j] > threshold:
                G.add_edge(i, j, weight=float(sim[i][j]))

    nodes = [{"id": int(n), "label": f"Verse {n+1}",
              "cluster": int(s.theme_labels[n]) if s.theme_labels is not None else 0} for n in G.nodes()]
    edges = [{"source": int(e[0]), "target": int(e[1]), "weight": float(e[2]["weight"])} for e in G.edges(data=True)]

    return {"nodes": nodes, "edges": edges, "threshold": threshold,
            "stats": {"total_nodes": len(nodes), "total_edges": len(edges), "density": nx.density(G)}}


@app.get("/api/analytics/wordcloud")
async def generate_wordcloud_endpoint(request: Request):
    s = request.app.state
    if s.df_verses is None:
        raise HTTPException(status_code=503, detail="Dataset not available")

    # Cache wordcloud
    if 'wordcloud' not in s.cache:
        all_text = " ".join(s.df_verses[_col(s.df_verses, 'ayah_en', 'translation')].fillna(''))
        wc = _wordcloud()(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(all_text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        s.cache['wordcloud'] = {
            "image": f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
            "frequencies": dict(wc.words_)
        }

    return {"wordcloud_image": s.cache['wordcloud']['image'],
            "word_frequencies": s.cache['wordcloud']['frequencies']}


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
        "Prophets & Messengers": ["prophet", "messenger", "Moses", "Jesus", "Abraham"]
    }

    en_col = _col(df, 'ayah_en', 'translation')
    distribution = {}
    for theme, keywords in themes.items():
        count = sum(df[en_col].str.contains(kw, case=False, na=False).sum() for kw in keywords)
        distribution[theme] = int(count)

    return {"themes": distribution, "total_verses": len(df),
            "chart_data": {"labels": list(distribution.keys()), "values": list(distribution.values())}}


# ---- Utility endpoints ----

@app.post("/api/verse")
async def get_verse(req: Dict[str, Any], request: Request):
    df = request.app.state.df_verses
    if df is None:
        raise HTTPException(status_code=503, detail="System loading")

    reference = req.get("reference", "")
    if not reference:
        raise HTTPException(status_code=400, detail="Reference required")

    match = re.match(r'^(\d+)[:\s]+(\d+)$', reference.strip())
    if not match:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'surah:ayah' (e.g. '2:255')")

    surah_num, ayah_num = int(match.group(1)), int(match.group(2))
    sc, ac = _col(df, 'surah_no', 'surah'), _col(df, 'ayah_no_surah', 'ayah')
    row = df[(df[sc] == surah_num) & (df[ac] == ayah_num)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Verse {surah_num}:{ayah_num} not found")

    r = row.iloc[0]
    return {"verse": {
        "reference": f"{surah_num}:{ayah_num}", "surah": int(r[sc]), "ayah": int(r[ac]),
        "arabic": r.get('ayah_ar', ''), "translation": r.get('ayah_en', ''),
        "juz": int(r.get('juz_no', 0)) if pd.notna(r.get('juz_no')) else None
    }}


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
    reference = req.get("reference")
    if reference:
        m = re.match(r'^(\d+)[:\s]+(\d+)$', reference)
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

    sc, ac = _col(df, 'surah_no', 'surah'), _col(df, 'ayah_no_surah', 'ayah')
    row = df[(df[sc] == surah) & (df[ac] == ayah)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Verse {surah}:{ayah} not found")

    verse_text = row.iloc[0].get('ayah_ar', '')
    verse_trans = row.iloc[0].get('ayah_en', '')

    # Find semantically related verses
    qe = get_embedding(s.model, verse_trans or verse_text)
    similar = search_quran_vec(s.db, qe, 5)
    related = [{"reference": f"{r['surah']}:{r['ayah']}", "translation": r['translation'], "score": r['score']}
               for r in similar if not (r['surah'] == surah and r['ayah'] == ayah)][:3]

    return {
        "reference": f"{surah}:{ayah}", "surah": surah, "ayah": ayah,
        "verse": verse_text, "translation": verse_trans,
        "related_verses": related,
        "note": "Tafsir data not yet available. Showing semantically related verses."
    }


@app.post("/api/export")
@app.post("/api/analytics/export")
async def export_search(req: dict, request: Request):
    s = request.app.state
    if not s.quran_ready or s.df_verses is None:
        raise HTTPException(status_code=503, detail="System loading")

    search_query = req.get("query", "")
    fmt = req.get("format", "json")
    limit = req.get("limit", 50)

    qe = get_embedding(s.model, search_query)
    results = search_quran_vec(s.db, qe, limit)
    data = [{"surah": r['surah'], "ayah": r['ayah'], "arabic_text": r['text'],
             "english_translation": r['translation'], "similarity_score": r['score']}
            for r in results]

    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', search_query[:20])

    if fmt == "csv":
        import csv
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return {"format": "csv", "content": output.getvalue(),
                "filename": f"noor_{safe_name}.csv", "count": len(data)}

    return {"format": "json", "data": data, "filename": f"noor_{safe_name}.json", "count": len(data)}


@app.get("/api/visualize/similarity")
async def visualize_similarity(request: Request, query: str = ""):
    if not query:
        return {"nodes": [], "edges": []}
    return await create_similarity_network(request, threshold=0.5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
