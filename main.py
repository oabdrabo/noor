#!/usr/bin/env python3
import base64
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("noor")

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_DIM = 384
BATCH = 64

DATA_DIR = os.environ.get("NOOR_DATA_DIR", "/config")
DB_PATH = os.path.join(DATA_DIR, "vectors.db")
CSV_PATH = os.path.join(DATA_DIR, "quran-dataset.csv")
MODELS_DIR = os.path.join(DATA_DIR, "models")

AIGW_URL = os.environ.get("AIGW_URL", "http://aigw.aigw.svc.cluster.local:5006")
AIGW_KEY = os.environ.get("AIGW_KEY", "")
AIGW_MODEL = os.environ.get("AIGW_MODEL", "sonnet")
AIGW_TIMEOUT = 120

STT_URL = os.environ.get("STT_URL", "http://stt.stt.svc.cluster.local:8000")
STT_MODEL = os.environ.get("STT_MODEL", "Systran/faster-whisper-small")

# In-memory per-IP rate limit for the expensive public endpoints (embeddings, Claude,
# STT) - an unauthenticated flood would otherwise be a cost + resource DoS.
_RL: dict[str, tuple[int, float]] = {}


def _rate_limit(request: Request, limit: int, window: float = 60.0) -> None:
    ip = (
        request.headers.get("x-real-ip")
        or request.headers.get("x-forwarded-for", "").split(",")[0]
        or (request.client.host if request.client else "?")
    ).strip()
    now = time.monotonic()
    if len(_RL) > 10000:  # bound memory: drop windows that have elapsed
        for k in [k for k, (_, r) in _RL.items() if now > r]:
            del _RL[k]
    count, reset = _RL.get(ip, (0, now + window))
    if now > reset:
        count, reset = 0, now + window
    count += 1
    _RL[ip] = (count, reset)
    if count > limit:
        raise HTTPException(status_code=429, detail="Too many requests - slow down.")


COL_SURAH = "surah_no"
COL_AYAH = "ayah_no_surah"
COL_AR = "ayah_ar"
COL_EN = "ayah_en"

QURAN_COLS = ("surah", "ayah", "text", "translation")
HADITH_COLS = ("collection", "hadith_number", "text", "reference")

HADITH_SOURCES = (
    (
        "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.json",
        "Sahih Bukhari",
    ),
    (
        "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.json",
        "Sahih Muslim",
    ),
)

THEMES = {
    "Faith & Belief": ["believe", "faith", "trust", "Allah", "Lord"],
    "Prayer & Worship": ["pray", "worship", "prostrate", "bow", "glorify"],
    "Charity & Kindness": ["give", "charity", "poor", "needy", "kindness"],
    "Justice & Truth": ["just", "fair", "judge", "right", "truth"],
    "Mercy & Forgiveness": ["mercy", "forgive", "compassion", "grace", "pardon"],
    "Guidance & Wisdom": ["guide", "wisdom", "knowledge", "understand", "reflect"],
    "Paradise & Hell": ["paradise", "garden", "hell", "fire", "punishment"],
    "Prophets & Messengers": ["prophet", "messenger", "Moses", "Jesus", "Abraham"],
}

QA_SYSTEM = (
    "You answer questions about Islam strictly from the Qur'an verses and hadith the user provides. "
    "Ground every statement in those passages and cite each with its bracketed reference, e.g. [2:255] "
    "or [Sahih Bukhari 1]. If the passages do not address the question, say so plainly. Do not issue "
    "legal rulings; report what the sources say and suggest consulting a qualified scholar for rulings. "
    "Answer in the language of the question, in at most three short paragraphs."
)


def serialize_f32(vector):
    return np.asarray(vector, dtype=np.float32).tobytes()


def _vec_search(db, vec_table, meta_table, id_col, columns, query_emb, limit):
    hits = db.execute(
        f"SELECT rowid, distance FROM {vec_table} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        [serialize_f32(query_emb), limit],
    ).fetchall()
    if not hits:
        return []
    ids = [h[0] for h in hits]
    score = {h[0]: max(0.0, 1.0 - h[1]) for h in hits}
    placeholders = ",".join("?" * len(ids))
    rows = db.execute(
        f"SELECT {id_col},{','.join(columns)} FROM {meta_table} WHERE {id_col} IN ({placeholders})",
        ids,
    ).fetchall()
    meta = {r[0]: r[1:] for r in rows}
    return [dict(zip(columns, meta[i]), score=score[i]) for i in ids if i in meta]


class SearchQuery(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=50)
    surah_filter: Optional[int] = Field(None, ge=1, le=114)
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)


class CountQuery(BaseModel):
    word: str
    case_sensitive: bool = False


class QAQuery(BaseModel):
    question: str
    limit: int = Field(5, ge=1, le=10)


class TranscribeQuery(BaseModel):
    audio: str = Field(..., max_length=2_700_000)  # ~2 MB base64; matches nginx body cap
    content_type: str = Field("audio/webm", max_length=80, pattern=r"^[\w.+-]+/[\w.+-]+$")


def _connect():
    import sqlite_vec

    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    for pragma in (
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA cache_size=10000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA mmap_size=268435456",
    ):
        db.execute(pragma)
    return db


def _create_hadith_metadata(db):
    db.execute(
        "CREATE TABLE IF NOT EXISTS hadith_metadata "
        "(hadith_id INTEGER PRIMARY KEY AUTOINCREMENT, collection TEXT, hadith_number TEXT, text TEXT, reference TEXT)"
    )


def _init_schema(db):
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS quran_metadata "
        "(verse_id INTEGER PRIMARY KEY, surah INTEGER, ayah INTEGER, text TEXT, translation TEXT)"
    )
    _create_hadith_metadata(db)
    db.commit()


def _ensure_model(db):
    stored = db.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    if stored is None or stored[0] != MODEL_NAME:
        logger.info(
            "Model changed (%s -> %s), rebuilding",
            stored[0] if stored else "none",
            MODEL_NAME,
        )
        for table in ("quran_vec", "hadith_vec"):
            db.execute(f"DROP TABLE IF EXISTS {table}")
        db.execute("DELETE FROM quran_metadata")
        db.execute("DELETE FROM hadith_metadata")
        db.execute("DELETE FROM meta WHERE key='hadith_total'")
    for table in ("quran_vec", "hadith_vec"):
        db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{EMBED_DIM}] distance_metric=cosine)"
        )
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('model', ?)", [MODEL_NAME]
    )
    db.commit()


def _run_batches(state, stop, texts, label, start, persist):
    total = len(texts)
    for i in range(start, total, BATCH):
        if stop.is_set():
            return False
        embeddings = state.model.encode(texts[i : i + BATCH], batch_size=32)
        persist(i, embeddings)
        state.db.commit()
        done = min(i + BATCH, total)
        state.index_progress = f"{label}: {done}/{total}"
        if done % 256 == 0 or done == total:
            logger.info("Indexed %s %d/%d", label, done, total)
    return True


def _index_quran(state, stop):
    db = state.db
    df = state.df_verses
    total = len(df)
    start = db.execute("SELECT COUNT(*) FROM quran_vec").fetchone()[0]
    if start >= total:
        state.quran_ready = True
        return

    ar = df[COL_AR].astype(str).to_numpy()
    en = df[COL_EN].astype(str).to_numpy()
    surah = df[COL_SURAH].to_numpy()
    ayah = df[COL_AYAH].to_numpy()
    texts = [f"passage: {e}" for e in en]

    def persist(i, embeddings):
        rows = range(i, min(i + BATCH, total))
        db.executemany(
            "INSERT OR REPLACE INTO quran_metadata VALUES (?,?,?,?,?)",
            [
                (j, int(surah[j]), int(ayah[j]), ar[j][:5000], en[j][:5000])
                for j in rows
            ],
        )
        db.executemany(
            "INSERT INTO quran_vec (rowid, embedding) VALUES (?,?)",
            [(j, serialize_f32(embeddings[j - i])) for j in rows],
        )

    logger.info("Indexing Quran (%d/%d)", start, total)
    if _run_batches(state, stop, texts, "Quran", start, persist):
        state.quran_ready = True
        logger.info("Quran index complete (%d)", total)


def _download_hadith(stop):
    records = []
    for url, name in HADITH_SOURCES:
        if stop.is_set():
            return None
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        logger.info("%s: %d hadiths", name, len(data["hadiths"]))
        short = name.split()[-1]
        for h in data["hadiths"]:
            number = str(h.get("hadithNumber", ""))
            records.append(
                {
                    "collection": name,
                    "hadith_number": number,
                    "text": h.get("text", "")[:9900],
                    "reference": f"{short} {number}",
                }
            )
    return records


def _index_hadith(state, stop):
    db = state.db
    target = db.execute("SELECT value FROM meta WHERE key='hadith_total'").fetchone()
    count = db.execute("SELECT COUNT(*) FROM hadith_vec").fetchone()[0]
    if target and count >= int(target[0]):
        state.hadith_ready = True
        return

    logger.info("Downloading hadith collections")
    records = _download_hadith(stop)
    if records is None:
        return
    total = len(records)

    if not target:
        db.execute("DELETE FROM hadith_vec")
        db.execute("DROP TABLE IF EXISTS hadith_metadata")
        _create_hadith_metadata(db)
        db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('hadith_total', ?)",
            [str(total)],
        )
        db.commit()
        count = 0

    texts = ["passage: " + (r["text"].strip()[:2000] or "[No text]") for r in records]

    def persist(i, embeddings):
        batch = records[i : i + BATCH]
        db.executemany(
            "INSERT INTO hadith_metadata (collection, hadith_number, text, reference) VALUES (?,?,?,?)",
            [
                (r["collection"], r["hadith_number"], r["text"], r["reference"])
                for r in batch
            ],
        )
        last = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        first = last - len(batch) + 1
        db.executemany(
            "INSERT INTO hadith_vec (rowid, embedding) VALUES (?,?)",
            [(first + k, serialize_f32(embeddings[k])) for k in range(len(batch))],
        )

    logger.info("Indexing Hadith (%d/%d)", count, total)
    if _run_batches(state, stop, texts, "Hadith", count, persist):
        state.hadith_ready = True
        logger.info("Hadith index complete (%d)", total)


def _index_all(state, stop):
    try:
        _index_quran(state, stop)
        if stop.is_set():
            return
        _index_hadith(state, stop)
        if not stop.is_set():
            state.index_progress = "Ready"
            logger.info("Indexing complete")
    except Exception:
        if not stop.is_set():
            logger.exception("Indexing failed")
            state.index_progress = "Error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Noor")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODELS_DIR)
    os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "huggingface"))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded (%dd)", EMBED_DIM)

    db = _connect()
    _init_schema(db)
    _ensure_model(db)

    df = pd.read_csv(CSV_PATH)
    df[COL_AR] = df[COL_AR].fillna("")
    df[COL_EN] = df[COL_EN].fillna("")
    logger.info("%d verses loaded", len(df))

    @lru_cache(maxsize=2048)
    def embed(text):
        return np.asarray(model.encode("query: " + text), dtype=np.float32)

    s = app.state
    s.model = model
    s.db = db
    s.df_verses = df
    s.embed = embed
    s.quran_ready = False
    s.hadith_ready = False
    s.index_progress = "Starting"
    s.cache = {}

    def _warm_and_index(state, ev):
        _index_all(state, ev)
        try:
            # Full warm-up (embed model + sqlite-vec index) - warming the model alone
            # still left the first vector MATCH cold (~6s). Run one real search.
            _vec_search(
                state.db, "quran_vec", "quran_metadata", "verse_id", QURAN_COLS,
                state.embed("نور"), 1,
            )
        except Exception:
            logger.exception("search warm-up failed")

    stop = threading.Event()
    thread = threading.Thread(target=_warm_and_index, args=(s, stop), daemon=True)
    thread.start()
    logger.info("API ready, indexing in background")

    yield

    stop.set()
    thread.join(timeout=5)
    db.close()


app = FastAPI(title="Noor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"https://noor.{os.environ.get('BASE_DOMAIN', 'pyxis3.ai')}"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _indexing(progress):
    return {"results": [], "total": 0, "indexing": True, "progress": progress}


def _format_context(verses, hadith):
    lines = [f"[{v['surah']}:{v['ayah']}] {v['translation']}" for v in verses]
    lines += [
        f"[{h['collection']} {h['hadith_number']}] {h['text'][:600]}" for h in hadith
    ]
    return "\n".join(lines)


def _aigw_chat(messages):
    body = json.dumps(
        {"model": AIGW_MODEL, "messages": messages, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{AIGW_URL}/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIGW_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=AIGW_TIMEOUT) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError) as e:
        logger.error("aigw call failed: %s", e)
        raise HTTPException(status_code=502, detail="Q&A backend unavailable")


def _stt(audio, filename, content_type):
    boundary = "----noortranscribe"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{STT_MODEL}\r\n'.encode()
        + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{STT_URL}/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("text", "").strip()
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        logger.error("stt call failed: %s", e)
        raise HTTPException(status_code=502, detail="Transcription unavailable")


@app.get("/api/health")
async def health(request: Request):
    s = request.app.state
    return {
        "status": "ready" if s.quran_ready else "indexing",
        "quran_ready": s.quran_ready,
        "hadith_ready": s.hadith_ready,
        "progress": s.index_progress,
    }


@app.get("/api/stats")
def stats(request: Request):
    s = request.app.state
    return {
        "total_verses": len(s.df_verses),
        "status": "Ready" if s.quran_ready else "Loading",
        "indexed_verses": s.db.execute("SELECT COUNT(*) FROM quran_vec").fetchone()[0],
        "total_hadiths": s.db.execute("SELECT COUNT(*) FROM hadith_vec").fetchone()[0],
    }


@app.post("/api/search")
def search_verses(query: SearchQuery, request: Request):
    _rate_limit(request, 60)
    s = request.app.state
    if not s.quran_ready:
        return _indexing(s.index_progress)
    q = query.query.strip()
    if not q:
        return {"results": [], "total": 0}

    ref = re.match(r"^(\d+)[:\s]+(\d+)$", q)
    if ref:
        surah, ayah = int(ref.group(1)), int(ref.group(2))
        row = s.df_verses[
            (s.df_verses[COL_SURAH] == surah) & (s.df_verses[COL_AYAH] == ayah)
        ]
        if not row.empty:
            r = row.iloc[0]
            return {
                "results": [
                    {
                        "surah": surah,
                        "ayah": ayah,
                        "text": r[COL_AR],
                        "translation": r[COL_EN],
                        "score": 1.0,
                    }
                ],
                "total": 1,
            }

    hits = _vec_search(
        s.db,
        "quran_vec",
        "quran_metadata",
        "verse_id",
        QURAN_COLS,
        s.embed(q),
        query.limit * 2,
    )
    results = [
        r
        for r in hits
        if r["score"] >= query.similarity_threshold
        and (query.surah_filter is None or r["surah"] == query.surah_filter)
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[: query.limit]
    return {"results": results, "total": len(results)}


@app.post("/api/hadith/search")
def search_hadith(query: SearchQuery, request: Request):
    _rate_limit(request, 60)
    s = request.app.state
    if not s.hadith_ready:
        return _indexing(s.index_progress)
    q = query.query.strip()
    if not q:
        return {"results": [], "total": 0}
    hits = _vec_search(
        s.db,
        "hadith_vec",
        "hadith_metadata",
        "hadith_id",
        HADITH_COLS,
        s.embed(q),
        query.limit,
    )
    results = [r for r in hits if r["score"] >= query.similarity_threshold]
    return {"results": results, "total": len(results)}


@app.post("/api/qa")
def qa(query: QAQuery, request: Request):
    _rate_limit(request, 20)
    s = request.app.state
    if not s.quran_ready:
        return {
            "answer": "",
            "citations": [],
            "indexing": True,
            "progress": s.index_progress,
        }
    q = query.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    embedding = s.embed(q)
    verses = _vec_search(
        s.db,
        "quran_vec",
        "quran_metadata",
        "verse_id",
        QURAN_COLS,
        embedding,
        query.limit,
    )
    hadith = (
        _vec_search(
            s.db,
            "hadith_vec",
            "hadith_metadata",
            "hadith_id",
            HADITH_COLS,
            embedding,
            3,
        )
        if s.hadith_ready
        else []
    )
    if not verses and not hadith:
        return {
            "answer": "No relevant passages were found for that question.",
            "citations": [],
        }
    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {q}\n\nPassages:\n{_format_context(verses, hadith)}",
        },
    ]
    return {"answer": _aigw_chat(messages), "citations": verses + hadith}


@app.post("/api/transcribe")
def transcribe(query: TranscribeQuery, request: Request):
    _rate_limit(request, 15)
    try:
        audio = base64.b64decode(query.audio, validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid audio")
    return {"text": _stt(audio, "audio.webm", query.content_type)}


@app.post("/api/count")
def count_word(query: CountQuery, request: Request):
    df = request.app.state.df_verses
    word = query.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    combined = df[COL_AR] + " " + df[COL_EN]
    counts = combined.str.count(
        re.escape(word), flags=0 if query.case_sensitive else re.IGNORECASE
    )
    mask = counts > 0
    examples = [
        {
            "surah": int(df.at[idx, COL_SURAH]),
            "ayah": int(df.at[idx, COL_AYAH]),
            "text": df.at[idx, COL_AR][:100] + "..."
            if len(df.at[idx, COL_AR]) > 100
            else df.at[idx, COL_AR],
            "matches": int(counts[idx]),
        }
        for idx in df.index[mask][:10]
    ]
    return {
        "word": word,
        "count": int(counts.sum()),
        "examples": examples,
        "total_verses_with_word": int(mask.sum()),
    }


@app.get("/api/analytics/surah")
def surah_analytics(request: Request):
    df = request.app.state.df_verses
    sizes = df.groupby(COL_SURAH).size()
    return {
        "total_surahs": int(sizes.size),
        "total_verses": int(len(df)),
        "average_verses_per_surah": round(len(df) / sizes.size, 1),
        "longest_surah": {"number": int(sizes.idxmax()), "verses": int(sizes.max())},
    }


@app.get("/api/analytics/frequency")
def word_frequency(request: Request):
    df = request.app.state.df_verses
    words = re.findall(r"\b\w+\b", " ".join(df[COL_AR]).lower())
    return {
        "words": dict(Counter(words).most_common(20)),
        "total_unique_words": len(set(words)),
    }


@app.get("/api/analytics/distribution")
def theme_distribution(request: Request):
    en = request.app.state.df_verses[COL_EN]
    distribution = {
        theme: int(
            sum(en.str.contains(kw, case=False, regex=False).sum() for kw in keywords)
        )
        for theme, keywords in THEMES.items()
    }
    return {"themes": distribution}


@app.get("/api/analytics/themes")
def theme_clusters(request: Request):
    s = request.app.state
    if "themes" not in s.cache:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        sample = s.df_verses.sample(n=min(400, len(s.df_verses)), random_state=42)
        texts = ["passage: " + e for e in sample[COL_EN].astype(str)]
        embeddings = np.asarray(s.model.encode(texts, batch_size=32), dtype=np.float32)
        labels = KMeans(n_clusters=15, random_state=42, n_init=10).fit_predict(
            embeddings
        )

        themes = {}
        for cid in np.unique(labels):
            members = sample[labels == cid]
            try:
                vec = TfidfVectorizer(max_features=10, stop_words="english").fit(
                    members[COL_EN]
                )
                keywords = list(vec.get_feature_names_out()[:5])
            except ValueError:
                keywords = ["mixed"]
            themes[int(cid)] = {"keywords": keywords, "size": int(len(members))}
        s.cache["themes"] = themes
    return {"themes": s.cache["themes"]}


@app.get("/api/analytics/wordcloud")
def wordcloud(request: Request):
    s = request.app.state
    if "wordcloud" not in s.cache:
        from wordcloud import WordCloud

        image = (
            WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                max_words=100,
            )
            .generate(" ".join(s.df_verses[COL_EN]))
            .to_image()
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        s.cache["wordcloud"] = (
            "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        )
    return {"wordcloud_image": s.cache["wordcloud"]}


@app.get("/api/search/surah/{surah_no}")
def get_surah(surah_no: int, request: Request):
    df = request.app.state.df_verses
    verses = df[df[COL_SURAH] == surah_no]
    if verses.empty:
        raise HTTPException(status_code=404, detail=f"Surah {surah_no} not found")
    return {
        "surah": surah_no,
        "total_verses": int(len(verses)),
        "verses": [
            {"ayah": int(r[COL_AYAH]), "text": r[COL_AR], "translation": r[COL_EN]}
            for _, r in verses.iterrows()
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
