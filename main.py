#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import io
import base64
import struct
import sqlite3
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import re
from collections import Counter
import torch

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import umap
import networkx as nx
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Noor - Islamic Knowledge Search API",
    version="3.0.0",
    description="AI-powered Quran and Hadith search with semantic analysis and clustering"
)

# Enable CORS
base_domain = os.environ.get("BASE_DOMAIN", "jsr.bz")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"https://{base_domain}", f"https://noor.{base_domain}"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global variables for AI models and data
model = None
db = None
arabic_model = None
arabic_tokenizer = None
collection = None
hadith_collection = None
df_verses = None
df_hadith = None
verse_embeddings = None
cluster_model = None
theme_labels = None

# Advanced AI/ML helper functions
def prepare_arabic_text(text):
    # Prepare Arabic text for display
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except Exception:
        return text

def generate_theme_clusters(embeddings, n_clusters=20):
    # Generate thematic clusters from embeddings
    print(f"[{datetime.now()}] Running KMeans clustering with {n_clusters} clusters...", flush=True)
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        print(f"[{datetime.now()}] Clustering complete", flush=True)
        return kmeans, cluster_labels
    except Exception as e:
        print(f"[{datetime.now()}] Clustering error: {e}", flush=True)
        return None, np.zeros(len(embeddings))

def create_verse_network(embeddings, threshold=0.7):
    # Create network graph of semantically similar verses
    similarity_matrix = cosine_similarity(embeddings)
    G = nx.Graph()

    for i in range(len(embeddings)):
        G.add_node(i)
        for j in range(i+1, len(embeddings)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    return G

def analyze_themes(df, cluster_labels):
    # Analyze and label thematic clusters
    theme_analysis = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_verses = df[cluster_labels == cluster_id]
        # Extract key themes using TF-IDF
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_verses['ayah_en'].fillna(''))
            feature_names = vectorizer.get_feature_names_out()
            theme_analysis[int(cluster_id)] = {
                'keywords': list(feature_names[:5]),
                'size': len(cluster_verses),
                'sample_verses': cluster_verses.head(3)[['surah_no', 'ayah_no_surah', 'ayah_en']].to_dict('records')
            }
        except Exception:
            theme_analysis[int(cluster_id)] = {
                'keywords': ['mixed_theme'],
                'size': len(cluster_verses),
                'sample_verses': []
            }
    return theme_analysis

# Pydantic models
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

# SQLite-vec helpers
def serialize_f32(vector):
    """Serialize a vector to bytes for sqlite-vec"""
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32).tobytes()
    return struct.pack(f'{len(vector)}f', *vector)

def search_quran_vec(query_embedding, limit):
    """Search quran vectors and return results with metadata"""
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

def search_hadith_vec(query_embedding, limit):
    """Search hadith vectors and return results with metadata"""
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

# Embedding functions
def get_embedding(text):
    """Get embedding using local sentence-transformers model"""
    global model
    if not text or not str(text).strip():
        return np.zeros(384, dtype=np.float32)
    try:
        text_str = str(text).strip()[:5000]
        embedding = model.encode(text_str)
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.zeros(384, dtype=np.float32)

def get_batch_embeddings(texts):
    """Get embeddings for multiple texts using local model"""
    global model
    print(f"[{datetime.now()}] Generating embeddings for {len(texts)} texts...", flush=True)
    try:
        clean_texts = [str(t).strip()[:5000] if t and str(t).strip() else "" for t in texts]
        embeddings = model.encode(clean_texts, show_progress_bar=True, batch_size=64)
        print(f"[{datetime.now()}] Completed generating {len(embeddings)} embeddings", flush=True)
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return np.zeros((len(texts), 384), dtype=np.float32)

# Helper function to normalize Arabic text
def normalize_arabic(text):
    """Normalize Arabic text for better matching"""
    import unicodedata
    # Remove diacritics
    text = unicodedata.normalize('NFD', text)
    # Remove Arabic diacritical marks
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    # Replace special Arabic characters with standard ones
    replacements = {
        '\u0671': '\u0627',  # Alif with waslah to regular alif
        '\u0623': '\u0627',  # Alif with hamza above
        '\u0625': '\u0627',  # Alif with hamza below
        '\u0622': '\u0627',  # Alif with madda
        '\u0670': '',   # Arabic small high ligature alif
        '\u06DA': '',   # Arabic small high jeem
        '\u06D6': '',   # Arabic small high ligature sad with lam with alif maksura
        '\u06D7': '',   # Arabic small high ligature qaf with lam with alif maksura
        '\u06D9': '',   # Arabic small high meem initial form
        '\u06DB': '',   # Arabic small high sad
        '\u06DC': '',   # Arabic small high ain
        '\u0649': '\u064A',  # Alif maksura to ya
        '\u0629': '\u0647',  # Ta marbuta to ha
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Global embedding function with dimension matching
def get_hybrid_embedding(text, target_dim=384):
    """Generate hybrid embedding combining Arabic-BERT + multilingual"""
    global model, arabic_model, arabic_tokenizer
    try:
        # Check if text contains Arabic characters
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)

        # Normalize Arabic text if present
        if has_arabic:
            text = normalize_arabic(text)

        if arabic_model and arabic_tokenizer and has_arabic:
            # Use Arabic-BERT for Arabic text (768 dimensions)
            inputs = arabic_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = arabic_model(**inputs)
                # Get CLS token embedding (768 dimensions)
                arabic_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

            # Ensure correct dimension - ALWAYS resize to target_dim
            if len(arabic_embedding) != target_dim:
                if len(arabic_embedding) > target_dim:
                    return arabic_embedding[:target_dim]
                else:
                    return np.pad(arabic_embedding, (0, target_dim - len(arabic_embedding)))
            else:
                return arabic_embedding
        else:
            # Use local sentence-transformers model (384 dimensions natively)
            return get_embedding(text)

    except Exception as e:
        print(f"Hybrid embedding error: {e}")
        return get_embedding(text)

@app.on_event("startup")
async def startup_event():
    global model, arabic_model, arabic_tokenizer, collection, db, df_verses, verse_embeddings, cluster_model, theme_labels, hadith_collection, df_hadith

    print("Starting Noor - Islamic Knowledge Search API...", flush=True)
    print(f"[{datetime.now()}] Starting startup_event function", flush=True)

    # Set model cache directory to PVC for persistence
    os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', '/config/models')
    os.environ.setdefault('HF_HOME', '/config/models/huggingface')
    os.environ.setdefault('TRANSFORMERS_CACHE', '/config/models/huggingface')

    # Load Arabic-BERT model
    print(f"[{datetime.now()}] Loading Arabic-optimized transformers...", flush=True)
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"[{datetime.now()}] Loading Arabic-BERT model...", flush=True)
        arabic_tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
        print(f"[{datetime.now()}] Tokenizer loaded", flush=True)
        arabic_model = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')
        print(f"[{datetime.now()}] Arabic-BERT loaded successfully", flush=True)
    except Exception as e:
        print(f"Arabic-BERT failed, using fallback: {e}", flush=True)
        arabic_model = None
        arabic_tokenizer = None

    # Load sentence-transformers model for embeddings
    print(f"[{datetime.now()}] Loading sentence-transformers model...", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"[{datetime.now()}] sentence-transformers model loaded (all-MiniLM-L6-v2, 384d)", flush=True)
    except Exception as e:
        print(f"sentence-transformers failed: {e}", flush=True)
        # Emergency fallback to TF-IDF
        class TfidfModel:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=384)
                self.fitted = False

            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                if not self.fitted:
                    self.vectorizer.fit(texts)
                    self.fitted = True
                sparse = self.vectorizer.transform(texts)
                dense = sparse.todense()
                result = np.zeros((len(texts), 384))
                result[:, :min(384, dense.shape[1])] = dense[:, :min(384, dense.shape[1])]
                if len(texts) == 1:
                    return result[0].astype(np.float32)
                return result.astype(np.float32)

        model = TfidfModel()
        print("Using TF-IDF embeddings as fallback", flush=True)

    # Initialize SQLite with sqlite-vec
    print(f"[{datetime.now()}] Initializing SQLite vector database...", flush=True)
    import sqlite_vec
    db = sqlite3.connect('/config/vectors.db', check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Create tables
    db.execute('''CREATE TABLE IF NOT EXISTS quran_metadata (
        verse_id INTEGER PRIMARY KEY,
        surah INTEGER,
        ayah INTEGER,
        text TEXT,
        translation TEXT
    )''')
    db.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS quran_vec USING vec0(
        embedding float[384] distance_metric=cosine
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS hadith_metadata (
        hadith_id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT,
        hadith_number TEXT,
        text TEXT,
        reference TEXT
    )''')
    db.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS hadith_vec USING vec0(
        embedding float[384] distance_metric=cosine
    )''')
    db.commit()
    print(f"[{datetime.now()}] SQLite tables ready", flush=True)

    # Load dataset
    print(f"[{datetime.now()}] Loading Quran dataset...", flush=True)
    dataset_path = '/config/quran-dataset.csv'
    df_verses = pd.read_csv(dataset_path)
    print(f"[{datetime.now()}] Loaded {len(df_verses)} verses from local file", flush=True)

    # Check if quran vectors already exist (skip rebuild on restart)
    quran_count = db.execute('SELECT COUNT(*) FROM quran_vec').fetchone()[0]

    if quran_count == 0:
        print(f"[{datetime.now()}] Building quran vector index from scratch...", flush=True)

        # Prepare texts for embedding
        texts = []
        verse_ids = []
        for idx, row in df_verses.iterrows():
            arabic_text = row.get('ayah_ar', row.get('text', ''))
            english_text = row.get('ayah_en', row.get('translation', ''))
            combined_text = f"{arabic_text} {english_text}" if pd.notna(english_text) else str(arabic_text)
            texts.append(combined_text)
            verse_ids.append(idx)

        # Generate embeddings locally (much faster than HTTP calls)
        print(f"[{datetime.now()}] Generating {len(texts)} embeddings (384d)...", flush=True)
        embeddings = get_batch_embeddings(texts)
        print(f"[{datetime.now()}] Generated {len(embeddings)} embeddings with shape {embeddings.shape}", flush=True)

        # Insert into SQLite
        print(f"[{datetime.now()}] Inserting verses into SQLite...", flush=True)
        for i, idx in enumerate(verse_ids):
            row = df_verses.iloc[idx]
            surah = int(row.get('surah_no', row.get('surah', 1)))
            ayah = int(row.get('ayah_no_surah', row.get('ayah', 1)))
            text = str(row.get('ayah_ar', row.get('text', '')))[:5000]
            translation = str(row.get('ayah_en', row.get('translation', '')))[:5000]

            db.execute(
                'INSERT OR REPLACE INTO quran_metadata (verse_id, surah, ayah, text, translation) VALUES (?, ?, ?, ?, ?)',
                [i, surah, ayah, text, translation]
            )
            db.execute(
                'INSERT INTO quran_vec (rowid, embedding) VALUES (?, ?)',
                [i, serialize_f32(embeddings[i])]
            )

            if (i + 1) % 1000 == 0:
                print(f"[{datetime.now()}]   Inserted {i + 1}/{len(verse_ids)} verses...", flush=True)

        db.commit()
        quran_count = len(verse_ids)
        print(f"[{datetime.now()}] Quran vector index built with {quran_count} verses", flush=True)
    else:
        print(f"[{datetime.now()}] Quran vectors already indexed ({quran_count} vectors), skipping rebuild", flush=True)

    collection = True  # Flag: quran vectors ready
    print(f"[{datetime.now()}] Quran vector database ready!", flush=True)

    # Initialize advanced analytics
    try:
        print(f"[{datetime.now()}] Initializing advanced analytics...", flush=True)

        # Generate embeddings for clustering (use subset for performance)
        sample_size = min(1000, len(df_verses))
        sample_df = df_verses.sample(n=sample_size, random_state=42)

        combined_texts = [f"{row['ayah_ar']} {row['ayah_en']}"
                        for _, row in sample_df.iterrows()]
        verse_embeddings = get_batch_embeddings(combined_texts)

        # Generate theme clusters
        print(f"[{datetime.now()}] Generating thematic clusters...", flush=True)
        cluster_model, theme_labels = generate_theme_clusters(verse_embeddings, n_clusters=15)
        print(f"[{datetime.now()}] Advanced AI analytics ready!", flush=True)

        # Initialize Hadith collection
        print(f"[{datetime.now()}] Initializing Hadith Database...", flush=True)

        hadith_count = db.execute('SELECT COUNT(*) FROM hadith_vec').fetchone()[0]

        if hadith_count == 0:
            try:
                # Download Hadith datasets
                print(f"[{datetime.now()}] Downloading Hadith collections...", flush=True)
                import urllib.request

                bukhari_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.json"
                muslim_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.json"

                print(f"[{datetime.now()}] Downloading Bukhari...", flush=True)
                with urllib.request.urlopen(bukhari_url) as response:
                    bukhari_data = json.loads(response.read())
                print(f"[{datetime.now()}] Downloaded {len(bukhari_data['hadiths'])} Bukhari hadiths", flush=True)

                print(f"[{datetime.now()}] Downloading Muslim...", flush=True)
                with urllib.request.urlopen(muslim_url) as response:
                    muslim_data = json.loads(response.read())
                print(f"[{datetime.now()}] Downloaded {len(muslim_data['hadiths'])} Muslim hadiths", flush=True)

                # Build hadith records
                hadith_records = []
                for hadith in bukhari_data['hadiths']:
                    text = hadith.get('text', '')
                    if len(text) > 9900:
                        text = text[:9897] + "..."
                    hadith_records.append({
                        'collection': 'Sahih Bukhari',
                        'hadith_number': str(hadith.get('hadithNumber', '')),
                        'text': text,
                        'reference': f"Bukhari {hadith.get('hadithNumber', '')}"
                    })

                for hadith in muslim_data['hadiths']:
                    text = hadith.get('text', '')
                    if len(text) > 9900:
                        text = text[:9897] + "..."
                    hadith_records.append({
                        'collection': 'Sahih Muslim',
                        'hadith_number': str(hadith.get('hadithNumber', '')),
                        'text': text,
                        'reference': f"Muslim {hadith.get('hadithNumber', '')}"
                    })

                df_hadith = pd.DataFrame(hadith_records)
                print(f"Loaded {len(df_hadith)} hadiths total", flush=True)

                # Generate embeddings and insert in batches
                print("Generating Hadith embeddings...", flush=True)
                hadith_texts = [
                    str(text) if text and str(text).strip() else "[No text available]"
                    for text in df_hadith['text'].tolist()
                ]

                batch_size = 500
                for i in range(0, len(hadith_texts), batch_size):
                    try:
                        batch_texts = hadith_texts[i:i+batch_size]
                        batch_embeddings = get_batch_embeddings(batch_texts)
                        batch_df = df_hadith.iloc[i:i+len(batch_texts)]

                        for j in range(len(batch_texts)):
                            row = batch_df.iloc[j]
                            text = str(row['text']) if row['text'] and str(row['text']).strip() else "[No text available]"

                            db.execute(
                                'INSERT INTO hadith_metadata (collection_name, hadith_number, text, reference) VALUES (?, ?, ?, ?)',
                                [row['collection'], str(row['hadith_number']), text, row['reference']]
                            )
                            hadith_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
                            db.execute(
                                'INSERT INTO hadith_vec (rowid, embedding) VALUES (?, ?)',
                                [hadith_id, serialize_f32(batch_embeddings[j])]
                            )

                        if (i + batch_size) % 2000 == 0:
                            print(f"  Indexed {min(i + batch_size, len(hadith_texts))}/{len(hadith_texts)} hadiths...", flush=True)
                    except Exception as batch_error:
                        print(f"  Error inserting hadith batch {i//batch_size}: {batch_error}", flush=True)
                        continue

                db.commit()
                hadith_count = db.execute('SELECT COUNT(*) FROM hadith_vec').fetchone()[0]
                print(f"Hadith collection ready with {hadith_count} hadiths", flush=True)

            except Exception as e:
                print(f"Hadith initialization failed: {e}", flush=True)
                import traceback
                print(traceback.format_exc())
                hadith_collection = None
                df_hadith = None
        else:
            print(f"[{datetime.now()}] Hadith vectors already indexed ({hadith_count} vectors), skipping rebuild", flush=True)
            df_hadith = True  # Sentinel: hadiths are available

        hadith_collection = True
        print("Hadith database initialization complete!", flush=True)

    except Exception as e:
        print(f"Advanced analytics/hadith initialization failed: {e}", flush=True)
        import traceback
        print(traceback.format_exc())
        verse_embeddings = None
        cluster_model = None
        theme_labels = None

    print(f"[{datetime.now()}] STARTUP COMPLETE - API READY!", flush=True)

@app.get("/")
async def root():
    return {"message": "Noor API", "status": "ready", "verses": len(df_verses) if df_verses is not None else 0}

@app.get("/api")
async def api_root():
    return {"message": "Noor API", "version": "3.0", "endpoints": ["/api/search", "/api/hadith/search", "/api/qa", "/api/count", "/api/similar", "/api/tafsir", "/api/export", "/api/analytics/themes"]}

@app.post("/api/hadith/search")
async def search_hadith(query: SearchQuery):
    global model, hadith_collection

    if not hadith_collection or not model:
        raise HTTPException(status_code=400, detail="Hadith system not ready")

    try:
        query_embedding = get_embedding(query.query)
        results = search_hadith_vec(query_embedding, query.limit)

        search_results = [r for r in results if r['score'] > query.similarity_threshold]

        return {
            "results": search_results,
            "total": len(search_results),
            "query": query.query,
            "source": "hadith"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_verses(query: SearchQuery):
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=400, detail="System not fully loaded")

    try:
        # Handle empty queries gracefully
        if not query.query or query.query.strip() == "":
            return {
                "results": [],
                "total": 0,
                "message": "Please enter a search query"
            }

        # Check if query is a verse reference (e.g., "2:255" or "2 255")
        verse_pattern = re.match(r'^(\d+)[:\s]+(\d+)$', query.query.strip())
        if verse_pattern:
            surah_num = int(verse_pattern.group(1))
            ayah_num = int(verse_pattern.group(2))

            # Find the specific verse - handle both column name formats
            surah_col = 'surah_no' if 'surah_no' in df_verses.columns else 'surah'
            ayah_col = 'ayah_no_surah' if 'ayah_no_surah' in df_verses.columns else 'ayah'

            verse_row = df_verses[(df_verses[surah_col] == surah_num) &
                                 (df_verses[ayah_col] == ayah_num)]

            if not verse_row.empty:
                try:
                    result = {
                        "surah": int(verse_row.iloc[0]['surah_no']),
                        "ayah": int(verse_row.iloc[0]['ayah_no_surah']),
                        "text": verse_row.iloc[0]['ayah_ar'],
                        "translation": verse_row.iloc[0]['ayah_en'],
                        "score": 1.0  # Perfect match
                    }

                    return {
                        "results": [result],
                        "total": 1
                    }
                except KeyError as e:
                    logger.error(f"Column not found: {e}")
                    raise HTTPException(status_code=500, detail=f"Dataset column error: {e}")

        # Check if query contains Arabic
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in query.query)

        # If Arabic query, also do keyword search as fallback
        keyword_results = []
        if has_arabic:
            normalized_query = normalize_arabic(query.query)
            # Search in dataframe directly for Arabic text
            for idx, row in df_verses.iterrows():
                arabic_text = row.get('ayah_ar', '')
                if pd.notna(arabic_text):
                    normalized_arabic = normalize_arabic(arabic_text)
                    if normalized_query in normalized_arabic:
                        keyword_results.append({
                            "surah": int(row.get('surah_no', 1)),
                            "ayah": int(row.get('ayah_no_surah', 1)),
                            "text": arabic_text,
                            "translation": row.get('ayah_en', ''),
                            "score": 0.95  # High score for exact match
                        })
                        if len(keyword_results) >= query.limit:
                            break

        # Semantic vector search
        query_embedding = get_hybrid_embedding(query.query, target_dim=384)
        search_results = search_quran_vec(query_embedding, query.limit * 2)

        # Filter by threshold and surah
        filtered = []
        for r in search_results:
            if r['score'] >= query.similarity_threshold:
                if query.surah_filter and r['surah'] != query.surah_filter:
                    continue
                filtered.append(r)

        # Combine keyword results with search results
        if keyword_results:
            all_results = keyword_results + filtered
            seen = set()
            unique_results = []
            for result in all_results:
                key = (result['surah'], result.get('ayah', result.get('verse', 0)))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            filtered = unique_results

        filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:query.limit]
        return {"results": filtered, "total": len(filtered)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/search/advanced")
async def advanced_search(query: SearchQuery):
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    try:
        query_embedding = get_hybrid_embedding(query.query, target_dim=384)
        search_results_raw = search_quran_vec(query_embedding, query.limit * 3)

        search_results = []
        confidence_scores = []

        for r in search_results_raw:
            if r['score'] >= query.similarity_threshold:
                if query.surah_filter and r['surah'] != query.surah_filter:
                    continue

                result = {
                    "surah": r['surah'],
                    "ayah": r['ayah'],
                    "text": r['text'],
                    "translation": r['translation'],
                    "score": r['score'],
                    "confidence": "high" if r['score'] > 0.8 else "medium" if r['score'] > 0.6 else "low",
                    "relevance_percentage": round(r['score'] * 100, 1)
                }
                search_results.append(result)
                confidence_scores.append(r['score'])

        # Sort and limit results
        search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)[:query.limit]

        # Analytics
        analytics = {
            "total_matches": len(search_results),
            "avg_confidence": round(np.mean(confidence_scores) * 100, 1) if confidence_scores else 0,
            "high_confidence_matches": len([s for s in confidence_scores if s > 0.8]),
            "query_processed": query.query,
            "search_method": "semantic_vector"
        }

        return {
            "results": search_results,
            "analytics": analytics,
            "filters_applied": {
                "surah_filter": query.surah_filter,
                "similarity_threshold": query.similarity_threshold,
                "language": query.language
            }
        }

    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")

@app.post("/api/search/multi-vector")
async def multi_vector_search(query: dict):
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    try:
        query_text = query.get("query", "")
        limit = query.get("limit", 10)
        semantic_weight = query.get("semantic_weight", 0.7)
        keyword_weight = query.get("keyword_weight", 0.3)

        # Semantic search
        query_embedding = get_embedding(query_text)
        semantic_results = search_quran_vec(query_embedding, limit * 2)

        # Keyword search
        keyword_matches = []
        query_words = query_text.lower().split()

        for idx, verse in df_verses.iterrows():
            text_content = f"{verse.get('ayah_ar', '')} {verse.get('ayah_en', '')}".lower()
            keyword_score = sum(1 for word in query_words if word in text_content) / max(len(query_words), 1)

            if keyword_score > 0:
                keyword_matches.append({
                    "index": idx,
                    "score": keyword_score,
                    "verse": verse
                })

        keyword_matches = sorted(keyword_matches, key=lambda x: x['score'], reverse=True)[:limit]

        # Combine results with weighted scoring
        combined_results = {}

        # Add semantic results
        for r in semantic_results:
            verse_key = f"{r['surah']}:{r['ayah']}"
            combined_results[verse_key] = {
                "semantic_score": r['score'],
                "keyword_score": 0,
                "verse": r
            }

        # Add keyword results
        for match in keyword_matches:
            verse = match["verse"]
            verse_key = f"{verse.get('surah_no', verse.get('surah', 1))}:{verse.get('ayah_no_surah', verse.get('ayah', 1))}"

            if verse_key in combined_results:
                combined_results[verse_key]["keyword_score"] = match["score"]
            else:
                combined_results[verse_key] = {
                    "semantic_score": 0,
                    "keyword_score": match["score"],
                    "verse": {
                        "surah": int(verse.get('surah_no', verse.get('surah', 1))),
                        "ayah": int(verse.get('ayah_no_surah', verse.get('ayah', 1))),
                        "text": verse.get('ayah_ar', verse.get('text', '')),
                        "translation": verse.get('ayah_en', verse.get('translation', ''))
                    }
                }

        # Calculate final weighted scores
        final_results = []
        for verse_idx, data in combined_results.items():
            final_score = (semantic_weight * data["semantic_score"]) + (keyword_weight * data["keyword_score"])
            verse = data["verse"]

            if final_score > 0.1:  # Minimum threshold
                final_results.append({
                    "surah": int(verse.get('surah', 1)),
                    "ayah": int(verse.get('ayah', 1)),
                    "text": verse.get('text', ''),
                    "translation": verse.get('translation', ''),
                    "final_score": round(final_score, 3),
                    "semantic_score": round(data["semantic_score"], 3),
                    "keyword_score": round(data["keyword_score"], 3),
                    "search_method": "multi_vector_hybrid"
                })

        # Sort by final score
        final_results = sorted(final_results, key=lambda x: x['final_score'], reverse=True)[:limit]

        return {
            "results": final_results,
            "search_config": {
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                "total_combined": len(combined_results)
            },
            "performance": {
                "semantic_matches": len(semantic_results),
                "keyword_matches": len(keyword_matches),
                "final_results": len(final_results)
            }
        }

    except Exception as e:
        logger.error(f"Multi-vector search error: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-vector search failed: {str(e)}")

@app.post("/api/count")
async def count_word_occurrences(request: CountQuery):
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    try:
        word = request.word.strip()
        if not word:
            raise HTTPException(status_code=400, detail="Word cannot be empty")

        count = 0
        examples = []
        flags = 0 if request.case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(word), flags)

        for _, verse in df_verses.iterrows():
            text = verse.get('ayah_ar', verse.get('text', ''))
            translation = verse.get('ayah_en', verse.get('translation', ''))
            text = text if pd.notna(text) else ""
            translation = translation if pd.notna(translation) else ""
            combined_text = f"{text} {translation}"

            matches = len(pattern.findall(combined_text))
            if matches > 0:
                count += matches
                examples.append({
                    "surah": int(verse.get('surah_no', verse.get('surah', 1))),
                    "ayah": int(verse.get('ayah_no_surah', verse.get('ayah', 1))),
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "matches": matches
                })

        return {
            "word": word,
            "count": count,
            "examples": examples[:10],
            "total_verses_with_word": len(examples)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Count failed: {str(e)}")

@app.post("/api/qa/islamic")
async def islamic_qa(query: QAQuery):
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    try:
        query_embedding = get_hybrid_embedding(query.question, target_dim=384)
        results = search_quran_vec(query_embedding, query.context_limit)

        relevant_verses = []
        total_relevance = 0

        for r in results:
            total_relevance += r['score']
            relevant_verses.append({
                "surah": r['surah'],
                "ayah": r['ayah'],
                "text": r['text'],
                "translation": r['translation'],
                "relevance": r['score'],
                "confidence": "high" if r['score'] > 0.7 else "medium" if r['score'] > 0.5 else "low"
            })

        # Enhanced answer generation
        question_type = "guidance" if any(word in query.question.lower() for word in ["how", "should", "guide"]) else "definition" if any(word in query.question.lower() for word in ["what", "who", "define"]) else "general"

        answer_intro = {
            "guidance": "The Quran provides guidance on this matter through these verses:",
            "definition": "The Quran describes this concept in the following way:",
            "general": "Based on the Quranic teachings, regarding your question:"
        }

        context = " ".join([v.get('translation', '') for v in relevant_verses[:3]])
        answer = f"{answer_intro.get(question_type, answer_intro['general'])} {context[:300]}..."

        return {
            "question": query.question,
            "answer": answer,
            "question_type": question_type,
            "relevant_verses": relevant_verses,
            "confidence": round((total_relevance / query.context_limit) * 100, 1) if query.context_limit > 0 else 0,
            "sources_count": len(relevant_verses),
            "high_confidence_sources": len([v for v in relevant_verses if v["confidence"] == "high"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Islamic QA error: {e}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

@app.post("/api/arabic/analyze")
async def analyze_arabic_text(text_data: dict):
    try:
        text = text_data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Character analysis
        arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
        diacritics = len([c for c in text if '\u064B' <= c <= '\u065F'])

        # Word analysis
        words = text.split()
        word_count = len(words)

        # Basic morphological patterns
        definite_articles = len([w for w in words if w.startswith('\u0627\u0644')])  # al-
        conjunctions = len([w for w in words if w.startswith('\u0648')])  # wa

        # Linguistic features
        features = {
            "character_analysis": {
                "total_chars": len(text),
                "arabic_chars": arabic_chars,
                "diacritics": diacritics,
                "diacritic_ratio": round(diacritics / max(arabic_chars, 1), 3)
            },
            "word_analysis": {
                "word_count": word_count,
                "avg_word_length": round(sum(len(w) for w in words) / max(word_count, 1), 2),
                "unique_words": len(set(words))
            },
            "morphological": {
                "definite_articles": definite_articles,
                "conjunctions": conjunctions,
                "article_frequency": round(definite_articles / max(word_count, 1), 3)
            },
            "text_type": "classical" if diacritics > arabic_chars * 0.1 else "modern"
        }

        # Simplified root analysis for common patterns
        trilateral_patterns = []
        for word in words[:5]:  # Analyze first 5 words
            clean_word = ''.join([c for c in word if '\u0621' <= c <= '\u064A'])
            if len(clean_word) >= 3:
                trilateral_patterns.append({
                    "word": word,
                    "clean": clean_word,
                    "length": len(clean_word),
                    "pattern": "CCC" if len(clean_word) == 3 else "CCCC+" if len(clean_word) > 3 else "CC"
                })

        features["trilateral_analysis"] = trilateral_patterns

        return {
            "analysis": features,
            "summary": f"Arabic text with {arabic_chars} Arabic characters, {word_count} words, {diacritics} diacritics",
            "linguistic_score": round((arabic_chars / max(len(text), 1)) * 100, 1)
        }

    except Exception as e:
        logger.error(f"Arabic analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Arabic analysis failed: {str(e)}")

@app.get("/api/analytics/surah")
async def surah_analytics():
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    try:
        # Handle both dataset formats
        surah_col = 'surah_no' if 'surah_no' in df_verses.columns else 'surah'
        surah_stats = df_verses.groupby(surah_col).size().reset_index(name='verse_count')
        total_surahs = len(surah_stats)
        total_verses = len(df_verses)
        longest_surah = surah_stats.loc[surah_stats['verse_count'].idxmax()]

        return {
            "total_surahs": total_surahs,
            "total_verses": total_verses,
            "average_verses_per_surah": round(total_verses / total_surahs, 1),
            "longest_surah": {
                "number": int(longest_surah[surah_col]),
                "verses": int(longest_surah['verse_count'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.get("/api/analytics/frequency")
async def word_frequency_analytics():
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    try:
        # Handle both dataset formats
        text_col = 'ayah_ar' if 'ayah_ar' in df_verses.columns else 'text'
        all_text = " ".join(df_verses[text_col].fillna(''))
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = dict(Counter(words).most_common(20))

        return {"words": word_freq, "total_unique_words": len(set(words))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frequency analysis failed: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    global df_verses, collection, hadith_collection, db

    try:
        quran_count = db.execute('SELECT COUNT(*) FROM quran_vec').fetchone()[0] if db else 0
        hadith_count = db.execute('SELECT COUNT(*) FROM hadith_vec').fetchone()[0] if db else 0
        stats = {
            "total_verses": len(df_verses) if df_verses is not None else 0,
            "status": "Ready" if collection and df_verses is not None else "Loading",
            "indexed_verses": quran_count,
            "total_hadiths": hadith_count,
            "indexed_hadiths": hadith_count
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/surah/{surah_no}")
async def get_surah_info(surah_no: int):
    """Get information about a specific surah"""
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    if surah_no < 1 or surah_no > 114:
        raise HTTPException(status_code=400, detail="Invalid surah number. Must be between 1 and 114")

    try:
        surah_verses = df_verses[df_verses['surah_no'] == surah_no]
        if surah_verses.empty:
            raise HTTPException(status_code=404, detail=f"Surah {surah_no} not found")

        first_verse = surah_verses.iloc[0]
        return {
            "surah_no": surah_no,
            "name_en": first_verse.get('surah_name_en', f'Surah {surah_no}'),
            "name_ar": first_verse.get('surah_name_ar', ''),
            "total_verses": len(surah_verses),
            "place_of_revelation": first_verse.get('place_of_revelation', ''),
            "juz_no": int(first_verse.get('juz_no', 0)) if pd.notna(first_verse.get('juz_no')) else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching surah info: {str(e)}")

@app.get("/api/juz/{juz_no}")
async def get_juz_info(juz_no: int):
    """Get verses in a specific juz"""
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    if juz_no < 1 or juz_no > 30:
        raise HTTPException(status_code=400, detail="Invalid juz number. Must be between 1 and 30")

    try:
        juz_verses = df_verses[df_verses['juz_no'] == juz_no]
        if juz_verses.empty:
            raise HTTPException(status_code=404, detail=f"Juz {juz_no} not found")

        return {
            "juz_no": juz_no,
            "total_verses": len(juz_verses),
            "verses": [
                {
                    "reference": f"{int(row['surah_no'])}:{int(row['ayah_no_surah'])}",
                    "arabic": row.get('ayah_ar', ''),
                    "translation": row.get('ayah_en', '')
                }
                for _, row in juz_verses.head(10).iterrows()  # Return first 10 verses as sample
            ],
            "surahs_included": sorted(juz_verses['surah_no'].unique().tolist())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching juz info: {str(e)}")

# ==================== ADVANCED AI/ML ENDPOINTS ====================

@app.get("/api/analytics/themes")
async def get_theme_clusters():
    # Get thematic clusters analysis
    global df_verses, verse_embeddings, cluster_model, theme_labels

    if df_verses is None or verse_embeddings is None:
        raise HTTPException(status_code=503, detail="Analytics data not available")

    try:
        # Generate theme analysis
        sample_df = df_verses.sample(n=min(1000, len(df_verses)), random_state=42)
        theme_analysis = analyze_themes(sample_df, theme_labels)

        return {
            "themes": theme_analysis,
            "total_themes": len(theme_analysis),
            "sample_size": len(verse_embeddings)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/embeddings/visualization")
async def get_embeddings_visualization():
    # Generate 2D visualization of verse embeddings using UMAP
    global verse_embeddings, theme_labels

    if verse_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not available")

    try:
        umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d = umap_model.fit_transform(verse_embeddings)

        # Create visualization data
        viz_data = {
            "embeddings_2d": embedding_2d.tolist(),
            "cluster_labels": theme_labels.tolist() if theme_labels is not None else None,
            "method": "UMAP",
            "dimensions": embedding_2d.shape
        }

        return viz_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analytics/similarity/network")
async def create_similarity_network(threshold: float = 0.8):
    # Create network graph of semantically similar verses
    global verse_embeddings, df_verses

    if verse_embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not available")

    try:
        G = create_verse_network(verse_embeddings, threshold=threshold)

        # Extract network data
        nodes = []
        edges = []

        for node in G.nodes():
            nodes.append({
                "id": int(node),
                "label": f"Verse {node+1}",
                "cluster": int(theme_labels[node]) if theme_labels is not None else 0
            })

        for edge in G.edges(data=True):
            edges.append({
                "source": int(edge[0]),
                "target": int(edge[1]),
                "weight": float(edge[2]["weight"])
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "threshold": threshold,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "density": nx.density(G)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/wordcloud")
async def generate_wordcloud():
    # Generate word cloud from Quran text
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=503, detail="Dataset not available")

    try:
        all_text = " ".join(df_verses['ayah_en'].fillna('').tolist())

        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(all_text)

        # Convert to base64 for web display
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)

        img_str = base64.b64encode(img_buffer.read()).decode()

        return {
            "wordcloud_image": f"data:image/png;base64,{img_str}",
            "word_frequencies": dict(wordcloud.words_)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analytics/export")
async def export_search_results(query: dict):
    # Export search results in various formats
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=503, detail="System not ready")

    try:
        search_query = query.get("query", "")
        export_format = query.get("format", "json")  # json, csv
        limit = query.get("limit", 50)

        # Perform search
        query_embedding = get_embedding(search_query)
        results = search_quran_vec(query_embedding, limit)

        # Process results
        export_data = []
        for r in results:
            export_data.append({
                "surah": r['surah'],
                "ayah": r['ayah'],
                "arabic_text": r['text'],
                "english_translation": r['translation'],
                "similarity_score": r['score'],
                "search_query": search_query
            })

        if export_format == "csv":
            # Convert to CSV format
            import csv

            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)

            csv_content = output.getvalue()
            output.close()

            return {
                "format": "csv",
                "content": csv_content,
                "filename": f"noor_search_{re.sub(r'[^a-zA-Z0-9]', '_', search_query[:20])}.csv",
                "count": len(export_data)
            }
        else:
            return {
                "format": "json",
                "data": export_data,
                "filename": f"noor_search_{re.sub(r'[^a-zA-Z0-9]', '_', search_query[:20])}.json",
                "count": len(export_data)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ ADDITIONAL ENDPOINTS FOR MISSING FEATURES ============

@app.post("/api/verse")
async def get_verse(request: Dict[str, Any]):
    """Get a specific verse by reference (e.g., '2:255' or '114:1')"""
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    reference = request.get("reference", "")
    if not reference:
        raise HTTPException(status_code=400, detail="Reference required")

    # Parse reference (e.g., "2:255" or "2 255")
    match = re.match(r'^(\d+)[:\s]+(\d+)$', reference.strip())
    if not match:
        raise HTTPException(status_code=400, detail="Invalid reference format. Use 'surah:ayah' (e.g., '2:255')")

    surah_num = int(match.group(1))
    ayah_num = int(match.group(2))

    # Find the verse
    surah_col = 'surah_no' if 'surah_no' in df_verses.columns else 'surah'
    ayah_col = 'ayah_no_surah' if 'ayah_no_surah' in df_verses.columns else 'ayah'

    verse_row = df_verses[(df_verses[surah_col] == surah_num) &
                          (df_verses[ayah_col] == ayah_num)]

    if verse_row.empty:
        raise HTTPException(status_code=404, detail=f"Verse {surah_num}:{ayah_num} not found")

    # Return verse details
    row = verse_row.iloc[0]
    return {
        "verse": {
            "reference": f"{surah_num}:{ayah_num}",
            "surah": int(row.get('surah_no', surah_num)),
            "ayah": int(row.get('ayah_no_surah', ayah_num)),
            "arabic": row.get('ayah_ar', ''),
            "translation": row.get('ayah_en', ''),
            "surah_name": row.get('surah_name_en', ''),
            "surah_name_arabic": row.get('surah_name_ar', ''),
            "juz": int(row.get('juz_no', 0)) if pd.notna(row.get('juz_no')) else None,
            "place": row.get('place_of_revelation', '')
        }
    }

@app.post("/api/qa")
async def qa_endpoint(query: QAQuery):
    """Alias for /api/qa/islamic"""
    return await islamic_qa(query)

@app.post("/api/similar")
async def similar_verses(request: Dict[str, Any]):
    """Find similar verses using semantic search"""
    # Support both 'reference' and 'query' parameters
    query_text = request.get("reference", request.get("query", ""))
    limit = request.get("limit", 10)

    if not query_text:
        raise HTTPException(status_code=422, detail=[{"type": "missing", "loc": ["body", "query"], "msg": "Field required", "input": request}])

    # Convert to SearchQuery
    search_query = SearchQuery(query=query_text, limit=limit)
    result = await search_verses(search_query)

    # Rename 'results' to 'similar_verses' for API consistency
    if "results" in result:
        result["similar_verses"] = result.pop("results")

    return result

@app.post("/api/tafsir")
async def tafsir_endpoint(request: Dict[str, Any]):
    """Get tafsir (interpretation) for a verse"""
    # Support both reference format ("2:255") and separate surah/ayah
    reference = request.get("reference")
    if reference:
        # Parse reference like "2:255"
        match = re.match(r'^(\d+)[:\s]+(\d+)$', reference)
        if match:
            surah = int(match.group(1))
            ayah = int(match.group(2))
        else:
            raise HTTPException(status_code=400, detail="Invalid reference format")
    else:
        surah = request.get("surah")
        ayah = request.get("ayah")

    if not surah or not ayah:
        raise HTTPException(status_code=400, detail="Surah and ayah required (or use reference like '2:255')")

    global df_verses
    if df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    # Get the verse
    verse_row = df_verses[(df_verses['surah_no'] == surah) & (df_verses['ayah_no_surah'] == ayah)]

    if verse_row.empty:
        raise HTTPException(status_code=404, detail=f"Verse {surah}:{ayah} not found")

    verse_text = verse_row.iloc[0]['ayah_ar']
    verse_trans = verse_row.iloc[0].get('ayah_en', '')

    # Find semantically similar verses for context
    query_embedding = get_embedding(verse_trans if verse_trans else verse_text)
    similar = search_quran_vec(query_embedding, 5)
    related_verses = [
        {"reference": f"{r['surah']}:{r['ayah']}", "translation": r['translation'], "score": r['score']}
        for r in similar if not (r['surah'] == surah and r['ayah'] == ayah)
    ][:3]

    return {
        "reference": f"{surah}:{ayah}",
        "surah": surah,
        "ayah": ayah,
        "verse": verse_text,
        "translation": verse_trans,
        "related_verses": related_verses,
        "note": "Tafsir data not yet available. Showing semantically related verses instead."
    }

@app.get("/api/search/surah/{surah_no}")
async def get_surah_verses(surah_no: int):
    """Get all verses from a specific surah"""
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    surah_verses = df_verses[df_verses['surah_no'] == surah_no]

    if surah_verses.empty:
        raise HTTPException(status_code=404, detail=f"Surah {surah_no} not found")

    # Get surah name if available
    surah_name = surah_verses.iloc[0].get('surah_name_en', f'Surah {surah_no}')

    return {
        "surah": surah_no,
        "surah_name": surah_name,
        "total_verses": len(surah_verses),
        "verses": [
            {
                "ayah": int(row['ayah_no_surah']),
                "text": row['ayah_ar'],
                "translation": row.get('ayah_en', '')
            }
            for _, row in surah_verses.iterrows()
        ]
    }

@app.post("/api/export")
async def export_search(request: Dict[str, Any]):
    """Export search results"""
    result = await export_search_results(request)

    # Add 'data' wrapper for API consistency
    if "export_data" in result:
        result["data"] = result.pop("export_data")
    elif "results" in result:
        result["data"] = result.pop("results")

    # Add metadata if requested
    if request.get("include_metadata"):
        result["metadata"] = {
            "export_date": datetime.now().isoformat(),
            "total_results": len(result.get("data", [])),
            "query": request.get("query", "")
        }

    return result

@app.get("/api/visualize/similarity")
async def visualize_similarity(query: str = ""):
    """Get similarity visualization data"""
    if not query:
        return {"nodes": [], "edges": []}

    # Use the similarity network endpoint
    threshold = 0.5
    return await create_similarity_network(threshold)

@app.get("/api/analytics/distribution")
async def theme_distribution():
    """Get theme distribution across the Quran"""
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    # Theme distribution based on keywords
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

    distribution = {}
    for theme, keywords in themes.items():
        count = 0
        for keyword in keywords:
            count += df_verses['ayah_en'].str.contains(keyword, case=False, na=False).sum()
        distribution[theme] = int(count)

    return {
        "themes": distribution,
        "total_verses": len(df_verses),
        "chart_data": {
            "labels": list(distribution.keys()),
            "values": list(distribution.values())
        }
    }

@app.get("/api/health")
@app.get("/api/status")
async def health_status():
    """Health/status endpoint"""
    global collection, df_verses, db
    return {
        "status": "healthy",
        "collection_ready": collection is not None,
        "verses_loaded": df_verses is not None,
        "total_verses": len(df_verses) if df_verses is not None else 0,
        "sqlite_connected": db is not None,
        "endpoints": [
            "/api/search", "/api/qa", "/api/similar", "/api/tafsir",
            "/api/count", "/api/analytics/themes", "/api/export", "/api/hadith/search"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
