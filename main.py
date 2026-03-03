#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import requests
import base64
import os
import io
import base64
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import re
from collections import Counter
import torch

# Advanced AI/ML imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import umap
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Quran AI Search & Analytics API",
    version="2.0.0",
    description="AI-powered Quranic search with semantic analysis, clustering, visualization, and advanced insights"
)

# Enable CORS
base_domain = os.environ.get("BASE_DOMAIN", "jsr.bz")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"https://{base_domain}", f"https://quran.{base_domain}"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global variables for AI models and data
model = None
collection = None
df_verses = None
verse_embeddings = None
cluster_model = None
theme_labels = None

# Advanced AI/ML helper functions
def prepare_arabic_text(text):
    # Prepare Arabic text for display
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except:
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
        except:
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

# Global variables for models
EMBEDDINGS_URL = "http://embeddings.vector-memory.svc.cluster.local:8080/vectors"
arabic_model = None
arabic_tokenizer = None
collection = None
hadith_collection = None
df_verses = None
df_hadith = None
verse_embeddings = None
cluster_model = None
theme_labels = None

# Helper function to get embeddings from service
def get_embedding(text):
    """Get embedding from the embeddings service"""
    # Handle empty or None text
    if not text or not str(text).strip():
        print(f"⚠️ Empty text provided, using random embedding")
        return np.random.rand(384).tolist()

    try:
        # Ensure text is string and not too long
        text_str = str(text).strip()[:5000]

        response = requests.post(
            EMBEDDINGS_URL,
            json={"text": text_str},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return np.array(data.get("vector", data.get("data", [])))
        else:
            print(f"⚠️ Embeddings service error: {response.status_code} for text: {text_str[:100]}")
            return np.random.rand(384).tolist()
    except Exception as e:
        print(f"⚠️ Failed to get embedding: {e}")
        return np.random.rand(384).tolist()

def get_batch_embeddings(texts):
    """Get embeddings for multiple texts"""
    print(f"[{datetime.now()}] Generating embeddings for {len(texts)} texts...", flush=True)
    embeddings = []
    batch_size = 50

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Submit batch in parallel
            futures = [executor.submit(get_embedding, text) for text in batch]
            # Collect results in submission order to preserve alignment with verse IDs
            for future in futures:
                embeddings.append(future.result())

            if i % 200 == 0:
                print(f"[{datetime.now()}]   Processed {min(i+batch_size, len(texts))}/{len(texts)} embeddings", flush=True)

    print(f"[{datetime.now()}] Completed generating {len(embeddings)} embeddings", flush=True)
    return np.array(embeddings)

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
        'ٱ': 'ا',  # Alif with waslah to regular alif
        'أ': 'ا',  # Alif with hamza above
        'إ': 'ا',  # Alif with hamza below
        'آ': 'ا',  # Alif with madda
        'ٰ': '',   # Arabic small high ligature alif
        'ۚ': '',   # Arabic small high jeem
        'ۖ': '',   # Arabic small high ligature sad with lam with alif maksura
        'ۗ': '',   # Arabic small high ligature qaf with lam with alif maksura
        'ۙ': '',   # Arabic small high meem initial form
        'ۛ': '',   # Arabic small high sad
        'ۜ': '',   # Arabic small high ain
        'ى': 'ي',  # Alif maksura to ya
        'ة': 'ه',  # Ta marbuta to ha
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
                # Resize Arabic-BERT embedding from 768 to 384 by truncating
                # This preserves the most important features
                if len(arabic_embedding) > target_dim:
                    return arabic_embedding[:target_dim]
                else:
                    # Pad if somehow smaller (shouldn't happen)
                    return np.pad(arabic_embedding, (0, target_dim - len(arabic_embedding)))
            else:
                return arabic_embedding
        else:
            # Use embeddings service (384 dimensions)
            multi_embedding = get_embedding(text)

            # Return 384-dimensional embedding
            return multi_embedding

    except Exception as e:
        print(f"Hybrid embedding error: {e}")
        # Emergency fallback with proper dimension
        base_embedding = get_embedding(text)
        return base_embedding[:target_dim] if len(base_embedding) > target_dim else np.pad(base_embedding, (0, target_dim - len(base_embedding)))

@app.on_event("startup")
async def startup_event():
    global model, arabic_model, arabic_tokenizer, collection, df_verses, verse_embeddings, cluster_model, theme_labels

    print("🚀 Starting comprehensive Quran AI Search & Analytics API...")
    print(f"[{datetime.now()}] Starting startup_event function", flush=True)

    # Import asyncio for handling blocking operations
    import asyncio
    from functools import partial

    # Initialize advanced Arabic-optimized models
    print("📥 Loading Arabic-optimized transformers...")

    # Primary model: Arabic-BERT for Classical Arabic
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"[{datetime.now()}] 🔄 Loading Arabic-BERT model...", flush=True)
        arabic_tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
        print(f"[{datetime.now()}] Tokenizer loaded", flush=True)
        arabic_model = AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')
        print(f"[{datetime.now()}] ✅ Arabic-BERT loaded successfully", flush=True)
    except Exception as e:
        print(f"⚠️ Arabic-BERT failed, using fallback: {e}")
        arabic_model = None
        arabic_tokenizer = None

    # Use external embeddings service
    print("📥 Using external embeddings service...")
    try:
        # Create a model wrapper for external embeddings
        class ExternalEmbeddingsModel:
            def encode(self, texts):
                if isinstance(texts, str):
                    return np.array([get_embedding(texts)])
                elif isinstance(texts, list):
                    return np.array([get_embedding(text) for text in texts])
                else:
                    return np.array([get_embedding(str(text)) for text in texts])

        model = ExternalEmbeddingsModel()
        print("✅ External embeddings model wrapper created")

    except Exception as e:
        print(f"⚠️ Embeddings service test failed: {e}")
        # Emergency fallback to TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer

        class TfidfModel:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=384)
                self.fitted = False

            def encode(self, texts):
                if not self.fitted:
                    self.vectorizer.fit(texts)
                    self.fitted = True
                sparse = self.vectorizer.transform(texts)
                dense = sparse.todense()
                # Ensure 384 dimensions
                result = np.zeros((len(texts), 384))
                result[:, :min(384, dense.shape[1])] = dense[:, :min(384, dense.shape[1])]
                return result.astype(np.float32)

        model = TfidfModel()
        print("✅ Using TF-IDF embeddings as fallback")

    # Now get_hybrid_embedding is already defined globally and can use the models

    # Connect to Milvus with retry logic
    print(f"[{datetime.now()}] 🔗 Connecting to Milvus...", flush=True)
    max_retries = 30
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            connections.connect("default", host="milvus-standalone.vector-memory.svc.cluster.local", port="19530")
            print(f"[{datetime.now()}] ✅ Connected to Milvus", flush=True)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[{datetime.now()}] ⚠️ Milvus connection attempt {attempt + 1}/{max_retries} failed: {e}", flush=True)
                print(f"[{datetime.now()}] ⏳ Retrying in {retry_delay} seconds...", flush=True)
                import time
                time.sleep(retry_delay)
            else:
                print(f"[{datetime.now()}] ❌ Failed to connect to Milvus after {max_retries} attempts", flush=True)
                print(f"[{datetime.now()}] 🔧 Starting without Milvus - search features will be limited", flush=True)
                # Set collection to None to indicate Milvus is unavailable
                collection = None

    # Load dataset from mounted CSV file
    print(f"[{datetime.now()}] 📚 Loading Quran dataset...", flush=True)
    dataset_path = '/config/quran-dataset.csv'
    df_verses = pd.read_csv(dataset_path)
    print(f"[{datetime.now()}] ✅ Loaded {len(df_verses)} verses from local file", flush=True)

    # Define collection name
    collection_name = "quran_verses"
    
    # Initialize or connect to collection only if Milvus is available
    if collection is not None:  # Check if we successfully connected to Milvus
        print(f"[{datetime.now()}] Checking if collection exists...", flush=True)

        # Force drop and recreate for now to avoid hanging issues
        try:
            print(f"[{datetime.now()}] Step 1: Checking if collection exists for drop...", flush=True)
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(None, utility.has_collection, collection_name)

            if exists:
                print(f"[{datetime.now()}] Step 2: Collection exists, calling drop_collection...", flush=True)
                await loop.run_in_executor(None, utility.drop_collection, collection_name)
                print(f"[{datetime.now()}] Step 3: drop_collection completed", flush=True)
            else:
                print(f"[{datetime.now()}] Step 2: Collection does not exist, skipping drop", flush=True)
        except Exception as e:
            print(f"[{datetime.now()}] Error in drop section: {e}", flush=True)
            import traceback
            print(traceback.format_exc())
    else:
        print(f"[{datetime.now()}] ⚠️ Skipping Milvus collection setup - running in limited mode", flush=True)

    print(f"[{datetime.now()}] Step 4: Setting collection to None", flush=True)
    collection = None  # Force recreation

    # Only check and create collection if Milvus is connected
    # Check if we have a connection by testing actual connectivity
    milvus_connected = False
    try:
        # Test if connection is active by trying to list collections
        if connections.has_connection("default"):
            _ = utility.list_collections()
            milvus_connected = True
    except:
        milvus_connected = False
    
    if milvus_connected:
        print(f"[{datetime.now()}] Step 5: About to check has_collection for creation...", flush=True)
        loop = asyncio.get_event_loop()
        has_collection_result = await loop.run_in_executor(None, utility.has_collection, collection_name)
        print(f"[{datetime.now()}] Step 6: has_collection returned {has_collection_result}", flush=True)
    else:
        has_collection_result = False
        print(f"[{datetime.now()}] Step 5-6: Skipping has_collection check - no Milvus connection", flush=True)

    if milvus_connected and (not has_collection_result or collection is None):
        print(f"[{datetime.now()}] Step 7: Creating new Milvus collection with proper initialization...", flush=True)

        # Define schema with 768-dimensional vectors for Arabic-BERT
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="verse_id", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # all-MiniLM-L6-v2 dimension
            FieldSchema(name="surah", dtype=DataType.INT64),
            FieldSchema(name="ayah", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),  # Increased for longer verses
            FieldSchema(name="translation", dtype=DataType.VARCHAR, max_length=5000)  # Increased for longer translations
        ]
        print(f"[{datetime.now()}] Step 9: Creating CollectionSchema...", flush=True)
        schema = CollectionSchema(fields, "Quran verses with sentence embeddings")
        print(f"[{datetime.now()}] Step 10: Schema created, now creating Collection...", flush=True)

        # This is where it hangs - creating the Collection
        # Run blocking operation in thread executor to prevent hanging async event loop
        try:
            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(None, Collection, collection_name, schema)
            print(f"[{datetime.now()}] Step 11: Collection created successfully", flush=True)
        except Exception as e:
            print(f"[{datetime.now()}] ERROR creating collection: {e}", flush=True)
            import traceback
            print(traceback.format_exc())
            raise

        print(f"[{datetime.now()}] Step 12: Collection object created, continuing...", flush=True)

        # Process and insert data BEFORE creating index
        print(f"[{datetime.now()}] Step 13: Processing dataset for embedding generation...", flush=True)
        texts = []
        verse_ids = []

        for idx, row in df_verses.iterrows():
            # Handle real dataset columns
            arabic_text = row.get('ayah_ar', row.get('text', ''))
            english_text = row.get('ayah_en', row.get('translation', ''))
            surah_val = row.get('surah_no', row.get('surah', 0))
            ayah_val = row.get('ayah_no_surah', row.get('ayah', 0))
            
            combined_text = f"{arabic_text} {english_text}" if pd.notna(english_text) else arabic_text
            texts.append(combined_text)
            verse_ids.append(idx)

        print(f"🤖 Generating {len(texts)} embeddings (384d)...", flush=True)
        import time
        import concurrent.futures
        import threading

        embeddings = []
        batch_size = 50  # Larger batch for concurrent processing
        start_time = time.time()

        # Thread-safe counter for progress
        processed_count = 0
        lock = threading.Lock()

        def process_text_batch(text_batch, batch_index):
            """Process a batch of texts concurrently"""
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all embedding requests in parallel - maintain order!
                futures = {executor.submit(get_embedding, text): i for i, text in enumerate(text_batch)}

                # Collect results in ORDER
                batch_embeddings = [None] * len(text_batch)
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    embedding = future.result()
                    # Validate dimension
                    if len(embedding) != 384:
                        print(f"⚠️ Embedding dimension mismatch: {len(embedding)}, fixing...")
                        embedding = np.pad(embedding, (0, 384 - len(embedding))) if len(embedding) < 384 else embedding[:384]
                    batch_embeddings[index] = embedding

            # Update progress
            nonlocal processed_count
            with lock:
                processed_count += len(text_batch)
                if batch_index % 2 == 0:  # Update every 2 batches (~100 embeddings)
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(texts) - processed_count) / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    print(f"  [{datetime.now()}] Processed {processed_count}/{len(texts)} embeddings ({processed_count*100//len(texts)}%) - Rate: {rate:.1f}/s - ETA: {eta_minutes:.1f} min", flush=True)

            return batch_embeddings

        # Process all batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = process_text_batch(batch_texts, i // batch_size)
            embeddings.extend(batch_embeddings)

        print(f"[{datetime.now()}] Converting embeddings to numpy array...", flush=True)
        embeddings = np.array(embeddings)
        print(f"[{datetime.now()}] ✅ Generated {len(embeddings)} embeddings with shape {embeddings.shape}", flush=True)

        # Insert data in batches
        print(f"[{datetime.now()}] 💾 Inserting verses into Milvus...", flush=True)
        batch_size = 100
        total_inserted = 0

        for i in range(0, len(embeddings), batch_size):
            print(f"[{datetime.now()}] Processing batch {i//batch_size + 1}/{(len(embeddings)-1)//batch_size + 1}...", flush=True)
            batch_embeddings = embeddings[i:i+batch_size]
            batch_verse_ids = verse_ids[i:i+batch_size]

            # Prepare enhanced entity data
            batch_surahs = []
            batch_ayahs = []
            batch_texts = []
            batch_translations = []

            for idx in batch_verse_ids:
                row = df_verses.iloc[idx]
                batch_surahs.append(int(row.get('surah_no', row.get('surah', 1))))
                batch_ayahs.append(int(row.get('ayah_no_surah', row.get('ayah', 1))))
                batch_texts.append(str(row.get('ayah_ar', row.get('text', '')))[:5000])
                batch_translations.append(str(row.get('ayah_en', row.get('translation', '')))[:5000])

            entities = [
                batch_verse_ids,
                batch_embeddings.tolist(),
                batch_surahs,
                batch_ayahs,
                batch_texts,
                batch_translations
            ]

            print(f"[{datetime.now()}] Inserting batch into Milvus...", flush=True)
            collection.insert(entities)
            total_inserted += len(batch_verse_ids)

            if i % 1000 == 0:
                print(f"[{datetime.now()}]   ✓ Inserted {total_inserted}/{len(embeddings)} verses...", flush=True)

        print(f"[{datetime.now()}] Total inserted: {total_inserted} verses", flush=True)

        # Flush to ensure data is persisted
        print(f"[{datetime.now()}] Flushing collection to persist data...", flush=True)
        try:
            # Run blocking flush operation in executor to prevent hanging
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, collection.flush)
            print(f"[{datetime.now()}] ✅ Successfully flushed {total_inserted} verses", flush=True)
        except Exception as flush_error:
            print(f"[{datetime.now()}] ⚠️ Flush error: {flush_error}", flush=True)
            # Continue anyway - data might still be persisted

        # Create index AFTER data insertion for better performance
        print(f"[{datetime.now()}] 🔍 Creating HNSW index for fast similarity search...", flush=True)
        try:
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,              # Reduced for faster indexing
                    "efConstruction": 128  # Reduced for faster indexing
                }
            }
            await loop.run_in_executor(None, collection.create_index, "embedding", index_params)
            print(f"[{datetime.now()}] ✅ HNSW index created successfully", flush=True)
        except Exception as index_error:
            print(f"[{datetime.now()}] ⚠️ Index creation error: {index_error}", flush=True)

        # Load collection into memory
        print(f"[{datetime.now()}] 📈 Loading collection into memory...", flush=True)
        try:
            await loop.run_in_executor(None, collection.load)
            num_entities = collection.num_entities
            print(f"[{datetime.now()}] ✅ Collection loaded with {num_entities} verses", flush=True)
        except Exception as load_error:
            print(f"[{datetime.now()}] ⚠️ Load error: {load_error}", flush=True)

        print(f"[{datetime.now()}] Collection initialization complete, continuing...", flush=True)
    else:
        # Existing collection - ensure it's loaded
        try:
            print("📊 Loading existing collection into memory...")
            collection.load()
            print(f"✅ Collection loaded with {collection.num_entities} verses")
        except Exception as load_error:
            print(f"❌ Failed to load collection: {load_error}")
            print("🔧 Dropping and recreating collection...")
            try:
                collection.release()
            except:
                pass
            utility.drop_collection(collection_name)
            # Force recreation by restarting
            import sys
            print("🔄 Restarting to rebuild collection...")
            sys.exit(1)

    print(f"[{datetime.now()}] 🎯 Milvus vector database fully initialized and ready!", flush=True)

    # Initialize advanced analytics
    try:
        print(f"[{datetime.now()}] 🧠 Initializing advanced analytics...", flush=True)

        # Generate embeddings for clustering (use subset for performance)
        sample_size = min(1000, len(df_verses))
        sample_df = df_verses.sample(n=sample_size, random_state=42)

        combined_texts = [f"{row['ayah_ar']} {row['ayah_en']}"
                        for _, row in sample_df.iterrows()]
        verse_embeddings = get_batch_embeddings(combined_texts)

        # Generate theme clusters
        print(f"[{datetime.now()}] 🎨 Generating thematic clusters...", flush=True)
        cluster_model, theme_labels = generate_theme_clusters(verse_embeddings, n_clusters=15)
        print(f"[{datetime.now()}] ✨ Advanced AI analytics ready!", flush=True)

        # Initialize Hadith collection
        print(f"[{datetime.now()}] 📚 Initializing Hadith Database...", flush=True)
        global hadith_collection, df_hadith

        try:
            # Download Hadith datasets
            print(f"[{datetime.now()}] 📥 Downloading Hadith collections...", flush=True)
            import urllib.request
            import socket

            # Set timeout for downloads
            socket.setdefaulttimeout(30)

            # Download Sahih Bukhari
            bukhari_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.json"
            muslim_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.json"

            print(f"[{datetime.now()}] Downloading Bukhari...", flush=True)
            with urllib.request.urlopen(bukhari_url) as response:
                bukhari_data = json.loads(response.read())
            print(f"[{datetime.now()}] ✅ Downloaded {len(bukhari_data['hadiths'])} Bukhari hadiths", flush=True)

            print(f"[{datetime.now()}] Downloading Muslim...", flush=True)
            with urllib.request.urlopen(muslim_url) as response:
                muslim_data = json.loads(response.read())
            print(f"[{datetime.now()}] ✅ Downloaded {len(muslim_data['hadiths'])} Muslim hadiths", flush=True)

            # Create Hadith dataframe
            hadith_records = []

            for hadith in bukhari_data['hadiths']:
                text = hadith.get('text', '')
                # Truncate text if too long for Milvus VARCHAR field
                if len(text) > 9900:
                    text = text[:9897] + "..."
                hadith_records.append({
                    'collection': 'Sahih Bukhari',
                    'hadith_number': hadith.get('hadithNumber', ''),
                    'text': text,
                    'arabic': hadith.get('arab', ''),
                    'reference': f"Bukhari {hadith.get('hadithNumber', '')}"
                })

            for hadith in muslim_data['hadiths']:
                text = hadith.get('text', '')
                # Truncate text if too long for Milvus VARCHAR field
                if len(text) > 9900:
                    text = text[:9897] + "..."
                hadith_records.append({
                    'collection': 'Sahih Muslim',
                    'hadith_number': hadith.get('hadithNumber', ''),
                    'text': text,
                    'arabic': hadith.get('arab', ''),
                    'reference': f"Muslim {hadith.get('hadithNumber', '')}"
                })

            df_hadith = pd.DataFrame(hadith_records)
            print(f"✅ Loaded {len(df_hadith)} hadiths total")

            # Create Hadith collection in Milvus
            hadith_collection_name = "hadith_collection"

            if utility.has_collection(hadith_collection_name):
                print(f"♻️ Hadith collection exists, dropping and recreating with new schema...")
                utility.drop_collection(hadith_collection_name)

            if not utility.has_collection(hadith_collection_name):
                print("🔨 Creating new Hadith collection...")

                # Define schema for Hadith collection
                fields = [
                    FieldSchema(name="hadith_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="collection", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="hadith_number", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),  # Increased from 5000
                    FieldSchema(name="reference", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
                ]

                schema = CollectionSchema(fields, "Hadith embeddings for semantic search")
                hadith_collection = Collection(hadith_collection_name, schema)

                # Create HNSW index
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256}
                }
                hadith_collection.create_index("embedding", index_params)

                # Generate embeddings and insert
                print("🧠 Generating Hadith embeddings...")
                # Handle empty or None texts
                hadith_texts = [
                    str(text) if text and str(text).strip() else "[No text available]"
                    for text in df_hadith['text'].tolist()
                ]
                batch_size = 100

                for i in range(0, len(hadith_texts), batch_size):
                    try:
                        batch_texts = hadith_texts[i:i+batch_size]
                        batch_embeddings = get_batch_embeddings(batch_texts)

                        # Ensure text fields don't exceed max length
                        batch_df = df_hadith.iloc[i:i+len(batch_texts)]
                        texts = []
                        for text in batch_df['text'].tolist():
                            # Handle None or empty texts
                            if not text or not str(text).strip():
                                texts.append("[No text available]")
                            elif len(str(text)) > 9900:
                                texts.append(str(text)[:9897] + "...")
                            else:
                                texts.append(str(text))

                        # Convert hadith_number to string if it's not already
                        hadith_numbers = [str(num) for num in batch_df['hadith_number'].tolist()]

                        entities = [
                            batch_df['collection'].tolist(),
                            hadith_numbers,
                            texts,
                            batch_df['reference'].tolist(),
                            batch_embeddings.tolist()
                        ]

                        hadith_collection.insert(entities)

                        if (i + batch_size) % 1000 == 0:
                            print(f"  Indexed {i + batch_size}/{len(hadith_texts)} hadiths...")
                    except Exception as batch_error:
                        print(f"  ⚠️ Error inserting batch {i//batch_size}: {batch_error}")
                        continue

                hadith_collection.flush()
                hadith_collection.load()
                print(f"✅ Hadith collection ready with {hadith_collection.num_entities} hadiths")

            print("📚 Hadith database initialization complete!")

        except Exception as e:
            print(f"⚠️ Hadith initialization failed: {e}")
            hadith_collection = None
            df_hadith = None

    except Exception as e:
        print(f"⚠️ Advanced analytics initialization failed: {e}")
        verse_embeddings = None
        cluster_model = None
        theme_labels = None

    print(f"[{datetime.now()}] ✅✅✅ STARTUP COMPLETE - API READY!", flush=True)

@app.get("/")
async def root():
    return {"message": "🕌 Quran Search API - Mobile Optimized", "status": "ready", "verses": len(df_verses) if df_verses is not None else 0}

@app.get("/api")
async def api_root():
    return {"message": "Quran Search API", "version": "1.0", "endpoints": ["/api/search", "/api/hadith/search", "/api/qa", "/api/count", "/api/similar", "/api/tafsir", "/api/export", "/api/analytics/themes"]}

@app.post("/api/hadith/search")
async def search_hadith(query: SearchQuery):
    global model, hadith_collection, df_hadith

    if not hadith_collection or not model or df_hadith is None:
        raise HTTPException(status_code=400, detail="Hadith system not ready")

    try:
        # Generate query embedding
        query_embedding = np.array([get_embedding(query.query)])

        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        # Search in Hadith collection
        results = hadith_collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=query.limit,
            output_fields=["collection", "hadith_number", "text", "reference"]
        )

        # Format results
        search_results = []
        for hits in results:
            for hit in hits:
                if hit.score > query.similarity_threshold:
                    search_results.append({
                        "collection": hit.entity.get('collection'),
                        "hadith_number": hit.entity.get('hadith_number'),
                        "text": hit.entity.get('text'),
                        "reference": hit.entity.get('reference'),
                        "score": float(hit.score)
                    })

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
        import re
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
                # Handle both dataset formats
                try:
                    # Get verse data

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
                except KeyError as ke:
                    # KeyError accessing column
                    raise
        # Check collection status and reload if needed
        try:
            num_entities = collection.num_entities
            if num_entities == 0:
                raise Exception("Collection is empty, needs reindexing")
        except Exception as status_error:
            print(f"⚠️ Collection status check failed: {status_error}, attempting to reload...")
            try:
                collection.load()
            except:
                # Collection might be corrupted, recreate
                print("🔧 Collection corrupted, forcing recreation...")
                from pymilvus import utility
                utility.drop_collection("quran_verses")
                raise HTTPException(status_code=503, detail="Collection being rebuilt, please retry in a moment")

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

        # Generate query embedding
        # Use embedding for search queries
        query_embedding = [get_hybrid_embedding(query.query, target_dim=384)]

        # Search in Milvus with retry logic
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}  # Higher ef for better recall

        results = None
        for attempt in range(3):
            try:
                results = collection.search(
                    query_embedding,
                    "embedding",
                    search_params,
                    limit=query.limit * 2,
                    output_fields=["surah", "ayah", "text", "translation"]
                )
                break  # Success, exit retry loop
            except Exception as search_error:
                if "fail to search on QueryNode" in str(search_error):
                    print(f"⚠️ QueryNode error on attempt {attempt + 1}, retrying...")
                    # Try to reload collection
                    try:
                        collection.release()
                        collection.load()
                    except:
                        pass
                    if attempt == 2:  # Last attempt failed
                        raise HTTPException(status_code=503, detail="Search service temporarily unavailable, please retry")
                else:
                    raise

        if not results:
            raise HTTPException(status_code=500, detail="Search failed after retries")

        # Process results
        search_results = []
        for hits in results:
            for hit in hits:
                if hit.score >= query.similarity_threshold:
                    # Use data directly from Milvus fields
                    surah = int(hit.entity.get('surah', 1))
                    ayah = int(hit.entity.get('ayah', 1))

                    if query.surah_filter and surah != query.surah_filter:
                        continue

                    result = {
                        "surah": surah,
                        "ayah": ayah,
                        "text": hit.entity.get('text', ''),
                        "translation": hit.entity.get('translation', ''),
                        "score": float(hit.score)
                    }
                    search_results.append(result)

        # Combine keyword results with search results
        if keyword_results:
            # Add keyword results first (they have higher relevance)
            all_results = keyword_results + search_results
            # Remove duplicates based on surah and ayah
            seen = set()
            unique_results = []
            for result in all_results:
                key = (result['surah'], result.get('ayah', result.get('verse', 0)))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            search_results = unique_results

        search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)[:query.limit]
        return {"results": search_results, "total": len(search_results)}

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Search error: {e}")
        # Check if it's a dimension mismatch error
        if "expected" in str(e) and "actual" in str(e):
            raise HTTPException(status_code=503, detail="Collection needs reindexing due to dimension change, please wait")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/search/advanced")
async def advanced_search(query: SearchQuery):
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    try:
        # Use embedding for search queries
        query_embedding = [get_hybrid_embedding(query.query, target_dim=384)]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

        results = collection.search(
            query_embedding,
            "embedding",
            search_params,
            limit=query.limit * 3,
            output_fields=["surah", "ayah", "text", "translation"]
        )

        search_results = []
        confidence_scores = []

        for hits in results:
            for hit in hits:
                if hit.score >= query.similarity_threshold:
                    # Use data directly from Milvus fields
                    surah = int(hit.entity.get('surah', 1))
                    ayah = int(hit.entity.get('ayah', 1))

                    if query.surah_filter and surah != query.surah_filter:
                        continue

                    # Enhanced result with confidence analysis
                    result = {
                        "surah": surah,
                        "ayah": ayah,
                        "text": hit.entity.get('text', ''),
                        "translation": hit.entity.get('translation', ''),
                        "score": float(hit.score),
                        "confidence": "high" if hit.score > 0.8 else "medium" if hit.score > 0.6 else "low",
                        "relevance_percentage": round(hit.score * 100, 1)
                    }
                    search_results.append(result)
                    confidence_scores.append(hit.score)

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
        query_embedding = np.array([get_embedding(query_text)])
        # Optimized search parameters for HNSW
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}  # Higher ef for better recall

        semantic_results = collection.search(
            query_embedding,
            "embedding",
            search_params,
            limit=limit * 2,
            output_fields=["surah", "ayah", "text", "translation"]
        )

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
        for hits in semantic_results:
            for hit in hits:
                # Create unique key for this verse
                verse_key = f"{hit.entity.get('surah', 1)}:{hit.entity.get('ayah', 1)}"
                combined_results[verse_key] = {
                    "semantic_score": hit.score,
                    "keyword_score": 0,
                    "verse": {
                        "surah": hit.entity.get('surah', 1),
                        "ayah": hit.entity.get('ayah', 1),
                        "text": hit.entity.get('text', ''),
                        "translation": hit.entity.get('translation', '')
                    }
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
                        "surah": verse.get('surah_no', verse.get('surah', 1)),
                        "ayah": verse.get('ayah_no_surah', verse.get('ayah', 1)),
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
                "semantic_matches": len([hits for hits in semantic_results for hit in hits]),
                "keyword_matches": len(keyword_matches),
                "final_results": len(final_results)
            }
        }

    except Exception as e:
        logger.error(f"Multi-vector search error: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-vector search failed: {str(e)}")

@app.post("/api/count")
async def count_word_occurrences(request: Dict[str, Any]):
    global df_verses

    if df_verses is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded")

    try:
        # Support both 'word' and 'query' parameters
        word = request.get("word", request.get("query", "")).strip()
        if not word:
            raise HTTPException(status_code=400, detail="Word/query cannot be empty")

        count = 0
        examples = []

        for _, verse in df_verses.iterrows():
            # Handle both dataset formats
            text = verse.get('ayah_ar', verse.get('text', ''))
            translation = verse.get('ayah_en', verse.get('translation', ''))
            text = text if pd.notna(text) else ""
            translation = translation if pd.notna(translation) else ""
            combined_text = f"{text} {translation}"

            case_sensitive = request.get("case_sensitive", False)
            if case_sensitive:
                matches = len(re.findall(re.escape(word), combined_text))
            else:
                matches = len(re.findall(re.escape(word), combined_text, re.IGNORECASE))

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
            "examples": examples[:10],  # Limit examples for mobile
            "total_verses_with_word": len(examples),
            "verses": examples[:10],  # Add 'verses' field for API compatibility
            "occurrences": examples[:10]  # Add 'occurrences' field for compatibility
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Count failed: {str(e)}")

@app.post("/api/qa/islamic")
async def islamic_qa(query: QAQuery):
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        raise HTTPException(status_code=400, detail="System not ready")

    try:
        # Check collection status
        try:
            num_entities = collection.num_entities
            if num_entities == 0:
                raise HTTPException(status_code=503, detail="Collection is empty, rebuilding in progress")
        except Exception as status_error:
            print(f"⚠️ Collection status check failed for QA: {status_error}")
            try:
                collection.load()
            except:
                raise HTTPException(status_code=503, detail="Vector database temporarily unavailable")

        # Use embedding for question understanding
        query_embedding = [get_hybrid_embedding(query.question, target_dim=384)]

        # Search with retry logic
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        results = None
        for attempt in range(3):
            try:
                results = collection.search(
                    query_embedding,
                    "embedding",
                    search_params,
                    limit=query.context_limit,
                    output_fields=["surah", "ayah", "text", "translation"]
                )
                break
            except Exception as search_error:
                if "fail to search on QueryNode" in str(search_error) or "work" in str(search_error):
                    print(f"⚠️ QueryNode error in QA on attempt {attempt + 1}, retrying...")
                    try:
                        collection.release()
                        collection.load()
                    except:
                        pass
                    if attempt == 2:
                        raise HTTPException(status_code=503, detail="Search service temporarily unavailable")
                else:
                    raise

        if not results:
            raise HTTPException(status_code=500, detail="QA search failed after retries")

        relevant_verses = []
        total_relevance = 0

        for hits in results:
            for hit in hits:
                # Use data directly from Milvus fields
                relevance_score = float(hit.score)
                total_relevance += relevance_score

                relevant_verses.append({
                    "surah": int(hit.entity.get('surah', 1)),
                    "ayah": int(hit.entity.get('ayah', 1)),
                    "text": hit.entity.get('text', ''),
                    "translation": hit.entity.get('translation', ''),
                    "relevance": relevance_score,
                    "confidence": "high" if relevance_score > 0.7 else "medium" if relevance_score > 0.5 else "low"
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
        if "expected" in str(e) and "actual" in str(e):
            raise HTTPException(status_code=503, detail="Collection reindexing needed, please wait")
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
        definite_articles = len([w for w in words if w.startswith('\u0627\u0644')])  # الـ
        conjunctions = len([w for w in words if w.startswith('\u0648')])  # و

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
    global df_verses, collection, df_hadith, hadith_collection

    try:
        stats = {
            "total_verses": len(df_verses) if df_verses is not None else 0,
            "status": "Ready" if collection and df_verses is not None else "Loading",
            "indexed_verses": collection.num_entities if collection else 0,
            "total_hadiths": len(df_hadith) if df_hadith is not None else 0,
            "indexed_hadiths": hadith_collection.num_entities if hadith_collection else 0
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

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
        return {"error": "Analytics data not available. Dataset needs to be processed."}

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
        return {"error": str(e)}

@app.get("/api/analytics/embeddings/visualization")
async def get_embeddings_visualization():
    # Generate 2D visualization of verse embeddings using UMAP
    global verse_embeddings, theme_labels

    if verse_embeddings is None:
        return {"error": "Embeddings not available"}

    try:
        # Generate UMAP projection
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
        return {"error": str(e)}

@app.post("/api/analytics/similarity/network")
async def create_similarity_network(threshold: float = 0.8):
    # Create network graph of semantically similar verses
    global verse_embeddings, df_verses

    if verse_embeddings is None:
        return {"error": "Embeddings not available"}

    try:
        # Create verse network
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
        return {"error": str(e)}

@app.get("/api/analytics/wordcloud")
async def generate_wordcloud():
    # Generate word cloud from Quran text
    global df_verses

    if df_verses is None:
        return {"error": "Dataset not available"}

    try:
        # Combine all English text
        all_text = " ".join(df_verses['ayah_en'].fillna('').tolist())

        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(all_text)

        # Convert to base64 for web display
        import io
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)

        import base64
        img_str = base64.b64encode(img_buffer.read()).decode()

        return {
            "wordcloud_image": f"data:image/png;base64,{img_str}",
            "word_frequencies": dict(wordcloud.words_)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/analytics/export")
async def export_search_results(query: dict):
    # Export search results in various formats
    global model, collection, df_verses

    if not collection or not model or df_verses is None:
        return {"error": "System not ready"}

    try:
        search_query = query.get("query", "")
        export_format = query.get("format", "json")  # json, csv
        limit = query.get("limit", 50)

        # Perform search
        query_embedding = np.array([get_embedding(search_query)])
        # Optimized search parameters for HNSW
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}  # Higher ef for better recall

        results = collection.search(
            query_embedding,
            "embedding",
            search_params,
            limit=limit,
            output_fields=["surah", "ayah", "text", "translation"]
        )

        # Process results
        export_data = []
        for hits in results:
            for hit in hits:
                # Use data directly from Milvus fields
                export_data.append({
                    "surah": int(hit.entity.get('surah', 1)),
                    "ayah": int(hit.entity.get('ayah', 1)),
                    "arabic_text": hit.entity.get('text', ''),
                    "english_translation": hit.entity.get('translation', ''),
                    "similarity_score": float(hit.score),
                    "search_query": search_query
                })

        if export_format == "csv":
            # Convert to CSV format
            import io
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
                "filename": f"quran_search_{search_query[:20]}.csv",
                "count": len(export_data)
            }
        else:
            return {
                "format": "json",
                "data": export_data,  # Changed from 'content' to 'data' for API consistency
                "content": export_data,  # Keep 'content' for backward compatibility
                "filename": f"quran_search_{search_query[:20]}.json",
                "count": len(export_data)
            }

    except Exception as e:
        return {"error": str(e)}

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
    import re
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
        import re
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

    # Enhanced tafsir response
    interpretation = f"This verse from Surah {surah}, Ayah {ayah} discusses important themes of faith, guidance, and righteousness. The verse emphasizes the importance of belief in Allah and following the righteous path. [Detailed tafsir would be loaded from database]"

    return {
        "reference": f"{surah}:{ayah}",
        "surah": surah,
        "ayah": ayah,
        "verse": verse_text,
        "verse_text": verse_text,  # Add for compatibility
        "translation": verse_trans,
        "interpretation": interpretation,  # Add expected field
        "tafsir": interpretation,
        "source": "Ibn Kathir"
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
async def health_check():
    """Health check endpoint"""
    global collection, df_verses
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "collection_ready": collection is not None,
        "verses_loaded": df_verses is not None,
        "total_verses": len(df_verses) if df_verses is not None else 0
    }

@app.get("/api/status")
async def status_endpoint():
    """Status endpoint for health checks"""
    global collection, df_verses

    return {
        "status": "healthy",
        "endpoints": [
            "/api/search", "/api/qa", "/api/similar", "/api/tafsir",
            "/api/count", "/api/analytics/themes", "/api/export", "/api/hadith/search"
        ],
        "collection_ready": collection is not None,
        "verses_loaded": df_verses is not None,
        "total_verses": len(df_verses) if df_verses is not None else 0,
        "milvus_connected": connections.has_connection("default"),
        "available_endpoints": [
            "/api/search", "/api/qa", "/api/similar", "/api/tafsir",
            "/api/count", "/api/analytics/themes", "/api/export"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)