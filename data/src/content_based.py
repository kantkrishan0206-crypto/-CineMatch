# src/content_based.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

MOVIES_CLEAN = DATA_DIR / "movies_clean.csv"
CB_MODEL_PATH = MODELS_DIR / "cb_model.pkl"

def train_content_model():
    movies = pd.read_csv(MOVIES_CLEAN)
    # metadata column created by data_prep
    docs = movies['metadata'].fillna('')
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(docs)
    # store index mapping for fast lookup
    item_index = pd.Series(movies.index, index=movies['movieId']).to_dict()
    payload = {
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'movies': movies,
        'item_index': item_index
    }
    joblib.dump(payload, CB_MODEL_PATH, compress=3)
    print(f"Saved content-based model -> {CB_MODEL_PATH}")
    return payload

def similar_movies(movie_id, top_n=10, cb_model=None):
    if cb_model is None:
        cb_model = joblib.load(CB_MODEL_PATH)
    movies = cb_model['movies']
    idx_map = cb_model['item_index']
    if movie_id not in idx_map:
        return []
    idx = idx_map[movie_id]
    tfidf_matrix = cb_model['tfidf_matrix']
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_idx = cosine_sim.argsort()[::-1][1: top_n+1]
    results = movies.iloc[sim_idx][['movieId','title']].copy()
    results['score'] = cosine_sim[sim_idx]
    return results.to_dict(orient='records')

if __name__ == "__main__":
    train_content_model()
    # quick sanity check
    # print(similar_movies(1, top_n=5))
