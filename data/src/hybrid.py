# src/hybrid.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

CB_MODEL_PATH = MODELS_DIR / "cb_model.pkl"
CF_MODEL_PATH = MODELS_DIR / "cf_model.pkl"
MOVIES_CLEAN = DATA_DIR / "movies_clean.csv"
RATINGS_CLEAN = DATA_DIR / "ratings_clean.csv"

# load on import for API speed
cb_store = joblib.load(CB_MODEL_PATH)
cf_model = joblib.load(CF_MODEL_PATH)
movies = cb_store['movies']
item_index = cb_store['item_index']
tfidf_matrix = cb_store['tfidf_matrix']

def user_top_movies(user_id, min_ratings=1, top_k=10):
    ratings = pd.read_csv(RATINGS_CLEAN)
    user_r = ratings[ratings['userId'] == int(user_id)]
    if user_r.empty:
        return []
    top = user_r.sort_values('rating', ascending=False).head(top_k)
    return top[['movieId','rating']].to_dict(orient='records')

def aggregate_cb_score_for_user(user_id, candidate_movie_ids, top_k=5):
    # get user's top rated movies
    liked = user_top_movies(user_id, top_k=top_k)
    if not liked:
        # fallback: return zeros
        return {m: 0.0 for m in candidate_movie_ids}
    # compute similarity of each candidate to each liked movie and weight by rating
    scores = {m: 0.0 for m in candidate_movie_ids}
    denom = {m: 0.0 for m in candidate_movie_ids}
    for liked_item in liked:
        lid = liked_item['movieId']
        rating = liked_item['rating']
        if lid not in item_index:
            continue
        li_idx = item_index[lid]
        sim = linear_kernel(tfidf_matrix[li_idx:li_idx+1], tfidf_matrix).flatten()
        for m in candidate_movie_ids:
            if m not in item_index:
                continue
            idx = item_index[m]
            s = float(sim[idx])
            scores[m] += s * (rating / 5.0)  # weight by normalized rating
            denom[m] += (rating / 5.0)
    # normalize
    final = {}
    for m in candidate_movie_ids:
        if denom[m] > 0:
            final[m] = scores[m] / denom[m]
        else:
            final[m] = 0.0
    return final

def hybrid_recommend_for_user(user_id, top_n=10, cb_weight=0.5, cf_weight=0.5, candidate_pool=5000):
    # select candidate pool (popular movies or entire catalog if small)
    movie_ids = movies['movieId'].tolist()
    # optional: sample or use popularity; here use entire list for ml-latest-small
    candidates = movie_ids
    # compute cb aggregated scores
    cb_scores = aggregate_cb_score_for_user(user_id, candidates, top_k=5)
    # compute cf predicted scores
    cf_scores = {}
    for m in candidates:
        try:
            est = cf_model.predict(uid=int(user_id), iid=int(m)).est
            cf_scores[m] = (est / 5.0)  # normalized
        except Exception:
            cf_scores[m] = 0.0
    # combine
    hybrid_scores = []
    for m in candidates:
        score = cb_weight * cb_scores.get(m, 0.0) + cf_weight * cf_scores.get(m, 0.0)
        hybrid_scores.append((m, score))
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    # filter out movies user already rated
    rated = set(pd.read_csv(RATINGS_CLEAN).query("userId == @user_id")['movieId'].tolist())
    out = []
    for movie_id, score in hybrid_scores:
        if movie_id in rated:
            continue
        row = movies.loc[movies['movieId'] == movie_id].iloc[0]
        out.append({'movieId': int(movie_id), 'title': row['title'], 'score': float(score)})
        if len(out) >= top_n:
            break
    return out

if __name__ == "__main__":
    print(hybrid_recommend_for_user(user_id=1, top_n=10)[:10])
