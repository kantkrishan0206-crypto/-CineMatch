# web/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MOVIES_CSV = DATA_DIR / "movies_clean.csv"

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 MovieLens Hybrid Recommender")

API_BASE = st.text_input("API base URL", "http://localhost:8000")

col1, col2 = st.columns([2,3])

with col1:
    mode = st.selectbox("Mode", ["By User", "By Movie"])
    if mode == "By User":
        user_id = st.number_input("User ID", min_value=1, step=1, value=1)
        n = st.slider("Top-N recommendations", 1, 25, 10)
        cb_w = st.slider("Content weight", 0.0, 1.0, 0.5)
        cf_w = 1.0 - cb_w
        if st.button("Get Recommendations"):
            try:
                resp = requests.get(f"{API_BASE}/recommend/user/{user_id}?n={n}&cb_weight={cb_w}&cf_weight={cf_w}", timeout=10)
                if resp.status_code == 200:
                    recs = resp.json()
                    if not recs:
                        st.info("No recommendations (user cold-start or error). Try a different user ID.")
                    else:
                        df = pd.DataFrame(recs)
                        st.table(df[['title','score']].assign(score=lambda d: d['score'].map(lambda x: f"{x:.3f}")))
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))
    else:
        movie_id = st.number_input("Movie ID", min_value=1, step=1, value=1)
        n = st.slider("Similar movies", 1, 25, 10)
        if st.button("Find Similar"):
            try:
                resp = requests.get(f"{API_BASE}/recommend/movie/{movie_id}?n={n}", timeout=10)
                if resp.status_code == 200:
                    recs = resp.json()
                    df = pd.DataFrame(recs)
                    st.table(df[['title','score']].assign(score=lambda d: d['score'].map(lambda x: f"{x:.3f}")))
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))

with col2:
    st.subheader("Movie Catalog (sample)")
    if MOVIES_CSV.exists():
        movies = pd.read_csv(MOVIES_CSV)
        st.dataframe(movies[['movieId','title','genres']].head(200))
    else:
        st.info("Run data/download_movielens.py and src/data_prep.py then train models first.")
