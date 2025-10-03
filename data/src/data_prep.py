# src/data_prep.py
import pandas as pd
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
TAGS_CSV = DATA_DIR / "tags.csv"

OUT_MOVIES = DATA_DIR / "movies_clean.csv"
OUT_RATINGS = DATA_DIR / "ratings_clean.csv"
OUT_TAGS = DATA_DIR / "movie_tags.csv"

def load_raw():
    assert MOVIES_CSV.exists() and RATINGS_CSV.exists(), \
        f"Place movies.csv and ratings.csv in {DATA_DIR} or run data/download_movielens.py"
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    tags = pd.read_csv(TAGS_CSV) if TAGS_CSV.exists() else pd.DataFrame(columns=["userId","movieId","tag","timestamp"])
    return movies, ratings, tags

def clean_movies(movies: pd.DataFrame, tags: pd.DataFrame):
    # Clean title, extract year if present, normalize genres
    movies = movies.copy()
    movies['title'] = movies['title'].astype(str).str.strip()
    movies['genres'] = movies['genres'].fillna('')
    # optionally extract year from title (e.g., Toy Story (1995))
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna('').astype(str)
    # aggregate tags into a single string per movie
    if not tags.empty:
        tag_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(map(str, x.unique()))).reset_index()
        movies = movies.merge(tag_agg, how='left', left_on='movieId', right_on='movieId')
        movies.rename(columns={'tag':'tags'}, inplace=True)
        movies['tags'] = movies['tags'].fillna('')
    else:
        movies['tags'] = ''
    # create a metadata column used for TF-IDF later
    movies['metadata'] = (movies['title'] + ' ' + movies['genres'].str.replace('|', ' ') + ' ' + movies['tags']).str.lower()
    return movies

def clean_ratings(ratings: pd.DataFrame):
    ratings = ratings.copy()
    # ensure columns consistent
    if 'userId' not in ratings.columns or 'movieId' not in ratings.columns or 'rating' not in ratings.columns:
        raise ValueError("ratings.csv must contain userId, movieId, rating columns")
    # drop NaNs and duplicates
    ratings = ratings.dropna(subset=['userId','movieId','rating'])
    ratings = ratings.drop_duplicates(subset=['userId','movieId','timestamp'], keep='last')
    # convert types
    ratings['userId'] = ratings['userId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    return ratings

def main():
    movies, ratings, tags = load_raw()
    movies_clean = clean_movies(movies, tags)
    ratings_clean = clean_ratings(ratings)
    movies_clean.to_csv(OUT_MOVIES, index=False)
    ratings_clean.to_csv(OUT_RATINGS, index=False)
    # also write aggregated tag file
    tags_out = movies_clean[['movieId','tags']].copy()
    tags_out.to_csv(OUT_TAGS, index=False)
    print(f"Wrote: {OUT_MOVIES}, {OUT_RATINGS}, {OUT_TAGS}")

if __name__ == "__main__":
    main()
