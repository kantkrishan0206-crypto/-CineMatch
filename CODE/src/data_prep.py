"""Simple data prep for MovieLens CSVs. Produces movies.csv, ratings.csv cleaned."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

def load_movielens(path: Path = DATA_DIR):
    # expects movies.csv and ratings.csv already downloaded (MovieLens 100k/1m)
    movies = pd.read_csv(path / 'movies.csv')
    ratings = pd.read_csv(path / 'ratings.csv')
    # basic clean
    movies['title'] = movies['title'].astype(str)
    return movies, ratings

if __name__ == '__main__':
    movies, ratings = load_movielens()
    print(f"Movies: {len(movies)}, Ratings: {len(ratings)}")
    movies.to_csv(DATA_DIR / 'movies_clean.csv', index=False)
    ratings.to_csv(DATA_DIR / 'ratings_clean.csv', index=False)
    print('Saved cleaned CSVs to data/')
