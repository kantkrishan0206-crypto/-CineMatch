# src/collaborative.py
import joblib
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

RATINGS_CLEAN = DATA_DIR / "ratings_clean.csv"
CF_MODEL_PATH = MODELS_DIR / "cf_model.pkl"

def train_cf(n_factors=50, n_epochs=25, random_state=42):
    ratings = pd.read_csv(RATINGS_CLEAN)
    reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.15, random_state=random_state)
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    algo.fit(trainset)
    # optional quick cross-validation
    # cv = cross_validate(algo, data, measures=['rmse'], cv=3, verbose=True)
    joblib.dump(algo, CF_MODEL_PATH, compress=3)
    print(f"Saved collaborative model -> {CF_MODEL_PATH}")
    return algo

def predict(user_id, movie_id, cf_model=None):
    if cf_model is None:
        cf_model = joblib.load(CF_MODEL_PATH)
    try:
        pred = cf_model.predict(uid=int(user_id), iid=int(movie_id))
        return pred.est
    except Exception:
        return None

if __name__ == "__main__":
    train_cf()
