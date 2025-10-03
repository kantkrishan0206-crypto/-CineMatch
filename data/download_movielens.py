# data/download_movielens.py
import os
import zipfile
import requests
from pathlib import Path

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parents[0]

def download_and_extract(url=ML_URL, dst=DATA_DIR):
    dst.mkdir(parents=True, exist_ok=True)
    zip_path = dst / "ml-latest-small.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens small dataset (~1MB) from {url} ...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Zip already exists.")
    # extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst)
    # files are in ml-latest-small/
    extracted = dst / "ml-latest-small"
    # move main CSVs to data root
    for fname in ("movies.csv", "ratings.csv", "tags.csv", "links.csv"):
        src = extracted / fname
        if src.exists():
            dst_path = dst / fname
            src.rename(dst_path)
    # cleanup extracted folder (optional)
    if extracted.exists():
        try:
            for p in extracted.iterdir():
                if p.exists():
                    p.unlink()
            extracted.rmdir()
        except Exception:
            pass
    print(f"MovieLens CSVs are in {dst.resolve()}")

if __name__ == "__main__":
    download_and_extract()
