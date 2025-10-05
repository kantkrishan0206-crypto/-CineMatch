# -CineMatch
Movie Recommender  AI System: "An intelligent Movie Recommender System that suggests films using content-based, collaborative, and hybrid filtering. Built with Python and ML algorithms, it delivers personalized recommendations with data insights and a simple web interface."
# 🎬 Movie Recommender System
https://labelyourdata.com/cms/wp-content/uploads/2022/04/movie-recommendation-with-machine-learning_4.png
An intelligent **Movie Recommender System** that suggests films using **content-based, collaborative, and hybrid filtering**. Built with Python and ML algorithms, it delivers **personalized recommendations**, **data insights**, and a **simple interactive web interface**.

---

## 🚀 Features

* **Content-Based Filtering**: Recommends movies based on genres, tags, and descriptions.
* **Collaborative Filtering**: Suggests movies by analyzing user behavior and ratings.
* **Hybrid Model**: Combines content-based and collaborative approaches for better accuracy.
* **Interactive Web Interface**: Explore recommendations using Streamlit.
* **API Ready**: Built with FastAPI for easy integration.

---

## 📂 Project Structure

```
movie-recommender/
├── README.md
├── requirements.txt
├── data/             # MovieLens datasets
├── notebooks/        # EDA and exploration
├── src/              # Training scripts & API
├── models/           # Saved ML models
├── web/              # Streamlit app
└── Dockerfile        # Containerization
```

---

## 💻 Installation & Setup

1. Clone the repo:

```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download MovieLens dataset (100k or 1M) and place `ratings.csv` and `movies.csv` in `data/`.

---

## 🛠️ Usage

### Train Models

```bash
python src/data_prep.py
python src/content_based.py
python src/collaborative.py
python src/hybrid.py
```

### Run API

```bash
uvicorn src.api:app --reload
```

### Run Web App

```bash
streamlit run web/streamlit_app.py
```

---

## 📊 Tech Stack

* **Python**: Pandas, NumPy, Scikit-learn
* **Recommendation Algorithms**: Content-Based, Collaborative, Hybrid
* **Surprise Library**: SVD for collaborative filtering
* **Web Frameworks**: FastAPI, Streamlit
* **Model Storage**: Joblib
* **Containerization**: Docker

---

## 🎯 Goals

* Build a **scalable and accurate movie recommendation engine**
* Deliver a **friendly UI for interactive recommendations**
* Explore **different ML techniques** for personalized suggestions

---

## 🌟 Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.

---

## 📜 License

This project is **MIT Licensed**.

---
