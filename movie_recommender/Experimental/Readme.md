# Movie Recommender System

A Python-based movie recommendation system using the **Surprise** library's SVD algorithm.  
Includes exploratory data analysis (EDA) and a Streamlit web app interface for interactive movie recommendations with genre, language, and title filtering.

---

## Features

- Loads and processes MovieLens dataset (`u.data` and `u.item`).
- Exploratory Data Analysis (EDA) with visualizations of ratings distribution and user rating counts.
- Trains or loads an SVD collaborative filtering model for recommendations.
- Provides top-N personalized movie recommendations for any user.
- Interactive Streamlit app for filtering by genres, language, and searching movie titles.
- Allows users to add new ratings (updates dataset in-memory, no retraining).
- Saves and loads trained models with pickle.

---

## Project Structure

- `data/` — contains MovieLens dataset files `u.data` and `u.item`.
- `svd_model.pkl` — saved SVD model file after training.
- `ratings_distribution.png`, `ratings_per_user.png` — generated EDA plots.
- `movie_recommender.py` — main Python code with `MovieRecommender` class and CLI usage.
- `app.py` — Streamlit web app for interactive recommendations.

---

## Dependencies

- Python 3.x  
- pandas  
- matplotlib  
- seaborn  
- scikit-surprise (`surprise`)  
- streamlit

Install dependencies with:

```bash
pip install pandas matplotlib seaborn scikit-surprise streamlit

```

