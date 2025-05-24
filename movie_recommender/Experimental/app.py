import os
import pandas as pd
import streamlit as st
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class MovieRecommender:
    def __init__(self, data_path="data", model_path="svd_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.movies_df = None
        self.ratings_df = None
        self.genre_columns = []
        self.reader = Reader(rating_scale=(1, 5))
        self.load_data()
        self.train_or_load_model()

    def load_data(self):
        ratings_file = os.path.join(self.data_path, "u.data")
        movies_file = os.path.join(self.data_path, "u.item")

        # Load ratings
        self.ratings_df = pd.read_csv(
            ratings_file,
            sep="\t",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding='latin-1'
        )

        # Load movies and genre flags
        movies_cols = ["movieId", "title", "release_date", "video_release_date", "IMDb_URL",
                       "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                       "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                       "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

        self.movies_df = pd.read_csv(
            movies_file,
            sep='|',
            names=movies_cols,
            encoding='latin-1',
            usecols=range(len(movies_cols))
        )

        # Simulate language column
        self.movies_df['language'] = 'English'

        # Save genre column names
        self.genre_columns = movies_cols[5:]

    def train_or_load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], self.reader)
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            self.model = SVD()
            self.model.fit(trainset)
            predictions = self.model.test(testset)
            accuracy.rmse(predictions)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def recommend_movies(self, user_id, top_n=10):
        all_movie_ids = self.movies_df['movieId'].unique()
        rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values
        movies_to_predict = [mid for mid in all_movie_ids if mid not in rated_movies]
        predictions = [(mid, self.model.predict(user_id, mid).est) for mid in movies_to_predict]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:top_n]
        recommended = []
        for movie_id, est_rating in top_movies:
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                title = movie_row['title'].values[0]
                release_date = movie_row['release_date'].values[0] if 'release_date' in movie_row else "N/A"
                recommended.append((movie_id, title, est_rating, release_date))
        return recommended

    def add_rating(self, user_id, movie_id, rating):
        new_row = {'userId': user_id, 'movieId': movie_id, 'rating': rating, 'timestamp': 0}
        self.ratings_df = pd.concat([self.ratings_df, pd.DataFrame([new_row])], ignore_index=True)
        # Note: Not retraining on-the-fly for speed

# ---------- Streamlit App ----------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# Instantiate recommender
recommender = MovieRecommender()

# ---------- Sidebar ----------
st.sidebar.header("📂 Filters")

# Genre filtering
genre_options = sorted(recommender.genre_columns)
selected_genres = st.sidebar.multiselect("🎭 Select Genres", genre_options)

# Language filter
available_languages = sorted(recommender.movies_df['language'].dropna().unique().tolist())
selected_language = st.sidebar.selectbox("🌐 Select Language", ["All"] + available_languages)

# Movie title search
search_query = st.sidebar.text_input("🔎 Search Movie Title")

# ---------- Filter Function ----------
def filter_movies(df, language, genres, query):
    filtered = df.copy()

    if language != "All":
        filtered = filtered[filtered['language'] == language]

    if genres:
        genre_filter = filtered[genres].any(axis=1)
        filtered = filtered[genre_filter]

    if query:
        filtered = filtered[filtered['title'].str.contains(query, case=False, na=False)]

    return filtered

filtered_movies_df = filter_movies(recommender.movies_df, selected_language, selected_genres, search_query)

# ---------- User ID Input ----------
user_id = st.sidebar.number_input("👤 Enter User ID", min_value=1, step=1)

# ---------- Auto Show Recommendations ----------
if user_id > 0:
    recommendations = recommender.recommend_movies(user_id, top_n=50)
    recommendations = [
        rec for rec in recommendations
        if rec[0] in filtered_movies_df['movieId'].values
    ][:10]

    if not recommendations:
        st.warning("No matching movies found in this category.")
    else:
        for movie_id, title, est_rating, release_date in recommendations:
            movie = filtered_movies_df[filtered_movies_df['movieId'] == movie_id]

            # Extract genres
            genre_flags = movie[recommender.genre_columns].iloc[0]
            genres = ', '.join(genre for genre, is_genre in genre_flags.items() if is_genre == 1)

            # Placeholder poster
            poster_url = "https://via.placeholder.com/100x150?text=Movie"

            # Display info
            st.markdown(f"### {title} ({release_date})")
            st.markdown(f"**Genres:** {genres}")
            st.markdown(f"**Estimated Rating:** {est_rating:.2f}")
            st.image(poster_url, width=100)

            # Rating input
            rating_input = st.number_input(f"Rate {title}", min_value=0.0, max_value=5.0, step=0.5, key=f"input_{movie_id}")
            if st.button(f"Add Rating for {title}", key=f"rate_{movie_id}"):
                recommender.add_rating(user_id, movie_id, rating_input)
                st.success(f"✅ Rating {rating_input} added for '{title}'!")

        st.success("✅ Recommendations displayed successfully!")
