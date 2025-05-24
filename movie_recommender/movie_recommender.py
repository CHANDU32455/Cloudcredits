import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.reader = Reader(rating_scale=(1, 5))
        
        self.load_data()
        self.train_or_load_model()

    def load_data(self):
        ratings_file = os.path.join(self.data_path, "u.data")
        movies_file = os.path.join(self.data_path, "u.item")
        
        # Load ratings
        self.ratings_df = pd.read_csv(ratings_file, sep="\t", names=["userId", "movieId", "rating", "timestamp"], encoding='latin-1')
        
        # Load movies (only movieId and title)
        movies_cols = ["movieId", "title"]
        self.movies_df = pd.read_csv(movies_file, sep='|', names=movies_cols, encoding='latin-1', usecols=[0, 1])
        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies.")

    def eda(self):
        print("Performing exploratory data analysis...")
            
        # Ratings distribution
        plt.figure(figsize=(8,4))
        sns.countplot(x='rating', data=self.ratings_df, palette="viridis")
        plt.title("Distribution of Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.savefig("ratings_distribution.png")  # save plot as PNG locally
        plt.show()
            
        # Number of ratings per user
        plt.figure(figsize=(8,4))
        ratings_per_user = self.ratings_df.groupby('userId').size()
        sns.histplot(ratings_per_user, bins=30, kde=True)
        plt.title("Number of Ratings per User")
        plt.xlabel("Ratings Count")
        plt.ylabel("Number of Users")
        plt.savefig("ratings_per_user.png")  # save plot as PNG locally
        plt.show()

    def train_or_load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path} ...")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("Training new SVD model...")
            data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], self.reader)
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            
            self.model = SVD()
            self.model.fit(trainset)
            
            print("Evaluating model on test set...")
            predictions = self.model.test(testset)
            rmse = accuracy.rmse(predictions)
            print(f"Test RMSE: {rmse:.4f}")
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_path}")

    def recommend_movies(self, user_id, top_n=10):
        # Get all movie ids
        all_movie_ids = self.movies_df['movieId'].unique()
        
        # Get movies already rated by user
        rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values
        
        # Filter out already rated movies
        movies_to_predict = [mid for mid in all_movie_ids if mid not in rated_movies]
        
        # Predict ratings for unseen movies
        predictions = []
        for movie_id in movies_to_predict:
            pred = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
        
        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_movies = predictions[:top_n]
        
        # Fetch movie titles
        recommended = []
        for movie_id, est_rating in top_movies:
            title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
            recommended.append((title, est_rating))
        
        return recommended


if __name__ == "__main__":
    recommender = MovieRecommender()
    
    # Optional: Uncomment to run EDA plots
    recommender.eda()
    
    user_to_test = 50
    print(f"\nTop 10 movie recommendations for user {user_to_test}:")
    recommendations = recommender.recommend_movies(user_to_test, top_n=10)
    for idx, (title, rating) in enumerate(recommendations, 1):
        print(f"{idx}. {title} - Predicted Rating: {rating:.2f}")
