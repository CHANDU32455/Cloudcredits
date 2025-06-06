import os
import joblib
import logging
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "house_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
POLY_PATH = os.path.join(MODEL_DIR, "poly_transformer.pkl")
PLOT_DIR = pathlib.Path("reports")
PLOT_DIR.mkdir(exist_ok=True)

def load_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    logging.info("Dataset loaded successfully.")
    return X, y

def preprocess_features(X_train, X_test):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    logging.info("Polynomial features created and scaled successfully.")
    return X_train_scaled, X_test_scaled, scaler, poly

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # inverse of log1p

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model Evaluation - MSE: {mse:.4f}, R² Score: {r2:.4f}")
    save_diagnostics(y_test, y_pred)

def save_diagnostics(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Median Value (100k $)")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.savefig(PLOT_DIR / "actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    residuals = y_test - y_pred
    plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.savefig(PLOT_DIR / "residual_hist.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Diagnostic plots saved to /reports/")

def save_objects(model, scaler, poly):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(poly, POLY_PATH)
    logging.info("Model, scaler, and polynomial transformer saved successfully.")

def main():
    try:
        X, y = load_data()
        # Split raw data first (not scaled)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess (poly features + scaling)
        X_train_scaled, X_test_scaled, scaler, poly = preprocess_features(X_train_raw, X_test_raw)

        # Log-transform target
        y_train = np.log1p(y_train_raw)
        y_test = y_test_raw  # keep actual target for evaluation

        # Train and evaluate
        model = train_model(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test)

        # Save model objects
        save_objects(model, scaler, poly)

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
