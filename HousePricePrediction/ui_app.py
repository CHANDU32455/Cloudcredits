import os
import glob
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import base64
from datetime import datetime
from fpdf import FPDF

MODEL_PATH = "model/house_model.pkl"
SCALER_PATH = "model/scaler.pkl"
PREDICTION_DIR = "model/predictions"

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("Model or scaler not found. Please run 'train_model.py' first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_price(model, scaler, features):
    try:
        input_array = np.array([features])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        return round(prediction * 100000, 2)  # Convert to USD
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def save_prediction_report(features, predicted_price):
    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Prediction_Report_{timestamp}.pdf"
    filepath = os.path.join(PREDICTION_DIR, filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "House Price Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Prediction Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    feature_names = ["Median Income (10k USD)", "House Age (years)", "Average Rooms per House", "Average Bedrooms per House",
                     "Population", "Average Occupants per Household", "Latitude", "Longitude"]

    pdf.ln(5)
    pdf.cell(0, 10, "Input Features:", ln=True)
    pdf.set_font("Arial", '', 11)
    for name, val in zip(feature_names, features):
        pdf.cell(0, 8, f"{name}: {val}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Predicted House Price: ${predicted_price:,.2f}", ln=True, align="C")

    pdf.output(filepath)

    return filepath

def show_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        st.warning("PDF not found.")
        return
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")
    st.title("🏡 California House Price Prediction")
    st.markdown("Enter details below to predict the price of a house.")

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.stop()

    # Input fields
    MedInc = st.number_input('Median Income (10k USD)', min_value=0.0, max_value=20.0, value=5.0)
    HouseAge = st.slider('House Age (years)', 1, 50, 20)
    AveRooms = st.number_input('Average Rooms per House', value=5.0)
    AveBedrms = st.number_input('Average Bedrooms per House', value=1.0)
    Population = st.number_input('Population in the area', value=1000.0)
    AveOccup = st.number_input('Average Occupants per Household', value=3.0)
    Latitude = st.number_input('Latitude', value=34.0)
    Longitude = st.number_input('Longitude', value=-118.0)

    if st.button("Predict House Price"):
        features = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
        price = predict_price(model, scaler, features)
        if price is not None:
            st.success(f"🏠 Estimated House Price: **${price:,.2f}**")

            # Save the report PDF
            saved_pdf = save_prediction_report(features, price)
            st.info(f"Prediction report saved as: {os.path.basename(saved_pdf)}")

            # Show the generated PDF right away
            show_pdf(saved_pdf)

    # Sidebar extras
    st.sidebar.header("📎 Extras")

    with st.sidebar.expander("📁 Model Files"):
        if st.button("View Feature Weights"):
            feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
            st.dataframe(pd.DataFrame({"Feature": feature_names, "Weight": model.coef_}))

    with st.sidebar.expander("📄 View Prediction Reports"):
        if not os.path.exists(PREDICTION_DIR):
            st.info("No prediction reports available yet.")
        else:
            prediction_pdfs = sorted(glob.glob(os.path.join(PREDICTION_DIR, "Prediction_Report_*.pdf")), reverse=True)
            if not prediction_pdfs:
                st.info("No prediction reports available yet.")
            else:
                for i, pdf_file in enumerate(prediction_pdfs[:5]):  # show latest 5 reports
                    if st.button(f"📄 Report {i+1} - {os.path.basename(pdf_file)}"):
                        show_pdf(pdf_file)

if __name__ == "__main__":
    main()
