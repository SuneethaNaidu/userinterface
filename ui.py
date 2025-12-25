import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="House Price Prediction", layout="centered")

# --------------------------------------------------
# DATASET PATH (YOUR LOCAL PATH)
# --------------------------------------------------
DATASET_PATH = r"C:\tasks\archive (3)\UCI_Real_Estate_Valuation.xlsx"

# --------------------------------------------------
# MODEL PATH (same folder as this file)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATASET_PATH)

# --------------------------------------------------
# TRAIN MODEL & SAVE
# --------------------------------------------------
def train_and_save_model(df):
    X = df[
        [
            "X1 transaction date",
            "X2 house age",
            "X3 distance to the nearest MRT station",
            "X4 number of convenience stores",
            "X5 latitude",
            "X6 longitude",
        ]
    ]

    y = df["Y house price of unit area"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model

# --------------------------------------------------
# LOAD OR TRAIN MODEL
# --------------------------------------------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    df = load_data()
    model = train_and_save_model(df)

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🏠 House Price Prediction")
st.write("Predict house price using UCI Real Estate Valuation dataset")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
transaction_date = st.number_input("Transaction Date (e.g., 2013.5)", value=2013.5)
house_age = st.number_input("House Age (years)", min_value=0.0, value=10.0)
distance_mrt = st.number_input(
    "Distance to nearest MRT station (meters)", min_value=0.0, value=500.0
)
stores = st.number_input("Number of convenience stores", min_value=0, value=5)
latitude = st.number_input("Latitude", value=24.98)
longitude = st.number_input("Longitude", value=121.54)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("Predict Price"):
    input_data = np.array(
        [[transaction_date, house_age, distance_mrt, stores, latitude, longitude]]
    )

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price (per unit area): {prediction[0]:.2f}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("✅ **Task 3 – Streamlit House Price Prediction App**")
