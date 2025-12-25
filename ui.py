import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="House Price Prediction", layout="centered")

# -------------------------------
# Load dataset from repo
# -------------------------------
DATASET_PATH = "C:\tasks\archive (3)\UCI_Real_Estate_Valuation.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATASET_PATH)
    return df

# -------------------------------
# Train model
# -------------------------------
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "house_price_model.pkl")
    return model

# -------------------------------
# Load or train model
# -------------------------------
if os.path.exists("house_price_model.pkl"):
    model = joblib.load("house_price_model.pkl")
else:
    data = load_data()
    model = train_and_save_model(data)

# -------------------------------
# App UI
# -------------------------------
st.title("🏠 House Price Prediction")
st.write("Predict house price (UCI Real Estate dataset)")

transaction_date = st.number_input("Transaction Date (e.g., 2013.5)", value=2013.5)
house_age = st.number_input("House Age (years)", value=10.0)
distance_mrt = st.number_input("Distance to MRT station (meters)", value=500.0)
stores = st.number_input("Convenience stores count", value=5)
latitude = st.number_input("Latitude", value=24.98)
longitude = st.number_input("Longitude", value=121.54)

if st.button("Predict Price"):
    input_data = np.array([[transaction_date, house_age, distance_mrt, stores, latitude, longitude]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted House Price (per unit area): {prediction[0]:.2f}")

st.markdown("---")
st.markdown("📌 **Task 3 – House Price Prediction Streamlit App**")
