import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction", layout="centered")

DATASET_PATH = "UCI_Real_Estate_Valuation.xlsx"

MODEL_PATH = "house_price_model.pkl"

@st.cache_data
def load_data():
    return pd.read_excel(DATASET_PATH)

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

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    df = load_data()
    model = train_and_save_model(df)

st.title("🏠 House Price Prediction")
st.write("UCI Real Estate Valuation Dataset")

transaction_date = st.number_input("Transaction Date", value=2013.5)
house_age = st.number_input("House Age (years)", value=10.0)
distance_mrt = st.number_input("Distance to MRT (meters)", value=500.0)
stores = st.number_input("No. of Convenience Stores", value=5)
latitude = st.number_input("Latitude", value=24.98)
longitude = st.number_input("Longitude", value=121.54)

if st.button("Predict Price"):
    input_data = np.array([[transaction_date, house_age, distance_mrt, stores, latitude, longitude]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted House Price: {prediction[0]:.2f}")

