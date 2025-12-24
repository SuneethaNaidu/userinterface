import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

# -----------------------------
# SAFE DATA LOADING
# -----------------------------
def load_data():
    try:
        file_name = "C:/tasks/archive (3)/UCI_Real_Estate_Valuation.xlsx"

        if not os.path.exists(file_name):
            st.error(f"❌ Dataset not found: {file_name}")
            st.stop()

        df = pd.read_excel(file_name)

        if "No" in df.columns:
            df.drop(columns=["No"], inplace=True)

        df = df.rename(columns={
            "X1 transaction date": "transaction_date",
            "X2 house age": "house_age",
            "X3 distance to the nearest MRT station": "distance_to_mrt",
            "X4 number of convenience stores": "num_convenience_stores",
            "X5 latitude": "latitude",
            "X6 longitude": "longitude",
            "Y house price of unit area": "house_price"
        })

        df.fillna(df.median(numeric_only=True), inplace=True)
        return df

    except Exception as e:
        st.error("❌ Error while loading dataset")
        st.exception(e)
        st.stop()

df = load_data()

# -----------------------------
# MODEL TRAINING
# -----------------------------
try:
    X = df.drop("house_price", axis=1)
    y = df["house_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

except Exception as e:
    st.error("❌ Error during model training")
    st.exception(e)
    st.stop()

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("📋 Enter House Details")

inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.number_input(
        col.replace("_", " ").title(),
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

input_df = pd.DataFrame([inputs])

# -----------------------------
# PREDICTION
# -----------------------------
try:
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: {prediction:.2f}")
except Exception as e:
    st.error("❌ Prediction error")
    st.exception(e)

# -----------------------------
# VISUALIZATION
# -----------------------------
st.subheader("📊 House Price Distribution")

try:
    fig, ax = plt.subplots()
    ax.hist(df["house_price"], bins=30)
    ax.set_xlabel("House Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
except Exception as e:
    st.error("❌ Plot error")
    st.exception(e)

st.subheader("📌 Feature Importance")

try:
    fig2, ax2 = plt.subplots()
    ax2.barh(X.columns, model.feature_importances_)
    st.pyplot(fig2)
except Exception as e:
    st.error("❌ Feature importance plot error")
    st.exception(e)
