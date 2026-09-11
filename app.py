import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

st.write("Adjust the sliders to predict the median house value.")

st.subheader("Enter House Information")

col1, col2, col3 = st.columns(3)

with col1:
    CRIM = st.slider("CRIM", 0.0, 90.0, 1.0)
    ZN = st.slider("ZN", 0.0, 100.0, 0.0)
    INDUS = st.slider("INDUS", 0.0, 30.0, 10.0)
    CHAS = st.slider("CHAS", 0.0, 1.0, 0.0, step=1.0)
    NOX = st.slider("NOX", 0.0, 1.0, 0.5)

with col2:
    RM = st.slider("RM", 0.0, 10.0, 6.0)
    AGE = st.slider("AGE", 0.0, 100.0, 50.0)
    DIS = st.slider("DIS", 0.0, 15.0, 4.0)
    RAD = st.slider("RAD", 0, 25, 5)

with col3:
    TAX = st.slider("TAX", 0, 800, 300)
    PTRATIO = st.slider("PTRATIO", 0.0, 25.0, 18.0)
    B = st.slider("B", 0.0, 400.0, 350.0)
    LSTAT = st.slider("LSTAT", 0.0, 40.0, 12.0)

input_data = pd.DataFrame({
    "CRIM": [CRIM],
    "ZN": [ZN],
    "INDUS": [INDUS],
    "CHAS": [CHAS],
    "NOX": [NOX],
    "RM": [RM],
    "AGE": [AGE],
    "DIS": [DIS],
    "RAD": [RAD],
    "TAX": [TAX],
    "PTRATIO": [PTRATIO],
    "B": [B],
    "LSTAT": [LSTAT]
})

prediction = model.predict(input_data)[0]

st.success(f"Predicted House Value: {prediction:.2f}")