# app.py

import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it is **fraudulent** or **legitimate**.")

# User inputs
input_values = []
for i in range(1, 29):
    input_values.append(st.number_input(f"V{i}", value=0.0))

amount = st.number_input("Transaction Amount", value=0.0)
input_values.append(amount)

# Predict
if st.button("Predict"):
    features = np.array(input_values).reshape(1, -1)
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Legitimate Transaction")

