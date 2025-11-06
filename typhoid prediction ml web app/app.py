import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ----------------------------
# Load trained model safely
# ----------------------------
with open("typhoid_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Typhoid Prediction", page_icon="🧬", layout="centered")
st.title("🧫 Typhoid Prediction System")
st.write("Enter the patient details and test results below to predict the likelihood of Typhoid infection.")

# ----------------------------
# Get feature names exactly as used during training
# ----------------------------
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    # fallback: use dataset columns if model lacks names
    data = pd.read_csv("Processed Dataset.csv")
    feature_names = [col for col in data.columns if col.lower() != "typhoid"]

# ----------------------------
# Smart input section
# ----------------------------
st.subheader("Enter Parameters")
inputs = {}

for col in feature_names:
    col_lower = col.lower()

    # Numeric columns
    if "age" in col_lower:
        inputs[col] = st.number_input("Age", 0, 120, 25)

    elif "temp" in col_lower or "temperature" in col_lower:
        inputs[col] = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)

    # Gender selector
    elif "gender" in col_lower:
        gender = st.selectbox("Gender", ["Male", "Female"])
        inputs[col] = 1 if gender == "Male" else 0

    # Test result selectors
    elif any(x in col_lower for x in ["to", "th", "bh", "ox", "a", "m", "acute", "para", "rickettsia"]):
        result = st.selectbox(f"{col.replace('_', ' ').upper()} Result", ["Negative", "Positive"])
        inputs[col] = 1 if result == "Positive" else 0

    # All others treated as binary symptoms
    else:
        label = col.replace("_", " ").title()
        inputs[col] = 1 if st.checkbox(label) else 0

# ----------------------------
# Convert input to DataFrame in correct feature order
# ----------------------------
input_df = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Typhoid"):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1 or prediction is True:
            st.error("⚠️ The model predicts Typhoid Positive. Please consult a doctor.")
        else:
            st.success("✅ The model predicts Typhoid Negative.")

        if proba is not None:
            st.info(f"Model Confidence: {proba*100:.2f}%")

    except Exception as e:
        st.warning(f"Error: {e}")

# ----------------------------
# Optional debug view
# ----------------------------
if st.checkbox("Show Input Summary"):
    st.dataframe(input_df)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning.")
