# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# === Load Model and Preprocessing Components ===
model = joblib.load("models/mlp_stroke_model.pkl")
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/chi2_selector.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
selected_features = joblib.load("models/selected_features.pkl")

# === Streamlit Interface ===
st.set_page_config(page_title="Stroke Prediction System", layout="centered")

st.title("🧠 Stroke Prediction System")
st.markdown("Fill in the patient's information below to predict the **risk of stroke**.")

# === Collect User Inputs ===
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100, 25)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.slider("Average Glucose Level", 40.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 22.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    data = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    return pd.DataFrame([data])

# === Input Data ===
input_df = user_input_features()

# === Encode Categorical Columns ===
for col in label_encoders:
    if col in input_df.columns:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

# === Select & Scale Features ===
X_input = input_df[selected_features]
X_input_selected = selector.transform(X_input)
X_input_scaled = scaler.transform(X_input_selected)

# === Predict ===
if st.button("Predict Stroke Risk"):
    prediction = model.predict(X_input_scaled)[0]
    probability = model.predict_proba(X_input_scaled)[0][1]

    if prediction == 1:
        st.error(f"✅ Low Risk of Stroke! (Probability: {probability:.2f})")
    else:
        st.success(f"⚠️ High Risk of Stroke. (Probability: {probability:.2f})")

# === Sidebar for Metrics and Docs ===
st.sidebar.title("📊 Model Evaluation")

# if os.path.exists("models/evaluation_report.txt"):
#     with open("models/evaluation_report.txt", "r") as file:
#         st.sidebar.text(file.read())

if os.path.exists("models/confusion_matrix.png"):
    st.sidebar.image("models/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

if os.path.exists("models/roc_curve.png"):
    st.sidebar.image("models/roc_curve.png", caption="ROC Curve", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Developed using Chi-Square + MLP Classifier")

