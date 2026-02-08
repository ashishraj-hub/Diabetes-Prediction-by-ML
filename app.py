import joblib
import streamlit as st
import numpy as np
import pandas as pd

model=joblib.load("diabetes_xgboost_pipeline.joblib")

FEATURES= ["Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"]

THRESHOLD = 0.55  # tuned threshold

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Disease Prediction App")
st.write("Enter patient details to assess diabetes risk.")

inputs = {
    "Pregnancies": st.number_input("Pregnancies", 0, 20, 1),
    "Glucose": st.number_input("Glucose", 0, 300, 85),
    "BloodPressure": st.number_input("Blood Pressure", 0, 200, 66),
    "SkinThickness": st.number_input("Skin Thickness", 0, 100, 29),
    "Insulin": st.number_input("Insulin", 0, 900, 80),
    "BMI": st.number_input("BMI", 0.0, 70.0, 26.6),
    "DiabetesPedigreeFunction": st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.351),
    "Age": st.number_input("Age", 1, 120, 31)
}

input_df=pd.DataFrame([inputs], columns=FEATURES)

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict"):
    probs=model.predict_proba(input_df)[0]
    prob_non_diabetic=probs[0]
    prob_diabetic=probs[1]

    st.subheader("Prediction Probability")
    st.write({
        "Non-diabetic": round(prob_non_diabetic, 3),
        "Diabetic": round(prob_diabetic, 3)
    })

    if prob_diabetic >= THRESHOLD:
        st.error(f"⚠️ High risk of diabetes (Probability: {prob_diabetic:.2f})")
    else:
        st.success(f"✅ Low risk of diabetes (Probability: {prob_diabetic:.2f})")



