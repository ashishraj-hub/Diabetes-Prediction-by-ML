import joblib
import streamlit as st
import numpy as np
import pandas as pd

model=joblib.load("diabetes_xgboost_pipeline.joblib")

st.title("🩺 Diabetes Disease Prediction")

st.info(
    "Note: Zero values are not physiologically valid for medical features. "
    "Predictions are reliable only for realistic inputs."
)

preg=st.number_input("Pregnancies",0,20)
glucose = st.slider("Glucose", 50, 200, 120)
bp = st.slider("Blood Pressure", 40, 120, 70)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

input_data=pd.DataFrame([{
    "Pregnancies":preg,
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}])

if glucose == 0 or bmi == 0 or bp == 0:
    st.warning("Please enter valid medical values. Zero is not a realistic input.")
    st.stop()

st.write("Prediction probability:", model.predict_proba(input_data))
prob = model.predict_proba(input_data)[0][1]

if st.button("Prediction"):
    if prob > 0.6:
        st.error("⚠ High chance of Diabetes")
    else:
        st.success("✅ Low chance of Diabetes") 




