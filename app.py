import joblib
import streamlit as st
import numpy as np
import pandas as pd

model=joblib.load("diabetes_xgboost_pipeline.joblib")

st.title("🩺 Diabetes Disease Prediction")

preg=st.number_input("Pregnancies",0,20)
glucose=st.number_input("Glucose Level",0,300)
bp=st.number_input("Blood Pressure",0,200)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
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

prediction=model.predict(input_data)

st.write("Prediction probability:", model.predict_proba(input_data))

prob = model.predict_proba(input_data)[0][1]

if prob > 0.6:
    st.error("⚠ High chance of Diabetes")
else:
    st.success("✅ Low chance of Diabetes")
    


