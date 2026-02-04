import joblib
import streamlit as st
import numpy as np

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

if st.button("Predict"):
    input_data=np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
    prediction=model.predict(input_data)
    if prediction[0]==1:
        st.error("⚠ High chance of Diabetes")
    else:
        st.success("✅ Low chance of Diabetes")