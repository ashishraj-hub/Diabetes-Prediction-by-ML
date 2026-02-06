import joblib
import streamlit as st
import numpy as np
import pandas as pd

pipeline=joblib.load("diabetes_xgboost_pipeline.joblib")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

st.title("🩺 Diabetes Disease Prediction")

st.info(
    "Note: Zero values are not physiologically valid for medical features. "
    "Predictions are reliable only for realistic inputs."
)

preg = st.number_input("Pregnancies",0,20)
glucose= st.number_input("Glucose", min_value=50, max_value=300, value=85)
bp  = st.number_input("Blood Pressure", min_value=40, max_value=200, value=70)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", min_value=15, max_value=900, value=120)
bmi = st.number_input("BMI", min_value=15.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.00, 3.00)
age = st.number_input("Age", 1, 120)

def clean_input(df):
    zero_as_missing = [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"
    ]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    return df

input_df = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                        columns=FEATURES)
st.write("Input Data:",input_df)

if glucose == 0 or bmi == 0 or bp == 0:
    st.warning("Please enter valid medical values. Zero is not a realistic input.")
    st.stop()

input_df = clean_input(input_df)

st.write("Prediction probability:", pipeline.predict_proba(input_df))
prob = pipeline.predict_proba(input_df)[0][1]

THRESHOLD=0.55

if st.button("Prediction"):
    if prob >= THRESHOLD:
        st.error(f"💀 High risk of diabetes (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of diabetes (Probability: {prob:.2f})")


st.write("Probability:", prob)
st.write("Input summary:", input_df.describe())







