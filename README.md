# 🩺 Disease Prediction from Medical Data (Diabetes)

## 📌 Project Overview
This project focuses on predicting the likelihood of **diabetes** in patients using structured medical data and machine learning classification techniques.  
The goal is to build an **accurate, reliable, and deployable ML system** with a strong emphasis on **medical safety and interpretability**.

The project follows **industry-standard ML practices**, including data preprocessing, model comparison, evaluation using medical-relevant metrics, pipeline creation, and deployment as a web application.

---

## 🎯 Objective
- Predict whether a patient is diabetic or non-diabetic based on medical attributes.
- Compare multiple machine learning models.
- Prioritize **recall for diabetic patients** to minimize false negatives.
- Deploy the final model as a user-friendly web application.

---

## 📂 Dataset
- **Source:** Kaggle – Pima Indians Diabetes Dataset  
- **Type:** Structured medical dataset  
- **Target Variable:** `Outcome`
  - `0` → No Diabetes
  - `1` → Diabetes

### Features Used:
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

---

## 🛠️ Technologies & Tools
- **Programming Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - XGBoost  
  - Matplotlib  
  - Seaborn  
  - Streamlit  
- **Model Serialization:** Joblib  

---

## 🧠 Machine Learning Models Used
The following classification models were trained and evaluated:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- **XGBoost (Final Model)**  

---

## 📊 Model Evaluation Strategy
Models were evaluated using multiple metrics to ensure reliable medical predictions:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC–AUC Curve  

⚠️ **Why Recall Matters:**  
In medical diagnosis, predicting a diabetic patient as non-diabetic (false negative) can be dangerous.  
Therefore, **recall for class 1 (Diabetes)** was prioritized over accuracy.

---

## 🏆 Final Model Selection
**XGBoost** was selected as the final model because:
- It achieved the **highest recall** for diabetic patients.
- It showed strong ROC–AUC performance.
- It effectively handled non-linear relationships in medical data.
- It provided better generalization compared to other models.

---

## 🔗 Machine Learning Pipeline
A **Scikit-learn Pipeline** was created to combine:
1. Feature Scaling (`StandardScaler`)
2. XGBoost Classifier

### Benefits of Using Pipeline:
- Prevents data leakage  
- Ensures consistent preprocessing  
- Simplifies deployment  
- Follows professional ML engineering practices  

---

## 🚀 Deployment
The trained pipeline was serialized using `joblib` and deployed using **Streamlit**.

### Run the Application Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
Disease-Prediction-Diabetes/
│
├── data/
│   └── diabetes.csv
│
├── model/
│   └── diabetes_xgboost_pipeline.joblib
│
├── Diabetes Prediction Model Train.ipynb
├── app.py
├── requirements.txt
└── README.md

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV

- Threshold optimization to further reduce false negatives

- Deployment on cloud platforms (Streamlit Cloud / HuggingFace Spaces)

- Addition of explainability tools (SHAP)

---

## 👤 Author

Name: **Ashish Raj**

📌GitHub: [Ashish Raj](https://github.com/ashishraj-hub)

📌LinkedIn: [Ashish Raj](https://www.linkedin.com/in/ashish-raj-ashishraj/)

---

## 📜 Disclaimer

This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis.

---
