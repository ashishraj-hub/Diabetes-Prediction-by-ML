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

## 📊 Dataset Information

- **Dataset Name:** Pima Indians Diabetes Dataset  
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

### 🔢 Features Used

| Feature | Description |
|------|------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes risk |
| Age | Age of the patient |
| Outcome | Target variable (0 = Non-diabetic, 1 = Diabetic) |

---

## ⚙️ Machine Learning Pipeline

The entire ML workflow is implemented using a **Scikit-learn Pipeline** to prevent data leakage and ensure consistency between training and deployment.

### 🧠 Pipeline Steps

1. **Data Cleaning**
   - Replaced medically invalid zero values with `NaN`
   - Used **median imputation** (robust to skewed medical data)

2. **Feature Scaling**
   - StandardScaler used for normalization

3. **Model**
   - **XGBoost Classifier**
   - Handles non-linear relationships effectively

4. **Class Imbalance Handling**
   - Used `scale_pos_weight` to reduce bias toward diabetic class

5. **Probability Calibration**
   - Applied **Isotonic Calibration**
   - Ensures predicted probabilities reflect real risk

6. **Threshold Tuning**
   - Custom decision threshold instead of default 0.5
   - Reduces false positives in non-diabetic cases

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

## 📈 Model Performance (Approx.)

- **Accuracy:** ~75%
- **ROC-AUC:** ~0.78
- **Balanced recall** for diabetic class
- Stable and realistic probability outputs

> ⚠️ Note: In healthcare ML, **probability reliability matters more than raw accuracy**.

---

## 🚀 Deployment

- **Framework:** Streamlit
- **Model Serialization:** joblib
- **Hosting:** Streamlit Cloud
- **Python Version:** 3.10 (pinned for compatibility)

The deployed app:
- Accepts patient inputs
- Displays both **diabetic & non-diabetic probabilities**
- Uses a **custom risk threshold** for final decision

🌐Live App:- [Click here](https://diabetes-prediction-by-ml-44ubbvsaqdycgfe5rxuejm.streamlit.app/)

---

## 🖥️ Application Interface

Users can:
- Enter patient medical data
- View prediction probabilities
- Receive a clear risk assessment (Low / High Risk)

---

### Run the Application Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
diabetes-prediction-by-ml/
│
├── app.py # Streamlit application
├── diabetes_xgboost_pipeline.joblib # Trained & calibrated ML pipeline
├── requirements.txt # Dependency versions (pinned)
├── runtime.txt # Python version for deployment
├── README.md # Project documentation
├── Different Models Training And Evaluation.ipynb # Some models comparision
├── Diabetes FinalPrediction Model Training #Final model selection and Training

---

## 🧠 Key Design Decisions 

- Used median imputation instead of mean due to skewed medical data

- Avoided training-test leakage using a pipeline

- Handled class imbalance explicitly instead of relying on accuracy

- Calibrated probabilities for healthcare reliability

- Pinned Python & library versions to ensure reproducible deployment

---

## 🧪 Sample Input Used for Testing

```python
Pregnancies: 1
Glucose: 85
BloodPressure: 66
SkinThickness: 29
Insulin: 80
BMI: 26.6
DiabetesPedigreeFunction: 0.351
Age: 31
```

---

## 👤 Author

Name: **Ashish Raj**

📌GitHub: [Ashish Raj](https://github.com/ashishraj-hub)

📌LinkedIn: [Ashish Raj](https://www.linkedin.com/in/ashish-raj-ashishraj/)

---
## 📬 Contact

If you have feedback, suggestions, or collaboration ideas, feel free to connect on LinkedIn.

