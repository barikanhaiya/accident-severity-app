# 🚧 Road Accident Severity Predictor

A machine learning web app that predicts the severity of a road accident (Fatal / Serious / Slight Injury) based on accident conditions such as weather, road type, driver details, and vehicle information.

## 🔍 About the Project

This project uses a **Random Forest Classifier** trained on a road traffic accident dataset. The pipeline includes:

- Missing value imputation (`SimpleImputer`)
- Categorical encoding (`OneHotEncoder`)
- Feature selection (`SelectKBest` with chi2)
- Classification (`RandomForestClassifier`)

The model achieves ~98% accuracy on the test set.

## 🚀 

[Add

## 🛠️ Tech Stack

- Python
- scikit-learn
- Streamlit
- Pandas / NumPy

## 📊 Model Performance

- Accuracy: ~98%
- Balanced precision/recall across Fatal, Serious, and Slight injury classes

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Files

- `app.py` — Streamlit web application
- `pipe.pkl` — Trained ML pipeline (preprocessing + model)
- `requirements.txt` — Python dependencies

## 👤 Author

Built as a beginner machine learning project to learn end-to-end ML deployment.
