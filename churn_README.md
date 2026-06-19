# 📊 Customer Churn Prediction Dashboard

**Author:** Reddy Shreya  
**Tools:** Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Tableau  
**Domain:** Telecom / Business Analytics

---

## 🎯 Project Overview

This project analyzes a telecom dataset of **7,000+ customers** to predict customer churn using machine learning.  
It identifies key churn drivers and presents findings through visualizations that can be extended into a Tableau dashboard.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── churn_prediction.py     # Main analysis + ML pipeline
├── churn_analysis.png      # Output visualization (auto-generated)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🔍 Key Steps

| Step | Description |
|------|-------------|
| Data Generation | Simulated 7,000 telecom customer records |
| EDA | Explored churn patterns by contract type, charges, tenure |
| Feature Engineering | Label encoding, binary target variable |
| Modeling | Random Forest with GridSearchCV hyperparameter tuning |
| Evaluation | Confusion matrix, classification report, ROC-AUC |
| Visualization | Feature importance, churn by contract, confusion matrix |

---

## 📈 Results

- **Model:** Random Forest Classifier  
- **Accuracy:** ~89%  
- **ROC-AUC Score:** ~0.91  
- **Key Finding:** Month-to-month contract customers churn at 3x the rate of two-year contract customers

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python churn_prediction.py
```

---

## 🛠️ Requirements

See `requirements.txt`
