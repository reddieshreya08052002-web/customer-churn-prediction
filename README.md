# Customer Churn Prediction

**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn
**Domain:** Telecom / Business Analytics

---

Customer churn is one of those problems where the business already knows it's happening — they just don't know which customers to act on before it's too late. This project builds a pipeline that flags at-risk customers in a telecom dataset of 7,000+ records, using contract type, monthly charges, tenure, and service usage patterns as predictors.

The modeling part was straightforward once the features were cleaned up. The more interesting piece was digging into what the model was actually weighting — which turned out to tell a cleaner story than the accuracy number alone.

---

## Results

- **Model:** Random Forest Classifier (tuned with GridSearchCV)
- **Accuracy:** ~89%
- **ROC-AUC:** ~0.91
- Month-to-month contract customers churn at roughly 3x the rate of customers on two-year contracts — the single strongest signal in the data
- Customers with high monthly charges and short tenure (under 12 months) form the highest-risk segment; they're expensive to serve and haven't built up switching costs yet
- Overtime and add-on services show up as secondary churn drivers, but the contract type effect dominates everything else

---

## Project structure

```
customer-churn-prediction/
├── churn_prediction.py      # EDA, feature engineering, ML pipeline
├── churn_analysis.png       # feature importance, churn by contract, confusion matrix
├── churn_requirements.txt
└── README.md
```

---

## Running it

```bash
pip install -r churn_requirements.txt
python churn_prediction.py
```

Trains the model, prints evaluation metrics, and saves the visualization.

---

## Skills demonstrated

Classification modeling · Random Forest · GridSearchCV · Feature importance analysis · ROC-AUC evaluation · Scikit-learn pipelines
