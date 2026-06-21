# ============================================================
# Customer Churn Prediction
# Author: Reddy Shreya
# Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. GENERATE SAMPLE DATASET ──────────────────────────────
np.random.seed(42)
n = 7000

data = pd.DataFrame({
    'CustomerID':       range(1, n+1),
    'Tenure':           np.random.randint(1, 72, n),
    'MonthlyCharges':   np.round(np.random.uniform(20, 120, n), 2),
    'TotalCharges':     np.round(np.random.uniform(100, 8000, n), 2),
    'Contract':         np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
    'InternetService':  np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20]),
    'PaymentMethod':    np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'SeniorCitizen':    np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'Dependents':       np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
    'TechSupport':      np.random.choice(['Yes', 'No'], n, p=[0.40, 0.60]),
    'Churn':            np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73]),
})

print("Dataset Shape:", data.shape)
print("\nChurn Distribution:\n", data['Churn'].value_counts())
print("\nMissing Values:\n", data.isnull().sum().sum())

# ── 2. PREPROCESSING ────────────────────────────────────────
le = LabelEncoder()
cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'Dependents', 'TechSupport']
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

data['Churn_Binary'] = (data['Churn'] == 'Yes').astype(int)

features = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
            'InternetService', 'PaymentMethod', 'SeniorCitizen', 'Dependents', 'TechSupport']
X = data[features]
y = data['Churn_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 3. MODEL TRAINING ───────────────────────────────────────
param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\nBest Params:", grid.best_params_)

# ── 4. EVALUATION ───────────────────────────────────────────
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))

# ── 5. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Customer Churn Analysis — Reddy Shreya', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Feature Importance
importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)
importances.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Feature Importances')
axes[1].set_xlabel('Importance Score')

# Churn by Contract Type
contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
data['Contract_Label'] = data['Contract'].map(contract_map)
churn_by_contract = data.groupby('Contract_Label')['Churn_Binary'].mean() * 100
churn_by_contract.plot(kind='bar', ax=axes[2], color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
axes[2].set_title('Churn Rate by Contract Type')
axes[2].set_ylabel('Churn Rate (%)')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('churn_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as churn_analysis.png")
