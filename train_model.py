"""
train_model.py
Trains a Logistic Regression model to predict P(team_a wins).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

df = pd.read_csv("ipl_matches.csv")

FEATURES = [
    "win_rate_a", "win_rate_b",
    "form_a", "form_b",
    "h2h_a",
    "nrr_a", "nrr_b",
    "home_a", "home_b",
    "toss_a",
]
X = df[FEATURES]
y = df["winner_is_a"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))
print()
print(classification_report(y_test, pred))

print("Feature weights (higher |coef| = more influence on win probability):")
for f, c in sorted(zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {f:12s} {c:+.3f}")

joblib.dump(model, "ipl_model.joblib")
joblib.dump(scaler, "ipl_scaler.joblib")
print("\nSaved model -> ipl_model.joblib, scaler -> ipl_scaler.joblib")