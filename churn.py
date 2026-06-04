import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import joblib

# Load customer segments
rfm = pd.read_csv(
    "data/customer_segments.csv"
)

print("Data Loaded Successfully")

# Create churn target
rfm["Churn"] = (
    rfm["Recency"] > 90
).astype(int)

print(
    rfm["Churn"].value_counts()
)

# Features
X = rfm[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

# Target
y = rfm["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(
    X_train,
    y_train
)

print("Model Trained")

# Predictions
predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nAccuracy:",
    accuracy
)

print(
    "\nClassification Report"
)

print(
    classification_report(
        y_test,
        predictions
    )
)

cm = confusion_matrix(
    y_test,
    predictions
)

print(
    "\nConfusion Matrix"
)

print(cm)

joblib.dump(
    model,
    "models/churn_model.pkl"
)

print(
    "\nModel Saved"
)

import matplotlib.pyplot as plt

importance = model.feature_importances_

features = [
    "Recency",
    "Frequency",
    "Monetary"
]

plt.figure(figsize=(8,5))

plt.bar(
    features,
    importance
)

plt.title(
    "Feature Importance"
)

plt.show()

rfm["Risk_Probability"] = model.predict_proba(
    X
)[:,1]

rfm.to_csv(
    "data/churn_predictions.csv",
    index=False
)

print(
    "\nRisk Scores Saved"
)

high_risk = rfm[
    rfm["Risk_Probability"] > 0.80
]

print(
    "\nHigh Risk Customers:"
)

print(
    high_risk.head()
)