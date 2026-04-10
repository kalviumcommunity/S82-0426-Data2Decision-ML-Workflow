"""
Module: problem_type_analysis.py
Responsibility: Independent module for analyzing Supervised Learning problem types, 
including classification and regression examples with synthetic data.
Assignment: Understanding Supervised Learning Problem Types.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def run_scenario_analysis():
    print("\n" + "="*60)
    print("📈 PART 1: SUPERVISED LEARNING SCENARIO ANALYSIS")
    print("="*60)

    scenarios = [
        {
            "id": 1,
            "title": "Fraud Detection",
            "type": "Classification",
            "subtype": "Binary Classification (Imbalanced)",
            "target": "is_fraud (0 or 1)",
            "justification": "The goal is to decide between two distinct categories: Fraudulent or Legitimate.",
            "algorithms": "Random Forest, XGBoost, Logistic Regression",
            "metrics": "AUPRC, Recall, F1-Score (Accuracy is deceptive here)",
            "wrong_approach": "Regression is wrong because probability of fraud isn't a continuous scale like price; we need a categorical decision."
        },
        {
            "id": 2,
            "title": "House Price Prediction",
            "type": "Regression",
            "subtype": "Simple/Multiple Regression",
            "target": "SalePrice (Continuous Currency)",
            "justification": "We are predicting a continuous numerical value that can vary infinitely within a range.",
            "algorithms": "Linear Regression, Gradient Boosting, Ridge",
            "metrics": "MAE, RMSE, R-squared",
            "wrong_approach": "Classification is wrong because rounding prices into 'buckets' (e.g., $100k-$200k) loses precise value information."
        },
        {
            "id": 3,
            "title": "Movie Genre Tagging",
            "type": "Classification",
            "subtype": "Multi-Label Classification",
            "target": "Genres (Action, Comedy, Drama, etc.)",
            "justification": "A single movie can belong to multiple categories simultaneously (e.g., Action AND Comedy).",
            "algorithms": "OneVsRest Classifier, Binary Relevance",
            "metrics": "Micro/Macro F1-Score, Hamming Loss",
            "wrong_approach": "Multi-class classification (predicting only one genre) is wrong because it fails to capture the true nature of films."
        },
        {
            "id": 4,
            "title": "Product Demand Prediction",
            "type": "Regression",
            "subtype": "Time Series / Forecasting",
            "target": "Quantity_Sold (Integer/Continuous)",
            "justification": "Predicting the volume of sales based on historical trends for inventory management.",
            "algorithms": "XGBoost Regressor, ARIMA, Prophet",
            "metrics": "MAPE, MAE",
            "wrong_approach": "Classification is wrong because demand fluctuates on a spectrum; treating each number of units as a class is impossible."
        },
        {
            "id": 5,
            "title": "Disease Classification",
            "type": "Classification",
            "subtype": "Multi-Class Classification",
            "target": "Disease_Type (e.g., Type A, Type B, Type C)",
            "justification": "Predicting which specific disease a person has out of a finite set of possibilities.",
            "algorithms": "Support Vector Machines, Neural Networks",
            "metrics": "Recall (Critical), Precision, Confusion Matrix",
            "wrong_approach": "Regression is wrong because disease types have no mathematical order (Type A is not 'less than' Type B)."
        }
    ]

    for s in scenarios:
        print(f"\n🔹 SCENARIO {s['id']}: {s['title']}")
        print(f"   - Type: {s['type']} ({s['subtype']})")
        print(f"   - Target Variable: {s['target']}")
        print(f"   - Justification: {s['justification']}")
        print(f"   - Algorithms: {s['algorithms']}")
        print(f"   - Evaluation Metrics: {s['metrics']}")
        print(f"   - Why others fail: {s['wrong_approach']}")

def run_classification_example():
    print("\n" + "="*60)
    print("🎯 PART 2: CLASSIFICATION EXAMPLE (Synthetic)")
    print("="*60)

    # 1. Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = clf.predict(X_test)
    print(f"✅ Model: Logistic Regression")
    print(f"📊 Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"📊 Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"📊 Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"📊 F1 Score:  {f1_score(y_test, y_pred):.4f}")

def run_regression_example():
    print("\n" + "="*60)
    print("📏 PART 3: REGRESSION EXAMPLE (Synthetic)")
    print("="*60)

    # 1. Create synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = reg.predict(X_test)
    print(f"✅ Model: Linear Regression")
    print(f"📊 MAE:   {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"📊 RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"📊 R²:    {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    run_scenario_analysis()
    run_classification_example()
    run_regression_example()
    print("\n" + "="*60)
    print("🏁 ANALYSIS MODULE COMPLETED")
    print("="*60)
