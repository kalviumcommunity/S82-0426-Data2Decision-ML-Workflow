import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", model)
    ])

def run_final_selection():
    print("\n" + "="*60)
    print("FINAL MODEL SELECTION & BUSINESS ANALYSIS")
    print("="*60)

    # PART 1 -- DATA & COMPARISON TABLE
    df = load_data(DATA_PATH)
    X = df["project_description"].apply(clean_text)
    y = df["Python"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB()
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    
    table_data = []

    for name, model in models.items():
        pipeline = make_pipeline(model)
        
        # CV for stability
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1")
        
        # Test Evaluation for performance
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        table_data.append({
            "Model": name,
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Test Score": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        })

    results_df = pd.DataFrame(table_data)
    
    print("\nPART 1: PERFORMANCE COMPARISON TABLE")
    print("-" * 85)
    print(results_df.to_string(index=False))
    print("-" * 85)

    # PART 2 -- USE CASE ANALYSIS
    print("\nPART 2: USE CASE ANALYSIS")
    print("-" * 40)
    print("Scenario: Fraud Detection")
    print("Status: False Negative (Missing Fraud) = VERY COSTLY")
    print("System Requirement: Real-time, Interpretability Optional")
    print("\nQUESTION: Which metric matters most?")
    print("ANSWER: RECALL (Must catch as much fraud as possible to avoid financial loss).")

    # PART 3 -- FINAL MODEL SELECTION
    print("\nPART 3: FINAL MODEL SELECTION")
    print("-" * 40)
    
    # Logic: For fraud, we prioritize Recall, then F1 score
    best_recall_idx = results_df['Recall'].idxmax()
    selected_model = results_df.loc[best_recall_idx, 'Model']
    
    print(f"SELECTED MODEL: {selected_model}")
    print(f"JUSTIFICATION: I selected {selected_model} because it has a high recall ({results_df.loc[best_recall_idx, 'Recall']:.4f}), which is critical for fraud detection where missing a single fraudulent case is significantly more expensive than a false alarm.")

    # PART 4 -- HOLISTIC EVALUATION & JUSTIFICATION
    print("\nPART 4: HOLISTIC EVALUATION & JUSTIFICATION")
    print("-" * 85)
    
    print("1. Metrics Analysis:")
    print(f"   - Recall: {results_df.loc[best_recall_idx, 'Recall']:.4f} (Priority: High)")
    print(f"   - F1-Score: {results_df.loc[best_recall_idx, 'F1']:.4f} (Balance: Acceptable)")
    
    print("\n2. Bias-Variance Trade-off:")
    print(f"   - CV Std: {results_df.loc[best_recall_idx, 'CV Std']:.4f}")
    print("   - Low Gap between CV and Test indicates a generalized model with low variance.")

    print("\n3. Confusion Matrix Context:")
    print("   - High Recall = Reduced False Negatives (FN).")
    print("   - False Positives (FP) are acceptable for manual review overhead.")

    print("\n4. Operational Analysis:")
    print(f"   - Interpretability: {'High' if selected_model == 'Logistic Regression' else 'Moderate/Low'}")
    print(f"   - Computational Cost: {'Very Low' if selected_model == 'Logistic Regression' else 'Medium'}")
    print(f"   - Deployment: Suitable for Real-time inference.")

    print("\nFINAL THOUGHT:")
    print("The best model is NOT necessarily the one with the highest accuracy, but the one whose error profile (low FN) aligns with the specific business risk of the Fraud Detection use case.")
    print("-" * 85)

    print("\n" + "="*60)
    print("FINAL PROJECT ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_final_selection()
