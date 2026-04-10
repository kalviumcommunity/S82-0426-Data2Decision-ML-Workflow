import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_f1_score_demo():
    print("\n" + "="*60)
    print("⚖️  EVALUATION: F1-SCORE & BALANCED PERFORMANCE")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    
    # Using 'Python' as the representative target for clean binary metrics
    y = df['Python']
    
    # 2. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Pipelines
    # Logistic Regression Pipeline
    model_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # Baseline Pipeline (Most Frequent)
    baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])

    # 4. Training
    model_pipe.fit(X_train, y_train)
    baseline_pipe.fit(X_train, y_train)

    # 5. Predictions
    y_pred_model = model_pipe.predict(X_test)
    y_pred_base = baseline_pipe.predict(X_test)

    # 6. Metrics Computation
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_true, y_pred, zero_division=0)
        }

    model_metrics = get_metrics(y_test, y_pred_model)
    base_metrics = get_metrics(y_test, y_pred_base)

    # 7. PRINT OUTPUT
    print("\n=== BASELINE ===")
    for m, v in base_metrics.items():
        print(f"{m:<10}: {v:.4f}")

    print("\n=== MODEL ===")
    for m, v in model_metrics.items():
        print(f"{m:<10}: {v:.4f}")

    print("\n📑 Classification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_model, zero_division=0))

    print("\n📊 Confusion Matrix (Logistic Regression):")
    cm = confusion_matrix(y_test, y_pred_model)
    print(cm)
    print("\n   [ TN  FP ]\n   [ FN  TP ]")

    # STEP 8: ASSIGNMENT COMMENTS & JUSTIFICATIONS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: What is the F1-Score?")
    print("A: It is the harmonic mean of Precision and Recall. It provides a single")
    print("   score that balances the two, specially useful when classes are imbalanced.")
    
    print("\nQ: Why is F1 Score important for imbalanced data?")
    print("A: Accuracy can be high even if the model fails on the minority class.")
    print("   F1-Score 'punishes' the model if either precision or recall is low.")
    
    print("\nQ: Disease Detection Scenario: Prioritize Recall or F1?")
    print("A: Prioritize Recall (to avoid missing sick people), but use F1 to ensure")
    print("   you don't have so many False Positives that the healthcare system is")
    print("   overwhelmed by healthy people being tested.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 EVALUATION COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_f1_score_demo()
