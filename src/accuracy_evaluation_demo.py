import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_accuracy_evaluation():
    print("\n" + "="*60)
    print("🎯 CLASSIFICATION EVALUATION: ACCURACY & CONFUSION MATRIX")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    
    # Focusing on a single label ('Python') for clear stratification and binary evaluation
    y = df['Python']
    
    # 2. Train-Test Split (with stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Model Pipeline (TF-IDF + Logistic Regression)
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # 4. Baseline Pipeline (Most Frequent Class)
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])

    # 5. Training
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)

    # 6. Predictions
    model_pred = model.predict(X_test)
    baseline_pred = baseline.predict(X_test)

    # 7. Accuracy Calculation
    model_acc = accuracy_score(y_test, model_pred)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # 8. Additional Metric: F1-score
    f1 = f1_score(y_test, model_pred, zero_division=0)

    # 9. Cross Validation (Accuracy)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    # 10. PRINT OUTPUT
    print("\n📊 Metric Comparison:")
    print("-" * 40)
    print(f"🏁 Baseline Accuracy: {baseline_acc:.4f}")
    print(f"🎯 Model Accuracy:    {model_acc:.4f}")
    print(f"🚀 Improvement:      {model_acc - baseline_acc:+.4f}")
    print(f"📈 F1-score:         {f1:.4f}")
    print("-" * 40)

    print(f"\n🔄 5-Fold Cross-Validation (Accuracy):")
    print(f"   Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 11. Confusion Matrix
    print("\n📑 Confusion Matrix (Logistic Regression):")
    cm = confusion_matrix(y_test, model_pred)
    print(cm)
    print("\n   [ Interpretation: [ [TN, FP], [FN, TP] ] ]")
    
    # Optional comment for Visual display Requirement
    # ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    # STEP 12: ASSIGNMENT COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Why is accuracy alone not enough?")
    print("A: In an imbalanced dataset (e.g., 90% Python projects), a classifier could")
    print("   reach 90% accuracy by saying NO project has other skills. This is useless.")
    
    print("\nQ: What does the Confusion Matrix tell us?")
    print("A: It breaks down errors into False Positives (over-detection) and")
    print("   False Negatives (missed detection). It shows exactly where the model fails.")
    
    print("\nQ: Why use F1-score?")
    print("A: F1-score balances precision and recall. It is only high if BOTH precision")
    print("   and recall are high, making it more honest for imbalanced data.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 EVALUATION COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_accuracy_evaluation()
