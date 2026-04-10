import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def perform_diagnosis(train_acc, test_acc):
    """Diagnoses model state based on accuracy gaps."""
    if train_acc < 0.7 and test_acc < 0.7:
        return "High Bias"
    elif train_acc > 0.9 and (train_acc - test_acc) > 0.1:
        return "High Variance"
    else:
        return "Balanced"

def run_bias_variance_analysis():
    print("\n" + "="*60)
    print("📉 BIAS-VARIANCE DIAGNOSIS & FIX DEMONSTRATION")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    y = df['Python'] # Using Python as the representative target

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. INITIAL MODEL (Intentionally High Bias - using very high regularization)
    print("\n🚀 Phase 1: Training Initial Model (Inducing High Bias)...")
    initial_model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(C=0.1, max_iter=1000)) # High regularization (low C)
    ])
    initial_model.fit(X_train, y_train)

    # Compute Initial Metrics
    init_train_acc = initial_model.score(X_train, y_train)
    init_test_acc = initial_model.score(X_test, y_test)
    init_cv_scores = cross_val_score(initial_model, X_train, y_train, cv=3)
    
    init_diagnosis = perform_diagnosis(init_train_acc, init_test_acc)

    print("\n📊 INITIAL METRICS:")
    print("-" * 40)
    print(f"Train Accuracy: {init_train_acc:.4f}")
    print(f"Test Accuracy:  {init_test_acc:.4f}")
    print(f"CV Mean Score:  {init_cv_scores.mean():.4f} ± {init_cv_scores.std():.4f}")
    print(f"👉 Diagnosis:    {init_diagnosis}")
    print("-" * 40)

    # 4. FIX MODEL (Based on Diagnosis)
    print(f"\n🛠️  Applying Fix for {init_diagnosis}...")
    if init_diagnosis == "High Bias":
        # Fix: Reduce regularization (increase C) or add features
        fixed_c = 10.0
        reason = "Increased complexity (Reduced Regularization)"
    elif init_diagnosis == "High Variance":
        # Fix: Increase regularization (decrease C) or simplify
        fixed_c = 0.5
        reason = "Increased Regularization"
    else:
        fixed_c = 1.0
        reason = "Standard baseline check"

    # 5. RETRAIN
    fixed_model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(C=fixed_c, max_iter=1000))
    ])
    fixed_model.fit(X_train, y_train)

    # Recompute Final Metrics
    final_train_acc = fixed_model.score(X_train, y_train)
    final_test_acc = fixed_model.score(X_test, y_test)
    final_cv_scores = cross_val_score(fixed_model, X_train, y_train, cv=3)
    
    final_diagnosis = perform_diagnosis(final_train_acc, final_test_acc)

    print("\n📊 FINAL RESULTS (After Fix):")
    print("-" * 40)
    print(f"Fix Applied:     {reason}")
    print(f"Train Accuracy:  {final_train_acc:.4f}")
    print(f"Test Accuracy:   {final_test_acc:.4f}")
    print(f"CV Mean Score:   {final_cv_scores.mean():.4f} ± {final_cv_scores.std():.4f}")
    print(f"👉 Diagnosis:     {final_diagnosis}")
    print("-" * 40)

    # STEP 6: ASSIGNMENT COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Decision Tree has Train=100% and Test=71%. What is this?")
    print("A: This is High Variance (Overfitting). The model memorized the training")
    print("   data but failed to generalize to the test data.")
    
    print("\nQ: Why did limiting depth help?")
    print("A: It simplified the model, preventing it from learning noise as patterns.")
    
    print("\nQ: Does more data help?")
    print("A: Yes, for High Variance. More data helps the model learn the true signal.")
    print("   For High Bias, more data rarely helps; you need a better model.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_bias_variance_analysis()
