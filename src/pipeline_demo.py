import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_pipeline_demo():
    print("\n" + "="*60)
    print("⛓️ SCIKIT-LEARN PIPELINE: LIFE CYCLE DEMONSTRATION")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X_raw = df['project_description'].apply(clean_text)
    y = df['Python'] # Focus on 'Python' label

    # 2. Split (Proper Workflow)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # --- PART 1: BASELINE (WITHOUT PIPELINE) ---
    print("\n🚀 Phase 1: Manual Workflow (Potential Leakage Risk)...")
    manual_vectorizer = TfidfVectorizer()
    X_train_manual = manual_vectorizer.fit_transform(X_train_raw)
    X_test_manual = manual_vectorizer.transform(X_test_raw)

    manual_model = LogisticRegression(random_state=RANDOM_STATE)
    manual_model.fit(X_train_manual, y_train)
    
    manual_train_acc = manual_model.score(X_train_manual, y_train)
    manual_test_acc = manual_model.score(X_test_manual, y_test)

    print("\n📊 MANUAL METRICS:")
    print("-" * 40)
    print(f"Manual Train Acc: {manual_train_acc:.4f}")
    print(f"Manual Test Acc:  {manual_test_acc:.4f}")
    print("⚠️ Risk: If you run CV on X_train_manual now, leakage is already present.")
    print("-" * 40)

    # --- PART 2: BUILD PIPELINE ---
    print("\n📦 Phase 2: Building Atomic Pipeline...")
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("model", LogisticRegression(random_state=RANDOM_STATE))
    ])

    # --- PART 3: CROSS-VALIDATION WITH PIPELINE ---
    print("\n🔍 Phase 3: Leakage-Free Cross-Validation...")
    # Preprocessing happens INSIDE each fold automatically
    cv_scores = cross_val_score(pipeline, X_train_raw, y_train, cv=3)
    
    print("\n📊 PIPELINE CV RESULTS:")
    print("-" * 40)
    print(f"CV Mean Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("-" * 40)

    # --- PART 4: HYPERPARAMETER TUNING ---
    print("\n🎯 Phase 4: GridSearchCV on Full Pipeline...")
    param_grid = {
        "model__C": [0.1, 1, 10],
        "vectorizer__max_features": [50, 100, None]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3)
    grid_search.fit(X_train_raw, y_train)
    
    best_pipe = grid_search.best_estimator_
    test_acc_pipe = best_pipe.score(X_test_raw, y_test)

    print("\n📊 TUNED PIPELINE METRICS:")
    print("-" * 40)
    print(f"Best Params:    {grid_search.best_params_}")
    print(f"Final Test Acc: {test_acc_pipe:.4f}")
    print("-" * 40)

    # --- PART 5: SAVE PIPELINE ---
    print("\n💾 Phase 5: Saving Pipeline Artifact...")
    joblib.dump(best_pipe, "model_pipeline.pkl")
    print("✅ Saved to: model_pipeline.pkl")

    # STEP 6: SCENARIO QUESTION (Requirement)
    print("\n🎯 SCENARIO QUESTION:")
    print("-" * 60)
    print("Scenario: Preprocessing before split. Test=92%, Production=78%.")
    print("\nDiagnosis:")
    print("👉 Data Leakage occurred. The scaler/vectorizer 'saw' the test data")
    print("   statistics before the split, inflating the metrics.")
    print("\nSolution:")
    print("👉 Use a Pipeline. It ensures fitting only happens on training folds.")
    print("   The test set remains truly unseen until the final predict() call.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 PIPELINE DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_pipeline_demo()
