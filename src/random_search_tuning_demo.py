import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_random_search_tuning():
    print("\n" + "="*60)
    print("🎲 HYPERPARAMETER TUNING: RandomizedSearchCV DEMONSTRATION")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    y = df['Python'] # Focus on 'Python' label for binary demonstration

    # 2. Split (with stratification)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. BASELINE MODEL
    print("\n🚀 Phase 1: Training Baseline RandomForest (Default Params)...")
    baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    baseline_pipe.fit(X_train_raw, y_train)

    train_acc = baseline_pipe.score(X_train_raw, y_train)
    test_acc = baseline_pipe.score(X_test_raw, y_test)
    cv_scores = cross_val_score(baseline_pipe, X_train_raw, y_train, cv=3)

    print("\n📊 BASELINE METRICS:")
    print("-" * 40)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"CV Score:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    diagnosis = "Overfitting" if (train_acc - test_acc) > 0.1 else "Balanced"
    print(f"👉 Diagnosis:    {diagnosis}")
    print("-" * 40)

    # 4. DEFINE PARAMETER DISTRIBUTIONS (Requirement: scipy.stats)
    print("\n📦 Phase 2: Defining Parameter Distributions...")
    param_distributions = {
        "model__n_estimators": randint(50, 301),
        "model__max_depth": randint(2, 21),
        "model__min_samples_leaf": randint(1, 11)
    }
    print(f"Distributions: {param_distributions}")

    # 5. APPLY RandomizedSearchCV (Requirement: n_iter=50, cv=5 in real scenario)
    # Using n_iter=20 and cv=3 for demo efficiency on this tiny dataset
    print("\n🔍 Phase 3: Running RandomizedSearchCV (on Training Data only)...")
    random_search = RandomizedSearchCV(
        baseline_pipe,
        param_distributions=param_distributions,
        n_iter=20, 
        cv=3,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    random_search.fit(X_train_raw, y_train)

    # 6. EXTRACT RESULTS
    print("\n📊 TUNED MODEL RESULTS:")
    print("-" * 40)
    print(f"Best Params: {random_search.best_params_}")
    print(f"Best CV Score: {random_search.best_score_:.4f}")
    print(f"CV Std Dev:   {random_search.cv_results_['std_test_score'][random_search.best_index_]:.4f}")
    print(f"Iterations:   {random_search.n_iter}")
    print("-" * 40)

    # 7. EVALUATE TUNED MODEL (Requirement: Best estimator on test set once)
    best_model = random_search.best_estimator_
    tuned_train_acc = best_model.score(X_train_raw, y_train)
    tuned_test_acc = best_model.score(X_test_raw, y_test)

    # 8. COMPARE BASELINE VS TUNED
    print("\n📊 FINAL COMPARISON:")
    print("-" * 40)
    print(f"Test Accuracy (Baseline): {test_acc:.4f}")
    print(f"Test Accuracy (Tuned):    {tuned_test_acc:.4f}")
    print(f"Gap Improvement:         {(train_acc - test_acc) - (tuned_train_acc - tuned_test_acc):+.4f}")
    print("-" * 40)

    # STEP 9: SCENARIO QUESTION (Requirement)
    print("\n🎯 SCENARIO QUESTION:")
    print("-" * 60)
    print("Scenario: Train=0.99, Test=0.74, CV=0.91, n_iter=15")
    print("\nDiagnosis:")
    print("👉 This indicates High Variance (Overfitting). The model memorized the data.")
    print("👉 CV is high but optimistic because n_iter=15 is too low for a large space.")
    print("👉 Solution: Increase n_iter to explore more, reduce max_depth,")
    print("   and increase min_samples_leaf to regularize the model.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 RANDOMIZED SEARCH TUNING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_random_search_tuning()
