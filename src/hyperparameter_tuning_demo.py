import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_tuning_demo():
    print("\n" + "="*60)
    print("🎯 HYPERPARAMETER TUNING: GridSearchCV DEMONSTRATION")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    y = df['Python'] # Focus on 'Python' label

    # 2. Split (with stratification)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. BASELINE MODEL
    print("\n🚀 Phase 1: Training Baseline Decision Tree (Default Params)...")
    baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
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
    diagnosis = "High Variance (Overfitting)" if (train_acc - test_acc) > 0.1 else "Balanced"
    print(f"👉 Diagnosis:    {diagnosis}")
    print("-" * 40)

    # 4. DEFINE GRID
    print("\n📦 Phase 2: Defining Hyperparameter Grid...")
    param_grid = {
        "model__max_depth": [2, 4, 6, 8, 10],
        "model__min_samples_leaf": [1, 5, 10]
    }
    print(f"Grid: {param_grid}")

    # 5. GRID SEARCH
    print("\n🔍 Phase 3: Running GridSearchCV (on Training Data only)...")
    grid_search = GridSearchCV(
        baseline_pipe,
        param_grid,
        cv=3,
        scoring="accuracy",
        return_train_score=True
    )
    grid_search.fit(X_train_raw, y_train)

    # 6. BEST MODEL
    best_model = grid_search.best_estimator_
    print("\n📊 GRID SEARCH RESULTS:")
    print("-" * 40)
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print("-" * 40)

    # 7. FINAL TEST
    y_pred = best_model.predict(X_test_raw)
    tuned_train_acc = best_model.score(X_train_raw, y_train)
    tuned_test_acc = best_model.score(X_test_raw, y_test)

    # 8. COMPARISON
    print("\n📊 TUNED MODEL PERFORMANCE:")
    print("-" * 40)
    print(f"Train Accuracy: {tuned_train_acc:.4f}")
    print(f"Test Accuracy:  {tuned_test_acc:.4f}")
    print(f"Improvement:    {tuned_test_acc - test_acc:+.4f}")
    print(f"Gap Reduced:    {(train_acc - test_acc) - (tuned_train_acc - tuned_test_acc):.4f}")
    print("-" * 40)

    # STEP 9: ASSIGNMENT COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Train 100%, Test 73%. What is this?")
    print("A: This is clear Overfitting (High Variance). The model has memorized the")
    print("   training set but fails to generalize to the test set.")
    
    print("\nQ: Why is CV still high if the model overfits?")
    print("A: Because the model is overfitting every folder independently. High CV")
    print("   accuracy without checking the gap between CV and Train doesn't reveal")
    print("   overfitting by itself.")
    
    print("\nQ: How to fix the grid if overfitting persists?")
    print("A: Reduce the values of 'max_depth' or increase 'min_samples_leaf'.")
    print("   This 'prunes' the tree and forces it to be simpler.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 TUNING DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_tuning_demo()
