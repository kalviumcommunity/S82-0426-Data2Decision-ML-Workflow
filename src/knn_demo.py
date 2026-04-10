import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

# Transformer to convert sparse TF-IDF to dense for StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

class SparseToDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray()

def run_knn_demo():
    print("\n" + "="*60)
    print("📏 K-NEAREST NEIGHBORS (KNN) MODEL DEMONSTRATION")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    
    # Focusing on 'Python' label for clean binary identification
    y = df['Python']

    # 2. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. PIPELINE (Requirement: Scaling + KNN)
    # We add TF-IDF and SparseToDense to make it a true end-to-end demo
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("dense", SparseToDense()), # Conversion for StandardScaler
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # 4. FIND BEST K (Requirement: GridSearchCV)
    # Small dataset implies small K is better, but we test up to 20 or max available
    max_k = min(len(X_train) - 1, 20)
    param_grid = {
        "knn__n_neighbors": range(1, max_k + 1)
    }

    print(f"\n🚀 Running GridSearchCV to find best K (Range: 1 to {max_k})...")
    # Low CV folds because of tiny dataset
    grid = GridSearchCV(pipeline, param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_k = grid.best_params_['knn__n_neighbors']

    # 5. PREDICTION
    y_pred = best_model.predict(X_test)

    # 6. BASELINE (Requirement)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    # 7. EVALUATION
    accuracy = accuracy_score(y_test, y_pred)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # 8. OUTPUT
    print("\n📊 MODEL RESULTS:")
    print("-" * 40)
    print(f"✅ Best K selected:  {best_k}")
    print(f"🎯 KNN Accuracy:    {accuracy:.4f}")
    print(f"🏁 Baseline Acc:    {baseline_acc:.4f}")
    print(f"🚀 Improvement:     {accuracy - baseline_acc:+.4f}")
    print("-" * 40)

    print("\n📑 Classification Report (Best KNN):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # STEP 9: ASSIGNMENT COMMENTS & JUSTIFICATIONS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: What is KNN?")
    print("A: KNN is a 'lazy learner'. It doesn't learn a global function during")
    print("   training but instead looks up the nearest neighbors during prediction.")
    
    print("\nQ: Why is scaling crucial for KNN?")
    print("A: KNN uses distance (often Euclidean). If one feature has a range of [0, 1]")
    print("   and another has [0, 1000], the second feature will dominate the distance")
    print("   calculation even if it's less important.")
    
    print("\nQ: Small K vs. Large K?")
    print("A: A small K (e.g., 1) is sensitive to noise and overfits. A very large K")
    print("   (e.g., N) effectively just predicts the majority class (underfitting).")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 KNN DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_knn_demo()
