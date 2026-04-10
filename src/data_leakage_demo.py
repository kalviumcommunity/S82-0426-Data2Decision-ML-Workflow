import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_leakage_demo():
    print("\n" + "="*60)
    print("🚨 DATA LEAKAGE DEMONSTRATION: INCORRECT VS. CORRECT")
    print("="*60)

    # Load Data
    df = load_data(DATA_PATH)
    X_raw = df['project_description'].apply(clean_text)
    y = df['Python'] # Focus on 'Python' label

    # ---------------------------------------------------------
    # 🔴 PART 1: INCORRECT WORKFLOW (LEAKAGE)
    # ---------------------------------------------------------
    print("\n🔴 PHASE 1: Introducing Data Leakage (The WRONG Way)...")
    
    # ❌ Step 1: TF-IDF on FULL dataset BEFORE split
    # Leakage: The vectorizer has 'seen' word distributions of the entire set.
    wrong_vectorizer = TfidfVectorizer()
    X_all_leaky = wrong_vectorizer.fit_transform(X_raw)
    
    # ❌ Step 2: Split vectorized data
    # The test set is no longer truly unseen.
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_all_leaky, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train model
    leaky_model = LogisticRegression(random_state=RANDOM_STATE)
    leaky_model.fit(X_train_l, y_train_l)
    
    # Report Results
    leaky_train_acc = leaky_model.score(X_train_l, y_train_l)
    leaky_test_acc = leaky_model.score(X_test_l, y_test_l)
    leaky_cv_score = cross_val_score(leaky_model, X_all_leaky, y, cv=3).mean()
    
    print("\n📊 LEAKY METRICS (Inflated):")
    print("-" * 40)
    print(f"Train Accuracy: {leaky_train_acc:.4f}")
    print(f"Test Accuracy:  {leaky_test_acc:.4f}")
    print(f"CV Avg Score:   {leaky_cv_score:.4f}")
    print("-" * 40)

    # ---------------------------------------------------------
    # 🟢 PART 2: CORRECT WORKFLOW (PIPELINE)
    # ---------------------------------------------------------
    print("\n🟢 PHASE 2: Proper Preprocessing with Pipeline (The RIGHT Way)...")
    
    # ✅ Step 1: Split Raw Data FIRST
    # The test set remains completely isolated.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # ✅ Step 2: Build Pipeline (Preprocessing inside)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(random_state=RANDOM_STATE))
    ])
    
    # ✅ Step 3: CV on Pipeline (Safe)
    # In each fold, TF-IDF is refitted only on the training portion.
    safe_cv_scores = cross_val_score(pipeline, X_train_raw, y_train, cv=3)
    
    # ✅ Step 4: Train + Evaluate
    pipeline.fit(X_train_raw, y_train)
    pipeline_test_acc = pipeline.score(X_test_raw, y_test)
    
    print("\n📊 PIPELINE METRICS (Honest):")
    print("-" * 40)
    print(f"CV Mean Scores: {safe_cv_scores.mean():.4f} ± {safe_cv_scores.std():.4f}")
    print(f"Test Accuracy:  {pipeline_test_acc:.4f}")
    print("-" * 40)

    # ---------------------------------------------------------
    # 📈 COMPARISON TABLE
    # ---------------------------------------------------------
    print("\n📈 FINAL COMPARISON:")
    print("-" * 65)
    print(f"{'Approach':<15} | {'CV Score':<10} | {'Test Score':<10} | {'Trustworthy?':<12}")
    print("-" * 65)
    print(f"{'🔴 Leaky':<15} | {leaky_cv_score:<10.4f} | {leaky_test_acc:<10.4f} | ❌ No")
    print(f"{'🟢 Pipeline':<15} | {safe_cv_scores.mean():<10.4f} | {pipeline_test_acc:<10.4f} | ✅ Yes")
    print("-" * 65)

    # MANDATORY CONCEPT SECTION
    print("\n🧠 CONCEPT RECAP:")
    print("-" * 60)
    print("Q: Why must preprocessing be fitted only on training data?")
    print("A: Because test data must remain unseen. Otherwise, the model 'cheats' by")
    print("   learning from future information during feature extraction.")
    
    print("\nQ: How does the pipeline fix leakage?")
    print("A: It fits the entire chain (TF-IDF + Model) separately inside each CV")
    print("   fold, preventing any contamination between folds.")
    
    print("\nQ: Why lower honest score is better?")
    print("A: It reflects true generalization performance. A high leaky score breeds")
    print("   false confidence and will lead to catastrophic production failure.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 LEAKAGE DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_leakage_demo()
