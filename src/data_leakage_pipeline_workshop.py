import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_workshop():
    print("\n" + "="*60)
    print("DATA LEAKAGE & PIPELINE WORKSHOP")
    print("="*60)

    # Load and clean data
    df = load_data(DATA_PATH)
    # For demonstration, we'll use 'project_description' as feature and 'Python' as target
    X_raw = df["project_description"].apply(clean_text)
    y = df["Python"]

    # ---------------------------------------------------------
    # PART 1 -- INCORRECT WORKFLOW (LEAKAGE)
    # ---------------------------------------------------------
    print("\nPART 1: INCORRECT WORKFLOW (LEAKAGE)")
    print("-" * 40)
    
    # Step 1: Introduce Leakage
    vectorizer = TfidfVectorizer()
    X_all = vectorizer.fit_transform(X_raw)  # LEAKAGE: Fit on FULL dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model_leaky = LogisticRegression(max_iter=1000)
    model_leaky.fit(X_train, y_train)

    # Step 2: Cross-validation (WRONG)
    # Performing CV on already transformed data that saw the whole set
    wrong_cv_scores = cross_val_score(model_leaky, X_all, y, cv=5)

    # Step 3: Report Results
    train_acc_leaky = model_leaky.score(X_train, y_train)
    test_acc_leaky = model_leaky.score(X_test, y_test)
    cv_mean_leaky = wrong_cv_scores.mean()

    print(f"Train Accuracy: {train_acc_leaky:.4f}")
    print(f"Test Accuracy:  {test_acc_leaky:.4f}")
    print(f"CV Score (WRONG): {cv_mean_leaky:.4f}")

    # Step 4: Explanation (MANDATORY)
    print("\nEXPLANATION:")
    print("- Leakage occurred because preprocessing (TF-IDF) was fitted on full dataset.")
    print("- Test data influenced feature extraction (the vocabulary and IDF weights).")
    print("- Model indirectly saw test information during the training phase.")
    print("- Metrics are inflated and misleading.")
    print("- This will fail in production because the real-world data won't be seen beforehand.")

    # ---------------------------------------------------------
    # PART 2 -- CORRECT WORKFLOW (PIPELINE)
    # ---------------------------------------------------------
    print("\nPART 2: CORRECT WORKFLOW (PIPELINE)")
    print("-" * 40)

    # Step 1: Proper Split FIRST
    # We split the RAW text data
    X_train_raw, X_test_raw, y_train_safe, y_test_safe = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 2: Build Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Step 3: Cross-validation (CORRECT)
    cv_scores_safe = cross_val_score(pipeline, X_train_raw, y_train_safe, cv=5)

    # Step 4: Train + Evaluate
    pipeline.fit(X_train_raw, y_train_safe)
    y_pred = pipeline.predict(X_test_raw)
    test_acc_safe = pipeline.score(X_test_raw, y_test_safe)

    # Step 5: Report Results
    print(f"CV Score (Safe): {cv_scores_safe.mean():.4f} ± {cv_scores_safe.std():.4f}")
    print(f"Test Accuracy:   {test_acc_safe:.4f}")

    # Step 6: Compare
    print("\nFINAL COMPARISON:")
    print("-" * 65)
    print(f"{'Approach':<15} | {'CV Score':<12} | {'Test Score':<10} | {'Trustworthy':<12}")
    print("-" * 65)
    print(f"{'Leaky':<15} | {cv_mean_leaky:<12.4f} | {test_acc_leaky:<10.4f} | No")
    print(f"{'Pipeline':<15} | {cv_scores_safe.mean():<12.4f} | {test_acc_safe:<10.4f} | Yes")
    print("-" * 65)

    # CONCEPT ANSWERS (MANDATORY)
    print("\nCONCEPT ANSWERS:")
    print("-" * 60)
    print("1. Why preprocessing must be fitted on training data?")
    print("   Because test data must remain unseen. Otherwise model learns from future information.")
    
    print("\n2. How pipeline fixes leakage?")
    print("   Pipeline fits preprocessing separately inside each CV fold, preventing contamination.")
    
    print("\n3. Why scaling/TF-IDF before GridSearch causes leakage?")
    print("   Because validation folds are already used during preprocessing.")
    
    print("\n4. Why leakage is dangerous?")
    print("   Because it gives false confidence and fails in production.")
    
    print("\n5. Why lower honest score is better?")
    print("   Because it reflects true generalization performance.")
    print("-" * 60)

    # SCENARIO ANSWER
    print("\nSCENARIO ANSWER:")
    print("-" * 60)
    print("The 92% -> 78% drop happened because:")
    print("- Preprocessing used full dataset (Data Leakage).")
    print("- Model saw test data indirectly through statistics (IDF/Range).")
    print("- Metrics were inflated during testing.")
    print("- Pipeline would prevent this by isolating training data during every step.")
    print("-" * 60)

    print("\n" + "="*60)
    print("WORKSHOP COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_workshop()
