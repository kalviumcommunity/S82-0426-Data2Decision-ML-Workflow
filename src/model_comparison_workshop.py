import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def make_pipeline(model):
    """
    Creates a fair preprocessing pipeline for any model.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", model)
    ])

def run_model_comparison():
    print("\n" + "="*60)
    print("FAIR MODEL COMPARISON WORKSHOP")
    print("="*60)

    # ---------------------------------------------------------
    # PART 1 -- DATA PREPARATION
    # ---------------------------------------------------------
    df = load_data(DATA_PATH)
    # We'll use 'Python' as the target for this comparison
    X = df["project_description"].apply(clean_text)
    y = df["Python"]

    # Step 1: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\nPART 1: DATA PREPARATION")
    print("-" * 40)
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print("Pre-requisite: Fair comparison requires same preprocessing.")

    # ---------------------------------------------------------
    # PART 2 -- MULTIPLE MODELS DEFINITION
    # ---------------------------------------------------------
    print("\nPART 2: MULTIPLE MODELS DEFINITION")
    print("-" * 40)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB()
    }
    for name in models:
        print(f"Ready: {name}")

    # ---------------------------------------------------------
    # PART 3 -- CROSS VALIDATION
    # ---------------------------------------------------------
    print("\nPART 3: CROSS VALIDATION (Evaluating Mean & Std)")
    print("-" * 40)
    
    # Low splits for small dataset demo
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    
    cv_results = []
    
    for name, model in models.items():
        pipeline = make_pipeline(model)
        # Using F1 as the metric for imbalanced text data
        scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1")

        cv_results.append({
            "Model": name,
            "CV Mean": scores.mean(),
            "CV Std": scores.std()
        })

    results_df = pd.DataFrame(cv_results)
    print(results_df)

    print("\nINSIGHT:")
    print("- Highest mean -> Best average performance.")
    print("- Lowest std -> Most stable/consistent model across folds.")

    # ---------------------------------------------------------
    # PART 4 -- TEST SET EVALUATION
    # ---------------------------------------------------------
    print("\nPART 4: TEST SET EVALUATION (Using the Best Model)")
    print("-" * 40)
    
    # Selecting the best model based on CV Mean
    best_idx = results_df['CV Mean'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    print(f"Best Model Selected: {best_model_name}")

    final_model = make_pipeline(models[best_model_name])
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred))

    print("\nREMINDER:")
    print("- Test set was used ONLY ONCE.")
    print("- It is NOT used for selecting the best model (CV handles that).")

    # ---------------------------------------------------------
    # PART 5 -- COMPARISON TABLE
    # ---------------------------------------------------------
    print("\nPART 5: FINAL COMPARISON TABLE")
    print("-" * 65)
    print(f"{'Model':<20} | {'CV Mean':<10} | {'CV Std':<10} | {'Status':<12}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        status = "Best Candidate" if row['Model'] == best_model_name else "Baseline"
        print(f"{row['Model']:<20} | {row['CV Mean']:<10.4f} | {row['CV Std']:<10.4f} | {status:<12}")
    print("-" * 65)

    # ---------------------------------------------------------
    # PART 6 -- ANALYSIS QUESTIONS
    # ---------------------------------------------------------
    print("\nPART 6: ANALYSIS QUESTIONS")
    print("-" * 60)
    print("Q1: Why CV > train/test?")
    print("   More reliable and less prone to randomness, as every sample gets to be in a validation fold once.")
    
    print("\nQ2: Why not use test repeatedly?")
    print("   It causes data leakage/overfitting to the test set, making the results overly optimistic and invalid.")
    
    print("\nQ3: Small difference (0.01), which to choose?")
    print("   Choose the one with the lower std, as it demonstrates better stability and consistency.")
    
    print("\nQ4: Bias vs Variance?")
    print("   Simple models (LR) often have high bias/low variance. Complex models (RF) often have low bias/high variance.")
    
    print("\nQ5: When choose lower score model?")
    print("   When it is significantly faster, more interpretable, or more stable (lower variance).")
    print("-" * 60)

    print("\n" + "="*60)
    print("WORKSHOP COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_model_comparison()
