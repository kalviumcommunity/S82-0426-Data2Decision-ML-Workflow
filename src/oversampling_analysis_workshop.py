import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_oversampling_analysis():
    print("\n" + "="*60)
    print("OVERSAMPLING TECHNIQUES ANALYSIS WORKSHOP")
    print("="*60)

    # Load and Preprocess
    df = load_data(DATA_PATH)
    # We'll use 'Flask' as minority class (2 samples out of 10 in total data)
    X_raw = df["project_description"].apply(clean_text)
    y = df["Flask"]

    # Stratified Split (MANDATORY)
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------------------------------------------------
    # PART 1 -- BASELINE MODEL (NO OVERSAMPLING)
    # ---------------------------------------------------------
    print("\nPART 1: BASELINE MODEL (NO OVERSAMPLING)")
    print("-" * 40)
    
    baseline_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)

    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base, pos_label=1, zero_division=0)
    rec_base = recall_score(y_test, y_pred_base, pos_label=1, zero_division=0)
    f1_base = f1_score(y_test, y_pred_base, pos_label=1, zero_division=0)
    cm_base = confusion_matrix(y_test, y_pred_base)

    print(f"Accuracy:  {acc_base:.4f}")
    print(f"Precision: {prec_base:.4f}")
    print(f"Recall:    {rec_base:.4f}")
    print(f"F1-score:  {f1_base:.4f}")
    print(f"Confusion Matrix:\n{cm_base}")

    # ---------------------------------------------------------
    # PART 2 -- RANDOM OVERSAMPLING
    # ---------------------------------------------------------
    print("\nPART 2: RANDOM OVERSAMPLING")
    print("-" * 40)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    model_ros = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model_ros.fit(X_train_ros, y_train_ros)
    y_pred_ros = model_ros.predict(X_test)

    acc_ros = accuracy_score(y_test, y_pred_ros)
    prec_ros = precision_score(y_test, y_pred_ros, pos_label=1, zero_division=0)
    rec_ros = recall_score(y_test, y_pred_ros, pos_label=1, zero_division=0)
    f1_ros = f1_score(y_test, y_pred_ros, pos_label=1, zero_division=0)

    print(f"Accuracy:  {acc_ros:.4f}")
    print(f"Precision: {prec_ros:.4f}")
    print(f"Recall:    {rec_ros:.4f}")
    print(f"F1-score:  {f1_ros:.4f}")

    # ---------------------------------------------------------
    # PART 3 -- SMOTE
    # ---------------------------------------------------------
    print("\nPART 3: SMOTE")
    print("-" * 40)

    try:
        smote_k = max(1, y_train.sum()-1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=int(smote_k))
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        model_smote = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model_smote.fit(X_train_smote, y_train_smote)
        y_pred_smote = model_smote.predict(X_test)

        acc_smote = accuracy_score(y_test, y_pred_smote)
        prec_smote = precision_score(y_test, y_pred_smote, pos_label=1, zero_division=0)
        rec_smote = recall_score(y_test, y_pred_smote, pos_label=1, zero_division=0)
        f1_smote = f1_score(y_test, y_pred_smote, pos_label=1, zero_division=0)
    except Exception as e:
        print(f"SMOTE Error: {e}")
        acc_smote = prec_smote = rec_smote = f1_smote = 0.0

    print(f"Accuracy:  {acc_smote:.4f}")
    print(f"Precision: {prec_smote:.4f}")
    print(f"Recall:    {rec_smote:.4f}")
    print(f"F1-score:  {f1_smote:.4f}")

    # ---------------------------------------------------------
    # PART 4 -- COMPARISON TABLE
    # ---------------------------------------------------------
    print("\nPART 4: COMPARISON TABLE")
    print("-" * 65)
    print(f"{'Metric':<12} | {'Baseline':<10} | {'Random OS':<10} | {'SMOTE':<10}")
    print("-" * 65)
    print(f"{'Accuracy':<12} | {acc_base:<10.4f} | {acc_ros:<10.4f} | {acc_smote:<10.4f}")
    print(f"{'Precision':<12} | {prec_base:<10.4f} | {prec_ros:<10.4f} | {prec_smote:<10.4f}")
    print(f"{'Recall':<12} | {rec_base:<10.4f} | {rec_ros:<10.4f} | {rec_smote:<10.4f}")
    print(f"{'F1-score':<12} | {f1_base:<10.4f} | {f1_ros:<10.4f} | {f1_smote:<10.4f}")
    print("-" * 65)

    # ---------------------------------------------------------
    # PART 5 -- CROSS VALIDATION (imblearn Pipeline)
    # ---------------------------------------------------------
    print("\nPART 5: CROSS VALIDATION (imblearn Pipeline)")
    print("-" * 40)

    try:
        cv_pipeline = Pipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=int(max(1, y_train.sum()-1)))),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE) 
        cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=skf, scoring="f1")
        print(f"F1 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"CV Error: {e}")

    print("\nEXPLANATION:")
    print("Oversampling must happen INSIDE each fold.")
    print("Otherwise leakage occurs because the oversampler 'sees' information from the validation fold.")

    # ---------------------------------------------------------
    # PART 6 -- SCENARIO QUESTIONS
    # ---------------------------------------------------------
    print("\nPART 6: SCENARIO QUESTIONS")
    print("-" * 60)
    print("Q1: Why not oversample before split?")
    print("   Causes data leakage; the model essentially memorizes duplicates of samples it hasn't officially 'seen' yet.")
    
    print("\n2: SMOTE vs Random?")
    print("   Random uses exact duplicates (overfitting risk). SMOTE creates synthetic points between nearest neighbors.")
    
    print("\nQ3: Why precision decreases?")
    print("   The model becomes 'aggressive' at identifying minority cases, leading to more False Alarms (False Positives).")
    
    print("\nQ4: When prefer class weights?")
    print("   For very small datasets where synthetic data generation (SMOTE) might be noisy/unreliable.")
    
    print("\nQ5: Does oversampling create new info?")
    print("   No. It only amplifies existing patterns and densities in the current feature space.")
    print("-" * 60)

    print("\n" + "="*60)
    print("WORKSHOP COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_oversampling_analysis()
