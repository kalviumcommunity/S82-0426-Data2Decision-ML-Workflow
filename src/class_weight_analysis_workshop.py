import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_class_weight_analysis():
    print("\n" + "="*60)
    print("CLASS WEIGHTS ANALYSIS WORKSHOP")
    print("="*60)

    # 1. Load and Preprocess Data
    df = load_data(DATA_PATH)
    # We'll use 'Flask' as a minority class for demonstration
    X_raw = df["project_description"].apply(clean_text)
    y = df["Flask"]

    # ---------------------------------------------------------
    # PART 1 -- BASELINE MODEL (WITHOUT WEIGHTS)
    # ---------------------------------------------------------
    print("\nPART 1: BASELINE MODEL (WITHOUT WEIGHTS)")
    print("-" * 40)

    # Step 1: Stratified Split
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 2: Train Model (No Weights)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Step 3: Report Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # Using zero_division=0 to handle cases where precision might be undefined for small datasets
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"Accuracy:         {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision (Flask): {prec:.4f}")
    print(f"Recall (Flask):    {rec:.4f}")
    print(f"F1-score (Flask):  {f1:.4f}")

    print("\nEXPLANATION:")
    print("- Model favors majority class (often predicts '0' for everything).")
    print("- Minority recall is low (detects few or no 'Flask' projects).")
    print("- Accuracy is misleading (high error on minority is hidden).")
    print("Key idea: Majority class dominates learning.")

    # ---------------------------------------------------------
    # PART 2 -- APPLY CLASS WEIGHTS
    # ---------------------------------------------------------
    print("\nPART 2: APPLY CLASS WEIGHTS")
    print("-" * 40)

    # Step 1: Train Weighted Model
    model_weighted = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model_weighted.fit(X_train, y_train)
    y_pred_weighted = model_weighted.predict(X_test)

    # Step 2: Report Metrics
    acc_w = accuracy_score(y_test, y_pred_weighted)
    cm_w = confusion_matrix(y_test, y_pred_weighted)
    prec_w = precision_score(y_test, y_pred_weighted, pos_label=1, zero_division=0)
    rec_w = recall_score(y_test, y_pred_weighted, pos_label=1, zero_division=0)
    f1_w = f1_score(y_test, y_pred_weighted, pos_label=1, zero_division=0)

    print(f"Accuracy:         {acc_w:.4f}")
    print(f"Confusion Matrix:\n{cm_w}")
    print(f"Precision (Flask): {prec_w:.4f}")
    print(f"Recall (Flask):    {rec_w:.4f}")
    print(f"F1-score (Flask):  {f1_w:.4f}")

    print("\nEXPECTED BEHAVIOR:")
    print("- Recall  (important): Usually increases.")
    print("- Precision (trade-off): Usually decreases.")
    print("- Accuracy (normal): Usually drops slightly.")
    print("Reason: Model now focuses on minority class.")

    # ---------------------------------------------------------
    # PART 3 -- COMPARISON TABLE
    # ---------------------------------------------------------
    print("\nPART 3: COMPARISON TABLE")
    print("-" * 55)
    print(f"{'Metric':<12} | {'Without Weights':<16} | {'With Weights':<12}")
    print("-" * 55)
    print(f"{'Accuracy':<12} | {acc:<16.4f} | {acc_w:<12.4f}")
    print(f"{'Precision':<12} | {prec:<16.4f} | {prec_w:<12.4f}")
    print(f"{'Recall':<12} | {rec:<16.4f} | {rec_w:<12.4f}")
    print(f"{'F1-score':<12} | {f1:<16.4f} | {f1_w:<12.4f}")
    print("-" * 55)

    # PART 4 -- ANALYSIS
    print("\nPART 4: ANALYSIS")
    print("-" * 60)
    print("1. How did recall change?")
    print("   Increased significantly (Model became 'braver' at identifying minority samples).")
    
    print("\n2. Why precision decreased?")
    print("   More false positives (Correcting minority misses often leads to flagging majority samples incorrectly).")
    
    print("\n3. Why accuracy dropped?")
    print("   More mistakes on majority class (The penalty shift reduces focus on majority correctness).")
    
    print("\n4. Which model is better?")
    print("   Weighted model (Captures minority better, crucial for specialized detection).")
    
    print("\n5. Does class weight fully solve imbalance?")
    print("   No. It only adjusts importance; it doesn't create new data or information.")
    print("-" * 60)

    # PART 5 -- SCENARIO QUESTIONS
    print("\nPART 5: SCENARIO QUESTIONS")
    print("-" * 60)
    print("Q1: Why unweighted models favor majority?")
    print("   Loss function is dominated by majority samples; minimizing overall error naturally ignores minority noisy signals.")
    
    print("\nQ2: How class weights help?")
    print("   They increase the penalty for minority errors, forcing the loss optimizer to prioritize them.")
    
    print("\nQ3: Fraud detection -- FP vs FN?")
    print("   FN is worse (missing fraud = massive financial loss).")
    
    print("\nQ4: Why stratified split still needed?")
    print("   To maintain class distribution across train/test sets, ensuring representative evaluation.")
    
    print("\nQ5: Why not accuracy alone?")
    print("   It hides minority failure (A model that guesses only majority class can have 90% accuracy but 0% utility).")
    print("-" * 60)

    print("\n" + "="*60)
    print("WORKSHOP COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_class_weight_analysis()
