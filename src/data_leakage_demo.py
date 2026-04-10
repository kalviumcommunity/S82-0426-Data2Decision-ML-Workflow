"""
Module: data_leakage_demo.py
Assignment: Data Leakage Demonstration (StandardScaler)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_leakage_demo():
    print("\n" + "="*60)
    print("🚀 DATA LEAKAGE DEMONSTRATION")
    print("="*60)

    # STEP 1: Create Simple Synthetic Dataset
    # 1000 samples, 10 features
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # --- SCENARIO 1: WRONG VERSION (DATA LEAKAGE) ---
    print("\n🔴 SCENARIO 1: Workflow with Data Leakage")
    
    # WRONG: Scaling BEFORE splitting!
    # The scaler 'sees' the distribution of the entire dataset (including future test data)
    # Mean and Standard Deviation are calculated from both training and testing samples.
    scaler_leakage = StandardScaler()
    X_scaled_wrong = scaler_leakage.fit_transform(X) # ❌ LEAKAGE HAPPENS HERE
    
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_scaled_wrong, y, test_size=0.2, random_state=42
    )
    
    model_w = LogisticRegression()
    model_w.fit(X_train_w, y_train_w)
    y_pred_w = model_w.predict(X_test_w)
    accuracy_wrong = accuracy_score(y_test_w, y_pred_w)
    
    print(f"   [!] Error: Scaling applied before splitting.")
    print(f"   [!] Impact: Test data distribution leaked into training statistics.")

    # --- SCENARIO 2: CORRECT VERSION (NO LEAKAGE) ---
    print("\n🟢 SCENARIO 2: Correct ML Workflow")
    
    # CORRECT: Split first, then scale ONLY training data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit scaler ONLY on training data
    scaler_correct = StandardScaler()
    X_train_c = scaler_correct.fit_transform(X_train_c) # ✅ Correct fitting
    
    # Use training statistics to transform test data (no leakage)
    X_test_c = scaler_correct.transform(X_test_c) # ✅ Correct transformation
    
    model_c = LogisticRegression()
    model_c.fit(X_train_c, y_train_c)
    y_pred_c = model_c.predict(X_test_c)
    accuracy_correct = accuracy_score(y_test_c, y_pred_c)
    
    print(f"   [✓] Success: Split performed before Scaling.")
    print(f"   [✓] Outcome: Scaler only learned from training distribution.")

    # --- FINAL COMPARISON ---
    print("\n" + "-"*60)
    print("📊 PERFORMANCE COMPARISON")
    print(f"   - Accuracy with Leakage:    {accuracy_wrong:.4f}")
    print(f"   - Accuracy without Leakage: {accuracy_correct:.4f}")
    print("-"*60)

    print("\n💡 WHY IS LEAKAGE DANGEROUS?")
    print("1. DECEPTIVE RESULTS: Leakage makes models appear much more accurate than they actually are.")
    print("2. REAL-WORLD FAILURE: Since leakage won't exist during production (new data), the model will crash in performance.")
    print("3. STATISTICAL CONTAMINATION: The training phase 'steals' mean/std info from the test set, creating a bias.")

    print("\n" + "="*60)
    print("🏁 DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_leakage_demo()
