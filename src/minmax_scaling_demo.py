import joblib
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_minmax_demo():
    print("\n" + "="*60)
    print("📏 FEATURE NORMALIZATION: MINMAXSCALER DEMONSTRATION")
    print("="*60)

    # STEP 1: Create Synthetic Dataset
    # 1000 samples, 10 features with varied ranges
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X = X * 50 + 100 # Shift range to [100, 150] approx

    # STEP 2: Train WITHOUT normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🚀 Training LogisticRegression WITHOUT Normalization...")
    model_raw = LogisticRegression(max_iter=1000)
    model_raw.fit(X_train, y_train)
    y_pred_raw = model_raw.predict(X_test)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # STEP 3: Train WITH MinMaxScaler
    print("\n🚀 Training LogisticRegression WITH MinMaxScaler...")
    
    # Split data FIRST (Leakage Prevention)
    # Note: X_train, X_test already split above
    
    scaler = MinMaxScaler()
    
    # Fit strictly on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    model_scaled = LogisticRegression(max_iter=1000)
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    # STEP 4: Verify Scaling
    print("\n📈 SCALING VERIFICATION (Training Set):")
    print(f"   - Minimum values (target ≈ 0): {X_train_scaled.min():.4f}")
    print(f"   - Maximum values (target ≈ 1): {X_train_scaled.max():.4f}")

    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"   - Accuracy WITHOUT normalization: {acc_raw:.4f}")
    print(f"   - Accuracy WITH normalization:    {acc_scaled:.4f}")

    # STEP 5: Save Scaler
    os.makedirs("models", exist_ok=True)
    scaler_path = "models/minmax_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Scaler saved to: {scaler_path}")

    # STEP 6: IMPORTANT JUSTIFICATIONS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Why is MinMaxScaler useful?")
    print("A: It squashes all features into a fixed range [0, 1]. This is critical for")
    print("   distance-based algorithms (kNN, SVM) where features with larger magnitudes")
    print("   would otherwise dominate the distance calculation.")
    
    print("\nQ: How does it differ from StandardScaler?")
    print("A: MinMaxScaler preserves the original distribution shape but changes the range.")
    print("   StandardScaler center-scales data to Mean=0 and Std=1 (useful for algorithms")
    print("   that assume normally distributed data).")
    
    print("\nQ: Why is it NOT used in our NLP project?")
    print("A: TF-IDF already produces values within a normalized range relative to the")
    print("   corpus frequent. Additionally, tree-based models like Random Forest are")
    print("   insensitive to the absolute scale of features.")
    
    print("\nQ: Why do tree models not need scaling?")
    print("A: Tree models split based on value comparisons (X > threshold). The magnitude")
    print("   doesn't change the split point logic, only the rank order matters.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_minmax_demo()
