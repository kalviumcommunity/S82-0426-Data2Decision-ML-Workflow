import joblib
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_scaling_demo():
    print("\n" + "="*60)
    print("⚖️  NUMERICAL FEATURE SCALING DEMONSTRATION")
    print("="*60)

    # STEP 1: Create Synthetic Numerical Dataset
    # 1000 samples, 20 features with different scales
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    # Artificially shift scales to make the difference clear
    X[:, 0] = X[:, 0] * 1000 
    X[:, 1] = X[:, 1] * 0.01

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # STEP 2: Train WITHOUT scaling
    print("\n🚀 Training LogisticRegression WITHOUT Scaling...")
    model_unscaled = LogisticRegression(max_iter=1000)
    model_unscaled.fit(X_train, y_train)
    y_pred_unscaled = model_unscaled.predict(X_test)
    acc_unscaled = accuracy_score(y_test, y_pred_unscaled)
    
    # STEP 3: Train WITH scaling
    print("\n🚀 Training LogisticRegression WITH Scaling (StandardScaler)...")
    scaler = StandardScaler()
    
    # Correct usage: Fit only on training data (prevent leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_scaled = LogisticRegression(max_iter=1000)
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    # STEP 4: Compare Results
    print("\n" + "-"*40)
    print(f"📊 Accuracy WITHOUT scaling: {acc_unscaled:.4f}")
    print(f"📊 Accuracy WITH scaling:    {acc_scaled:.4f}")
    print("-"*40)

    # STEP 5: Save Scaler
    os.makedirs("models", exist_ok=True)
    scaler_path = "models/standard_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Scaler saved to: {scaler_path}")

    # STEP 6: IMPORTANT JUSTIFICATIONS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Why is scaling required for Logistic Regression?")
    print("A: Logistic Regression uses Gradient Descent or other solvers that rely on")
    print("   the magnitude of the weights. When features have vastly different scales,")
    print("   the solver might struggle to converge or take much longer.")
    
    print("\nQ: Why is scaling NOT required for tree-based models?")
    print("A: Tree models (like Random Forest) split data based on feature thresholds.")
    print("   The relative order of values matters, but the absolute magnitude does not.")
    
    print("\nQ: Why is scaling NOT used in our main NLP pipeline?")
    print("A: Our pipeline uses TF-IDF (Term Frequency-Inverse Document Frequency).")
    print("   TF-IDF already normalizes data by its nature (calculating ratios), and")
    print("   standard scaling sparse text vectors can destroy their sparsity and meaning.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_scaling_demo()
