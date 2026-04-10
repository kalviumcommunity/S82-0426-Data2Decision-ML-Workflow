from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def run_baseline_demo():
    print("\n" + "="*60)
    print("📊 BASELINE VS. TRAINED MODEL COMPARISON")
    print("="*60)

    # STEP 1: Create Imbalanced Synthetic Dataset
    # 1000 samples, 20 features, 85% majority class
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        weights=[0.85, 0.15], 
        random_state=42
    )

    # STEP 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # STEP 3: BASELINE MODEL
    # Strategy: Always predict the 'most_frequent' class
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    # STEP 4: REAL MODEL (Trained Intelligent Model)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)

    # STEP 5: EVALUATION
    # Baseline Metrics
    base_acc = accuracy_score(y_test, baseline_preds)
    base_f1 = f1_score(y_test, baseline_preds, zero_division=0)

    # Trained Model Metrics
    model_acc = accuracy_score(y_test, model_preds)
    model_f1 = f1_score(y_test, model_preds)

    print("\n📈 PERFORMANCE RESULTS:")
    print("-" * 40)
    print(f"🏁 Baseline Accuracy: {base_acc:.4f}")
    print(f"🎯 Model Accuracy:    {model_acc:.4f}")
    print(f"🚀 Improvement (Acc): {model_acc - base_acc:+.4f}")
    print("-" * 40)
    print(f"🏁 Baseline F1 Score: {base_f1:.4f}")
    print(f"🎯 Model F1 Score:    {model_f1:.4f}")
    print(f"🚀 Improvement (F1):  {model_f1 - base_f1:+.4f}")
    print("-" * 40)

    # STEP 6: DETAILED REPORT
    print("\n📑 DETAILED CLASSIFICATION REPORT (Model):")
    print(classification_report(y_test, model_preds))

    # STEP 7: ASSIGNMENT COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Why is a baseline needed?")
    print("A: To establish a minimum bar of performance. If a complex model cannot")
    print("   beat a random or 'most frequent' guess, it brings no value.")
    
    print("\nQ: Why is accuracy alone misleading?")
    print("A: In imbalanced data (e.g., 85% class A), a model can get 85% accuracy by")
    print("   simply guessing 'A' every time, while failing to detect the minority class.")
    
    print("\nQ: Why is F1 Score important?")
    print("A: F1 Score is the harmonic mean of Precision and Recall. It penalizes models")
    print("   that ignore the minority class, providing a better measure for imbalanced sets.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_baseline_demo()
