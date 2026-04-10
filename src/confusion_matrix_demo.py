import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_confusion_matrix_demo():
    print("\n" + "="*60)
    print("🧩 EVALUATION: CONFUSION MATRIX & ERROR TYPE ANALYSIS")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    
    # Focusing on 'Python' label for clean binary extraction (TP, TN, FP, FN)
    y = df['Python']
    
    # 2. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Pipelines
    # Logistic Regression Pipeline (TF-IDF + Model)
    model_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # Baseline Pipeline (Most Frequent)
    baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])

    # 4. Training
    model_pipe.fit(X_train, y_train)
    baseline_pipe.fit(X_train, y_train)

    # 5. Predictions
    y_pred_model = model_pipe.predict(X_test)
    y_pred_base = baseline_pipe.predict(X_test)

    # 6. CONFUSION MATRIX CALCULATION
    def get_cm_info(y_true, y_pred, name):
        cm = confusion_matrix(y_true, y_pred)
        # Handle cases where only one class is present in y_test (rare but possible in tiny datasets)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Fallback for non-2x2 matrices (e.g., test set only has one class)
            print(f"   [!] Note: Non-binary matrix detected for {name} due to small test size.")
            tn = fp = fn = tp = 0 
            if len(np.unique(y_true)) == 1:
                val = np.unique(y_true)[0]
                if val == 0: tn = cm[0,0]
                else: tp = cm[0,0]

        return cm, (tn, fp, fn, tp)

    model_cm, model_vals = get_cm_info(y_test, y_pred_model, "Model")
    base_cm, base_vals = get_cm_info(y_test, y_pred_base, "Baseline")

    # 7. METRICS
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_true, y_pred, zero_division=0)
        }

    model_metrics = get_metrics(y_test, y_pred_model)
    base_metrics = get_metrics(y_test, y_pred_base)

    # 8. PRINT OUTPUT
    print("\n=== BASELINE CONFUSION MATRIX ===")
    print(base_cm)
    btn, bfp, bfn, btp = base_vals
    print(f"TP: {btp} | FP: {bfp} | FN: {bfn} | TN: {btn}")

    print("\n=== MODEL CONFUSION MATRIX ===")
    print(model_cm)
    tn, fp, fn, tp = model_vals
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

    print("\n📊 Metric Comparison:")
    print("-" * 50)
    print(f"{'Metric':<12} | {'Baseline':<12} | {'Model':<12}")
    print("-" * 50)
    for m in model_metrics:
        print(f"{m:<12} | {base_metrics[m]:12.4f} | {model_metrics[m]:12.4f}")
    print("-" * 50)

    # STEP 9: ASSIGNMENT COMMENTS & JUSTIFICATIONS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: What is a Confusion Matrix?")
    print("A: A table showing the distribution of actual vs. predicted values.")
    print("   It reveals NOT JUST IF the model failed, but HOW it failed.")
    
    print("\nQ: Why is False Negative (FN) more dangerous here?")
    print("A: Missing a skill (FN) means a project misses its expert match.")
    print("   Giving a wrong skill (FP) is a managed 'false alarm' during review.")
    
    print("\nQ: Fraud Detection Scenario: Cost FN = $40k, Cost FP = $1k?")
    print("A: We must MINIMIZE FN at all costs. This is done by LOWERING the")
    print("   classification threshold, which boosts TP and reduces FN, even if FP increases.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_confusion_matrix_demo()
