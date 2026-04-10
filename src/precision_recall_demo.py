import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_precision_recall_demo():
    print("\n" + "="*60)
    print("⚖️  EVALUATION: PRECISION & RECALL TRADE-OFFS")
    print("="*60)

    # 1. Load Data
    df = load_data(DATA_PATH)
    X = df['project_description'].apply(clean_text)
    
    # Using 'Python' as the representative target for clean binary metrics
    y = df['Python']
    
    # 2. Train-Test Split (with stratify to preserve label distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Model Pipeline
    model_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # 4. Baseline Pipeline
    baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])

    # 5. Training
    model_pipe.fit(X_train, y_train)
    baseline_pipe.fit(X_train, y_train)

    # 6. Predictions
    y_pred_model = model_pipe.predict(X_test)
    y_pred_base = baseline_pipe.predict(X_test)

    # 7. Metrics Calculation
    model_prec = precision_score(y_test, y_pred_model, zero_division=0)
    model_rec = recall_score(y_test, y_pred_model, zero_division=0)
    
    base_prec = precision_score(y_test, y_pred_base, zero_division=0)
    base_rec = recall_score(y_test, y_pred_base, zero_division=0)

    # 8. PRINT OUTPUT
    print("\n=== BASELINE ===")
    print(f"Precision: {base_prec:.4f}")
    print(f"Recall:    {base_rec:.4f}")

    print("\n=== MODEL ===")
    print(f"Precision: {model_prec:.4f}")
    print(f"Recall:    {model_rec:.4f}")

    print("\n📊 Confusion Matrix (Logistic Regression):")
    cm = confusion_matrix(y_test, y_pred_model)
    print(cm)
    print("\n   [ Interpretation: [ [TN, FP], [FN, TP] ] ]")

    print("\n📑 Classification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_model, zero_division=0))

    # STEP 9: ASSIGNMENT COMMENTS & JUSTIFICATIONS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: What is the trade-off between Precision and Recall?")
    print("A: Generally, as you try to catch more cases (Increase Recall), you risk")
    print("   flagging more 'false alarms' (Lower Precision). Conversely, being more")
    print("   selective (High Precision) usually means you miss more cases (Low Recall).")
    
    print("\nQ: Why prioritize Recall in Fraud Detection?")
    print("A: In fraud, the cost of a 'False Negative' (missing fraud) is huge ($50k),")
    print("   while the cost of a 'False Positive' (investigating a safe user) is")
    print("   small ($1k). We'd rather investigate extra safe users than miss one thief.")
    
    print("\nQ: How does the classification threshold affect these?")
    print("A: Lowering the threshold (e.g., 0.3 instead of 0.5) makes the model more")
    print("   aggressive, boosting Recall. Raising it (e.g., 0.8) makes it more")
    print("   conservative, boosting Precision.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 EVALUATION COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_precision_recall_demo()
