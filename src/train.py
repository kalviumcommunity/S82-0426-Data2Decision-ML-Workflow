import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import build_vectorizer, transform_text
from src.evaluate import evaluate_model
from src.config import MODEL_PATH, DATA_PATH

def train_model(X_train_tfidf, y_train):
    """
    Trains a Multi-label classifier and saves it.
    """
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = OneVsRestClassifier(base_clf)
    model.fit(X_train_tfidf, y_train)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Skill Classifier saved to {MODEL_PATH}")
    
    return model

def run_training_pipeline():
    """
    High-level training pipeline: Load -> Clean -> Vectorize -> Fit -> Evaluate -> Save
    """
    print("\n" + "="*50)
    print("🛠️  STARTING TRAINING PIPELINE (Fit + Save)")
    print("="*50)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Preprocess & Split
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Vectorize (Fit + Transform)
    vectorizer = build_vectorizer(X_train)
    X_train_tfidf = transform_text(X_train, vectorizer)
    X_test_tfidf = transform_text(X_test, vectorizer)

    # 4. Train (Fit + Save)
    model = train_model(X_train_tfidf, y_train)

    # 5. Evaluate
    accuracy, report = evaluate_model(model, X_test_tfidf, y_test)
    print(f"📊 Training Accuracy: {accuracy:.2f}")

    print("\n" + "="*50)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Ensure src is in path if run directly from root
    import sys
    import os
    sys.path.append(os.getcwd())
    run_training_pipeline()