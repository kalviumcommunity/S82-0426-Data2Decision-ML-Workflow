from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict
from src.config import DATA_PATH, TARGET_COLUMN

def run_pipeline():
    """
    Main orchestration function to run the ML pipeline.
    """
    print("\n" + "="*40)
    print("🚀 STARTING PROFESSIONAL ML PIPELINE")
    print("="*40)

    # 1. DATA LOADING
    print(f"\n[1/5] Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    # 2. DATA PREPROCESSING
    print("[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = preprocess_data(df, TARGET_COLUMN)

    # 3. MODEL TRAINING
    print("[3/5] training model...")
    model = train_model(X_train, y_train)

    # 4. MODEL EVALUATION
    print("[4/5] Evaluating model performance...")
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"📊 Final Model Accuracy: {accuracy:.2f}")

    # 5. INFERENCE (Isolated Prediction)
    print("\n[5/5] Testing Isolated Prediction...")
    # Sample dictionary matches config.FEATURES
    sample_input = {
        "study_hours": 8,
        "attendance": 85
    }
    
    # We call predict module which loads the saved model
    result = predict(sample_input)
    status = "SUCCESS" if result == 1 else "FAIL"
    print(f"🔮 Input Sample: {sample_input}")
    print(f"🎯 Prediction result: {status} (Raw: {result})")

    print("\n" + "="*40)
    print("🏁 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_pipeline()