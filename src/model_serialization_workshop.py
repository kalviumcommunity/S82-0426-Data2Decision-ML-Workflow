import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from src.data_loader import load_data
from src.data_preprocessing import clean_text
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

def run_serialization_workshop():
    print("\n" + "="*60)
    print("MODEL SERIALIZATION WORKSHOP")
    print("="*60)

    # ---------------------------------------------------------
    # PART 1 -- TRAIN MODEL (Baseline)
    # ---------------------------------------------------------
    print("\nPART 1: TRAINING THE INITIAL MODEL")
    print("-" * 40)
    
    df = load_data(DATA_PATH)
    X = df["project_description"].apply(clean_text)
    y = df["Python"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Creating the FULL pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    pipeline.fit(X_train, y_train)
    original_f1 = f1_score(y_test, pipeline.predict(X_test), pos_label=1, zero_division=0)
    print(f"Original Model F1 Score: {original_f1:.4f}")

    # ---------------------------------------------------------
    # PART 2 -- SAVE MODEL (PICKLE)
    # ---------------------------------------------------------
    print("\nPART 2: SAVING THE FULL PIPELINE")
    print("-" * 40)
    
    model_dir = "models"
    model_path = os.path.join(model_dir, "final_pipeline.pkl")
    
    os.makedirs(model_dir, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Full pipeline saved successfully to: {model_path}")
    print("WARNING: Save the FULL pipeline (TF-IDF + Model), not just the model!")

    # ---------------------------------------------------------
    # PART 3 -- LOAD MODEL (NO RETRAINING)
    # ---------------------------------------------------------
    print("\nPART 3: LOADING THE MODEL FROM DISK")
    print("-" * 40)
    
    with open(model_path, "rb") as f:
        loaded_pipeline = pickle.load(f)

    print("Model loaded successfully!")
    print("Note: No retraining was performed after loading.")

    # ---------------------------------------------------------
    # PART 4 -- PREDICT USING LOADED MODEL
    # ---------------------------------------------------------
    print("\nPART 4: PREDICTING WITH LOADED MODEL")
    print("-" * 40)
    
    y_pred_loaded = loaded_pipeline.predict(X_test)
    loaded_f1 = f1_score(y_test, y_pred_loaded, pos_label=1, zero_division=0)
    
    print(f"Loaded Model F1 Score: {loaded_f1:.4f}")

    # ---------------------------------------------------------
    # PART 5 -- VERIFY
    # ---------------------------------------------------------
    print("\nPART 5: VERIFICATION OF CONSISTENCY")
    print("-" * 40)
    
    match = (original_f1 == loaded_f1)
    print(f"Original Score: {original_f1:.4f}")
    print(f"Loaded Score:   {loaded_f1:.4f}")
    print(f"Scores Match:   {'YES' if match else 'NO'}")

    if match:
        print("\nSUCCESS: The serialized pipeline is perfectly consistent with the original.")
    else:
        print("\nFAILURE: There is a discrepancy between original and loaded model results.")

    print("\n" + "="*60)
    print("WORKSHOP COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_serialization_workshop()
