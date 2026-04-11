import pickle
import pandas as pd
import numpy as np

def run_inference_workshop():
    print("\n" + "="*60)
    print("MODEL INFERENCE WORKSHOP")
    print("="*60)

    # ---------------------------------------------------------
    # PART 1 -- LOAD MODEL
    # ---------------------------------------------------------
    print("\nPART 1: LOADING THE SAVED MODEL")
    print("-" * 40)
    
    model_path = "models/final_pipeline.pkl"
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully from:", model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please run the serialization workshop first.")
        return

    # ---------------------------------------------------------
    # PART 2 -- PREPARE INPUT DATA
    # ---------------------------------------------------------
    print("\nPART 2: PREPARING NEW INPUT DATA")
    print("-" * 40)
    
    # We use a DataFrame as it is standard for production ML systems
    new_data = pd.DataFrame([
        {"description": "A web app using React and Node.js with database integration"},
        {"description": "Deep learning project using Python and TensorFlow for image recognition"},
        {"description": "Data analysis notebook with Pandas and SQL queries"}
    ])
    
    print("New data for inference:")
    print(new_data)
    print("\nIMPORTANT: Input format must match what the TF-IDF vectorizer expects.")

    # ---------------------------------------------------------
    # PART 3 -- MAKE PREDICTION
    # ---------------------------------------------------------
    print("\nPART 3: PERFORMING INFERENCE (PREDICTION)")
    print("-" * 40)
    
    # Note: We pass the column that contains the text data
    predictions = model.predict(new_data["description"])
    
    print("Predicted Classes (1 = Python, 0 = Other):")
    for desc, pred in zip(new_data["description"], predictions):
        print(f"-> Description: {desc[:30]}... | Prediction: {pred}")

    # ---------------------------------------------------------
    # PART 4 -- PROBABILITY
    # ---------------------------------------------------------
    print("\nPART 4: CONFIDENCE ESTIMATION (PROBABILITY)")
    print("-" * 40)
    
    probabilities = model.predict_proba(new_data["description"])
    
    print("Prediction Probabilities [Class 0, Class 1]:")
    for desc, prob in zip(new_data["description"], probabilities):
        print(f"-> Description: {desc[:30]}... | Prob: {prob}")

    # ---------------------------------------------------------
    # PART 5 -- EXPLANATION
    # ---------------------------------------------------------
    print("\nPART 5: EXPLANATION & COMPARISON")
    print("-" * 60)
    print("What is Inference?")
    print("   Using a trained (frozen) model to make predictions on new data.")
    
    print("\nTraining vs Inference:")
    print("-" * 45)
    print(f"{'Feature':<15} | {'Training':<15} | {'Inference':<15}")
    print("-" * 45)
    print(f"{'Goal':<15} | {'Model learns':<15} | {'Model predicts':<15}")
    print(f"{'Speed':<15} | {'Slow':<15} | {'Fast':<15}")
    print(f"{'Labels':<15} | {'Uses labels':<15} | {'No labels needed':<15}")
    print("-" * 45)

    print("\nWhy No Retraining?")
    print("   - The model has already 'learned' the patterns.")
    print("   - In production, we need fast, deterministic predictions.")
    print("   - Retraining on every request would be inefficient and risky.")
    print("-" * 60)

    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_inference_workshop()
