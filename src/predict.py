import joblib
from src.config import MODEL_PATH, VECTORIZER_PATH
from src.data_preprocessing import clean_text
from src.recommend import get_recommendations

def predict(readme_text: str):
    """
    Inference function: Load -> Transform -> Predict
    """
    # Load artifacts
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print("❌ Error: Saved model or vectorizer not found. Run training first.")
        return None

    # Process text (No Fitting!)
    cleaned_input = clean_text(readme_text)
    tfidf_input = vectorizer.transform([cleaned_input])
    
    # Predict
    prediction_binary = model.predict(tfidf_input)[0]
    
    # Map back to skill names
    from src.config import SKILL_LABELS
    predicted_skills = [
        SKILL_LABELS[i] for i, val in enumerate(prediction_binary) if val == 1
    ]
    
    return predicted_skills

def run_inference_pipeline(sample_text=None):
    """
    High-level inference pipeline: Load -> Predict -> Recommend
    """
    print("\n" + "="*50)
    print("🔮 STARTING INFERENCE PIPELINE (Load + Predict)")
    print("="*50)

    if sample_text is None:
        sample_text = "Experienced Python developer working with SQL databases and Flask APIs."

    print(f"📄 Input Description: \"{sample_text}\"")
    
    # Predict
    detected_skills = predict(sample_text)
    
    if detected_skills is not None:
        print(f"🔍 Detected Skills: {detected_skills}")
        
        # Recommend
        rec = get_recommendations(detected_skills)
        print("\n💡 Career Insights:")
        print(f"   - Missing Skills: {', '.join(rec['missing_skills'])}")
        print(f"   - Next Projects: {', '.join(rec['suggested_next_projects'])}")

    print("\n" + "="*50)
    print("🏁 INFERENCE COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Ensure src is in path
    import sys
    import os
    sys.path.append(os.getcwd())
    
    # Optional: Take CLI input or use sample
    text_input = sys.argv[1] if len(sys.argv) > 1 else None
    run_inference_pipeline(text_input)