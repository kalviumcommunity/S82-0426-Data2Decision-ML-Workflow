import joblib
import pandas as pd
from typing import List
from src.config import MODEL_PATH, VECTORIZER_PATH, SKILL_LABELS
from src.data_preprocessing import clean_text

def predict(readme_text: str) -> List[str]:
    """
    Predicts skills from a single project description.
    """
    # Load components
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Process text
    cleaned_input = clean_text(readme_text)
    tfidf_input = vectorizer.transform([cleaned_input])
    
    # Predict
    prediction_binary = model.predict(tfidf_input)[0]
    
    # Map back to skill names
    predicted_skills = [
        SKILL_LABELS[i] for i, val in enumerate(prediction_binary) if val == 1
    ]
    
    return predicted_skills