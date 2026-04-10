import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import VECTORIZER_PATH

def build_vectorizer(X_train: pd.Series) -> TfidfVectorizer:
    """
    Trains a TF-IDF vectorizer and saves it.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(X_train)
    
    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    return vectorizer

def transform_text(texts: pd.Series, vectorizer: TfidfVectorizer):
    """
    Transforms text data into TF-IDF vectors.
    """
    return vectorizer.transform(texts)