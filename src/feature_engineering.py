"""
Module: feature_engineering.py
Responsibility: Transforms cleaned text data into numerical vectors (TF-IDF) 
that can be processed by machine learning models. It handles both 
the creation/saving of the vectorizer and the transformation of new text.
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import VECTORIZER_PATH

def build_vectorizer(X_train: pd.Series) -> TfidfVectorizer:
    """
    Trains a TF-IDF vectorizer and saves it to the models directory.
    
    Args:
        X_train (pd.Series): Cleaned project descriptions for training.
        
    Returns:
        TfidfVectorizer: The fitted vectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(X_train)
    
    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    return vectorizer

def transform_text(texts: pd.Series, vectorizer: TfidfVectorizer):
    """
    Transforms text data into TF-IDF vectors using a pre-fitted vectorizer.
    
    Args:
        texts (pd.Series): Text data to transform.
        vectorizer (TfidfVectorizer): The fitted vectorizer object.
        
    Returns:
        sparse matrix: TF-IDF feature matrix.
    """
    return vectorizer.transform(texts)