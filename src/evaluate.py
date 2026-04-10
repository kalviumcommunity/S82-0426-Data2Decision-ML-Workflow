"""
Module: evaluate.py
Responsibility: Assesses the performance of the multi-label skill classifier 
using standard metrics like Accuracy and F1-score. It ensures that 
model quality is tracked without modifying any model parameters.
"""
from sklearn.metrics import accuracy_score, classification_report
from src.config import SKILL_LABELS

def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluates multi-label classification performance on unseen test data.
    
    Args:
        model: The trained multi-label classifier object.
        X_test_tfidf: TF-IDF transformed test features.
        y_test: True skill labels for the test set.
        
    Returns:
        float: Global accuracy score.
        str: Detailed classification report for each skill.
    """
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-skill reporting
    report = classification_report(y_test, y_pred, target_names=SKILL_LABELS, zero_division=0)
    
    return accuracy, report
