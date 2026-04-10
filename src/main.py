from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import build_vectorizer, transform_text
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict
from src.recommend import get_recommendations
from src.config import DATA_PATH

def run_pipeline():
    """
    Orchestrates the NLP Project Description Analyzer pipeline.
    """
    print("\n" + "="*50)
    print("🚀 STARTING PROJECT DESCRIPTION ANALYZER (NLP)")
    print("="*50)

    # 1. Load Data
    print(f"\n[1/6] Loading project data...")
    df = load_data(DATA_PATH)

    # 2. Preprocess (Cleaning & Train/Test Split)
    print("[2/6] cleaning text and splitting data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Feature Engineering (TF-IDF Vectorization)
    print("[3/6] Building Vectorizer & Transforming features...")
    vectorizer = build_vectorizer(X_train)
    X_train_tfidf = transform_text(X_train, vectorizer)
    X_test_tfidf = transform_text(X_test, vectorizer)

    # 4. Model Training
    print("[4/6] Training Multi-label Skill Classifier...")
    model = train_model(X_train_tfidf, y_train)

    # 5. Model Evaluation
    print("[5/6] Evaluating Performance...")
    accuracy, report = evaluate_model(model, X_test_tfidf, y_test)
    print(f"📊 Global Accuracy: {accuracy:.2f}")
    print("\nDetailed Classification Report:")
    print(report)

    # 6. INFERENCE & RECOMMENDATION
    print("\n[6/6] Analyzing Sample Project Description...")
    
    # Real-world sample input (like a README snippet)
    sample_readme = "A portfolio website showcasing design work using React components and custom SQL database integration."
    
    print(f"📄 Input Description: \"{sample_readme}\"")
    
    # Predict Skills
    detected_skills = predict(sample_readme)
    print(f"🔍 Detected Skills: {detected_skills}")
    
    # Get Recommendations
    rec = get_recommendations(detected_skills)
    
    print("\n💡 Career Insights:")
    print(f"   - Missing Skills to learn: {', '.join(rec['missing_skills'])}")
    print(f"   - Suggested Next Projects: {', '.join(rec['suggested_next_projects'])}")

    print("\n" + "="*50)
    print("🏁 ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline()