import pandas as pd
from src.data_loader import load_data
from src.config import (
    DATA_PATH, ALL_FEATURES, TARGET_COLUMN, NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES, TEXT_FEATURES, EXCLUDED_COLUMNS, SKILL_LABELS
)

def run_validation():
    print("\n" + "="*60)
    print("🛡️  FEATURE TYPE SELECTION & VALIDATION")
    print("="*60)

    # 1. Load Dataset
    df = load_data(DATA_PATH)

    # Conceptual Mapping for Assignment:
    # Since this is a multi-label project, 'skills' is the target.
    # We ensure a representation of the target exists for the validation snippet.
    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = df[SKILL_LABELS].values.tolist()

    # 2. Validation Snippet (Assignment Requirement)
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    print("\n📊 Feature Summary:")
    print(f"Numerical features:   {len(NUMERICAL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Text features:        {len(TEXT_FEATURES)}")
    print(f"Total features:       {len(ALL_FEATURES)}")

    # Assertions
    print("\n✅ Running Assertions...")
    assert TARGET_COLUMN not in ALL_FEATURES, f"Target '{TARGET_COLUMN}' should not be in features!"
    
    for col in EXCLUDED_COLUMNS:
        assert col not in ALL_FEATURES, f"Excluded column '{col}' found in feature set!"
    
    print("   [✓] Target separation verified.")
    print("   [✓] Excluded columns (IDs/Leakage) verified.")

    # 3. Important Concept Explanation
    print("\n💡 KEY NLP CONCEPT & JUSTIFICATION:")
    print("-" * 40)
    print("Q: Why are there no numerical or categorical features?")
    print("A: In raw NLP data, text (project_description) is neither categorical nor numerical.")
    print("   Categorical data represents a finite set of labels (like 'Red', 'Blue').")
    print("   Text is unstructured and must be transformed using algorithms like TF-IDF.")
    
    print("\nQ: How does the model see this data?")
    print("A: During preprocessing, TF-IDF converts the text into high-dimensional numerical vectors.")
    print("   After this transformation, the features effectively BECOME numerical, which the Random Forest can process.")
    
    print("\nQ: Why exclude project_id and manual_tags?")
    print("A: project_id is a unique identifier with no predictive value (it causes overfitting).")
    print("   manual_tags contains target information (leakage), which wouldn't be available during real prediction.")
    print("-" * 40)

    print("\n" + "="*60)
    print("🏁 VALIDATION COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    import sys
    import os
    # Ensure project root in path
    sys.path.append(os.getcwd())
    run_validation()
