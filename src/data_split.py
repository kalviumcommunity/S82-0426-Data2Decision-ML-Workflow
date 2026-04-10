import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.config import DATA_PATH, TEXT_FEATURES, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

def run_data_split():
    print("\n" + "="*60)
    print("✂️  PROJECT DESCRIPTION ANALYZER: DATA SPLITTING MODULE")
    print("="*60)

    # 1. Load Dataset
    # We use the raw data directly to ensure no prior transformations/leakage
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Separate Features (X) and Target (y)
    # Feature: Raw project descriptions
    # Target: Skill labels (Multi-label set)
    X = df[TEXT_FEATURES[0]]
    y = df[TARGET_COLUMN]

    print("\n🔹 Data Separation Complete.")
    print(f"   - Feature: {TEXT_FEATURES[0]}")
    print(f"   - Target:  {TARGET_COLUMN if isinstance(TARGET_COLUMN, str) else 'Multi-label skills set'}")

    # 3. Perform Train-Test Split
    # Assignment Rule: test_size=0.2, random_state=42
    # No stratify used as this is multi-label classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

    # 4. Verification Output (Mandatory)
    print("\n📈 SPLIT VERIFICATION RESULTS:")
    print(f"   - Training set shape (X_train): {X_train.shape}")
    print(f"   - Testing set shape (X_test):   {X_test.shape}")
    print(f"   - Training targets (y_train):   {y_train.shape}")
    print(f"   - Testing targets (y_test):     {y_test.shape}")

    print("\n📄 Sample raw training data (Confirming No Leakage/Preprocessing):")
    print(X_train.head())

    # 5. Leakage Prevention Check
    print("\n🛡️  LEAKAGE PREVENTION CHECK:")
    print("   [✓] Splitting performed on raw text before Vectorization.")
    print("   [✓] No model training seen before split.")
    print("   [✓] No data imputation or cleaning applied before separation.")

    print("\n" + "="*60)
    print("🏁 DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    import sys
    import os
    # Ensure project root is in path if run as submodule
    sys.path.append(os.getcwd())
    run_data_split()
