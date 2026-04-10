# Configuration for Project Description Analyzer (NLP)

# Data paths
RAW_DATA_PATH = "data/raw/projects.csv"
PROCESSED_DATA_PATH = "data/processed/"
DATA_PATH = RAW_DATA_PATH  # Main path used by data_loader

MODEL_PATH = "models/skill_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# --- Assignment: Selecting Numerical and Categorical Features ---
SKILL_LABELS = ["Python", "SQL", "React", "Machine_Learning", "Flask"] # Actual target columns
TARGET_COLUMN = "skills" # Conceptual target name for assignment

# No traditional numerical features in raw NLP data
NUMERICAL_FEATURES = []

# No categorical features (before NLP transformation)
CATEGORICAL_FEATURES = []

# Text feature (handled separately via NLP pipeline)
TEXT_FEATURES = ["project_description"]

# Columns to exclude due to identifiers or leakage risks
EXCLUDED_COLUMNS = [
    "project_id",     # unique identifier
    "manual_tags"     # leakage risk (contains target info)
]

ALL_FEATURES = TEXT_FEATURES

# Required Assignment Validation
assert TARGET_COLUMN not in ALL_FEATURES

# NLP Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2