# Data2Decision: Project Description Analyzer (NLP)

This project is a professional Machine Learning system designed to analyze GitHub project descriptions (like READMEs) and provide intelligent career insights. It identifies core skills present in a project and suggests missing skills or new projects to improve a developer's portfolio.

## 📁 Project Structure

```text
project_root/
│
├── data/
│   └── projects.csv      # Synthetic multi-label NLP dataset
│
├── models/
│   ├── skill_classifier.pkl   # Multi-label classifier (OneVsRest + RandomForest)
│   └── tfidf_vectorizer.pkl   # TF-IDF vectorization model
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized configuration (Paths, Skill Labels)
│   ├── data_loader.py         # Data reading logic
│   ├── data_preprocessing.py  # Text cleaning and splitting
│   ├── feature_engineering.py  # TF-IDF transformation
│   ├── train.py              # Multi-label training and serialization
│   ├── evaluate.py           # Classification reporting
│   ├── predict.py            # Isolated NLP inference
│   ├── recommend.py          # Career insight & recommendation logic
│   └── main.py               # Full pipeline orchestrator
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Installation
Install the necessary NLP and data science libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
Execute the full pipeline through the orchestrator:
```bash
python -m src.main
```

## 🧠 Core Features

### NLP Logic
The system uses **TF-IDF Vectorization** to convert project descriptions into numerical features. It then employs a **Multi-Output Classifier** to detect multiple skills from a single block of text (e.g., detecting both "Python" and "Flask" in a README).

### Intelligent Recommendations
The `recommend.py` module analyzes the detected skills and maps them against industry standards to provide:
- **Missing Skills**: Identifies gaps in technical stacks.
- **Project Suggestions**: Recommends the next step for portfolio growth.

## 🛠️ Design Principles
- **Modularity**: Every stage of the pipeline is isolated for testing and scalability.
- **Absolute Imports**: Professional package structure using `src` as the root.
- **Inference Isolation**: Predictions are made using saved model artifacts, ensuring no data leakage from the training set.
