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
├── requirements.txt      # Pinned dependency versions
└── README.md
```

## 🛠️ Environment Setup & Reproducibility

### 1. Requirements
- **Python Version**: 3.9+
- **Platform**: Windows, macOS, or Linux

### 2. Creation & Activation
We recommend using **Conda** for isolated environment management to ensure that library versions do not conflict with other projects.

**Using Conda (Recommended):**
```bash
# Create the environment
conda create -n ml_env python=3.9 -y

# Activate the environment
conda activate ml_env          # Windows/macOS/Linux
```

**Using venv:**
```bash
# Create the environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### 3. Install Dependencies
Install the pinned versions of the required libraries to ensure the model behaves exactly as intended:
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
Execute the full end-to-end analysis through the orchestrator:
```bash
python -m src.main
```

### 5. Deactivation
To exit the environment when finished:
```bash
conda deactivate   # If using Conda
deactivate         # If using venv
```
### Run Project
```bash
python -m src.main
```
---

## 🧠 Why Virtual Environments & Reproducibility?

### What is Reproducibility?
In Machine Learning, reproducibility means that if another researcher runs your code on a different machine, they should get the **exact same results** (accuracy, predictions, error rates).

### The Role of Virtual Environments
- **Isolation**: Prevents "Dependency Hell" where one project requires `pandas 1.0` and another requires `pandas 2.0`.
- **Consistency**: By using a `requirements.txt` with pinned versions (e.g., `pandas==2.3.3`), we guarantee that the math and logic inside those libraries don't change between machines.
- **Portability**: It makes it trivial to move a project from a local laptop to a cloud server or a colleague's machine.

### What happens if we DON'T use them?
1. **Broken Code**: Your project might work today but fail tomorrow after a global library update.
2. **Data Leakage/Drift**: Subtle changes in library implementations can lead to different prediction results, making your model unreliable.
3. **Collaboration Friction**: Colleagues will spend hours debugging setup issues instead of working on the code.

---

## ⚡ Core Features

### NLP Logic
The system uses **TF-IDF Vectorization** to convert project descriptions into numerical features. It then employs a **Multi-Output Classifier** to detect multiple skills from a single block of text.

### Intelligent Recommendations
The `recommend.py` module analyzes detected skills against industry standards to identify gaps and suggest the next stage of portfolio growth.
