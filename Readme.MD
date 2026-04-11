# Data2Decision: Project Description Analyzer

## 🚀 Project Overview

**Data2Decision** is an end-to-end NLP-based Machine Learning system designed to bridge the gap between developer portfolios and industry expectations. By analyzing project descriptions (such as GitHub READMEs), the system automatically:

*   **Detects technical skills** embedded in the project.
*   **Provides career recommendations** based on the detected stack.
*   **Suggests future projects** to help developers target specific industry demands.

The goal is to provide a "placement-ready" analysis that helps students and job seekers optimize their portfolios for maximum impact.

---

## 📌 Problem Statement

Developers, especially students, often struggle with three critical questions:
1.  **Relevance**: Are my current projects using technologies that top companies actually want?
2.  **Skill Gaps**: What specific libraries or tools should I learn next to improve my profile?
3.  **Roadmap**: Given my current stack, what project should I build next to become a more competitive candidate?

This system automates this analytical process, turning unstructured project descriptions into actionable career data.

---

## 👥 Target Users

*   **Students**: Preparing for campus placements and looking to validate their technical depth.
*   **Portfolio Builders**: Developers looking to ensure their GitHub repositories reflect a cohesive skill set.
*   **Job Seekers**: Candidates tailoring their portfolios for specific roles in Backend, Frontend, or Data Science.

---

## 📊 Dataset

*   **Source**: A carefully curated custom dataset of technical project descriptions.
*   **Labels**: Five key skill categories including **Python**, **SQL**, **React**, **Machine_Learning**, and **Flask**.
*   **Nature**: Multi-label classification (a single project can possess multiple skills).

---

## 🛠️ Tech Stack

*   **Core**: Python 3.9+
*   **ML Framework**: Scikit-learn
*   **Data Handling**: Pandas, NumPy
*   **Serialization**: Pickle / Joblib
*   **Web Interface**: Streamlit
*   **NLP**: TF-IDF Vectorization

---

## 🔄 ML Lifecycle & Pipeline

The project follows a rigorous industry-standard pipeline to ensure reliable predictions:

1.  **Text Cleaning**: Normalizing case, removing noise, and handling special characters.
2.  **TF-IDF Vectorization**: Converting unstructured text into weighted numerical vectors.
3.  **Multi-label Classification**: Training models capable of detecting non-exclusive skill tags.
4.  **Model Evaluation**: Using F1-Score, Precision, and Recall rather than just Accuracy.
5.  **Model Comparison**: Benchmarking multiple algorithms with consistent preprocessing.
6.  **Final Model Selection**: Choosing the candidate that best balances performance and stability.
7.  **Serialization**: Saving the full pipeline (preprocessor + model) for consistent reuse.
8.  **Inference**: Executing predictions on unseen data at high speed.
9.  **Deployment**: Presenting the model through an interactive Streamlit web application.

---

## 🚫 Data Leakage Prevention

To ensure "honest" evaluation, we implemented several guardrails:
*   **Pipeline Encapsulation**: All transformations (TF-IDF, Scaling) are performed *inside* the pipeline, ensuring no information from the test set "leaks" into the training process.
*   **Stratified Splitting**: Ensuring that training and test sets maintain representative distributions.
*   **Fold-Level Preprocessing**: Cross-validation is performed using `imblearn` pipelines to prevent leakage during oversampling or normalization steps.

---

## ⚖️ Model Comparison

We benchmarked three distinct algorithms using 5-Fold Stratified Cross-Validation:

| Model | CV Mean (F1) | CV Std | Stability |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7333 | 0.0667 | **High** |
| Random Forest | 0.7333 | 0.0667 | Moderate |
| Naive Bayes | 0.7333 | 0.0667 | Moderate |

**Insight**: For the current dataset scale, preprocessing quality (TF-IDF weights) was found to be the dominant performance factor.

---

## 🏆 Final Model Selection

**Logistic Regression** was selected for final deployment because:
*   **High Recall (1.00)**: It prioritized catching all relevant skills, which is critical for portfolio analysis.
*   **Interpretability**: It allowed us to see clearly which words influenced the career recommendations.
*   **Operational Speed**: Fast enough for real-time inference in the Streamlit UI.

---

## 💾 Model Serialization

The **FULL pipeline** is serialized into `models/final_pipeline.pkl`. This ensures that whenever the model is loaded, it carries its own TF-IDF vocabulary, preventing feature mismatch errors that occur when saving just the model object.

---

## 🔍 Inference

The inference engine is "frozen," meaning it makes predictions on new projects without needing to retrain.
*   **Speed**: <50ms per prediction.
*   **Input**: Unstructured project description.
*   **Output**: Class labels and confidence probabilities.

---

## 🌐 Streamlit App (UI)

The UI provides a user-friendly way to interact with the model.

### Features
*   Paste any project README or description.
*   View detected skills with confidence scores.
*   Get dynamic career recommendations (e.g., "Learn Flask to complement your Python skills").

### How to Run
```bash
streamlit run app.py
```

---

## ⚠️ Limitations

*   **Dataset Size**: Currently trained on a small demo dataset; accuracy will improve with more samples.
*   **Skill Vocabulary**: Restricted to five core skill categories.
*   **Text Processing Only**: Does not yet analyze GitHub API data (commit frequency, code density).
*   **Keyword Dependency**: Performance is sensitive to how technical the user's description is.

---

## 🧠 Assumptions

*   **Consistency**: We assume project READMEs generally contain enough technical keywords to represent the stack.
*   **Keyword Mapping**: We assume that certain keyword patterns (e.g., "state management") strongly correlate with specific libraries (e.g., React).
*   **Representative Data**: We assume the training descriptions are representative of real-world developer project summaries.

---

## 🔮 Future Improvements

1.  **GitHub API Integration**: Automatically pull and analyze a user's entire repository history.
2.  **Transformer-based NLP**: Move from TF-IDF to **BERT** or **GPT-based** embeddings for better semantic understanding.
3.  **Company-Specific Tracks**: Tailor recommendations to specific hiring patterns of companies like Google, Meta, or Amazon.
4.  **Skill Density Analysis**: Measure not just the presence of a skill, but the *depth* based on description length and complexity.

---

## 🤔 Reflections

### Challenges Faced
One of the primary challenges was ensuring **leakage prevention** during scaling and oversampling. Implementing `imblearn` pipelines was essential for maintaining evaluation integrity. Additionally, handling **multi-label classification** required a deeper look at F1-scores over simple accuracy.

### Key Learnings
*   **ML is a lifecycle**: Successful models are 20% training and 80% data Engineering, evaluation, and deployment.
*   **Evaluation is Critical**: Accuracy is often a "liar" in imbalanced datasets; Precision and Recall are the true benchmarks of success.
*   **Deployment Matters**: A model only delivers value when it is accessible to non-technical users via a clean UI.

### What I would improve
If I were to start over, I would prioritize building a larger, more diverse dataset first and integrate more advanced text augmentation techniques to improve model robustness.

---

## 🏁 Conclusion

This project demonstrates a complete, production-ready machine learning workflow:
**Raw Data → Clean Pipeline → Objective Evaluation → Stable Product.**

By turning complex NLP patterns into understandable career insights, **Data2Decision** serves as a powerful proof-of-concept for AI-driven portfolio optimization.
