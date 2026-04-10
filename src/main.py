from src.train import run_training_pipeline
from src.predict import run_inference_pipeline

def main():
    """
    Master orchestrator: High-level calls to Training and Inference.
    """
    # 1. Run Training (Fit + Save)
    run_training_pipeline()

    # 2. Run Inference (Load + Predict)
    sample_text = "Portfolio showcasing React designs with a SQL database and Python backend."
    run_inference_pipeline(sample_text)

if __name__ == "__main__":
    main()