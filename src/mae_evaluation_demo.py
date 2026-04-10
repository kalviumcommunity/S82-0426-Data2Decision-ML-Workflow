import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_mae_demo():
    print("\n" + "="*60)
    print("📏 REGRESSION EVALUATION: MAE & STABILITY DEMONSTRATION")
    print("="*60)

    # STEP 1: Create Dataset
    # 1000 samples, 10 features, moderate noise
    X, y = make_regression(n_samples=1000, n_features=10, noise=20.0, random_state=42)

    # STEP 2: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: BASELINE (Goal: Beat the mean predictor)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    # STEP 4: MODEL (Trained Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)

    # STEP 5: METRICS COMPUTATION
    # 1. MAE
    base_mae = mean_absolute_error(y_test, baseline_preds)
    model_mae = mean_absolute_error(y_test, model_preds)
    
    # 2. RMSE
    model_rmse = np.sqrt(mean_squared_error(y_test, model_preds))
    
    # 3. R2
    model_r2 = r2_score(y_test, model_preds)

    # STEP 6: COMPARISON PRINTING
    print("\n📊 Metric Comparison:")
    print("-" * 40)
    print(f"🏁 Baseline MAE: {base_mae:.4f}")
    print(f"🎯 Model MAE:    {model_mae:.4f}")
    improvement = ((base_mae - model_mae) / base_mae) * 100
    print(f"🚀 Improvement:  {improvement:.2f}%")
    print("-" * 40)

    # STEP 7: CROSS VALIDATION (Stability Check)
    # Using negative MAE because cross_val_score maximizes value
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    mae_scores = -cv_scores # Convert to positive MAE

    print("\n🔄 5-Fold Cross-Validation Result:")
    print(f"   - Mean MAE: {mae_scores.mean():.4f}")
    print(f"   - Std MAE:  {mae_scores.std():.4f}")
    print(f"   - Stability: {'High' if mae_scores.std() < 1.0 else 'Moderate'}")

    # STEP 8: ASSIGNMENT COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: What does MAE (Mean Absolute Error) mean?")
    print("A: It represents the average absolute difference between the predicted and")
    print("   actual values. It's essentially the 'average mistake' the model makes.")
    
    print("\nQ: Why is MAE useful?")
    print("A: It is highly intuitive because it's in the same unit as the target variable.")
    print("   Unlike MSE, it doesn't exponentially penalize outliers.")
    
    print("\nQ: Difference vs. RMSE?")
    print("A: RMSE (Root Mean Squared Error) penalizes larger errors more heavily than")
    print("   small ones. If RMSE >> MAE, it means the model is making some very large errors.")
    
    print("\nQ: Why does baseline comparison matter?")
    print("A: Without a baseline, you don't know if your error of '15 units' is good")
    print("   or bad. Beating the mean predictor proves the features are relevant.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 EVALUATION DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_mae_demo()
