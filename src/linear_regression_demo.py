import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_regression_demo():
    print("\n" + "="*60)
    print("📈 LINEAR REGRESSION PERFORMANCE DEMONSTRATION")
    print("="*60)

    # STEP 1: Create Synthetic Regression Dataset
    # 1000 samples, 10 features, with some noise
    X, y = make_regression(n_samples=1000, n_features=10, noise=15.5, random_state=42)

    # STEP 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: BASELINE (Mean Predictor)
    # Strategy: Always predict the average value of the target
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    # STEP 4: LINEAR REGRESSION (With Pipeline Bonus)
    # Using Pipeline: StandardScaler + LinearRegression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)

    # STEP 5: EVALUATION FUNCTION
    def get_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    base_rmse, base_mae, base_r2 = get_metrics(y_test, baseline_preds)
    model_rmse, model_mae, model_r2 = get_metrics(y_test, model_preds)

    # STEP 6: PRINT COMPARISON
    print("\n📊 REGRESSION COMPARISON RESULTS:")
    print("-" * 50)
    print(f"{'Metric':<10} | {'Baseline':<12} | {'Model':<12} | {'Improvement'}")
    print("-" * 50)
    print(f"{'RMSE':<10} | {base_rmse:12.4f} | {model_rmse:12.4f} | {base_rmse - model_rmse:+12.4f}")
    print(f"{'MAE':<10} | {base_mae:12.4f} | {model_mae:12.4f} | {base_mae - model_mae:+12.4f}")
    print(f"{'R² Score':<10} | {base_r2:12.4f} | {model_r2:12.4f} | {model_r2 - base_r2:+12.4f}")
    print("-" * 50)

    # STEP 8: ASSIGNMENT COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Why is a baseline needed for regression?")
    print("A: It defines the 'zero-knowledge' bar. A model that can't beat the raw average")
    print("   is not capturing any of the underlying data relationships.")
    
    print("\nQ: What does R² (Coefficient of Determination) mean?")
    print("A: It represents the proportion of variance in the target that is explained")
    print("   by the features. R²=1 is a perfect fit, and R²=0 is no better than the mean.")
    
    print("\nQ: Why does improvement matter?")
    print("A: Simple models have computational costs and complexity. The improvement")
    print("   must justify the effort of building and deploying the model.")
    
    print("\nQ: Why might a linear model fail?")
    print("A: It assumes a linear relationship. If the data has complex curves (non-linear)")
    print("   or heavy outliers, the model will have high error (bias).")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_regression_demo()
