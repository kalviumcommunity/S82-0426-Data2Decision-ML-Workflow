import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_mse_r2_demo():
    print("\n" + "="*60)
    print("📈 REGRESSION EVALUATION: MSE & R² PERFORMANCE")
    print("="*60)

    # STEP 1: Dataset
    # 1000 samples, 10 features, moderate noise
    X, y = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)

    # STEP 2: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: BASELINE
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    # STEP 4: MODEL
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_preds = model.predict(X_test)

    # STEP 5 & 6: METRICS
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, r2

    base_mse, base_rmse, base_r2 = calculate_metrics(y_test, baseline_preds)
    model_mse, model_rmse, model_r2 = calculate_metrics(y_test, model_preds)

    # STEP 7: PRINT COMPARISON
    print("\n📊 Metric Comparison:")
    print("-" * 60)
    print(f"{'Metric':<10} | {'Baseline':<15} | {'Model':<15}")
    print("-" * 60)
    print(f"{'MSE':<10} | {base_mse:15.4f} | {model_mse:15.4f}")
    print(f"{'RMSE':<10} | {base_rmse:15.4f} | {model_rmse:15.4f}")
    print(f"{'R²':<10} | {base_r2:15.4f} | {model_r2:15.4f}")
    print("-" * 60)

    # STEP 8: CROSS VALIDATION
    print("\n🔄 5-Fold Cross-Validation Result (Model):")
    
    # R² Stability
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"   - Mean R²:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # MSE Stability
    cv_neg_mse = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    cv_mse = -cv_neg_mse
    cv_rmse = np.sqrt(cv_mse)
    print(f"   - Mean MSE: {cv_mse.mean():.4f} ± {cv_mse.std():.4f}")
    print(f"   - Mean RMSE:{cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

    # STEP 9: COMMENTS
    print("\n💡 ASSIGNMENT JUSTIFICATIONS:")
    print("-" * 60)
    print("Q: Why does MSE penalize large errors?")
    print("A: Because the error is squared. A mistake of '10' becomes '100' in MSE,")
    print("   while a mistake of '1' stays '1'. This makes MSE highly sensitive to outliers.")
    
    print("\nQ: Why does R² show improvement over baseline?")
    print("A: R² compares the model's error to the baseline's error. If R² > 0, it means")
    print("   the model captures more of the target's logic than just guessing the mean.")
    
    print("\nQ: Why are both needed?")
    print("A: MSE tells you the 'size' of the error in units squared. R² tells you the")
    print("   'utility' of the model on a scale of 0 to 1 (or 0% to 100%).")
    
    print("\nQ: When do they disagree?")
    print("A: If a model has a great R² but a huge MSE, it might be perfect for most cases")
    print("   but failing catastrophically on a few specific outliers.")
    print("-" * 60)

    print("\n" + "="*60)
    print("🏁 EVALUATION DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_mse_r2_demo()
