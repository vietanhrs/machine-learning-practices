"""
Exercise 01 — Regression Metrics From Scratch
==============================================

Implement all standard regression evaluation metrics from scratch using NumPy.
After implementing each function, compare with scikit-learn via compare_all_metrics().

Learning objectives:
- Understand MSE, RMSE, MAE, MAPE, R², Adjusted R² computations.
- See how outliers affect different metrics differently.
- Diagnose model fit using residual plots and Q-Q plots.

Run:
    python 01_regression_metrics.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn import metrics as skmetrics
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Core metrics (all TODO)
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Formula:
        MSE = (1/n) * sum( (y_i - ŷ_i)² )

    Returns
    -------
    mse : float — always non-negative, in units of target².

    Hint: Use np.mean() on the squared difference array.
    """
    # TODO: implement
    pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Formula:
        RMSE = sqrt(MSE)

    Same units as the target variable — more interpretable than MSE.

    Hint: Reuse mse() and take the square root with np.sqrt().
    """
    # TODO: implement
    pass


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Formula:
        MAE = (1/n) * sum( |y_i - ŷ_i| )

    Robust to outliers (linear penalty vs MSE's quadratic).

    Hint: Use np.abs() and np.mean().
    """
    # TODO: implement
    pass


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Formula:
        MAPE = (100/n) * sum( |y_i - ŷ_i| / max(|y_i|, epsilon) )

    Returns the percentage value (e.g., 12.5 means 12.5%).

    Parameters
    ----------
    epsilon : float — small constant to avoid division by zero when y_i ≈ 0.

    Edge case: If y_i == 0 exactly, the percentage error is undefined.
               Use epsilon to avoid division by zero.

    Hint: np.maximum(np.abs(y_true), epsilon) for safe division.
    """
    # TODO: implement
    pass


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the R² (coefficient of determination).

    Formula:
        SS_res = sum( (y_i - ŷ_i)² )
        SS_tot = sum( (y_i - mean(y_true))² )
        R² = 1 - SS_res / SS_tot

    Interpretation:
        R² = 1.0  → perfect fit
        R² = 0.0  → predicts mean only
        R² < 0.0  → worse than predicting the mean

    Hint: Compute SS_res and SS_tot separately.
          ȳ = np.mean(y_true)
          SS_tot = np.sum((y_true - y_bar) ** 2)
    """
    # TODO: implement
    pass


def adjusted_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> float:
    """
    Compute the Adjusted R² score.

    Formula:
        R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)

    where n = number of samples, p = number of features.

    Adjusted R² penalizes for adding features that don't improve the fit.
    It can decrease when irrelevant features are added (unlike raw R²).

    Parameters
    ----------
    n_features : int — number of predictor variables (p), excluding intercept.

    Edge case: if n - p - 1 <= 0, return np.nan.

    Hint: Compute r_squared() first, then apply the adjustment formula.
    """
    # TODO: implement
    pass


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Median Absolute Error.

    Formula:
        MedAE = median( |y_i - ŷ_i| )

    Even more robust to outliers than MAE — uses the median instead of mean.
    The optimal constant predictor under MedAE is the median of y_true.

    Hint: Use np.median() on the absolute residuals.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Comparison (TODO)
# ---------------------------------------------------------------------------

def compare_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_outlier: np.ndarray,
    n_features: int = 1,
) -> None:
    """
    Print a comparison table showing how each metric responds differently to
    outliers.

    Parameters
    ----------
    y_true         : true target values.
    y_pred         : model predictions without outlier influence.
    y_pred_outlier : predictions from a model or modified predictions that are
                     heavily affected by outliers.
    n_features     : int — number of features (for Adjusted R²).

    Steps:
        1. For each of (y_pred, y_pred_outlier), compute:
               MSE, RMSE, MAE, MAPE, R², Adjusted R², Median AE
        2. Also compute sklearn's versions for verification.
        3. Print a table with three columns: metric name, clean predictions,
           outlier-affected predictions.
        4. Highlight which metrics change the most (MSE/RMSE should change most).

    Hint: Use f-strings for aligned formatting.
          Compare your values against:
              skmetrics.mean_squared_error
              skmetrics.mean_absolute_error
              skmetrics.r2_score
              skmetrics.median_absolute_error
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Visualization (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
) -> None:
    """
    Plot residuals vs fitted values and a Q-Q plot. Fully implemented.

    Two-panel plot:
    - Left:  Residuals vs Fitted Values (should be random horizontal band around 0)
    - Right: Q-Q plot of residuals (should be close to diagonal for normality)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Residuals vs Fitted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.4, s=20, color="steelblue", edgecolors="none")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--", label="Zero line")
    # Smooth trend line
    order = np.argsort(y_pred)
    from numpy.polynomial import polynomial as P
    try:
        coeffs = P.polyfit(y_pred[order], residuals[order], deg=2)
        trend = P.polyval(y_pred[order], coeffs)
        ax.plot(y_pred[order], trend, color="darkorange", linewidth=2, label="Trend")
    except Exception:
        pass
    ax.set_xlabel("Fitted Values (ŷ)")
    ax.set_ylabel("Residuals (y - ŷ)")
    ax.set_title("Residuals vs Fitted Values\n(Random band = good; Patterns = problem)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Q-Q Plot
    ax = axes[1]
    (quantiles, values), (slope, intercept, r) = scipy_stats.probplot(residuals, dist="norm")
    ax.scatter(quantiles, values, alpha=0.4, s=20, color="steelblue", edgecolors="none")
    line_x = np.array([quantiles.min(), quantiles.max()])
    ax.plot(line_x, slope * line_x + intercept, color="red", linewidth=2,
            label=f"Normal line (R²={r**2:.3f})")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot of Residuals\n(Points on line = Gaussian errors)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Compare regression metrics on California Housing dataset with and without
    artificially introduced outliers.
    """
    # --- Load data ---
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    n_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # --- Standardize ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- Fit models ---
    # Clean model
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_clean = lr.predict(X_test_sc)

    # Create "outlier-affected predictions" by manually corrupting some predictions
    np.random.seed(42)
    y_pred_outlier = y_pred_clean.copy()
    outlier_idx = np.random.choice(len(y_pred_outlier), size=50, replace=False)
    y_pred_outlier[outlier_idx] += np.random.uniform(5, 10, size=50)  # large errors

    # --- Metrics comparison ---
    print("=== Regression Metrics: Clean vs Outlier-Affected Predictions ===\n")
    compare_all_metrics(y_test, y_pred_clean, y_pred_outlier, n_features=n_features)

    # --- Residual plots ---
    plot_residuals(y_test, y_pred_clean, title="Residual Analysis — Linear Regression (Clean)")
    plot_residuals(y_test, y_pred_outlier, title="Residual Analysis — With Outlier Predictions")

    plt.show()

    print("\n--- Key takeaways ---")
    print("MSE/RMSE jump sharply with outliers (quadratic penalty).")
    print("MAE and Median AE are much more stable (linear / median penalty).")
    print("If RMSE >> MAE, investigate your largest residuals.")


if __name__ == "__main__":
    main()
