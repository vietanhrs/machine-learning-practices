"""
Exercise 02 — Log-Likelihood, AIC, BIC, and Model Selection
============================================================

Implement log-likelihood functions and information criteria from scratch.
Use them to select the optimal polynomial regression degree.

Learning objectives:
- Understand the probabilistic interpretation of regression (Gaussian MLE).
- Connect cross-entropy loss to Bernoulli log-likelihood.
- Use AIC and BIC for model comparison with complexity penalization.
- Observe how AIC and BIC differ in their penalty strength.

Run:
    python 02_log_likelihood_and_model_selection.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as skmetrics


# ---------------------------------------------------------------------------
# Log-likelihood functions (TODO)
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float = None,
) -> float:
    """
    Compute the Gaussian log-likelihood for regression.

    Assumes: y_i | x_i ~ N(ŷ_i, σ²)

    Formula:
        log L = -n/2 * log(2π) - n/2 * log(σ²)
                - (1 / 2σ²) * sum( (y_i - ŷ_i)² )

    Parameters
    ----------
    y_true : true target values, shape (n,).
    y_pred : model predictions, shape (n,).
    sigma  : float or None — standard deviation of residuals.
             If None, estimate sigma from residuals: sigma = std(y_true - y_pred).

    Returns
    -------
    log_likelihood : float (always <= 0 for proper distributions).

    Hint:
        n = len(y_true)
        if sigma is None:
            residuals = y_true - y_pred
            sigma = np.std(residuals)  # MLE estimate of sigma
        log_L = -n/2 * np.log(2 * np.pi * sigma**2)
                - (1 / (2 * sigma**2)) * np.sum(residuals**2)

    Note: This is the log of the product of n Gaussian PDFs.
    """
    # TODO: implement
    pass


def bernoulli_log_likelihood(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """
    Compute the Bernoulli log-likelihood for binary classification.

    Assumes: y_i | x_i ~ Bernoulli(p_i), where p_i = model probability.

    Formula:
        log L = sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )

    This is the NEGATIVE of binary cross-entropy loss:
        log L = -n * binary_cross_entropy(y_true, y_pred_proba)

    Parameters
    ----------
    y_true       : binary labels {0, 1}, shape (n,).
    y_pred_proba : predicted probabilities for class 1, shape (n,).
    epsilon      : small constant to prevent log(0) — clip probabilities to
                   [epsilon, 1 - epsilon].

    Returns
    -------
    log_likelihood : float (always <= 0).

    Hint:
        p = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        log_L = np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    Verification: -log_L / n  should equal sklearn's log_loss(y_true, y_pred_proba).
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Information criteria (TODO)
# ---------------------------------------------------------------------------

def aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute the Akaike Information Criterion (AIC).

    Formula:
        AIC = 2 * k - 2 * log_likelihood

    where k = number of model parameters (including intercept and sigma).

    Lower AIC is better. AIC penalizes complexity by 2 per parameter.

    Parameters
    ----------
    log_likelihood : float — the log-likelihood of the fitted model.
    n_params       : int — total number of model parameters (k).

    Hint: AIC = 2 * n_params - 2 * log_likelihood
    """
    # TODO: implement
    pass


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC).

    Formula:
        BIC = k * log(n) - 2 * log_likelihood

    where k = number of parameters, n = number of data points.

    Lower BIC is better. BIC penalizes complexity by log(n) per parameter,
    which is stricter than AIC when n > e² ≈ 7.4 (i.e., almost always).

    Parameters
    ----------
    log_likelihood : float — the log-likelihood of the fitted model.
    n_params       : int — total number of model parameters (k).
    n_samples      : int — number of data points (n).

    Hint: BIC = n_params * np.log(n_samples) - 2 * log_likelihood
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Model comparison (TODO)
# ---------------------------------------------------------------------------

def compare_models_aic_bic(
    X: np.ndarray,
    y: np.ndarray,
    model_complexities: list,
) -> dict:
    """
    Fit polynomial regression models of increasing degree and compare using
    AIC, BIC, and test MSE.

    Parameters
    ----------
    X                  : np.ndarray of shape (n, 1) — single input feature.
    y                  : np.ndarray of shape (n,) — targets.
    model_complexities : list of int — polynomial degrees to try (e.g., [1,2,...,10]).

    Returns
    -------
    results : dict with keys:
        'degrees'    : list of ints
        'aic'        : list of float — AIC for each degree
        'bic'        : list of float — BIC for each degree
        'train_mse'  : list of float
        'test_mse'   : list of float
        'log_likelihoods' : list of float

    Algorithm for each degree d:
        1. Create polynomial features of degree d.
        2. Fit LinearRegression on X_train.
        3. Predict on X_train and X_test.
        4. Compute:
               train_mse = mse(y_train, y_pred_train)
               test_mse  = mse(y_test, y_pred_test)
               n_params  = d + 2  (d+1 polynomial coefficients + 1 for sigma)
               log_L = gaussian_log_likelihood(y_train, y_pred_train)
               aic_score = aic(log_L, n_params)
               bic_score = bic(log_L, n_params, n_train)
        5. Append all values to results lists.

    Hint: Use sklearn.preprocessing.PolynomialFeatures(degree=d, include_bias=False)
          with LinearRegression(fit_intercept=True).
          Split X and y into train/test (80/20) before the loop.
          Do NOT refit the split inside the loop.

    Note: We compute AIC/BIC on the TRAINING log-likelihood (standard practice).
          Test MSE measures generalization. The optimal AIC/BIC degree should
          approximately match the lowest test MSE degree.
    """
    # TODO: implement
    pass


# ---------------------------------------------------------------------------
# Visualization (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_model_selection(
    degrees: list,
    aic_scores: list,
    bic_scores: list,
    test_mse: list,
    train_mse: list = None,
    title: str = "Model Selection: AIC / BIC vs Test MSE",
) -> None:
    """
    Plot AIC, BIC, and test MSE vs polynomial degree. Fully implemented.
    """
    degrees = np.array(degrees)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: AIC and BIC
    ax = axes[0]
    if aic_scores and None not in aic_scores:
        ax.plot(degrees, aic_scores, "o-", color="steelblue", lw=2, label="AIC")
        best_aic_idx = np.argmin(aic_scores)
        ax.axvline(degrees[best_aic_idx], color="steelblue", linestyle="--", alpha=0.6,
                   label=f"Best AIC: degree={degrees[best_aic_idx]}")
    if bic_scores and None not in bic_scores:
        ax.plot(degrees, bic_scores, "s-", color="darkorange", lw=2, label="BIC")
        best_bic_idx = np.argmin(bic_scores)
        ax.axvline(degrees[best_bic_idx], color="darkorange", linestyle="--", alpha=0.6,
                   label=f"Best BIC: degree={degrees[best_bic_idx]}")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Information Criterion (lower = better)")
    ax.set_title("AIC and BIC vs Model Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Train vs Test MSE
    ax = axes[1]
    if test_mse and None not in test_mse:
        ax.plot(degrees, test_mse, "o-", color="red", lw=2, label="Test MSE")
        best_test_idx = np.argmin(test_mse)
        ax.axvline(degrees[best_test_idx], color="red", linestyle="--", alpha=0.6,
                   label=f"Best test MSE: degree={degrees[best_test_idx]}")
    if train_mse and None not in train_mse:
        ax.plot(degrees, train_mse, "s--", color="green", lw=2, label="Train MSE", alpha=0.7)
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Mean Squared Error (lower = better)")
    ax.set_title("Train vs Test MSE (Bias-Variance Tradeoff)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # log scale helps visualize large MSE range

    plt.tight_layout()
    plt.savefig("model_selection_aic_bic.png", dpi=120, bbox_inches="tight")
    print("Saved: model_selection_aic_bic.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate AIC/BIC model selection on a polynomial fitting problem.

    True data-generating process: y = sin(x) + noise (degree ~3 Taylor approx)
    We fit polynomial models of degree 1 through 12 and compare AIC, BIC,
    and test MSE to find the optimal complexity.
    """
    np.random.seed(42)

    # --- Generate data from a nonlinear function ---
    n = 200
    X_1d = np.sort(np.random.uniform(-3, 3, n))
    y = np.sin(X_1d) + 0.3 * X_1d + np.random.normal(0, 0.3, n)

    X = X_1d.reshape(-1, 1)

    print("=== Log-Likelihood Verification ===")

    # --- Test Gaussian log-likelihood ---
    y_mean_pred = np.full_like(y, np.mean(y))  # null model
    ll_null = gaussian_log_likelihood(y, y_mean_pred)
    print(f"  Gaussian log-likelihood (null model): {ll_null}")

    # Simple polynomial fit for verification
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_poly, y)
    y_pred_check = lr.predict(X_poly)
    ll_deg3 = gaussian_log_likelihood(y, y_pred_check)
    print(f"  Gaussian log-likelihood (degree-3): {ll_deg3}")

    if ll_null is not None and ll_deg3 is not None:
        print(f"  Degree-3 is better: {ll_deg3 > ll_null}")

    # --- Test Bernoulli log-likelihood ---
    print("\n=== Bernoulli Log-Likelihood Verification ===")
    from sklearn.datasets import make_classification
    X_cls, y_cls = make_classification(n_samples=500, n_features=5, random_state=42)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_cls, y_cls)
    proba = clf.predict_proba(X_cls)[:, 1]
    bern_ll = bernoulli_log_likelihood(y_cls, proba)
    sk_logloss = skmetrics.log_loss(y_cls, proba)
    if bern_ll is not None:
        print(f"  Your log-likelihood:              {bern_ll:.4f}")
        print(f"  -n * sklearn log_loss:            {-len(y_cls) * sk_logloss:.4f}")
        print(f"  Match: {abs(bern_ll - (-len(y_cls) * sk_logloss)) < 0.01}")

    # --- AIC and BIC spot check ---
    print("\n=== AIC / BIC Spot Check ===")
    if ll_deg3 is not None:
        n_params_deg3 = 3 + 2  # 3 poly coefs + intercept + sigma
        aic_val = aic(ll_deg3, n_params_deg3)
        bic_val = bic(ll_deg3, n_params_deg3, n)
        print(f"  AIC (degree-3): {aic_val}")
        print(f"  BIC (degree-3): {bic_val}")

    # --- Full model selection sweep ---
    print("\n=== Polynomial Model Selection (AIC / BIC / Test MSE) ===")
    degrees = list(range(1, 13))
    results = compare_models_aic_bic(X, y, degrees)

    if results is not None:
        print(f"\n{'Degree':>7} {'Log-L':>12} {'AIC':>10} {'BIC':>10} "
              f"{'Train MSE':>11} {'Test MSE':>10}")
        print("-" * 65)
        for i, d in enumerate(results.get("degrees", [])):
            ll  = results["log_likelihoods"][i]
            a   = results["aic"][i]
            b   = results["bic"][i]
            tr  = results["train_mse"][i]
            te  = results["test_mse"][i]
            vals = [ll, a, b, tr, te]
            if any(v is None for v in vals):
                print(f"  {d:5d}  {'TODO':>10}")
            else:
                print(f"  {d:5d}  {ll:12.2f} {a:10.2f} {b:10.2f} {tr:11.4f} {te:10.4f}")

        plot_model_selection(
            results.get("degrees", degrees),
            results.get("aic", [None] * len(degrees)),
            results.get("bic", [None] * len(degrees)),
            results.get("test_mse", [None] * len(degrees)),
            results.get("train_mse", [None] * len(degrees)),
        )
    else:
        print("TODO: compare_models_aic_bic not implemented")

    plt.show()
    print("\n--- Key Takeaways ---")
    print("  AIC selects a slightly more complex model than BIC.")
    print("  BIC is more conservative (penalizes complexity more).")
    print("  Both should approximate the test-MSE-optimal degree.")
    print("  Training MSE always decreases; test MSE forms a U-shape (bias-variance tradeoff).")


if __name__ == "__main__":
    main()
