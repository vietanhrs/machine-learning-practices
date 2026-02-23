"""
Exercise 01: Bias-Variance Decomposition
==========================================
Empirically compute and visualize the bias-variance tradeoff by training
models of increasing complexity on multiple bootstrap samples.

Learning objectives:
- Understand that bias² + variance + noise = expected error
- Observe the U-shaped total error curve as model complexity increases
- Plot learning curves to diagnose over/underfitting

Dependencies: numpy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression


# ---------------------------------------------------------------------------
# Bias-Variance Decomposition
# ---------------------------------------------------------------------------

def bias_variance_decomposition(model_fn, X_train_sets, X_test, y_test, n_points=50):
    """
    Estimate bias and variance by training the model on multiple datasets
    and measuring the spread of predictions on a fixed test set.

    Theory:
        E[(y - ŷ)²] = Bias(ŷ)² + Variance(ŷ) + Noise
        
        Bias²    = ||mean_predictions - true_y||²  (how wrong is the average prediction?)
        Variance = mean variance across test points  (how spread are predictions?)

    Args:
        model_fn (callable): A function that returns a new unfitted model instance.
                             Called once per training set: model = model_fn()
        X_train_sets (list): List of training feature matrices — different bootstrap samples
                             of the same distribution. Each has shape (n_samples, n_features).
        X_test (np.ndarray): Fixed test set features, shape (n_test, n_features)
        y_test (np.ndarray): Fixed test set labels, shape (n_test,)
        n_points (int): Number of test points to use (for efficiency)

    Returns:
        dict: {
            "bias_sq": float,     # Average squared bias across test points
            "variance": float,   # Average variance across test points
            "total_error": float  # bias_sq + variance (noise excluded)
        }

    Algorithm:
        1. For each training set in X_train_sets:
           - Fit model_fn() on (X_train_i, y_train_i)
           - Predict on X_test → predictions_i
        2. Stack all predictions: shape (n_models, n_test)
        3. Mean prediction per test point: mean_pred = mean(predictions, axis=0)
        4. Bias² = mean((mean_pred - y_test)²) across test points
        5. Variance = mean(var(predictions, axis=0)) across test points

    Hints:
        - You also need y_train_sets corresponding to X_train_sets
          (pass y_train_sets as an additional argument or compute inside)
        - For the formula to work, X_train_sets[i] and y_train_sets[i] must correspond
        - Use np.var(all_preds, axis=0).mean() for variance
        - Use np.mean((mean_preds - y_test[:n_points])**2) for bias²
    """
    # TODO: Initialize list to store predictions from each training set
    # TODO: For each (X_train, y_train) pair:
    #   - model = model_fn()
    #   - model.fit(X_train, y_train)
    #   - preds = model.predict(X_test[:n_points])
    #   - Append preds to predictions list
    # TODO: Stack predictions: all_preds = np.array(predictions)  # shape (n_models, n_test)
    # TODO: Compute mean prediction per test point: mean_preds = all_preds.mean(axis=0)
    # TODO: Compute bias_sq = np.mean((mean_preds - y_test[:n_points]) ** 2)
    # TODO: Compute variance = np.mean(np.var(all_preds, axis=0))
    # TODO: Compute total_error = bias_sq + variance
    # TODO: Return {"bias_sq": bias_sq, "variance": variance, "total_error": total_error}
    pass


# ---------------------------------------------------------------------------
# Visualization (Complete — Do Not Modify)
# ---------------------------------------------------------------------------

def plot_bias_variance_tradeoff(complexities, biases, variances, total_errors,
                                complexity_label="Model Complexity"):
    """
    Plot the classic U-shaped bias-variance tradeoff curve.

    Args:
        complexities (list): Complexity values on x-axis (e.g., polynomial degree)
        biases (list): Squared bias for each complexity level
        variances (list): Variance for each complexity level
        total_errors (list): Total error (bias + variance) for each complexity level
        complexity_label (str): Label for the x-axis
    """
    plt.figure(figsize=(10, 6))
    plt.plot(complexities, biases, 'b-o', linewidth=2, markersize=5, label='Bias²')
    plt.plot(complexities, variances, 'r-o', linewidth=2, markersize=5, label='Variance')
    plt.plot(complexities, total_errors, 'g-o', linewidth=2, markersize=5,
             label='Total Error (Bias² + Variance)', linestyle='--')

    # Mark minimum total error
    min_idx = np.argmin(total_errors)
    plt.axvline(x=complexities[min_idx], color='gray', linestyle=':', alpha=0.7,
                label=f'Optimal complexity: {complexities[min_idx]}')
    plt.scatter([complexities[min_idx]], [total_errors[min_idx]],
                color='gold', s=150, zorder=5)

    plt.xlabel(complexity_label)
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff\n(Total Error = Bias² + Variance + Noise)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Overfitting / Underfitting Demonstration
# ---------------------------------------------------------------------------

def demonstrate_overfitting_underfitting(X, y, test_x=None):
    """
    Train polynomial regression models of degrees 1-15 on the same dataset.
    Plot predictions to visually show underfitting, good fit, and overfitting.

    Args:
        X (np.ndarray): 1D feature array, shape (N,) — a range of x values
        y (np.ndarray): Target values, shape (N,)
        test_x (np.ndarray or None): Dense x range for smooth prediction curves

    Hints:
        - Reshape X to (N, 1) for sklearn Pipeline
        - For each degree in [1, 3, 5, 9, 15]:
            pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear", LinearRegression())
            ])
            pipeline.fit(X.reshape(-1, 1), y)
            preds = pipeline.predict(test_x.reshape(-1, 1))
        - Plot all predictions on the same figure with scatter for raw data
        - Label each curve with its degree
        - Observe: degree=1 underfits, degree≈3-5 fits well, degree=15 overfits wildly
    """
    # TODO: Define degrees to test [1, 3, 5, 9, 15]
    # TODO: Create dense test x range for smooth curves (if test_x is None)
    # TODO: For each degree: fit pipeline, predict on dense x range, plot
    # TODO: Also scatter plot original data points
    # TODO: Add labels, title, legend, and show
    pass


# ---------------------------------------------------------------------------
# Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(estimator, X, y, cv=5, title="Learning Curve"):
    """
    Plot training and cross-validation accuracy as a function of training set size.

    Args:
        estimator: sklearn estimator (not yet fitted)
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        cv (int): Number of cross-validation folds
        title (str): Plot title

    Hints:
        - Use sklearn.model_selection.learning_curve:
          train_sizes, train_scores, val_scores = learning_curve(
              estimator, X, y,
              train_sizes=np.linspace(0.1, 1.0, 10),
              cv=cv, scoring='neg_mean_squared_error'
          )
        - Negate scores to get positive MSE: -train_scores
        - Compute mean and std across folds:
          train_mean = np.mean(-train_scores, axis=1)
          train_std = np.std(-train_scores, axis=1)
        - Plot mean ± 1 std as shaded region (plt.fill_between)
        - X-axis: train_sizes (actual number of training samples)
        - Y-axis: MSE
        - Interpret: large gap → high variance; both curves high → high bias
    """
    # TODO: Call sklearn learning_curve
    # TODO: Compute mean and std for train and val scores
    # TODO: Plot training curve (mean ± std shaded)
    # TODO: Plot validation curve (mean ± std shaded)
    # TODO: Add labels, title, legend
    # TODO: plt.show()
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Empirically demonstrate the bias-variance tradeoff using polynomial regression.

    Steps:
    1. Generate synthetic 1D regression data from a sine function with noise
    2. Create bootstrap samples of the training data
    3. Train polynomial models of degree 1 to 10 on each bootstrap sample
    4. Compute bias² and variance for each degree
    5. Plot the U-shaped bias-variance-total error curves
    6. Also demonstrate overfitting/underfitting visually
    7. Plot learning curves for underfitting and overfitting models
    """
    print("=" * 60)
    print("Bias-Variance Decomposition and Learning Curves")
    print("=" * 60)

    np.random.seed(42)

    # Generate data: y = sin(2*pi*x) + noise
    N = 30  # Small dataset to make bias-variance visible
    x_true = np.linspace(0, 1, 200)
    y_true = np.sin(2 * np.pi * x_true)

    # Training data (small, with noise)
    X_train = np.random.uniform(0, 1, N)
    y_train = np.sin(2 * np.pi * X_train) + 0.3 * np.random.randn(N)

    # Test data (fixed, for bias-variance computation)
    X_test = np.linspace(0, 1, 100)
    y_test = np.sin(2 * np.pi * X_test)

    # TODO: Create bootstrap samples for bias-variance estimation
    # Hint: For each of n_bootstrap=50 repetitions:
    #   indices = np.random.choice(N, size=N, replace=True)
    #   X_boot = X_train[indices], y_boot = y_train[indices]
    # bootstrap_samples = [(X_boot_i, y_boot_i) for each bootstrap]

    # TODO: For each polynomial degree (1 to 10):
    #   - Define model_fn using Pipeline with PolynomialFeatures + LinearRegression
    #   - Call bias_variance_decomposition
    #   - Record bias_sq, variance, total_error

    # TODO: Call plot_bias_variance_tradeoff with results

    # TODO: Call demonstrate_overfitting_underfitting(X_train, y_train, x_true)

    # TODO: Plot learning curves for underfitting model (degree=1) and overfitting (degree=10)
    # Hint:
    # degree1_model = Pipeline([("poly", PolynomialFeatures(1)), ("lr", LinearRegression())])
    # degree10_model = Pipeline([("poly", PolynomialFeatures(10)), ("lr", LinearRegression())])
    # X2d = X_train.reshape(-1, 1)
    # plot_learning_curves(degree1_model, X2d, y_train, title="Degree 1 — Underfitting")
    # plot_learning_curves(degree10_model, X2d, y_train, title="Degree 10 — Overfitting")


if __name__ == "__main__":
    main()
