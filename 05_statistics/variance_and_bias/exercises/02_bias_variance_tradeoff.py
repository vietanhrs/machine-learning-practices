"""
Exercise 02: Bias-Variance Tradeoff
=====================================
Topic: Bias-Variance Decomposition (Phương sai & Độ lệch trong ML)
Difficulty: Intermediate → Advanced

Learning Objectives:
    - Implement bias and variance computation from multiple model runs
    - Perform bias-variance decomposition using bootstrap sampling
    - Simulate the tradeoff curve by varying model complexity
    - Visualize how underfitting and overfitting relate to bias and variance

Key Concept:
    Expected Test MSE = Bias² + Variance + Irreducible Noise

    - Bias: systematic error from wrong model assumptions (high → underfitting)
    - Variance: error from sensitivity to training data (high → overfitting)
    - Irreducible Noise: cannot be reduced by any model

Instructions:
    Implement each TODO section. The plot_* functions are fully provided.
    Focus on understanding what each metric means as you implement it.

Dependencies:
    pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample


# ---------------------------------------------------------------------------
# 1. Compute Bias
# ---------------------------------------------------------------------------

def compute_bias(predictions_per_model: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the squared bias of a set of model predictions.

    Bias measures how far the **average** prediction (across many differently-
    trained models) is from the true value.

    Formula:
        Bias² = mean over test samples of (mean_prediction - y_true)²
        where mean_prediction[j] = (1/M) * sum over models of predictions[m, j]

    Args:
        predictions_per_model: Array of shape (n_models, n_test_samples).
            Entry [m, j] is the prediction of model m on test sample j.
        y_true: Array of shape (n_test_samples,) — the true target values.

    Returns:
        Scalar float: mean squared bias over all test samples.

    Example:
        >>> preds = np.array([[1.0, 2.0], [1.2, 1.8], [0.9, 2.1]])  # 3 models, 2 test pts
        >>> y_true = np.array([2.0, 2.0])  # true values
        >>> compute_bias(preds, y_true)  # average pred ≈ [1.03, 1.97], bias² ≈ 0.48
    """
    # TODO: Step 1 — Compute the mean prediction across models for each test sample.
    #       Hint: np.mean(predictions_per_model, axis=0) gives shape (n_test_samples,)
    # TODO: Step 2 — Compute (mean_prediction - y_true)^2 for each test sample.
    # TODO: Step 3 — Return the mean of those squared differences across all test samples.
    pass


# ---------------------------------------------------------------------------
# 2. Compute Variance
# ---------------------------------------------------------------------------

def compute_variance(predictions_per_model: np.ndarray) -> float:
    """
    Compute the variance of predictions across models.

    Variance measures how much predictions fluctuate when the model is
    retrained on different training sets. High variance = overfitting.

    Formula:
        Variance = mean over test samples of Var(predictions across models)
        where Var at sample j = mean over models of (pred[m,j] - mean_pred[j])²

    Args:
        predictions_per_model: Array of shape (n_models, n_test_samples).

    Returns:
        Scalar float: mean variance across all test samples.

    Example:
        >>> preds = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
        >>> compute_variance(preds)  # variance at each test point, then averaged
    """
    # TODO: Step 1 — Compute the mean prediction for each test sample (axis=0).
    # TODO: Step 2 — Compute variance for each test sample:
    #       np.var(predictions_per_model, axis=0) computes per-sample variance.
    #       IMPORTANT: Use ddof=0 here (we are averaging over model instantiations,
    #       not estimating a population parameter).
    # TODO: Step 3 — Return the mean variance over all test samples.
    pass


# ---------------------------------------------------------------------------
# 3. Bias-Variance Decomposition
# ---------------------------------------------------------------------------

def bias_variance_decomposition(model_fn, X_train_sets: list,
                                 X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Perform bias-variance decomposition by training a model on multiple
    different training sets and evaluating on a shared test set.

    In practice, we approximate multiple "different training sets" using
    bootstrap sampling (sampling with replacement from the original data).

    Args:
        model_fn: Callable that returns a new untrained model instance.
                  Example: lambda: DecisionTreeRegressor(max_depth=3)
        X_train_sets: List of training feature arrays, one per bootstrap sample.
                      Each element has shape (n_train, n_features).
        X_test: Test features of shape (n_test, n_features).
        y_test: True test labels of shape (n_test,).

    Returns:
        Dictionary with keys:
            'bias_squared': scalar float
            'variance': scalar float
            'total_error': bias_squared + variance (ignoring irreducible noise)
            'predictions': array of shape (n_models, n_test_samples)

    Notes:
        - This does NOT include irreducible noise (we cannot measure it directly).
        - y_train_sets is derived from X_train_sets via the true function + noise.
    """
    # TODO: Step 1 — For each X_train in X_train_sets:
    #           a. Generate y_train: you need a corresponding y for each X_train.
    #              (In the calling code, pass both X_train_sets and y_train_sets,
    #               or encode the true function. Adapt the signature if needed.)
    #           b. Create a new model: model = model_fn()
    #           c. Fit the model: model.fit(X_train, y_train)
    #           d. Predict on X_test: preds = model.predict(X_test)
    #           e. Collect preds into a list.
    # TODO: Step 2 — Stack all predictions: shape (n_models, n_test_samples).
    # TODO: Step 3 — Call compute_bias(predictions, y_test).
    # TODO: Step 4 — Call compute_variance(predictions).
    # TODO: Step 5 — Return the results dictionary.
    # HINT: You may need to update the function signature to accept y_train_sets.
    pass


# ---------------------------------------------------------------------------
# 4. Simulate Tradeoff
# ---------------------------------------------------------------------------

def simulate_tradeoff(X: np.ndarray, y: np.ndarray,
                      model_complexities: list,
                      n_bootstrap: int = 50,
                      test_size: float = 0.3) -> tuple[list, list, list]:
    """
    Simulate the bias-variance tradeoff by varying model complexity.

    For each complexity level, train many models (on bootstrap samples) and
    compute bias² and variance on a held-out test set.

    Args:
        X: Feature array of shape (n_samples, n_features) or (n_samples,).
        y: Target array of shape (n_samples,).
        model_complexities: List of complexity values to try.
            These are interpreted as polynomial degrees (int).
            E.g., [1, 2, 3, 5, 7, 10, 15]
        n_bootstrap: Number of bootstrap models to train per complexity level.
        test_size: Fraction of data to hold out for testing.

    Returns:
        Tuple of three lists (biases, variances, total_errors), one value
        per entry in model_complexities.

    Steps outline:
        1. Split X, y into train and test sets (do NOT bootstrap the test set).
        2. For each complexity (polynomial degree):
           a. Create n_bootstrap bootstrap samples from training data.
           b. For each bootstrap sample, fit a Pipeline([PolynomialFeatures(degree),
              LinearRegression()]) and collect predictions on X_test.
           c. Compute bias and variance.
        3. Append results to lists and return.
    """
    # TODO: Step 1 — Split data into train/test (use np.random or sklearn's
    #       train_test_split). Set a random seed for reproducibility.
    # TODO: Step 2 — Loop over each complexity in model_complexities:
    #       a. Initialize a list to store predictions.
    #       b. Loop n_bootstrap times:
    #           - Sample with replacement from training data using resample().
    #           - Build: model = Pipeline([
    #               ('poly', PolynomialFeatures(degree=complexity, include_bias=False)),
    #               ('lr', LinearRegression())
    #             ])
    #           - Fit model on bootstrap sample.
    #           - Collect model.predict(X_test.reshape(-1, 1)) if 1D.
    #       c. Stack predictions: shape (n_bootstrap, n_test).
    #       d. Compute bias_sq = compute_bias(predictions, y_test).
    #       e. Compute var = compute_variance(predictions).
    #       f. Append to biases, variances, total_errors.
    # TODO: Step 3 — Return (biases, variances, total_errors).
    pass


# ---------------------------------------------------------------------------
# 5. Plot Bias-Variance Curves (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_bias_variance_curves(complexities: list, biases: list,
                               variances: list, total_errors: list):
    """
    Plot the bias², variance, and total error curves against model complexity.

    This is the classic U-shaped total error curve that results from the
    bias-variance tradeoff. Fully implemented — study the output!

    Args:
        complexities: List of complexity values (x-axis).
        biases: List of bias² values.
        variances: List of variance values.
        total_errors: List of (bias² + variance) values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left plot: individual curves ---
    ax = axes[0]
    ax.plot(complexities, biases, 'b-o', label='Bias²', linewidth=2, markersize=5)
    ax.plot(complexities, variances, 'r-o', label='Variance', linewidth=2, markersize=5)
    ax.plot(complexities, total_errors, 'g-o', label='Bias² + Variance (Total)',
            linewidth=2.5, markersize=5)
    ax.axvline(x=complexities[np.argmin(total_errors)], color='green',
               linestyle='--', alpha=0.7, label='Optimal Complexity')
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Right plot: stacked area chart ---
    ax = axes[1]
    ax.stackplot(complexities, biases, variances,
                 labels=['Bias²', 'Variance'],
                 colors=['steelblue', 'tomato'], alpha=0.7)
    ax.plot(complexities, total_errors, 'k-', linewidth=2, label='Total Error')
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error Composition (Stacked)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotations
    opt_idx = np.argmin(total_errors)
    opt_complexity = complexities[opt_idx]
    opt_error = total_errors[opt_idx]
    axes[0].annotate(
        f'  Optimal\n  complexity={opt_complexity}\n  error={opt_error:.4f}',
        xy=(opt_complexity, opt_error),
        xytext=(opt_complexity + 1, opt_error + 0.01),
        arrowprops=dict(arrowstyle='->', color='green'),
        fontsize=9, color='green'
    )

    plt.tight_layout()
    plt.savefig('bias_variance_tradeoff.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Plot saved to: bias_variance_tradeoff.png")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate the full bias-variance tradeoff workflow.

    1. Generate a noisy polynomial dataset (true function + Gaussian noise).
    2. Run bias-variance decomposition for a range of polynomial degrees.
    3. Plot the tradeoff curves.
    4. Interpret the results.
    """
    print("=" * 60)
    print("Exercise 02: Bias-Variance Tradeoff")
    print("=" * 60)

    # --- Step 1: Generate Dataset ---
    print("\n--- Step 1: Generating noisy polynomial dataset ---")
    # TODO: Set a random seed for reproducibility: np.random.seed(42)
    # TODO: Generate X = np.linspace(-3, 3, 200) (or use np.random.uniform)
    # TODO: Define the TRUE function: y_true = 0.5*X^3 - 2*X^2 + X + 1 (or similar)
    # TODO: Add Gaussian noise: y = y_true + np.random.normal(0, noise_std, len(X))
    # TODO: Reshape X to (-1, 1) for sklearn compatibility.
    # TODO: Print: dataset size, noise level, true function description.

    # --- Step 2: Visualize the Dataset ---
    print("\n--- Step 2: Visualizing dataset ---")
    # TODO: Plot scatter(X, y) in light gray and line(X, y_true) in red.
    # TODO: Label axes, add legend, show plot.

    # --- Step 3: Bias-Variance Decomposition for Selected Models ---
    print("\n--- Step 3: Bias-Variance for specific model complexities ---")
    # TODO: For a low-complexity model (degree=1) and high-complexity model (degree=12):
    #       a. Create bootstrap training sets using resample().
    #       b. Call bias_variance_decomposition() (adapt as needed).
    #       c. Print and compare bias², variance, total error for both models.

    # --- Step 4: Simulate Full Tradeoff Curve ---
    print("\n--- Step 4: Simulating full tradeoff curve ---")
    # TODO: Set complexities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    # TODO: Call simulate_tradeoff(X, y, complexities, n_bootstrap=100)
    # TODO: Print a table: complexity | bias² | variance | total_error

    # --- Step 5: Plot ---
    print("\n--- Step 5: Plotting bias-variance curves ---")
    # TODO: Call plot_bias_variance_curves(complexities, biases, variances, total_errors)

    # --- Step 6: Interpretation ---
    print("\n--- Step 6: Interpretation ---")
    # TODO: Find the complexity with minimum total error.
    # TODO: Print: "Best complexity: X (bias²=Y, variance=Z, total=W)"
    # TODO: Print a brief interpretation:
    #       - Which degrees underfit (high bias)?
    #       - Which degrees overfit (high variance)?
    #       - Where is the sweet spot?

    print("\nDone! Review bias_variance_tradeoff.png for the tradeoff curves.")


if __name__ == "__main__":
    main()
