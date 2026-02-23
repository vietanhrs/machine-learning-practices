"""
Exercise 01: Variance & Statistics from Scratch
================================================
Topic: Variance (Phương sai), Covariance, Correlation
Difficulty: Beginner → Intermediate

Learning Objectives:
    - Implement population and sample variance from scratch
    - Understand Bessel's correction (why n-1 instead of n)
    - Implement covariance and Pearson correlation
    - Build a full correlation matrix
    - Visualize correlation patterns with scatter plots

Instructions:
    Each function has a TODO block. Implement the function body using only
    the hints provided. Do not use numpy/scipy shortcuts for the core
    statistics functions (build them from scratch), EXCEPT where noted.

Dependencies:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# 1. Population Variance
# ---------------------------------------------------------------------------

def population_variance(data: list | np.ndarray) -> float:
    """
    Compute the population variance of a dataset.

    Population variance divides by N (the total number of elements), using
    the true population mean mu. Use this when the data IS the full population.

    Formula:
        sigma^2 = (1/N) * sum((x_i - mu)^2)
        where mu = (1/N) * sum(x_i)

    Args:
        data: 1D array-like of numeric values.

    Returns:
        Population variance as a float.

    Example:
        >>> population_variance([2, 4, 4, 4, 5, 5, 7, 9])
        4.0
    """
    # TODO: Step 1 — Convert data to a numpy array of floats.
    # TODO: Step 2 — Compute the mean (mu) of the data.
    # TODO: Step 3 — Compute squared deviations: (x_i - mu)^2 for each element.
    # TODO: Step 4 — Return the mean of the squared deviations (divide by N).
    pass


# ---------------------------------------------------------------------------
# 2. Sample Variance
# ---------------------------------------------------------------------------

def sample_variance(data: list | np.ndarray) -> float:
    """
    Compute the sample variance using Bessel's correction.

    Sample variance divides by (n - 1) instead of n to produce an unbiased
    estimator of the population variance. Use this when data is a sample
    drawn from a larger population.

    Formula:
        s^2 = (1/(n-1)) * sum((x_i - x_bar)^2)
        where x_bar = (1/n) * sum(x_i)

    Args:
        data: 1D array-like of numeric values. Must have at least 2 elements.

    Returns:
        Sample variance as a float.

    Raises:
        ValueError: If data has fewer than 2 elements.

    Example:
        >>> sample_variance([2, 4, 4, 4, 5, 5, 7, 9])
        4.571428...
    """
    # TODO: Step 1 — Convert data to numpy array. Raise ValueError if len < 2.
    # TODO: Step 2 — Compute the sample mean x_bar.
    # TODO: Step 3 — Compute squared deviations from x_bar.
    # TODO: Step 4 — Return sum of squared deviations divided by (n - 1).
    # HINT: Compare your result with np.var(data, ddof=1) to verify.
    pass


# ---------------------------------------------------------------------------
# 3. Standard Deviation
# ---------------------------------------------------------------------------

def standard_deviation(data: list | np.ndarray, population: bool = True) -> float:
    """
    Compute the standard deviation of a dataset.

    Standard deviation is the square root of variance. It has the same units
    as the original data, making it easier to interpret than variance.

    Args:
        data: 1D array-like of numeric values.
        population: If True, compute population std (divide by N).
                    If False, compute sample std (divide by n-1).

    Returns:
        Standard deviation as a float.

    Example:
        >>> standard_deviation([2, 4, 4, 4, 5, 5, 7, 9], population=True)
        2.0
    """
    # TODO: Call population_variance() or sample_variance() depending on
    #       the `population` flag, then return the square root (use math.sqrt
    #       or np.sqrt).
    # HINT: np.sqrt(population_variance(data)) should work.
    pass


# ---------------------------------------------------------------------------
# 4. Covariance
# ---------------------------------------------------------------------------

def covariance(X: list | np.ndarray, Y: list | np.ndarray) -> float:
    """
    Compute the sample covariance between two variables X and Y.

    Covariance measures how two variables change together. A positive value
    means they tend to increase together; negative means they move in
    opposite directions.

    Formula:
        Cov(X, Y) = (1/(n-1)) * sum((x_i - x_bar)(y_i - y_bar))

    Args:
        X: 1D array-like, first variable.
        Y: 1D array-like, second variable. Must be the same length as X.

    Returns:
        Sample covariance as a float.

    Raises:
        ValueError: If X and Y have different lengths.

    Example:
        >>> X = [1, 2, 3, 4, 5]
        >>> Y = [2, 4, 6, 8, 10]  # perfect positive relationship
        >>> covariance(X, Y)  # should be positive and large
    """
    # TODO: Step 1 — Convert X and Y to numpy arrays. Validate same length.
    # TODO: Step 2 — Compute means x_bar and y_bar.
    # TODO: Step 3 — Compute element-wise product: (x_i - x_bar) * (y_i - y_bar)
    # TODO: Step 4 — Return the sum of those products divided by (n - 1).
    # HINT: np.cov(X, Y)[0, 1] gives the sample covariance for verification.
    pass


# ---------------------------------------------------------------------------
# 5. Pearson Correlation
# ---------------------------------------------------------------------------

def pearson_correlation(X: list | np.ndarray, Y: list | np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient between X and Y.

    Correlation normalizes covariance by the product of standard deviations,
    producing a dimensionless value in [-1, +1].

    Formula:
        r = Cov(X, Y) / (std_X * std_Y)

    Args:
        X: 1D array-like.
        Y: 1D array-like. Same length as X.

    Returns:
        Pearson correlation coefficient r in [-1, +1].

    Example:
        >>> pearson_correlation([1,2,3,4,5], [2,4,6,8,10])
        1.0  # perfect positive linear correlation
        >>> pearson_correlation([1,2,3,4,5], [10,8,6,4,2])
        -1.0  # perfect negative linear correlation
    """
    # TODO: Step 1 — Call covariance(X, Y) to get the covariance.
    # TODO: Step 2 — Compute sample standard deviations of X and Y using
    #       standard_deviation(X, population=False) and same for Y.
    # TODO: Step 3 — Return cov_xy / (std_x * std_y).
    # HINT: Handle the edge case where std_x or std_y is 0 (constant variable).
    # HINT: scipy_stats.pearsonr(X, Y)[0] for verification.
    pass


# ---------------------------------------------------------------------------
# 6. Correlation Matrix
# ---------------------------------------------------------------------------

def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the full correlation matrix for a 2D dataset.

    The correlation matrix has shape (p, p) where p is the number of variables
    (columns). Entry (i, j) is the Pearson correlation between column i and
    column j. The diagonal is always 1.0.

    Args:
        data: 2D numpy array of shape (n_samples, n_features).
              Each column is a variable; each row is an observation.

    Returns:
        Symmetric correlation matrix of shape (n_features, n_features).

    Example:
        >>> data = np.array([[1,2,3], [4,5,6], [7,8,9]]).T
        >>> correlation_matrix(data)  # 3x3 matrix with 1s on diagonal
    """
    # TODO: Step 1 — Get n_features = data.shape[1].
    # TODO: Step 2 — Initialize a (n_features, n_features) matrix of zeros.
    # TODO: Step 3 — Loop over all pairs (i, j) and fill in pearson_correlation
    #       of data[:, i] and data[:, j].
    # TODO: Step 4 — Return the matrix.
    # HINT: The matrix should be symmetric: corr[i,j] == corr[j,i].
    # HINT: np.corrcoef(data.T) for verification.
    pass


# ---------------------------------------------------------------------------
# 7. Demonstrate Bessel's Correction
# ---------------------------------------------------------------------------

def demonstrate_bessel_correction(true_variance: float, n: int = 10,
                                  n_simulations: int = 10_000) -> dict:
    """
    Empirically show that sample variance with (n-1) is an unbiased estimator.

    Draw many samples of size n from a known Normal(0, sqrt(true_variance))
    distribution. For each sample, compute both:
        - biased estimate: divide by n
        - unbiased estimate: divide by n-1 (Bessel's correction)
    Show that the mean of unbiased estimates converges to true_variance,
    while the mean of biased estimates is systematically lower.

    Args:
        true_variance: The known population variance to use.
        n: Size of each sample drawn.
        n_simulations: Number of Monte Carlo repetitions.

    Returns:
        Dictionary with keys:
            'true_variance': the input true_variance
            'mean_biased': mean of all biased estimates (should be < true_variance)
            'mean_unbiased': mean of all unbiased estimates (should ≈ true_variance)
            'bias_of_biased': mean_biased - true_variance (should be ≈ -sigma^2/n)
            'bias_of_unbiased': mean_unbiased - true_variance (should be ≈ 0)

    Example:
        >>> result = demonstrate_bessel_correction(true_variance=4.0, n=5)
        >>> print(result['mean_unbiased'])  # should be close to 4.0
        >>> print(result['mean_biased'])    # should be close to 4.0 * 4/5 = 3.2
    """
    # TODO: Step 1 — Draw n_simulations samples, each of size n, from
    #       np.random.normal(0, np.sqrt(true_variance), size=(n_simulations, n)).
    # TODO: Step 2 — For each simulation, compute:
    #           biased = sum of squared deviations from sample mean / n
    #           unbiased = sum of squared deviations from sample mean / (n - 1)
    #       Hint: Use axis=1 to operate row-wise.
    # TODO: Step 3 — Compute mean_biased and mean_unbiased across all simulations.
    # TODO: Step 4 — Return the results dictionary.
    pass


# ---------------------------------------------------------------------------
# 8. Plot Correlation Examples
# ---------------------------------------------------------------------------

def plot_correlation_examples():
    """
    Create a 2x2 grid of scatter plots showing different correlation patterns:
        1. Strong positive correlation (r ≈ +1)
        2. Strong negative correlation (r ≈ -1)
        3. Zero/no linear correlation (r ≈ 0)
        4. Non-linear relationship (r ≈ 0 but clearly related)

    For each plot, display the computed Pearson r value in the title.

    This illustrates what correlation does and does NOT capture.
    """
    # TODO: Step 1 — Generate x = np.linspace(-3, 3, 100).
    # TODO: Step 2 — Create y values for each scenario:
    #       - Positive: y = x + small_noise
    #       - Negative: y = -x + small_noise
    #       - Zero: y = random noise (independent of x)
    #       - Non-linear: y = x^2 + small_noise (quadratic)
    # TODO: Step 3 — Compute pearson_correlation() for each pair.
    # TODO: Step 4 — Create a plt.subplots(2, 2) figure.
    # TODO: Step 5 — Scatter plot each case with appropriate title including r value.
    # TODO: Step 6 — plt.tight_layout() and plt.show().
    pass


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate all variance and statistics functions using a real-world dataset.

    Uses the California Housing dataset (from sklearn) to compute:
        - Population and sample variance for each feature
        - Standard deviations
        - Covariance and correlation between key feature pairs
        - Full correlation matrix visualized as a heatmap
        - Bessel correction demonstration
        - Correlation scatter plot examples
    """
    print("=" * 60)
    print("Exercise 01: Variance & Statistics from Scratch")
    print("=" * 60)

    # Load dataset
    # TODO: Use sklearn.datasets.fetch_california_housing() or create synthetic data.
    # Hint:
    #   from sklearn.datasets import fetch_california_housing
    #   housing = fetch_california_housing()
    #   data = housing.data
    #   feature_names = housing.feature_names

    # --- Section 1: Basic Statistics ---
    print("\n--- Section 1: Basic Statistics ---")
    # TODO: For the first two features, compute and print:
    #   - population_variance()
    #   - sample_variance()
    #   - standard_deviation() (both population and sample)
    # TODO: Compare your results with numpy's np.var() and np.std().

    # --- Section 2: Bessel's Correction Demo ---
    print("\n--- Section 2: Bessel's Correction Demo ---")
    # TODO: Call demonstrate_bessel_correction(true_variance=9.0, n=5, n_simulations=10000)
    # TODO: Print the results table: true_variance, mean_biased, mean_unbiased, biases.

    # --- Section 3: Covariance and Correlation ---
    print("\n--- Section 3: Covariance and Correlation ---")
    # TODO: Pick two features and compute covariance() and pearson_correlation().
    # TODO: Verify against np.cov() and scipy_stats.pearsonr().

    # --- Section 4: Correlation Matrix ---
    print("\n--- Section 4: Correlation Matrix ---")
    # TODO: Compute correlation_matrix() on the first 4 features.
    # TODO: Print the matrix and verify against np.corrcoef().
    # TODO: Plot as a heatmap using matplotlib (plt.imshow or seaborn.heatmap).

    # --- Section 5: Visualization ---
    print("\n--- Section 5: Visualization ---")
    # TODO: Call plot_correlation_examples().


if __name__ == "__main__":
    main()
