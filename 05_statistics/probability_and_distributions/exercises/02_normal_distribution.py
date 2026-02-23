"""
Exercise 02: Normal Distribution & Central Limit Theorem
=========================================================
Topic: Gaussian Distribution (Phân phối chuẩn) and CLT
Difficulty: Beginner → Intermediate

Learning Objectives:
    - Implement the Normal PDF and CDF from scratch
    - Verify the 68-95-99.7 rule empirically
    - Demonstrate the Central Limit Theorem for arbitrary distributions
    - Standardize data (z-scores)
    - Visualize multiple normal distributions and CLT convergence

Key Facts:
    PDF: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))
    68-95-99.7 rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ
    CLT: mean of n i.i.d. samples → N(mu, sigma^2/n) as n → ∞

Instructions:
    Implement all TODO sections. The plot_* functions are fully provided.
    Focus on understanding WHY the CLT is so powerful and universal.

Dependencies:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import math


# ---------------------------------------------------------------------------
# 1. Normal PDF
# ---------------------------------------------------------------------------

def normal_pdf(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """
    Compute the probability density function of the Normal distribution.

    Formula:
        f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(-(x - mu)^2 / (2 * sigma^2))

    Args:
        x: Point(s) at which to evaluate the PDF. Can be scalar or numpy array.
        mu: Mean of the distribution.
        sigma: Standard deviation (must be > 0).

    Returns:
        PDF value(s) at x. Same shape as x.

    Raises:
        ValueError: If sigma <= 0.

    Example:
        >>> normal_pdf(0, mu=0, sigma=1)  # standard normal at 0
        0.3989422...  (= 1/sqrt(2*pi))
        >>> normal_pdf(1, mu=0, sigma=1)
        0.2419707...
    """
    # TODO: Step 1 — Validate sigma > 0, raise ValueError if not.
    # TODO: Step 2 — Compute the coefficient: 1 / (sigma * sqrt(2 * pi)).
    #       Use math.pi or np.pi, and math.sqrt or np.sqrt.
    # TODO: Step 3 — Compute the exponent: -(x - mu)^2 / (2 * sigma^2).
    # TODO: Step 4 — Return coefficient * np.exp(exponent).
    # HINT: Verify with scipy_stats.norm.pdf(x, loc=mu, scale=sigma).
    pass


# ---------------------------------------------------------------------------
# 2. Normal CDF
# ---------------------------------------------------------------------------

def normal_cdf(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """
    Compute the cumulative distribution function of the Normal distribution.

    The CDF gives P(X <= x) for X ~ N(mu, sigma^2).

    There is no closed-form formula for the Normal CDF, but it can be expressed
    using the error function (erf):
        CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

    Args:
        x: Point(s) at which to evaluate the CDF.
        mu: Mean of the distribution.
        sigma: Standard deviation (must be > 0).

    Returns:
        CDF value(s) P(X <= x). Values are in [0, 1].

    Example:
        >>> normal_cdf(0, mu=0, sigma=1)   # P(Z <= 0) = 0.5
        0.5
        >>> normal_cdf(1.96, mu=0, sigma=1)  # P(Z <= 1.96) ≈ 0.975
        0.975
    """
    # TODO: Use the math.erf or scipy.special.erf function.
    # Formula: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    # OPTION A (preferred): from scipy.special import erf; return 0.5*(1 + erf(...))
    # OPTION B: return scipy_stats.norm.cdf(x, loc=mu, scale=sigma)
    # HINT: math.erf works for scalars; scipy.special.erf works for arrays.
    pass


# ---------------------------------------------------------------------------
# 3. Sample from Normal
# ---------------------------------------------------------------------------

def sample_normal(mu: float, sigma: float, n: int) -> np.ndarray:
    """
    Draw n random samples from a Normal distribution N(mu, sigma^2).

    Args:
        mu: Mean of the distribution.
        sigma: Standard deviation (must be > 0).
        n: Number of samples to draw.

    Returns:
        1D numpy array of n samples.

    Example:
        >>> samples = sample_normal(mu=5.0, sigma=2.0, n=1000)
        >>> abs(samples.mean() - 5.0) < 0.1   # should be approximately 5.0
        True
    """
    # TODO: Use np.random.normal(loc=mu, scale=sigma, size=n).
    pass


# ---------------------------------------------------------------------------
# 4. Verify the 68-95-99.7 Rule
# ---------------------------------------------------------------------------

def verify_68_95_997_rule(mu: float, sigma: float,
                           n_samples: int = 1_000_000) -> dict:
    """
    Empirically verify the 68-95-99.7 rule for the Normal distribution.

    Draw a large number of samples and check what fraction fall within
    1σ, 2σ, and 3σ of the mean.

    Args:
        mu: Mean of the distribution.
        sigma: Standard deviation.
        n_samples: Number of samples to draw.

    Returns:
        Dictionary with keys:
            'within_1sigma': fraction of samples within [mu - sigma, mu + sigma]
            'within_2sigma': fraction of samples within [mu - 2*sigma, mu + 2*sigma]
            'within_3sigma': fraction of samples within [mu - 3*sigma, mu + 3*sigma]
            'expected_1sigma': 0.6827 (theoretical)
            'expected_2sigma': 0.9545 (theoretical)
            'expected_3sigma': 0.9973 (theoretical)

    Example:
        >>> result = verify_68_95_997_rule(mu=0, sigma=1, n_samples=1_000_000)
        >>> print(result['within_1sigma'])  # should be ~0.683
        >>> print(result['within_2sigma'])  # should be ~0.954
    """
    # TODO: Step 1 — Draw n_samples from N(mu, sigma): samples = sample_normal(mu, sigma, n_samples).
    # TODO: Step 2 — Compute fraction within mu ± k*sigma for k = 1, 2, 3.
    #       Hint: np.mean(np.abs(samples - mu) <= k * sigma) gives the fraction.
    # TODO: Step 3 — Return the results dictionary with both empirical and theoretical values.
    pass


# ---------------------------------------------------------------------------
# 5. Central Limit Theorem Demo
# ---------------------------------------------------------------------------

def central_limit_theorem_demo(distribution_fn, n_samples: int,
                                n_experiments: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Demonstrate the Central Limit Theorem for an arbitrary distribution.

    Draw n_samples from ANY distribution n_experiments times.
    Compute the mean of each experiment. Show that these means are
    approximately normally distributed, regardless of the original distribution.

    Args:
        distribution_fn: Callable with no arguments that returns an array of
                         n_samples values drawn from some distribution.
                         Example: lambda: np.random.uniform(0, 1, 1000)
        n_samples: Number of samples to draw in each experiment.
        n_experiments: Number of independent experiments to run.

    Returns:
        Tuple (original_samples, means) where:
            original_samples: One set of raw samples from the distribution.
            means: Array of length n_experiments — one sample mean per experiment.

    Example:
        >>> # CLT with Uniform distribution
        >>> samples, means = central_limit_theorem_demo(
        ...     lambda: np.random.uniform(0, 1, 500),
        ...     n_samples=500, n_experiments=10000
        ... )
        >>> # means should look normally distributed regardless of Uniform input
    """
    # TODO: Step 1 — Draw one set of original_samples = distribution_fn()
    #       for visualization purposes.
    # TODO: Step 2 — Run n_experiments iterations:
    #       For each, call distribution_fn() and compute its mean.
    # TODO: Step 3 — Return (original_samples, array of means).
    # HINT: Use a list comprehension or for loop.
    pass


# ---------------------------------------------------------------------------
# 6. Standardize (Z-scores)
# ---------------------------------------------------------------------------

def standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardize a dataset by computing z-scores.

    The z-score of each value measures how many standard deviations
    it is away from the mean. After standardization, the data has mean 0
    and standard deviation 1.

    Formula:
        z_i = (x_i - mean(X)) / std(X)

    Args:
        X: 1D or 2D numpy array. If 2D, standardize each column independently.

    Returns:
        Array of same shape as X with mean ≈ 0 and std ≈ 1 per column.

    Raises:
        ValueError: If any column has std = 0 (constant column).

    Example:
        >>> X = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        >>> z = standardize(X)
        >>> abs(z.mean()) < 1e-10   # mean should be ~0
        True
        >>> abs(z.std() - 1.0) < 1e-10  # std should be ~1
        True
    """
    # TODO: Step 1 — Compute mean and std along axis=0 (per column if 2D).
    # TODO: Step 2 — Check for zero std (constant columns) and raise ValueError.
    # TODO: Step 3 — Return (X - mean) / std.
    # HINT: For 2D, keepdims=True ensures broadcasting works correctly.
    pass


# ---------------------------------------------------------------------------
# 7. Plot Normal Distributions (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_normal_distributions(params_list: list[tuple[float, float]]):
    """
    Overlay multiple normal distribution PDFs on a single plot.

    Args:
        params_list: List of (mu, sigma) tuples, one per distribution to plot.
                     Example: [(0, 1), (0, 2), (2, 0.5), (-1, 1.5)]
    """
    x = np.linspace(-6, 8, 500)
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange',
              'purple', 'brown', 'deeppink']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (mu, sigma) in enumerate(params_list):
        y = normal_pdf(x, mu, sigma)
        color = colors[i % len(colors)]
        ax.plot(x, y, linewidth=2.5, color=color,
                label=f'N(μ={mu}, σ={sigma})')
        ax.fill_between(x, y, alpha=0.1, color=color)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Probability Density f(x)', fontsize=12)
    ax.set_title('Normal Distributions with Different Parameters', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('normal_distributions.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Plot saved to: normal_distributions.png")


# ---------------------------------------------------------------------------
# 8. Plot CLT Demo (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_clt_demo(original_samples: np.ndarray, means: np.ndarray,
                  distribution_name: str = "Unknown"):
    """
    Visualize the Central Limit Theorem:
        - Left: histogram of original samples (any distribution)
        - Right: histogram of sample means (should look Normal)

    Args:
        original_samples: Raw samples from the original distribution.
        means: Array of sample means from many experiments.
        distribution_name: Name of the distribution for the plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: original distribution ---
    ax = axes[0]
    ax.hist(original_samples, bins=50, density=True, alpha=0.7,
            color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_title(f'Original Distribution\n({distribution_name})', fontsize=13)
    ax.set_xlabel('Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Right: distribution of means ---
    ax = axes[1]
    ax.hist(means, bins=60, density=True, alpha=0.7,
            color='tomato', edgecolor='white', linewidth=0.5)

    # Overlay the theoretical Normal
    mu_means = np.mean(means)
    sigma_means = np.std(means)
    x_range = np.linspace(mu_means - 4 * sigma_means,
                          mu_means + 4 * sigma_means, 300)
    ax.plot(x_range, normal_pdf(x_range, mu_means, sigma_means),
            'k-', linewidth=2.5, label='Theoretical Normal')

    ax.set_title(f'Distribution of Sample Means\n(CLT: should be Normal)',
                 fontsize=13)
    ax.set_xlabel('Sample Mean', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Central Limit Theorem Demonstration', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'clt_demo_{distribution_name.lower().replace(" ", "_")}.png',
                dpi=100, bbox_inches='tight')
    plt.show()
    print(f"CLT plot saved.")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate Normal distribution properties and the Central Limit Theorem.

    Sections:
        1. PDF and CDF computation
        2. Empirical verification of the 68-95-99.7 rule
        3. Standardization (z-scores)
        4. Plot multiple Normal distributions
        5. CLT with Uniform distribution
        6. CLT with Exponential distribution
        7. CLT with a Bimodal distribution
    """
    print("=" * 60)
    print("Exercise 02: Normal Distribution & CLT")
    print("=" * 60)

    np.random.seed(42)

    # --- Section 1: PDF and CDF ---
    print("\n--- Section 1: PDF and CDF ---")
    # TODO: Compute and print normal_pdf(0, 0, 1)  → expected: 0.3989
    # TODO: Compute and print normal_pdf(1, 0, 1)  → expected: 0.2420
    # TODO: Compute and print normal_cdf(0, 0, 1)  → expected: 0.5
    # TODO: Compute and print normal_cdf(1.96, 0, 1) → expected: ~0.975
    # TODO: Verify: P(-1 < Z < 1) = CDF(1) - CDF(-1) → expected: ~0.6827
    # TODO: Verify: P(-2 < Z < 2) = CDF(2) - CDF(-2) → expected: ~0.9545

    # --- Section 2: 68-95-99.7 Rule ---
    print("\n--- Section 2: Verifying the 68-95-99.7 Rule ---")
    # TODO: Call verify_68_95_997_rule(mu=0, sigma=1, n_samples=1_000_000)
    # TODO: Print a table comparing empirical vs theoretical fractions.

    # --- Section 3: Standardization ---
    print("\n--- Section 3: Standardization (Z-scores) ---")
    # TODO: Create a sample dataset, e.g., heights in cm.
    # TODO: Call standardize() and verify: mean ≈ 0, std ≈ 1.
    # TODO: Show that the z-score tells you how many standard deviations a
    #       value is from the mean.

    # --- Section 4: Plot Normal Distributions ---
    print("\n--- Section 4: Plotting Normal Distributions ---")
    # TODO: Call plot_normal_distributions([(0, 1), (0, 2), (2, 0.5), (-1, 1.5)])

    # --- Section 5: CLT with Uniform Distribution ---
    print("\n--- Section 5: CLT with Uniform Distribution ---")
    # TODO: Demonstrate CLT with Uniform(0, 1):
    #       samples, means = central_limit_theorem_demo(
    #           lambda: np.random.uniform(0, 1, 500),
    #           n_samples=500, n_experiments=10000
    #       )
    # TODO: Call plot_clt_demo(samples, means, "Uniform(0,1)")
    # TODO: Print: theoretical mean of means (0.5), empirical mean.

    # --- Section 6: CLT with Exponential Distribution ---
    print("\n--- Section 6: CLT with Exponential Distribution ---")
    # TODO: Repeat CLT demo with Exponential(rate=1):
    #       lambda: np.random.exponential(scale=1.0, size=500)
    # TODO: Call plot_clt_demo(samples, means, "Exponential(1)")
    # TODO: Note: Exponential is highly skewed, yet means become Normal!

    # --- Section 7: CLT with Bimodal Distribution ---
    print("\n--- Section 7: CLT with Bimodal Distribution ---")
    # TODO: Create a bimodal distribution by mixing two Normals:
    #       0.5 * N(-3, 0.5) + 0.5 * N(3, 0.5)
    #       fn = lambda: np.where(np.random.random(500) < 0.5,
    #                             np.random.normal(-3, 0.5, 500),
    #                             np.random.normal(3, 0.5, 500))
    # TODO: Run CLT demo and call plot_clt_demo(samples, means, "Bimodal")
    # TODO: Observe: the bimodal original becomes Normal means!

    print("\nDone! Review the saved plots to see the Normal distribution and CLT in action.")


if __name__ == "__main__":
    main()
