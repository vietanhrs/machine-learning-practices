"""
Exercise 03: Long-Tail & Power Law Distributions
==================================================
Topic: Power Laws, Zipf's Law (Phân phối đuôi dài)
Difficulty: Intermediate → Advanced

Learning Objectives:
    - Implement the power-law PDF and sampling via inverse transform
    - Model word frequencies using Zipf's Law
    - Fit a power-law exponent from data using MLE
    - Compare Normal vs long-tail distributions on linear and log-log scales
    - Understand why long tails matter in ML (rare events, word frequencies)

Key Concepts:
    Power Law PDF: p(x) ∝ x^(-alpha), for x >= x_min, alpha > 1
    Zipf's Law: frequency of rank r ∝ 1/r^s  (s ≈ 1 for natural language)
    MLE for alpha: alpha_hat = 1 + n / sum(log(x_i / x_min))
    Sampling (Inverse Transform): x = x_min * (1 - U)^(-1/(alpha-1)), U ~ Uniform(0,1)

Instructions:
    Implement all TODO sections. The plot_log_log() function is fully provided.
    The word_frequency_zipf() function will be the most revealing — run it
    on English text to observe Zipf's Law in action.

Dependencies:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re


# ---------------------------------------------------------------------------
# 1. Power Law PDF
# ---------------------------------------------------------------------------

def power_law_pdf(x: float | np.ndarray, alpha: float,
                  x_min: float) -> float | np.ndarray:
    """
    Compute the normalized Power Law probability density function.

    Formula:
        p(x) = (alpha - 1) / x_min * (x / x_min)^(-alpha)
             = (alpha - 1) * x_min^(alpha-1) * x^(-alpha)

    for x >= x_min, and p(x) = 0 for x < x_min.

    The normalization constant ensures integral from x_min to infinity equals 1.

    Args:
        x: Point(s) at which to evaluate the PDF.
        alpha: Power law exponent. Must be > 1 for the distribution to be normalizable.
        x_min: Minimum value of x (lower bound of support). Must be > 0.

    Returns:
        PDF value(s) at x. Zero for x < x_min.

    Raises:
        ValueError: If alpha <= 1 or x_min <= 0.

    Example:
        >>> power_law_pdf(1.0, alpha=2.5, x_min=1.0)
        1.5  # (2.5-1)/1.0 * (1.0/1.0)^(-2.5) = 1.5
        >>> power_law_pdf(0.5, alpha=2.5, x_min=1.0)
        0.0  # below x_min
    """
    # TODO: Step 1 — Validate alpha > 1 and x_min > 0.
    # TODO: Step 2 — Convert x to numpy array.
    # TODO: Step 3 — Compute normalization constant: (alpha - 1) / x_min.
    # TODO: Step 4 — Compute pdf = normalization * (x / x_min)^(-alpha).
    # TODO: Step 5 — Set pdf to 0 where x < x_min.
    # TODO: Step 6 — Return pdf (or scalar if input was scalar).
    pass


# ---------------------------------------------------------------------------
# 2. Sample from Power Law
# ---------------------------------------------------------------------------

def sample_power_law(alpha: float, x_min: float, n: int) -> np.ndarray:
    """
    Draw n random samples from a Power Law distribution using inverse
    transform sampling.

    Inverse Transform Method:
        If U ~ Uniform(0, 1), then X = x_min * (1 - U)^(-1 / (alpha - 1))
        follows a Power Law with exponent alpha.

        Derivation: CDF(x) = 1 - (x_min/x)^(alpha-1).
        Set U = CDF(x) and solve for x: x = x_min * (1-U)^(-1/(alpha-1)).

    Args:
        alpha: Power law exponent (must be > 1).
        x_min: Minimum value (lower bound of support, > 0).
        n: Number of samples.

    Returns:
        1D numpy array of n samples, all >= x_min.

    Example:
        >>> samples = sample_power_law(alpha=2.5, x_min=1.0, n=10000)
        >>> samples.min() >= 1.0
        True
        >>> samples.mean()  # theoretical mean = alpha*x_min/(alpha-1) = 1.667
    """
    # TODO: Step 1 — Draw U = np.random.uniform(0, 1, n).
    # TODO: Step 2 — Apply inverse CDF: x = x_min * (1 - U)^(-1 / (alpha - 1)).
    #       Hint: Use (1 - U) instead of U for numerical stability.
    # TODO: Step 3 — Return the array of samples.
    pass


# ---------------------------------------------------------------------------
# 3. Zipf Distribution
# ---------------------------------------------------------------------------

def zipf_distribution(n_items: int, s: float = 1.0) -> np.ndarray:
    """
    Compute the Zipf frequency distribution for n_items items.

    Zipf's Law: The frequency of the r-th ranked item is proportional to 1/r^s.

    Formula (normalized so frequencies sum to 1):
        freq(r) = (1/r^s) / sum_{k=1}^{n} (1/k^s)    for r = 1, 2, ..., n

    This is also known as the Zipf-Mandelbrot distribution for s ≠ 1.

    Args:
        n_items: Number of items (vocabulary size, number of cities, etc.).
        s: Zipf exponent. s=1 is classic Zipf's Law. s>1 is more concentrated.

    Returns:
        1D numpy array of length n_items.
        Entry i is the probability (relative frequency) of the (i+1)-th ranked item.
        Array sums to 1.

    Example:
        >>> freqs = zipf_distribution(n_items=100, s=1.0)
        >>> freqs[0]  # most frequent item
        >>> freqs[-1]  # least frequent item (much smaller)
        >>> freqs.sum()  # should be 1.0
    """
    # TODO: Step 1 — Create ranks array: ranks = np.arange(1, n_items + 1)
    # TODO: Step 2 — Compute unnormalized frequencies: 1.0 / ranks^s
    # TODO: Step 3 — Normalize: divide by sum to get probabilities.
    # TODO: Step 4 — Return the normalized frequency array.
    pass


# ---------------------------------------------------------------------------
# 4. Fit Power Law Exponent (MLE)
# ---------------------------------------------------------------------------

def fit_power_law_exponent(data: np.ndarray, x_min: float = None) -> dict:
    """
    Estimate the power-law exponent alpha using the Maximum Likelihood Estimator.

    The MLE for the power law exponent (Clauset et al., 2009):
        alpha_hat = 1 + n / sum_{i=1}^{n} log(x_i / x_min)

    where x_min is the minimum value in the data (or a provided threshold).

    Args:
        data: 1D numpy array of positive values assumed to follow a power law.
        x_min: Minimum value threshold. If None, use min(data).

    Returns:
        Dictionary with keys:
            'alpha_hat': MLE estimate of the power law exponent
            'x_min': the x_min used
            'n': number of data points used (those >= x_min)
            'std_error': standard error of alpha_hat ≈ (alpha - 1) / sqrt(n)

    Example:
        >>> data = sample_power_law(alpha=2.5, x_min=1.0, n=10000)
        >>> result = fit_power_law_exponent(data, x_min=1.0)
        >>> print(result['alpha_hat'])  # should be close to 2.5
    """
    # TODO: Step 1 — If x_min is None, set x_min = np.min(data).
    # TODO: Step 2 — Filter data to only include values >= x_min.
    # TODO: Step 3 — Compute n = len(filtered data).
    # TODO: Step 4 — Compute alpha_hat = 1 + n / sum(log(x_i / x_min)).
    #       Hint: np.sum(np.log(filtered_data / x_min))
    # TODO: Step 5 — Compute std_error = (alpha_hat - 1) / np.sqrt(n).
    # TODO: Step 6 — Return the results dictionary.
    pass


# ---------------------------------------------------------------------------
# 5. Compare Normal vs Long-Tail
# ---------------------------------------------------------------------------

def compare_normal_vs_longtail(n_samples: int = 10_000) -> None:
    """
    Visually compare a Normal distribution and a Power Law distribution
    on both linear scale and log-log scale.

    This comparison shows how dramatically different the tail behavior is:
    - Normal: tails drop off exponentially fast.
    - Power Law: tails drop off polynomially (much heavier).

    Args:
        n_samples: Number of samples to generate for each distribution.

    Produces:
        A 2x2 grid of plots:
        - Row 1: Linear-scale histograms (normal vs power law)
        - Row 2: Log-log scale (should be roughly linear for power law)
    """
    # TODO: Step 1 — Generate Normal samples with mean=2, std=1:
    #       normal_samples = np.random.normal(loc=2, scale=1, size=n_samples)
    #       Clip to positive values: normal_samples = normal_samples[normal_samples > 0]
    # TODO: Step 2 — Generate Power Law samples:
    #       power_samples = sample_power_law(alpha=2.5, x_min=1.0, n=n_samples)
    # TODO: Step 3 — Create 2x2 subplot grid.
    # TODO: Step 4 — Top-left: Linear histogram of Normal samples.
    # TODO: Step 5 — Top-right: Linear histogram of Power Law samples.
    #       Note: the x-axis may extend very far due to heavy tail.
    # TODO: Step 6 — Bottom-left: Log-log plot of Normal samples.
    #       Hint: Plot np.log(bin_centers) vs np.log(bin_heights) as scatter.
    # TODO: Step 7 — Bottom-right: Log-log plot of Power Law samples.
    #       Should appear roughly linear (straight line on log-log = power law).
    # TODO: Step 8 — Add titles, labels, plt.tight_layout(), plt.show().
    # HINT: For log-log histograms, use log-spaced bins:
    #       bins = np.logspace(np.log10(data.min()+1e-9), np.log10(data.max()), 50)
    pass


# ---------------------------------------------------------------------------
# 6. Word Frequency Zipf
# ---------------------------------------------------------------------------

def word_frequency_zipf(text: str) -> dict:
    """
    Analyze word frequencies in a text and show Zipf's Law.

    Zipf's Law predicts: frequency of word at rank r ∝ 1/r.
    On a log-log plot of (rank, frequency), this appears as a straight line
    with slope ≈ -1.

    Args:
        text: Raw text string (can be a book excerpt, article, etc.).

    Returns:
        Dictionary with keys:
            'word_counts': Counter of {word: count}
            'ranks': numpy array of ranks [1, 2, 3, ..., vocabulary_size]
            'frequencies': numpy array of frequency counts, sorted descending
            'top_20': list of (word, count) tuples for the 20 most common words

    Example:
        >>> result = word_frequency_zipf("To be or not to be that is the question")
        >>> result['top_20'][:3]  # ('to', 2), ('be', 2), etc.
    """
    # TODO: Step 1 — Preprocess text:
    #       - Convert to lowercase.
    #       - Remove punctuation: re.sub(r'[^a-z\s]', '', text)
    #       - Split into words: text.split()
    # TODO: Step 2 — Count word frequencies using Counter(words).
    # TODO: Step 3 — Sort by frequency (descending) to get rank ordering.
    # TODO: Step 4 — Create ranks array: np.arange(1, len(word_counts) + 1)
    # TODO: Step 5 — Create frequencies array from the sorted counts.
    # TODO: Step 6 — Return the results dictionary.
    pass


# ---------------------------------------------------------------------------
# 7. Plot Log-Log (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_log_log(data: np.ndarray, title: str = "Log-Log Plot",
                 color: str = 'steelblue', fit_line: bool = True):
    """
    Create a log-log scale scatter plot of data values vs their frequency.

    Used to visually identify power-law distributions: they appear as a
    straight line on log-log axes.

    Args:
        data: 1D array of positive values to analyze.
        title: Plot title.
        color: Color for scatter points.
        fit_line: If True, overlay a fitted power law line.
    """
    # Compute empirical PDF using histogram with log-spaced bins
    data = data[data > 0]  # remove non-positive values
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)
    counts, edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Keep only non-zero counts
    mask = counts > 0
    x_vals = bin_centers[mask]
    y_vals = counts[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=25, zorder=3,
               label='Empirical data')

    if fit_line and len(x_vals) >= 3:
        # Fit a line in log-log space
        log_x = np.log(x_vals)
        log_y = np.log(y_vals)
        coeffs = np.polyfit(log_x, log_y, 1)
        slope = coeffs[0]
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_fit = np.exp(coeffs[1]) * x_fit ** slope
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8,
                label=f'Power law fit (slope={slope:.2f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Value (log scale)', fontsize=12)
    ax.set_ylabel('Density (log scale)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=100,
                bbox_inches='tight')
    plt.show()
    print(f"Log-log plot saved.")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate long-tail and power law distribution concepts.

    Sections:
        1. Power Law PDF — compute and plot
        2. Sampling from Power Law — verify MLE recovery
        3. Zipf distribution — compute theoretical frequencies
        4. Word frequency analysis — Zipf's Law on real text
        5. Normal vs Long-tail comparison — linear and log-log
    """
    print("=" * 60)
    print("Exercise 03: Long-Tail & Power Law Distributions")
    print("=" * 60)

    np.random.seed(42)

    # --- Section 1: Power Law PDF ---
    print("\n--- Section 1: Power Law PDF ---")
    # TODO: Compute power_law_pdf at x = [1, 2, 5, 10, 100] for alpha=2.5, x_min=1.
    # TODO: Plot the PDF on linear and log-log scale.

    # --- Section 2: Sampling and MLE ---
    print("\n--- Section 2: Sampling from Power Law & MLE ---")
    # TODO: Generate 10,000 samples from Power Law(alpha=2.5, x_min=1.0).
    # TODO: Call fit_power_law_exponent(samples, x_min=1.0).
    # TODO: Print: true alpha=2.5, estimated alpha, std error.
    # TODO: Plot samples using plot_log_log(samples, "Power Law Samples").

    # --- Section 3: Zipf Distribution ---
    print("\n--- Section 3: Zipf Distribution ---")
    # TODO: Compute zipf_distribution(n_items=100, s=1.0).
    # TODO: Plot rank vs frequency on log-log scale.
    # TODO: Compare s=0.5, s=1.0, s=2.0 on the same plot.

    # --- Section 4: Word Frequency Analysis ---
    print("\n--- Section 4: Word Frequency — Zipf's Law in Real Text ---")
    # Use a long English text sample to demonstrate Zipf's Law
    sample_text = """
    To be or not to be that is the question whether tis nobler in the mind to suffer
    the slings and arrows of outrageous fortune or to take arms against a sea of troubles
    and by opposing end them to die to sleep no more and by a sleep to say we end the
    heartache and the thousand natural shocks that flesh is heir to tis a consummation
    devoutly to be wished to die to sleep to sleep perchance to dream ay there the rub
    for in that sleep of death what dreams may come when we have shuffled off this mortal
    coil must give us pause there the respect that makes calamity of so long life for who
    would bear the whips and scorns of time the oppressors wrong the proud mans contumely
    the pangs of despised love the laws delay the insolence of office and the spurns that
    patient merit of the unworthy takes when he himself might his quietus make with a bare
    bodkin who would fardels bear to grunt and sweat under a weary life but that the dread
    of something after death the undiscovered country from whose bourn no traveler returns
    puzzles the will and makes us rather bear those ills we have than fly to others that we
    know not of thus conscience does make cowards of us all and thus the native hue of
    resolution is sicklied over with the pale cast of thought and enterprises of great pith
    and moment with this regard their currents turn awry and lose the name of action the
    machine learning model is trained on data to make predictions the neural network learns
    features from input data through multiple layers of transformations the optimization
    algorithm minimizes the loss function by updating the model parameters the training
    process involves many iterations over the dataset the model generalizes well when the
    training error and validation error are both low the bias variance tradeoff describes
    the fundamental tension between underfitting and overfitting in machine learning models
    """

    # TODO: Call word_frequency_zipf(sample_text).
    # TODO: Print the top 20 most frequent words.
    # TODO: Plot rank vs frequency on log-log scale.
    # TODO: Fit a power law to the word frequencies and print the exponent.
    # TODO: Comment on whether Zipf's Law holds (slope ≈ -1 on log-log).

    # --- Section 5: Normal vs Long-Tail ---
    print("\n--- Section 5: Normal vs Long-Tail Comparison ---")
    # TODO: Call compare_normal_vs_longtail(n_samples=50000).
    # TODO: Print key statistics for both:
    #       - Mean, std, max value
    #       - Fraction of values > 10 (should be tiny for Normal, larger for power law)

    print("\nDone! Key insight: power-law distributions have MUCH heavier tails than Normal.")
    print("In ML: word frequencies, social network degrees, and file sizes follow power laws.")
    print("Ignoring this leads to algorithms that fail on rare but important events.")


if __name__ == "__main__":
    main()
