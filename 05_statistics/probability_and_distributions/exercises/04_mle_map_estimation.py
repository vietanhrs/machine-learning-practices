"""
Exercise 04: Maximum Likelihood & MAP Estimation
=================================================
Topic: MLE and MAP (Ước lượng khả năng cực đại và ước lượng MAP)
Difficulty: Intermediate → Advanced

Learning Objectives:
    - Implement log-likelihood functions for Bernoulli and Gaussian distributions
    - Derive MLE estimates analytically and verify numerically
    - Implement MAP estimation with a Beta prior for Bernoulli
    - Understand why MAP = MLE + regularization
    - Observe how MLE converges to the true parameter as n increases
    - Visualize likelihood surfaces, priors, and posteriors

Key Formulas:
    Bernoulli log-likelihood: sum(k*log(p) + (1-k)*log(1-p))
    MLE for Bernoulli: p_hat = n_success / n_total
    MLE for Gaussian: mu_hat = mean(x), sigma_hat^2 = mean((x - mu_hat)^2)
    MAP for Bernoulli (Beta prior): p_hat_MAP = (n_success + alpha - 1) / (n_total + alpha + beta - 2)
    Beta posterior mean (Bayes estimate): (n_success + alpha) / (n_total + alpha + beta)

Instructions:
    Implement all TODO sections. The plot_* functions are fully provided.
    Pay attention to numerical stability when computing log-likelihoods.

Dependencies:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# 1. Bernoulli Log-Likelihood
# ---------------------------------------------------------------------------

def bernoulli_log_likelihood(p: float, n_success: int, n_total: int) -> float:
    """
    Compute the log-likelihood of observing n_success successes in n_total
    Bernoulli trials with success probability p.

    Formula:
        log L(p) = n_success * log(p) + (n_total - n_success) * log(1 - p)

    Note: The binomial coefficient C(n, k) is constant with respect to p and
    can be omitted when maximizing over p.

    Args:
        p: Success probability, in (0, 1). Must not be 0 or 1 to avoid log(0).
        n_success: Number of observed successes (0 <= n_success <= n_total).
        n_total: Total number of trials.

    Returns:
        Log-likelihood value (a non-positive float when p is in (0,1)).

    Raises:
        ValueError: If p <= 0 or p >= 1.
        ValueError: If n_success < 0 or n_success > n_total.

    Example:
        >>> bernoulli_log_likelihood(p=0.5, n_success=7, n_total=10)
        # log(0.5^7 * 0.5^3) = 10 * log(0.5) ≈ -6.931
        >>> bernoulli_log_likelihood(p=0.7, n_success=7, n_total=10)
        # 7*log(0.7) + 3*log(0.3) ≈ -6.108  (higher = better fit)
    """
    # TODO: Step 1 — Validate p in (0, 1), n_success >= 0, n_success <= n_total.
    # TODO: Step 2 — Compute n_failure = n_total - n_success.
    # TODO: Step 3 — Return n_success * np.log(p) + n_failure * np.log(1 - p).
    # HINT: Be careful about numerical stability — log(0) is -infinity.
    pass


# ---------------------------------------------------------------------------
# 2. MLE for Bernoulli
# ---------------------------------------------------------------------------

def mle_bernoulli(n_success: int, n_total: int) -> float:
    """
    Compute the Maximum Likelihood Estimate for the Bernoulli success probability p.

    The MLE is found by taking the derivative of the log-likelihood with respect to p,
    setting it to zero, and solving:
        d/dp [k*log(p) + (n-k)*log(1-p)] = k/p - (n-k)/(1-p) = 0
        => p_hat = k / n

    Args:
        n_success: Number of observed successes.
        n_total: Total number of trials.

    Returns:
        MLE estimate p_hat = n_success / n_total.

    Raises:
        ValueError: If n_total <= 0.

    Example:
        >>> mle_bernoulli(n_success=7, n_total=10)
        0.7
        >>> mle_bernoulli(n_success=0, n_total=10)
        0.0  # MLE is 0, but MAP would not be (prior regularizes)
    """
    # TODO: Validate n_total > 0.
    # TODO: Return n_success / n_total.
    pass


# ---------------------------------------------------------------------------
# 3. Gaussian Log-Likelihood
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(mu: float, sigma: float, data: np.ndarray) -> float:
    """
    Compute the log-likelihood of data under a Gaussian distribution N(mu, sigma^2).

    Formula:
        log L(mu, sigma) = -n/2 * log(2*pi) - n*log(sigma)
                           - 1/(2*sigma^2) * sum((x_i - mu)^2)

    Args:
        mu: Mean of the Gaussian.
        sigma: Standard deviation (must be > 0).
        data: 1D array of observed values.

    Returns:
        Log-likelihood value (scalar float, non-positive).

    Raises:
        ValueError: If sigma <= 0.

    Example:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> gaussian_log_likelihood(mu=3.0, sigma=1.0, data=data)
        # Should be higher than gaussian_log_likelihood(mu=0.0, sigma=1.0, data=data)
    """
    # TODO: Step 1 — Validate sigma > 0.
    # TODO: Step 2 — Compute n = len(data).
    # TODO: Step 3 — Compute the three terms:
    #       term1 = -n/2 * np.log(2 * np.pi)
    #       term2 = -n * np.log(sigma)
    #       term3 = -1/(2*sigma^2) * np.sum((data - mu)**2)
    # TODO: Step 4 — Return term1 + term2 + term3.
    # HINT: Verify with scipy_stats.norm.logpdf(data, loc=mu, scale=sigma).sum()
    pass


# ---------------------------------------------------------------------------
# 4. MLE for Gaussian
# ---------------------------------------------------------------------------

def mle_gaussian(data: np.ndarray) -> tuple[float, float]:
    """
    Compute the Maximum Likelihood Estimates for mu and sigma of a Gaussian.

    Analytical MLE (derived by setting gradient of log-likelihood to zero):
        mu_hat    = (1/n) * sum(x_i)                    [sample mean]
        sigma_hat = sqrt((1/n) * sum((x_i - mu_hat)^2)) [biased std, divide by n!]

    Note: The MLE for sigma uses n (not n-1). This is BIASED (underestimates sigma).
    In practice, we often use n-1 for an unbiased estimate, but MLE strictly gives n.

    Args:
        data: 1D array of observed values.

    Returns:
        Tuple (mu_hat, sigma_hat).

    Example:
        >>> data = np.random.normal(loc=5.0, scale=2.0, size=10000)
        >>> mu, sigma = mle_gaussian(data)
        >>> abs(mu - 5.0) < 0.05    # True
        >>> abs(sigma - 2.0) < 0.05  # True
    """
    # TODO: Step 1 — Compute mu_hat = np.mean(data).
    # TODO: Step 2 — Compute sigma_hat = np.sqrt(np.mean((data - mu_hat)**2)).
    #       NOTE: Use ddof=0 (divide by n, not n-1) — this is the true MLE.
    # TODO: Step 3 — Return (mu_hat, sigma_hat).
    pass


# ---------------------------------------------------------------------------
# 5. MAP for Bernoulli (Beta Prior)
# ---------------------------------------------------------------------------

def map_bernoulli(n_success: int, n_total: int,
                  alpha_prior: float, beta_prior: float) -> dict:
    """
    Compute the MAP estimate for Bernoulli probability p with a Beta prior.

    The Beta distribution Beta(alpha, beta) is the conjugate prior for
    the Bernoulli/Binomial likelihood. After observing n_success out of n_total:

        Posterior: Beta(alpha + n_success, beta + n_failure)

    MAP estimate (mode of the posterior, for alpha > 1 and beta > 1):
        p_hat_MAP = (n_success + alpha - 1) / (n_total + alpha + beta - 2)

    Bayesian posterior mean (Bayes estimate, minimizes expected squared error):
        p_hat_mean = (n_success + alpha) / (n_total + alpha + beta)

    Note: With Beta(1, 1) (uniform prior), MAP = MLE.
    With Alpha = Beta = 2 (adding pseudo-count of 1 success and 1 failure), MAP is regularized.

    Args:
        n_success: Number of observed successes.
        n_total: Total number of Bernoulli trials.
        alpha_prior: Alpha parameter of the Beta prior (pseudo-successes).
        beta_prior: Beta parameter of the Beta prior (pseudo-failures).

    Returns:
        Dictionary with keys:
            'mle': p_hat = n_success / n_total (no prior)
            'map': MAP estimate (mode of posterior)
            'posterior_mean': mean of Beta posterior
            'posterior_alpha': alpha of posterior = alpha_prior + n_success
            'posterior_beta': beta of posterior = beta_prior + n_failure

    Example:
        >>> # 0 heads in 5 flips, uniform prior
        >>> result = map_bernoulli(0, 5, alpha_prior=1, beta_prior=1)
        >>> result['mle']   # 0.0 — extreme, problematic
        >>> result['map']   # 0.0 (still, with uniform prior, MAP = MLE)

        >>> # 0 heads in 5 flips, informative prior (expect fair coin)
        >>> result = map_bernoulli(0, 5, alpha_prior=10, beta_prior=10)
        >>> result['map']   # ~0.39 — regularized toward 0.5 by the prior
    """
    # TODO: Step 1 — Compute n_failure = n_total - n_success.
    # TODO: Step 2 — Compute MLE = mle_bernoulli(n_success, n_total).
    # TODO: Step 3 — Compute posterior parameters:
    #       posterior_alpha = alpha_prior + n_success
    #       posterior_beta = beta_prior + n_failure
    # TODO: Step 4 — Compute MAP = (n_success + alpha_prior - 1) / (n_total + alpha_prior + beta_prior - 2)
    #       Handle edge case: if alpha_prior <= 1 or beta_prior <= 1, MAP may be at boundary.
    # TODO: Step 5 — Compute posterior_mean = (n_success + alpha_prior) / (n_total + alpha_prior + beta_prior)
    # TODO: Step 6 — Return results dictionary.
    pass


# ---------------------------------------------------------------------------
# 6. Plot Likelihood Surface (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_likelihood_surface(n_success: int, n_total: int):
    """
    Plot the Bernoulli log-likelihood curve as a function of p,
    with the MLE estimate marked.

    Args:
        n_success: Number of successes observed.
        n_total: Total number of trials.
    """
    p_values = np.linspace(0.001, 0.999, 500)
    log_likelihoods = [bernoulli_log_likelihood(p, n_success, n_total) for p in p_values]

    p_mle = mle_bernoulli(n_success, n_total)
    ll_mle = bernoulli_log_likelihood(max(min(p_mle, 0.999), 0.001), n_success, n_total)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p_values, log_likelihoods, 'steelblue', linewidth=2.5,
            label='Log-Likelihood')
    ax.axvline(x=p_mle, color='red', linestyle='--', linewidth=2,
               label=f'MLE: p̂ = {p_mle:.3f}')
    ax.scatter([p_mle], [ll_mle], color='red', s=100, zorder=5)

    ax.set_xlabel('p (success probability)', fontsize=12)
    ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.set_title(
        f'Bernoulli Log-Likelihood: {n_success} successes in {n_total} trials',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('likelihood_surface.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Plot saved to: likelihood_surface.png")


# ---------------------------------------------------------------------------
# 7. Plot Prior, Likelihood, Posterior (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_prior_likelihood_posterior(n_success: int, n_total: int,
                                     alpha: float, beta: float):
    """
    Visualize the three components of Bayes' theorem for the Bernoulli model:
        - Prior: Beta(alpha, beta)
        - Likelihood: Bernoulli likelihood (normalized for plotting)
        - Posterior: Beta(alpha + k, beta + n - k)

    Args:
        n_success: Observed number of successes.
        n_total: Total number of trials.
        alpha: Alpha parameter of the Beta prior.
        beta: Beta parameter of the Beta prior.
    """
    p_values = np.linspace(0.001, 0.999, 500)
    n_failure = n_total - n_success

    # Prior: Beta(alpha, beta)
    prior_pdf = scipy_stats.beta.pdf(p_values, alpha, beta)

    # Likelihood: proportional to p^k * (1-p)^(n-k)
    log_likelihood = n_success * np.log(p_values) + n_failure * np.log(1 - p_values)
    likelihood = np.exp(log_likelihood - log_likelihood.max())  # normalize for plotting

    # Posterior: Beta(alpha + k, beta + n - k)
    post_alpha = alpha + n_success
    post_beta = beta + n_failure
    posterior_pdf = scipy_stats.beta.pdf(p_values, post_alpha, post_beta)

    # MAP and posterior mean
    if post_alpha > 1 and post_beta > 1:
        p_map = (post_alpha - 1) / (post_alpha + post_beta - 2)
    else:
        p_map = None
    p_post_mean = post_alpha / (post_alpha + post_beta)
    p_mle = n_success / n_total if n_total > 0 else 0.5

    fig, ax = plt.subplots(figsize=(10, 5))

    # Normalize prior for visual comparison
    max_val = max(prior_pdf.max(), posterior_pdf.max(), likelihood.max() * posterior_pdf.max())
    ax.fill_between(p_values, prior_pdf / max_val, alpha=0.2, color='blue')
    ax.plot(p_values, prior_pdf / max_val, 'b-', linewidth=2,
            label=f'Prior: Beta({alpha:.0f}, {beta:.0f})')

    ax.fill_between(p_values, likelihood, alpha=0.2, color='green')
    ax.plot(p_values, likelihood, 'g--', linewidth=2,
            label=f'Likelihood (normalized): {n_success} of {n_total}')

    ax.fill_between(p_values, posterior_pdf / max_val, alpha=0.2, color='red')
    ax.plot(p_values, posterior_pdf / max_val, 'r-', linewidth=2.5,
            label=f'Posterior: Beta({post_alpha:.0f}, {post_beta:.0f})')

    ax.axvline(x=p_mle, color='green', linestyle=':', linewidth=1.5,
               label=f'MLE = {p_mle:.3f}')
    ax.axvline(x=p_post_mean, color='red', linestyle='--', linewidth=1.5,
               label=f'Posterior mean = {p_post_mean:.3f}')
    if p_map is not None:
        ax.axvline(x=p_map, color='darkred', linestyle='-.', linewidth=1.5,
                   label=f'MAP = {p_map:.3f}')

    ax.set_xlabel('p (success probability)', fontsize=12)
    ax.set_ylabel('Density (normalized)', fontsize=12)
    ax.set_title(
        f'Bayesian Inference: Prior × Likelihood → Posterior\n'
        f'({n_success} successes, {n_total} trials, Beta({alpha:.0f},{beta:.0f}) prior)',
        fontsize=12
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prior_likelihood_posterior.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Plot saved to: prior_likelihood_posterior.png")


# ---------------------------------------------------------------------------
# 8. Demonstrate MLE Convergence
# ---------------------------------------------------------------------------

def demonstrate_mle_convergence(true_p: float, n_values: list) -> dict:
    """
    Show that the MLE estimate for a Bernoulli parameter converges to the
    true value as the number of observations n increases.

    For each n in n_values:
        1. Simulate n coin flips with P(heads) = true_p.
        2. Compute the MLE estimate.
        3. Track how the estimate evolves.

    Args:
        true_p: True success probability (e.g., 0.7 for a biased coin).
        n_values: List of sample sizes to try (e.g., [1, 5, 10, 50, 100, 1000]).

    Returns:
        Dictionary with keys:
            'n_values': the input list
            'mle_estimates': list of MLE estimates for each n
            'errors': list of |mle - true_p| for each n

    Example:
        >>> result = demonstrate_mle_convergence(true_p=0.7, n_values=[1, 10, 100, 1000])
        >>> result['mle_estimates']  # should converge toward 0.7
    """
    # TODO: Step 1 — For each n in n_values:
    #       a. Simulate n flips: samples = np.random.binomial(1, true_p, n)
    #       b. Count successes: n_success = samples.sum()
    #       c. Compute MLE: mle_bernoulli(n_success, n)
    #       d. Compute error: |mle - true_p|
    # TODO: Step 2 — Return results dictionary.
    # HINT: Set a random seed before calling this function for reproducibility.
    pass


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate MLE and MAP estimation for Bernoulli and Gaussian models.

    Sections:
        1. Bernoulli log-likelihood — compute for several p values
        2. MLE for Bernoulli — coin flip example
        3. Gaussian log-likelihood — fit to synthetic data
        4. MLE for Gaussian — verify analytical formulas
        5. MAP for Bernoulli — Beta prior, effect of prior strength
        6. Prior vs likelihood vs posterior visualization
        7. MLE convergence — law of large numbers
    """
    print("=" * 60)
    print("Exercise 04: MLE and MAP Estimation")
    print("=" * 60)

    np.random.seed(42)

    # --- Section 1: Bernoulli Log-Likelihood ---
    print("\n--- Section 1: Bernoulli Log-Likelihood ---")
    # TODO: Compute bernoulli_log_likelihood(p, n_success=7, n_total=10) for
    #       p in [0.3, 0.5, 0.7, 0.9].
    # TODO: Print results and observe which p maximizes the log-likelihood.
    # Expected: p=0.7 should have the highest log-likelihood (since 7/10 = 0.7)

    # --- Section 2: MLE for Bernoulli --- Coin Flip ---
    print("\n--- Section 2: MLE for Bernoulli (Coin Flip) ---")
    # TODO: Simulate 100 flips of a biased coin (true p = 0.65).
    # TODO: Count successes and compute mle_bernoulli().
    # TODO: Print: true p, MLE estimate, error.
    # TODO: Call plot_likelihood_surface(n_success, n_total).

    # --- Section 3: Gaussian Log-Likelihood ---
    print("\n--- Section 3: Gaussian Log-Likelihood ---")
    # TODO: Generate 500 samples from N(mu=3.0, sigma=1.5).
    # TODO: Compute gaussian_log_likelihood for (mu, sigma) in:
    #       [(3.0, 1.5), (0.0, 1.5), (3.0, 5.0), (3.0, 0.5)]
    # TODO: Print results and observe which parameters give highest log-likelihood.

    # --- Section 4: MLE for Gaussian ---
    print("\n--- Section 4: MLE for Gaussian ---")
    # TODO: Call mle_gaussian() on the data from Section 3.
    # TODO: Print: true (mu, sigma), MLE estimates.
    # TODO: Verify that the MLE mu matches sample mean, MLE sigma matches
    #       np.std(data, ddof=0) (biased std).

    # --- Section 5: MAP for Bernoulli --- Effect of Prior ---
    print("\n--- Section 5: MAP Estimation with Beta Prior ---")
    # TODO: Scenario: Observe 0 successes in 5 trials. MLE says p=0 (problematic!).
    # TODO: Compute MAP with different priors:
    #       - Uniform prior: Beta(1, 1) → MAP should equal MLE = 0
    #       - Weak informative: Beta(2, 2) → MAP should pull toward 0.5
    #       - Strong informative: Beta(10, 10) → MAP should be close to 0.5
    # TODO: Print a table comparing MLE, MAP, and posterior mean for each prior.
    # TODO: Discuss: "When is MAP preferable to MLE?"

    # --- Section 6: Prior-Likelihood-Posterior Visualization ---
    print("\n--- Section 6: Prior × Likelihood → Posterior ---")
    # TODO: Call plot_prior_likelihood_posterior(n_success=7, n_total=10, alpha=2, beta=2)
    # TODO: Call plot_prior_likelihood_posterior(n_success=1, n_total=2, alpha=10, beta=10)
    # TODO: Observe: strong prior dominates when data is sparse;
    #       data dominates when n is large.

    # --- Section 7: MLE Convergence ---
    print("\n--- Section 7: MLE Convergence (Law of Large Numbers) ---")
    # TODO: Call demonstrate_mle_convergence(
    #           true_p=0.7,
    #           n_values=[1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000]
    #       )
    # TODO: Print a table: n | MLE estimate | |error|
    # TODO: Plot n_values (x-axis, log scale) vs MLE estimate (y-axis).
    #       Add horizontal line at true_p=0.7.
    # TODO: Observe: error shrinks as n grows (law of large numbers in action).

    print("\nDone! Key insights:")
    print("  - MLE maximizes P(data | theta) — purely data-driven.")
    print("  - MAP maximizes P(theta | data) = P(data | theta) * P(theta) — adds prior.")
    print("  - With enough data, MAP ≈ MLE (data overwhelms the prior).")
    print("  - With little data, MAP is more stable due to prior regularization.")
    print("  - Gaussian prior → L2 regularization; Laplace prior → L1 regularization.")


if __name__ == "__main__":
    main()
