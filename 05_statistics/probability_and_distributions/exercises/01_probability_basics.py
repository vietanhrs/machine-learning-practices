"""
Exercise 01: Probability Basics
================================
Topic: Conditional Probability, Bayes' Theorem, Law of Total Probability
Difficulty: Beginner → Intermediate

Learning Objectives:
    - Apply Bayes' theorem analytically
    - Understand why prevalence (prior) matters in medical testing
    - Estimate conditional probabilities via Monte Carlo simulation
    - Apply the Law of Total Probability
    - Explore the Monty Hall problem as a Bayesian puzzle

Instructions:
    Each function has a TODO block. Implement the body using the hints.
    Run main() and verify your numerical results match the analytical answers.

Dependencies:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Bayes' Theorem
# ---------------------------------------------------------------------------

def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """
    Compute the posterior probability using Bayes' Theorem.

    Formula:
        P(H | D) = P(D | H) * P(H) / P(D)

    Args:
        prior: P(H) — prior probability of hypothesis H.
        likelihood: P(D | H) — probability of data D given hypothesis H.
        evidence: P(D) — total probability of data D (normalizing constant).

    Returns:
        Posterior probability P(H | D) as a float.

    Raises:
        ValueError: If evidence is 0 (division by zero).

    Example:
        >>> bayes_theorem(prior=0.01, likelihood=0.99, evidence=0.1089)
        ~0.0909
    """
    # TODO: Validate that evidence != 0. Raise ValueError if so.
    # TODO: Return prior * likelihood / evidence.
    pass


# ---------------------------------------------------------------------------
# 2. Medical Test Example
# ---------------------------------------------------------------------------

def medical_test_example(sensitivity: float, specificity: float,
                          prevalence: float) -> dict:
    """
    Compute the probability of truly having a disease given a positive test
    result, using Bayes' theorem. Demonstrates the base rate fallacy.

    Definitions:
        - Sensitivity (True Positive Rate): P(positive test | disease)
        - Specificity (True Negative Rate): P(negative test | no disease)
        - Prevalence: P(disease) — prior probability of disease in the population
        - False Positive Rate: 1 - specificity = P(positive | no disease)

    Formula:
        P(disease | positive) = P(positive | disease) * P(disease) / P(positive)
        P(positive) = P(positive | disease) * P(disease)
                    + P(positive | no disease) * P(no disease)

    Args:
        sensitivity: P(positive test | disease), in [0, 1].
        specificity: P(negative test | no disease), in [0, 1].
        prevalence: P(disease), in [0, 1].

    Returns:
        Dictionary with keys:
            'prior': prevalence
            'false_positive_rate': 1 - specificity
            'p_positive': P(positive test), total probability of a positive result
            'ppv': Positive Predictive Value = P(disease | positive test)
            'npv': Negative Predictive Value = P(no disease | negative test)

    Example:
        >>> result = medical_test_example(0.99, 0.95, 0.01)
        >>> print(result['ppv'])  # ~0.167 — only 16.7% of positives are true positives!
    """
    # TODO: Step 1 — Compute false_positive_rate = 1 - specificity.
    # TODO: Step 2 — Compute P(positive) using the Law of Total Probability:
    #       p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)
    # TODO: Step 3 — Compute PPV = P(disease | positive) via bayes_theorem().
    # TODO: Step 4 — Compute NPV = P(no disease | negative test):
    #       false_negative_rate = 1 - sensitivity
    #       p_negative = 1 - p_positive
    #       NPV = specificity * (1 - prevalence) / p_negative
    # TODO: Step 5 — Return the results dictionary.
    # HINT: Try several prevalence values (0.001, 0.01, 0.1, 0.5) and observe
    #       how PPV changes drastically even with the same test quality.
    pass


# ---------------------------------------------------------------------------
# 3. Simulate Conditional Probability (Monte Carlo)
# ---------------------------------------------------------------------------

def simulate_conditional_probability(n_simulations: int = 100_000) -> dict:
    """
    Estimate P(A | B) via Monte Carlo simulation and compare to the analytical result.

    Scenario:
        Roll a fair six-sided die.
        A = {the result is 6}
        B = {the result is even} = {2, 4, 6}

    Analytical result:
        P(A | B) = P(A ∩ B) / P(B) = (1/6) / (3/6) = 1/3 ≈ 0.3333

    Args:
        n_simulations: Number of die roll simulations.

    Returns:
        Dictionary with keys:
            'analytical': exact P(A|B) = 1/3
            'simulated': estimated P(A|B) from simulation
            'error': abs(simulated - analytical)

    Example:
        >>> result = simulate_conditional_probability(100000)
        >>> print(result['simulated'])  # should be close to 0.3333
    """
    # TODO: Step 1 — Simulate n_simulations rolls: np.random.randint(1, 7, n_simulations).
    # TODO: Step 2 — Define event A: rolls == 6.
    # TODO: Step 3 — Define event B: rolls % 2 == 0 (even numbers).
    # TODO: Step 4 — Compute P(A|B) = count(A and B) / count(B).
    # TODO: Step 5 — Compute analytical = 1/3.
    # TODO: Step 6 — Return dictionary with results.
    pass


# ---------------------------------------------------------------------------
# 4. Law of Total Probability
# ---------------------------------------------------------------------------

def law_of_total_probability(priors: list, likelihoods: list) -> float:
    """
    Compute P(B) using the Law of Total Probability.

    Formula:
        P(B) = sum over i of P(B | A_i) * P(A_i)

    where {A_1, A_2, ..., A_n} is a partition of the sample space
    (mutually exclusive and exhaustive events).

    Args:
        priors: List of P(A_i) values — must sum to 1.
        likelihoods: List of P(B | A_i) values — same length as priors.

    Returns:
        P(B) as a float.

    Raises:
        ValueError: If priors and likelihoods have different lengths.
        ValueError: If priors do not sum to approximately 1.

    Example:
        >>> # Two factories: Factory 1 (60% of output, 2% defect rate)
        >>> #                Factory 2 (40% of output, 5% defect rate)
        >>> law_of_total_probability([0.6, 0.4], [0.02, 0.05])
        0.032  # total defect probability
    """
    # TODO: Step 1 — Convert priors and likelihoods to numpy arrays.
    # TODO: Step 2 — Validate: same length, priors sum to ~1 (use np.isclose).
    # TODO: Step 3 — Return np.dot(priors, likelihoods) = sum(P(A_i) * P(B|A_i)).
    pass


# ---------------------------------------------------------------------------
# 5. Plot Bayes Update (PROVIDED — do not modify)
# ---------------------------------------------------------------------------

def plot_bayes_update(prior_values: np.ndarray, posterior_values: np.ndarray,
                      observations: list):
    """
    Visualize how the posterior probability updates as we observe more data.

    For a sequence of observations (e.g., coin flips), show how the posterior
    probability of the hypothesis (e.g., P(fair coin)) evolves.

    This function is fully implemented. Study how the posterior shifts
    toward the truth as more evidence accumulates.

    Args:
        prior_values: Array of p values (x-axis, e.g., np.linspace(0, 1, 200)).
        posterior_values: 2D array (n_observations + 1, len(prior_values)).
            Row 0 = prior, row i = posterior after i-th observation.
        observations: List of observation labels (e.g., ['H', 'T', 'H', ...]).
    """
    n_steps = len(posterior_values)
    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: evolution of posterior ---
    ax = axes[0]
    for i, (posterior, color) in enumerate(zip(posterior_values, colors)):
        label = 'Prior' if i == 0 else f'After obs {i}: {observations[i-1]}'
        alpha = 0.4 if i < n_steps - 1 else 1.0
        lw = 1.5 if i < n_steps - 1 else 2.5
        ax.plot(prior_values, posterior, color=color, alpha=alpha,
                linewidth=lw, label=label)

    ax.set_xlabel('Parameter p (probability of Heads)', fontsize=12)
    ax.set_ylabel('Posterior density', fontsize=12)
    ax.set_title('Bayesian Updating: Prior → Posterior', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Right: MAP estimate over time ---
    ax = axes[1]
    map_estimates = [prior_values[np.argmax(post)] for post in posterior_values]
    ax.plot(range(n_steps), map_estimates, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='True p=0.5')
    ax.set_xlabel('Number of observations', fontsize=12)
    ax.set_ylabel('MAP estimate of p', fontsize=12)
    ax.set_title('MAP Estimate Convergence', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('bayes_update.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Plot saved to: bayes_update.png")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate Bayes' theorem, conditional probability, and probability puzzles.

    Sections:
        1. Bayes' theorem — basic computation
        2. Medical test example — base rate fallacy
        3. Monte Carlo conditional probability
        4. Law of total probability — factory defect example
        5. Bayesian updating visualization
        6. PUZZLE: Monty Hall problem (student must solve)
    """
    print("=" * 60)
    print("Exercise 01: Probability Basics")
    print("=" * 60)

    # --- Section 1: Bayes' Theorem ---
    print("\n--- Section 1: Bayes' Theorem ---")
    # TODO: Compute P(disease | positive) for a simple example:
    #   P(disease) = 0.01, P(positive | disease) = 0.99,
    #   P(positive) = 0.0099 + 0.0495 = 0.0594
    # TODO: Call bayes_theorem() and print the result.
    # TODO: Verify manually.

    # --- Section 2: Medical Test Example ---
    print("\n--- Section 2: Medical Test — Base Rate Fallacy ---")
    # TODO: Call medical_test_example(sensitivity=0.99, specificity=0.95, prevalence=0.01)
    # TODO: Print all returned values in a readable format.
    # TODO: Repeat for prevalence = 0.001, 0.1, 0.5 and show how PPV changes.
    # TODO: Print interpretation: "Even with a 99% sensitive test, only X% of positives
    #       are true positives when prevalence is 1%."

    # --- Section 3: Monte Carlo Simulation ---
    print("\n--- Section 3: Monte Carlo Conditional Probability ---")
    # TODO: Call simulate_conditional_probability(n_simulations=500_000).
    # TODO: Print analytical and simulated results, and the error.

    # --- Section 4: Law of Total Probability ---
    print("\n--- Section 4: Law of Total Probability ---")
    # TODO: Compute the overall defect rate for:
    #   Factory 1: 60% of output, 2% defect rate
    #   Factory 2: 40% of output, 5% defect rate
    # TODO: Call law_of_total_probability([0.6, 0.4], [0.02, 0.05])
    # TODO: Then use Bayes' theorem to answer:
    #   "Given a defective item, what is the probability it came from Factory 2?"

    # --- Section 5: Bayesian Updating Visualization ---
    print("\n--- Section 5: Bayesian Updating ---")
    # TODO: Simulate 10 fair coin flips.
    # TODO: Start with a Beta(1, 1) = Uniform prior over p in [0, 1].
    # TODO: After each flip, update to Beta(alpha + heads, beta + tails) posterior.
    #   Hint: from scipy.stats import beta; beta.pdf(p_values, alpha, beta_param)
    # TODO: Collect posterior after each observation and call plot_bayes_update().

    # --- Section 6: PUZZLE — Monty Hall Problem ---
    print("\n--- Section 6: PUZZLE — Monty Hall Problem ---")
    print("""
    The Monty Hall Problem:
    -----------------------
    You are on a game show. There are 3 doors. Behind one is a car (prize);
    behind the other two are goats. You pick door #1. The host (who knows
    where the car is) opens door #3, revealing a goat. He offers you the
    chance to switch to door #2.

    Question: Should you switch? Does it matter? What is P(win | switch)?

    TODO: Solve this analytically using Bayes' theorem:
        - Define events: C_i = car is behind door i, for i = 1, 2, 3.
        - Prior: P(C_1) = P(C_2) = P(C_3) = 1/3
        - Host opens door 3: P(host opens 3 | C_1) = 1/2, P(host opens 3 | C_2) = 1
        - Compute P(C_2 | host opens door 3) using Bayes' theorem.

    TODO: Verify by simulation:
        - Simulate 100,000 games.
        - In each: randomly place car, player picks door 1, host opens a goat door.
        - Compare win rates for "always switch" vs "always stay".
    """)
    # TODO: Implement the Monty Hall simulation and print results.
    # TODO: Print the Bayesian derivation results alongside simulation results.


if __name__ == "__main__":
    main()
