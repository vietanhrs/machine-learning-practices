# Probability & Distributions (Xác suất & Phân phối)

Probability is the mathematical language of uncertainty. Every machine learning model —
from a simple logistic regression to a deep neural network — rests on probabilistic assumptions.
This section covers the foundational concepts and the key distributions you will encounter in ML.

---

## 1. Probability Basics

### Sample Space and Events

- **Sample space** $\Omega$: The set of all possible outcomes of an experiment.
  - Example: Flipping a coin → $\Omega = \{H, T\}$
  - Example: Rolling a die → $\Omega = \{1, 2, 3, 4, 5, 6\}$
- **Event** $A$: A subset of the sample space ($A \subseteq \Omega$).
- **Probability** $P(A)$: A number assigned to each event satisfying the axioms.

### Kolmogorov's Axioms

1. **Non-negativity:** $P(A) \geq 0$ for all events $A$.
2. **Normalization:** $P(\Omega) = 1$ (something must happen).
3. **Additivity:** If $A$ and $B$ are mutually exclusive ($A \cap B = \emptyset$):
   $P(A \cup B) = P(A) + P(B)$

**Derived rules:**
- $P(\emptyset) = 0$
- $P(A^c) = 1 - P(A)$ (complement rule)
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion)

---

## 2. Conditional Probability (Xác suất có điều kiện)

The probability of event $A$ **given** that event $B$ has occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Intuition:** Conditioning on $B$ restricts the sample space to only those outcomes where $B$
is true, then renormalizes.

**Example:** What is the probability of rolling a 6, given the die shows an even number?
- $P(\text{even}) = 3/6 = 1/2$
- $P(6 \cap \text{even}) = 1/6$
- $P(6 \mid \text{even}) = (1/6) / (1/2) = 1/3$

**Independence:** $A$ and $B$ are independent if $P(A \mid B) = P(A)$, equivalently $P(A \cap B) = P(A)P(B)$.

---

## 3. Bayes' Theorem (Định lý Bayes)

Bayes' theorem relates the conditional probability in one direction to the other:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In ML and statistics, we typically write it as:

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})}$$

where:
- $P(\theta)$: **Prior** — our belief about parameter $\theta$ before seeing data.
- $P(\text{data} \mid \theta)$: **Likelihood** — how probable the data is given $\theta$.
- $P(\theta \mid \text{data})$: **Posterior** — updated belief after seeing data.
- $P(\text{data})$: **Evidence** (normalizing constant) — $\int P(\text{data} \mid \theta) P(\theta)\, d\theta$.

### Medical Test Example (Why Prevalence Matters)

Suppose a disease affects 1% of the population:
- Test **sensitivity**: $P(\text{pos} \mid \text{disease}) = 0.99$
- Test **specificity**: $P(\text{neg} \mid \text{no disease}) = 0.95$

What is $P(\text{disease} \mid \text{positive test})$?

$$P(\text{disease} \mid \text{pos}) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} \approx 0.167$$

Despite a 99% sensitive test, only ~17% of positive tests are true positives when prevalence is 1%.
This is the **base rate fallacy** — prior probability matters enormously.

---

## 4. Random Variables (Biến ngẫu nhiên)

A **random variable** $X$ maps outcomes from the sample space to real numbers.

| Type | Description | Example |
|------|-------------|---------|
| **Discrete** | Countable set of values | Number of heads in 10 flips |
| **Continuous** | Uncountably infinite values | Height of a random person |

- **PMF** (Probability Mass Function): For discrete $X$, $p(x) = P(X = x)$.
- **PDF** (Probability Density Function): For continuous $X$, $f(x)$ where $P(a \leq X \leq b) = \int_a^b f(x)\,dx$.
- **CDF** (Cumulative Distribution Function): $F(x) = P(X \leq x)$, valid for both types.

---

## 5. Key Distributions

### a. Bernoulli Distribution

Models a single binary trial (success/failure):

$$P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}$$

- $E[X] = p$
- $\text{Var}(X) = p(1-p)$
- **ML use:** Binary classification output, coin flip, any yes/no event.

### b. Binomial Distribution

Models the number of successes in $n$ independent Bernoulli trials:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

- $E[X] = np$
- $\text{Var}(X) = np(1-p)$
- **Relationship:** Sum of $n$ independent Bernoulli($p$) variables.
- **ML use:** Modeling count data with fixed number of trials.

### c. Uniform Distribution

All values in an interval $[a, b]$ are equally likely:

$$f(x) = \frac{1}{b - a}, \quad x \in [a, b]$$

- $E[X] = (a + b) / 2$
- $\text{Var}(X) = (b - a)^2 / 12$
- **ML use:** Flat/uninformative prior in Bayesian inference, random initialization of weights.

### d. Normal (Gaussian) Distribution (Phân phối chuẩn)

The most important distribution in statistics:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

- Parameters: mean $\mu$ (location) and standard deviation $\sigma$ (scale).
- Symmetric bell curve centered at $\mu$.
- $E[X] = \mu$, $\text{Var}(X) = \sigma^2$.

**68-95-99.7 Rule:**
- ~68% of values fall within $\pm 1\sigma$ of the mean.
- ~95% of values fall within $\pm 2\sigma$ of the mean.
- ~99.7% of values fall within $\pm 3\sigma$ of the mean.

**Standard Normal:** $Z = (X - \mu) / \sigma \sim \mathcal{N}(0, 1)$.

**Central Limit Theorem (CLT):** The sum (or mean) of a large number of independent,
identically distributed random variables — regardless of their individual distribution —
converges to a Normal distribution. This is why the Normal appears everywhere in nature.

### e. Log-Normal Distribution

If $\ln(X) \sim \mathcal{N}(\mu, \sigma^2)$, then $X$ is log-normally distributed:

$$f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

- **Heavy right tail**: large values are possible but rare.
- **Arises from multiplicative processes**: if $X = Z_1 \times Z_2 \times \cdots \times Z_n$ with
  $Z_i > 0$, then $\ln X = \sum \ln Z_i$, which is approximately Normal by CLT.
- **Examples:** Stock prices, income distributions, reaction times, biological measurements.
- **ML use:** Modeling positive skewed quantities; often better than Normal for right-skewed data.

### f. Long-Tail / Power Law Distributions (Phân phối đuôi dài)

A power law distribution has PDF:

$$p(x) \propto x^{-\alpha}, \quad x \geq x_{\min}, \quad \alpha > 1$$

The defining feature is an extremely heavy tail — extreme events are far more likely than in
a Normal or even Log-Normal distribution.

**Zipf's Law:** In natural language, the frequency of the $r$-th most common word is
proportional to $1/r^s$ (with $s \approx 1$). The most frequent word appears roughly twice
as often as the second most frequent, three times as often as the third, etc.

**Examples of power laws in ML and data:**
- Word frequencies in text corpora (Zipf's Law)
- Node degree in social networks (a few people have millions of followers)
- Pareto distribution of wealth (80/20 rule)
- File sizes on the internet
- Earthquake magnitudes (Gutenberg-Richter law)

**Why they matter in ML:**
- Standard algorithms assuming Gaussian data may fail badly on power-law data.
- Rare events (the "long tail") may be the most important cases (fraud detection, medical anomalies).
- Log-log plots linearize power-law relationships, making fitting easier.
- Vocabulary size in NLP follows power laws → rare words are common in aggregate.

---

## 6. Expectation and Variance of Distributions

| Distribution | $E[X]$ | $\text{Var}(X)$ |
|-------------|--------|----------------|
| Bernoulli($p$) | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $np$ | $np(1-p)$ |
| Uniform($a,b$) | $(a+b)/2$ | $(b-a)^2/12$ |
| Normal($\mu,\sigma^2$) | $\mu$ | $\sigma^2$ |
| Log-Normal($\mu,\sigma^2$) | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ |
| Power Law (Pareto, $\alpha>2$) | $\frac{\alpha x_{\min}}{\alpha-1}$ | $\frac{x_{\min}^2\alpha}{(\alpha-1)^2(\alpha-2)}$ |

---

## 7. Maximum Likelihood Estimation (MLE)

**Goal:** Find the parameters $\theta$ that make the observed data most probable.

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \, P(\text{data} \mid \theta) = \arg\max_\theta \prod_{i=1}^n p(x_i \mid \theta)$$

In practice, maximize the **log-likelihood** (avoids numerical underflow and turns products into sums):

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

**Examples:**
- **Gaussian MLE:** $\hat{\mu} = \bar{x}$ (sample mean), $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ (biased variance)
- **Bernoulli MLE:** $\hat{p} = \frac{\text{number of successes}}{n}$
- **Logistic Regression:** MLE of weights under Bernoulli likelihood with sigmoid link function

**Key insight:** MLE is equivalent to minimizing cross-entropy loss in classification and MSE loss
in Gaussian regression.

---

## 8. Maximum A Posteriori (MAP) Estimation

MAP adds a **prior distribution** $P(\theta)$ over parameters:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \, P(\theta \mid \text{data}) = \arg\max_\theta \left[\log P(\text{data} \mid \theta) + \log P(\theta)\right]$$

**Key differences from MLE:**

| | MLE | MAP |
|--|-----|-----|
| Prior | None (or flat) | Explicit prior $P(\theta)$ |
| With little data | Can overfit | Regularized by prior |
| With lots of data | Same as MAP | Converges to MLE |

**Regularization connection:**
- Gaussian prior on weights → L2 (Ridge) regularization
- Laplace prior on weights → L1 (Lasso) regularization

**Bayesian vs MAP:** MAP gives the single most likely parameter value (a point estimate).
Full Bayesian inference computes the entire posterior distribution $P(\theta \mid \text{data})$.

---

## 9. Reference Links

- [Normal Distribution — Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- [Power Law — Wikipedia](https://en.wikipedia.org/wiki/Power_law)
- [Bayes' Theorem — Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Zipf's Law — Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
- [Central Limit Theorem — Wikipedia](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [Maximum Likelihood Estimation — Wikipedia](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
- [Beta Distribution (conjugate prior for Bernoulli) — Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)
