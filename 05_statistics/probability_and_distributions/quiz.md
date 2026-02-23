# Quiz: Probability & Distributions

Test your understanding of probability theory, key distributions, and statistical estimation.

---

**Q1.** State Bayes' Theorem. Given events $A$ and $B$ with $P(B) > 0$, what is $P(A \mid B)$?

- A) $P(A \mid B) = P(B \mid A) \cdot P(B) / P(A)$
- B) $P(A \mid B) = P(A \cap B) / P(A)$
- C) $P(A \mid B) = P(B \mid A) \cdot P(A) / P(B)$
- D) $P(A \mid B) = P(A) + P(B) - P(A \cap B)$

<details><summary>Answer</summary>

**C) $P(A \mid B) = P(B \mid A) \cdot P(A) / P(B)$**

Bayes' Theorem: $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$.
This follows directly from the definition of conditional probability:
$P(A|B) = P(A \cap B)/P(B)$ and $P(B|A) = P(A \cap B)/P(A)$, hence $P(A \cap B) = P(B|A)P(A)$.

</details>

---

**Q2.** What does **conditional probability** $P(A \mid B)$ represent?

- A) The probability that both A and B occur simultaneously
- B) The probability that A occurs, given that B is known to have occurred
- C) The probability that A occurs after B in time
- D) The ratio of the probability of A to the probability of B

<details><summary>Answer</summary>

**B) The probability that A occurs, given that B is known to have occurred**

Conditioning on $B$ restricts the sample space to outcomes where $B$ is true, then asks
what fraction of those outcomes also have $A$. Formally: $P(A|B) = P(A \cap B)/P(B)$.

</details>

---

**Q3.** A Normal distribution $\mathcal{N}(\mu, \sigma^2)$ has parameters $\mu = 100$ and $\sigma = 15$.
According to the **68-95-99.7 rule**, approximately what percentage of values fall between 70 and 130?

- A) 68%
- B) 95%
- C) 99.7%
- D) 50%

<details><summary>Answer</summary>

**B) 95%**

The interval $[70, 130] = [\mu - 2\sigma, \mu + 2\sigma] = [100 - 30, 100 + 30]$.
The 68-95-99.7 rule states that approximately 95% of values fall within $\pm 2\sigma$ of the mean.
(This is approximately 95.45% precisely.)

</details>

---

**Q4.** What does the **Central Limit Theorem (CLT)** state?

- A) The sum of any two normal distributions is also normal
- B) The sample mean of any distribution is always exactly normal
- C) As sample size $n$ increases, the distribution of the sample mean approaches a Normal distribution, regardless of the original distribution
- D) Large datasets always follow a Normal distribution

<details><summary>Answer</summary>

**C) As sample size $n$ increases, the distribution of the sample mean approaches a Normal distribution, regardless of the original distribution**

The CLT says: if $X_1, X_2, \ldots, X_n$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$,
then $\bar{X}_n = \frac{1}{n}\sum X_i$ approaches $\mathcal{N}(\mu, \sigma^2/n)$ as $n \to \infty$.
This is why the Normal distribution appears so frequently — it is the limit of averages.

</details>

---

**Q5.** Which of the following are examples of **long-tail / power-law distributions** in ML and data?

- A) Height of adult humans
- B) Word frequencies in a large corpus (Zipf's Law)
- C) Measurement errors in a laboratory
- D) IQ scores in a population

<details><summary>Answer</summary>

**B) Word frequencies in a large corpus (Zipf's Law)**

Word frequencies follow Zipf's Law: the $r$-th most frequent word has frequency $\propto 1/r$.
The top few words ("the", "a", "is") appear enormously more often than the vast majority
of words. Other examples of power laws: social network follower counts, website traffic,
earthquake magnitudes, file sizes. Options A, C, D are all approximately Normally distributed.

</details>

---

**Q6.** What is another real-world example of power-law / long-tail distribution relevant to ML?

- A) Daily temperature fluctuations
- B) Random measurement noise
- C) Number of followers per user on a social network
- D) Blood pressure in a clinical trial

<details><summary>Answer</summary>

**C) Number of followers per user on a social network**

Social network degree distributions are classic power-law examples: most users have a few
followers, but a small number of accounts have millions. This "scale-free" property is a
hallmark of power-law distributions. Normal distribution applies to A, B, and D.

</details>

---

**Q7.** What is the key difference between **MLE** and **MAP** estimation?

- A) MLE maximizes likelihood; MAP minimizes it
- B) MLE uses only the likelihood; MAP incorporates a prior distribution over parameters
- C) MLE is for continuous data; MAP is for discrete data
- D) MLE requires labeled data; MAP is unsupervised

<details><summary>Answer</summary>

**B) MLE uses only the likelihood; MAP incorporates a prior distribution over parameters**

MLE: $\hat{\theta} = \arg\max P(\text{data} \mid \theta)$.
MAP: $\hat{\theta} = \arg\max P(\theta \mid \text{data}) = \arg\max [P(\text{data} \mid \theta) \cdot P(\theta)]$.
MAP adds $\log P(\theta)$ to the objective, which acts as regularization. With a flat prior,
MAP reduces to MLE.

</details>

---

**Q8.** Why does the **log-normal distribution** appear in so many real-world phenomena?

- A) Because the Central Limit Theorem applies to all sums of random variables
- B) Because it is the same as the normal distribution for positive values
- C) Because many quantities arise from multiplicative processes, and the log of a product of positive values is a sum, which tends toward normality by CLT
- D) Because log-normal variables have finite support like uniform distributions

<details><summary>Answer</summary>

**C) Because many quantities arise from multiplicative processes, and the log of a product of positive values is a sum, which tends toward normality by CLT**

If $X = Z_1 \cdot Z_2 \cdots Z_n$ (a product of many positive factors), then
$\ln X = \ln Z_1 + \ln Z_2 + \cdots + \ln Z_n$. By the CLT, this sum approaches normality,
so $X$ itself follows a log-normal distribution. Examples: stock prices (daily multiplicative
returns), income, biological measurements like body mass.

</details>

---

**Q9.** What is the relationship between **Bernoulli** and **Binomial** distributions?

- A) Binomial is a special case of Bernoulli with n > 1
- B) A Binomial(n, p) random variable is the sum of n independent Bernoulli(p) random variables
- C) Bernoulli is the continuous version of the Binomial
- D) They are identical for large n

<details><summary>Answer</summary>

**B) A Binomial(n, p) random variable is the sum of n independent Bernoulli(p) random variables**

Bernoulli($p$) models a single trial: $P(X=1) = p$, $P(X=0) = 1-p$.
Binomial($n$, $p$) models the number of successes in $n$ such independent trials.
Thus Binomial($n$, $p$) = $\sum_{i=1}^n X_i$ where each $X_i \sim \text{Bernoulli}(p)$.
Binomial($1$, $p$) is the same as Bernoulli($p$).

</details>

---

**Q10.** In Bayesian inference, define **prior**, **likelihood**, and **posterior**.

- A) Prior = data; likelihood = model output; posterior = loss
- B) Prior = belief before data; likelihood = P(data | parameters); posterior = updated belief after data
- C) Prior = training set; likelihood = validation set; posterior = test set
- D) Prior = mean; likelihood = variance; posterior = standard deviation

<details><summary>Answer</summary>

**B) Prior = belief before data; likelihood = P(data | parameters); posterior = updated belief after data**

- **Prior** $P(\theta)$: What we believe about $\theta$ before observing any data.
- **Likelihood** $P(\text{data} \mid \theta)$: How probable is the observed data for a given $\theta$?
- **Posterior** $P(\theta \mid \text{data})$: Updated belief after combining prior and likelihood via Bayes' theorem.

</details>

---

**Q11.** Why is the **Normal distribution** so ubiquitous in nature and statistics?

- A) Because nature is inherently symmetric
- B) Because it is the only distribution with finite mean and variance
- C) Because the Central Limit Theorem guarantees that sums/averages of many independent random variables converge to Normal, regardless of individual distributions
- D) Because it was defined to model natural phenomena

<details><summary>Answer</summary>

**C) Because the Central Limit Theorem guarantees that sums/averages of many independent random variables converge to Normal, regardless of individual distributions**

Most measurable traits in biology, physics, and social science are the result of many small,
independent contributing factors. By CLT, their sum is approximately Normal. For example:
height is influenced by hundreds of genetic and environmental factors, each adding a small
contribution — hence height is approximately normally distributed.

</details>

---

**Q12.** How do the tails of a **power-law** distribution compare to an **exponential** distribution for large $x$?

- A) Exponential has heavier tails than power law
- B) They have identical tail behavior
- C) Power law has much heavier tails — it decays as $x^{-\alpha}$ (polynomial), while exponential decays as $e^{-\lambda x}$ (much faster)
- D) Power law has lighter tails because $\alpha > 1$

<details><summary>Answer</summary>

**C) Power law has much heavier tails — it decays as $x^{-\alpha}$ (polynomial), while exponential decays as $e^{-\lambda x}$ (much faster)**

For large $x$: exponential $e^{-\lambda x}$ decays extremely fast (all moments are finite).
Power law $x^{-\alpha}$ decays very slowly — for $\alpha \leq 2$, the variance is infinite;
for $\alpha \leq 1$, even the mean is infinite. This "heavy tail" means extreme events are
far more common in power-law distributions than exponential distributions predict.
On a log-log plot, power law appears as a straight line; exponential does not.

</details>

---

**Q13.** What does **MLE optimize**?

- A) The posterior probability of the model parameters
- B) The probability of the parameters given the data
- C) The probability of the observed data given the model parameters (the likelihood)
- D) The squared difference between predicted and observed values

<details><summary>Answer</summary>

**C) The probability of the observed data given the model parameters (the likelihood)**

MLE finds $\hat{\theta} = \arg\max_\theta P(\text{data} \mid \theta)$.
It asks: "For which parameter values is this particular dataset most probable?"
Note: MLE does NOT incorporate any prior belief about $\theta$ — that is MAP's role.
Minimizing negative log-likelihood is equivalent to maximizing likelihood.

</details>

---

**Q14.** What is the difference between **PDF**, **PMF**, and **CDF**?

- A) They are three names for the same concept
- B) PMF is for discrete variables (gives P(X=x)); PDF is for continuous variables (gives density at x); CDF gives P(X ≤ x) for both
- C) PDF gives exact probabilities for continuous variables; PMF gives densities for discrete variables
- D) CDF is the derivative of the PDF; PMF is the integral of the PDF

<details><summary>Answer</summary>

**B) PMF is for discrete variables (gives P(X=x)); PDF is for continuous variables (gives density at x); CDF gives P(X ≤ x) for both**

- **PMF** (Probability Mass Function): $p(x) = P(X = x)$ for discrete $X$. Probabilities sum to 1.
- **PDF** (Probability Density Function): $f(x)$ for continuous $X$. $P(a \leq X \leq b) = \int_a^b f(x)\,dx$. Individual points have probability 0.
- **CDF** (Cumulative Distribution Function): $F(x) = P(X \leq x)$, valid for both types.
  For continuous: $F(x) = \int_{-\infty}^x f(t)\,dt$, so $f(x) = F'(x)$.
  For discrete: $F(x) = \sum_{t \leq x} p(t)$.

</details>

---

**Q15.** A medical test has sensitivity 0.95 and specificity 0.90 for a disease with prevalence 1%.
Approximately, what is $P(\text{disease} \mid \text{positive test})$?

- A) ~95%
- B) ~90%
- C) ~8.7%
- D) ~50%

<details><summary>Answer</summary>

**C) ~8.7%**

Using Bayes' theorem:
- $P(D) = 0.01$, $P(\text{pos} \mid D) = 0.95$, $P(\text{pos} \mid \neg D) = 1 - 0.90 = 0.10$
- $P(\text{pos}) = 0.95 \times 0.01 + 0.10 \times 0.99 = 0.0095 + 0.099 = 0.1085$
- $P(D \mid \text{pos}) = (0.95 \times 0.01) / 0.1085 \approx 0.0876 \approx 8.7\%$

This is the **base rate fallacy**: even with a good test, low prevalence means most
positive results are false positives. Prevalence (the prior) dominates when it is extreme.

</details>

---

**Q16.** What is the **evidence** (denominator) term in Bayes' theorem, and what role does it play?

- A) It represents the posterior distribution
- B) It is a normalizing constant that ensures the posterior integrates to 1
- C) It is the log-likelihood of the data
- D) It represents the prior distribution

<details><summary>Answer</summary>

**B) It is a normalizing constant that ensures the posterior integrates to 1**

$P(\text{data}) = \int P(\text{data} \mid \theta) P(\theta)\, d\theta$ (marginal likelihood).
It does not depend on $\theta$, so for MAP/MLE it can be ignored during optimization.
However, it is crucial for model comparison (Bayesian model selection) and full Bayesian
inference where you need $P(\theta \mid \text{data})$ to be a valid probability distribution.
Computing this integral is often intractable, motivating approximate inference methods
like MCMC or variational inference.

</details>

---

## Summary Table

| Distribution | Parameters | Mean | Variance | Use in ML |
|-------------|-----------|------|----------|-----------|
| Bernoulli | $p$ | $p$ | $p(1-p)$ | Binary classification |
| Binomial | $n, p$ | $np$ | $np(1-p)$ | Count of successes |
| Uniform | $a, b$ | $(a+b)/2$ | $(b-a)^2/12$ | Random init, flat prior |
| Normal | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | Everywhere (CLT) |
| Log-Normal | $\mu, \sigma^2$ | $e^{\mu+\sigma^2/2}$ | heavy right tail | Positive skewed data |
| Power Law | $\alpha, x_{\min}$ | $\frac{\alpha x_{\min}}{\alpha-1}$ | heavy tail | Word freq, degrees |
