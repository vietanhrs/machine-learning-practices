# Variance & Bias (Phương sai & Độ lệch)

Understanding where model error comes from is one of the most important skills in machine learning.
This section builds the statistical and conceptual foundation for the bias-variance tradeoff.

---

## 1. Variance (Phương sai)

**Definition:** Variance measures how spread out a random variable is around its mean.
It quantifies the average squared deviation of a value from the expected value.

**Formula:**

$$\text{Var}(X) = E\left[(X - \mu)^2\right] = E[X^2] - (E[X])^2$$

where $\mu = E[X]$ is the mean (expected value) of $X$.

**Standard Deviation:**

$$\sigma = \sqrt{\text{Var}(X)}$$

Standard deviation has the same units as the original variable, making it easier to interpret
than variance (which is in squared units).

**Intuition:**
- Low variance → values are tightly clustered around the mean.
- High variance → values are spread widely.
- Variance is always non-negative: $\text{Var}(X) \geq 0$.
- $\text{Var}(X) = 0$ if and only if $X$ is a constant.

---

## 2. Sample vs Population Variance

### Population Variance

When you have data for the **entire population** of $N$ elements:

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

### Sample Variance (Bessel's Correction)

When you have only a **sample** of $n$ observations drawn from a larger population,
dividing by $n$ produces a **biased** (systematically too small) estimate. Instead, we divide by $n - 1$:

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

### Why $n - 1$? (Bessel's Correction)

- The sample mean $\bar{x}$ is computed from the same data, so the deviations $(x_i - \bar{x})$
  are not fully independent — they sum to zero by construction.
- This "uses up" one degree of freedom, leaving only $n - 1$ independent pieces of information.
- Dividing by $n - 1$ instead of $n$ makes $s^2$ an **unbiased estimator** of the population variance:
  $E[s^2] = \sigma^2$.
- Dividing by $n$ would give $E[s^2] = \frac{n-1}{n}\sigma^2$, which is always too small.

**Rule of thumb:** Use $n$ when computing the variance of a complete dataset (descriptive statistics).
Use $n - 1$ when estimating the variance of the population from a sample (inferential statistics).

---

## 3. Covariance and Correlation

### Covariance

Covariance measures how two random variables **change together** (joint variability):

$$\text{Cov}(X, Y) = E\left[(X - \mu_X)(Y - \mu_Y)\right]$$

Sample covariance:

$$s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

- $\text{Cov}(X, Y) > 0$: $X$ and $Y$ tend to increase together.
- $\text{Cov}(X, Y) < 0$: When $X$ increases, $Y$ tends to decrease.
- $\text{Cov}(X, Y) = 0$: No **linear** relationship (may still have non-linear relationship).
- **Problem:** Covariance depends on the scale of the variables, making comparison difficult.

### Pearson Correlation

To remove the effect of scale, we normalize covariance by the product of standard deviations:

$$r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

**Properties:**
- Range: $r \in [-1, +1]$
- $r = +1$: Perfect positive linear relationship.
- $r = -1$: Perfect negative linear relationship.
- $r = 0$: No linear relationship.
- Correlation is **dimensionless** — it is scale-invariant.
- Correlation measures **linear** association only. Two variables can have $r = 0$ and still
  be strongly non-linearly related (e.g., $Y = X^2$ centered at 0).

**Correlation Matrix:** A symmetric matrix where entry $(i, j)$ is the Pearson correlation
between variable $i$ and variable $j$. Diagonal entries are always 1.

---

## 4. Bias in Machine Learning (Độ lệch trong Học máy)

In a statistical and ML context, **bias** refers to **systematic error** — the tendency of a
model's predictions to be consistently off in one direction.

**Formal definition:** The bias of an estimator $\hat{f}$ for the true function $f^*$ is:

$$\text{Bias}(\hat{f}) = E[\hat{f}(x)] - f^*(x)$$

**Sources of bias in ML:**
- **Wrong model family:** Using a linear model when the true relationship is non-linear.
- **Missing features:** Important predictors are absent from the model.
- **Wrong assumptions:** Assuming independence when variables are correlated.
- **Training data mismatch:** Training distribution differs from deployment distribution.

**High bias** means the model cannot capture the true underlying pattern — it is too simple
or makes wrong assumptions. This leads to **underfitting**.

---

## 5. Bias-Variance Decomposition

For a regression problem with true function $f^*(x)$ and noise $\epsilon$ where $E[\epsilon] = 0$
and $\text{Var}(\epsilon) = \sigma^2_\epsilon$, the expected test error of a learned model $\hat{f}$ is:

$$E\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(E[\hat{f}(x)] - f^*(x)\right)^2}_{\text{Bias}^2} + \underbrace{E\left[\left(\hat{f}(x) - E[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible Noise}}$$

### Interpreting Each Term

| Term | What it measures | Cause |
|------|-----------------|-------|
| **Bias²** | How far the average prediction is from the truth | Model assumptions / simplicity |
| **Variance** | How much predictions fluctuate across different training sets | Model sensitivity to training data |
| **Irreducible Noise** | Inherent randomness in the data | Nature of the problem — cannot be reduced |

### High Bias (Underfitting)

- Training error is **high**.
- Test error is **high** (and similar to training error).
- The model is too simple to capture the true pattern.
- Examples: Linear model on non-linear data, shallow decision tree on complex data.
- **Fix:** More complex model, more features, polynomial features, less regularization.

### High Variance (Overfitting)

- Training error is **low**.
- Test error is **much higher** than training error.
- The model memorizes training data but does not generalize.
- Examples: Very deep decision tree, k-NN with k=1, neural network with too many parameters
  trained on little data.
- **Fix:** More training data, regularization (L1/L2/dropout), simpler model, ensemble methods.

---

## 6. Visual Intuition: The Dartboard Analogy

Imagine throwing darts at a target (bullseye = true value):

```
         Low Variance          High Variance
         ___________           ___________
High    |  X X       |        | X     X   |
Bias    |   X        |        |     X     |
        |            |        | X       X |
        |____________|        |___________|
        (clustered, but off)  (scattered AND off)

         ___________           ___________
Low     |     X      |        | X     X   |
Bias    |   X X X    |        |   X X     |
        |     X      |        | X   X X   |
        |____________|        |___________|
        (clustered on target) (scattered around target)
```

- **Bias** = Is the average dart on the bullseye? (accuracy of aim)
- **Variance** = How tightly clustered are the darts? (consistency of throw)
- **Ideal:** Low bias AND low variance — clustered tightly on the bullseye.
- **Reality:** There is often a tradeoff between the two.

---

## 7. Practical Implications

### Model Selection

When choosing model complexity, aim for the **sweet spot** where bias² + variance is minimized:

```
Error
  |
  |  \          total error
  |   \        /
  |    \      /
  |     \    /   variance
  |      \  /
  |   bias²\/
  |_________________________
              Model Complexity
```

### Regularization

Regularization (L1/Lasso, L2/Ridge) adds a penalty for model complexity:
- Reduces **variance** by shrinking weights toward zero.
- Slightly increases **bias** (the model is constrained from fitting the data perfectly).
- Net effect: Often reduces total test error.

### Ensemble Methods

| Method | Primary Effect | How |
|--------|---------------|-----|
| **Bagging** (e.g., Random Forest) | Reduces **variance** | Average over many models trained on bootstrap samples |
| **Boosting** (e.g., AdaBoost, XGBoost) | Reduces **bias** | Sequentially correct errors of weak learners |
| **Stacking** | Reduces both | Combine diverse models with a meta-learner |

### Cross-Validation

Use k-fold cross-validation to estimate the true generalization error.
A large gap between train and validation error signals high variance.
High error on both signals high bias.

---

## 8. Reference Links

- [Bias-Variance Tradeoff — Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [Variance — Wikipedia](https://en.wikipedia.org/wiki/Variance)
- [Bessel's Correction — Wikipedia](https://en.wikipedia.org/wiki/Bessel%27s_correction)
- [Pearson Correlation — Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
