# Quiz: Variance & Bias

Test your understanding of variance, covariance, correlation, bias, and the bias-variance tradeoff.

---

**Q1.** What is the formula for **population variance** of a dataset with $N$ values and mean $\mu$?

- A) $\frac{1}{N} \sum (x_i - \bar{x})$
- B) $\frac{1}{N} \sum (x_i - \mu)^2$
- C) $\frac{1}{N-1} \sum (x_i - \mu)^2$
- D) $\sqrt{\frac{1}{N} \sum (x_i - \mu)^2}$

<details><summary>Answer</summary>

**B) $\frac{1}{N} \sum (x_i - \mu)^2$**

Population variance divides by the full count $N$ using the true population mean $\mu$.
Option C is sample variance. Option D is standard deviation.

</details>

---

**Q2.** What is the formula for **sample variance**, and how does it differ from population variance?

- A) Divide by $n$ instead of $N$; uses the sample mean $\bar{x}$
- B) Divide by $n - 1$ instead of $n$; uses the sample mean $\bar{x}$
- C) Divide by $n + 1$; uses the population mean $\mu$
- D) Divide by $n$; uses the population mean $\mu$

<details><summary>Answer</summary>

**B) Divide by $n - 1$ instead of $n$; uses the sample mean $\bar{x}$**

Sample variance is $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$.
The key differences are: (1) we use $\bar{x}$ (sample mean) not $\mu$, and (2) we divide by $n-1$.

</details>

---

**Q3.** Why do we use $n - 1$ (Bessel's correction) in sample variance instead of $n$?

- A) To make computation faster
- B) Because we only have $n - 1$ data points after discarding an outlier
- C) Because the sample mean uses up one degree of freedom, making the estimate biased low if we divide by $n$
- D) To ensure the result is always positive

<details><summary>Answer</summary>

**C) Because the sample mean uses up one degree of freedom, making the estimate biased low if we divide by $n$**

When computing deviations from the **sample** mean $\bar{x}$, the deviations $\sum(x_i - \bar{x}) = 0$
by definition, so only $n - 1$ of them are free to vary. Dividing by $n$ underestimates the true
population variance. Dividing by $n - 1$ corrects this, making the estimator unbiased: $E[s^2] = \sigma^2$.

</details>

---

**Q4.** What is the valid range of the **Pearson correlation coefficient** $r$?

- A) $[0, 1]$
- B) $(-\infty, +\infty)$
- C) $[-1, +1]$
- D) $[0, +\infty)$

<details><summary>Answer</summary>

**C) $[-1, +1]$**

$r = +1$ means perfect positive linear correlation, $r = -1$ means perfect negative linear correlation,
$r = 0$ means no linear relationship. Covariance is unbounded, but normalizing by standard deviations
constrains $r$ to $[-1, +1]$.

</details>

---

**Q5.** What is the main **difference between covariance and correlation**?

- A) Covariance is always positive; correlation can be negative
- B) Covariance is normalized to $[-1, 1]$; correlation is not
- C) Correlation is normalized by standard deviations, making it scale-invariant; covariance is not
- D) They measure different things: covariance measures linearity, correlation measures monotonicity

<details><summary>Answer</summary>

**C) Correlation is normalized by standard deviations, making it scale-invariant; covariance is not**

Covariance value depends on the units and scale of the variables (e.g., measuring height in cm vs meters
changes the covariance). Pearson correlation $r = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$ removes scale
dependence, allowing comparison across different pairs of variables.

</details>

---

**Q6.** Two variables have Pearson correlation $r = 0$. Which statement is **most accurate**?

- A) The variables are independent
- B) There is no relationship between the variables
- C) There is no **linear** relationship, but there could be a strong non-linear relationship
- D) The covariance between them must also be 0

<details><summary>Answer</summary>

**C) There is no linear relationship, but there could be a strong non-linear relationship**

$r = 0$ only means no linear relationship. For example, $Y = X^2$ with $X$ symmetric around 0
gives $r = 0$ (the parabola is "non-correlated" linearly) but $Y$ is perfectly determined by $X$.
Also, $r = 0$ does not imply statistical independence in general. Note: the covariance would also
be 0 in this case (D is also technically true but not "most accurate" as an insight).

</details>

---

**Q7.** In machine learning, **bias** of a model is best described as:

- A) The weight values in a neural network's bias neurons
- B) Unfairness in training data toward a demographic group
- C) The systematic error due to wrong assumptions in the model — the gap between the average prediction and the true value
- D) The difference between training error and test error

<details><summary>Answer</summary>

**C) The systematic error due to wrong assumptions in the model — the gap between the average prediction and the true value**

Statistical bias = $E[\hat{f}(x)] - f^*(x)$. It represents how far the model's average prediction
(across all possible training sets) is from the true function. Options A and B are different senses
of the word "bias" not related to statistical bias. Option D describes the variance effect.

</details>

---

**Q8.** The bias-variance decomposition states that expected test MSE equals:

- A) Bias + Variance
- B) Bias² + Variance + Irreducible Noise
- C) (Bias + Variance)² + Noise
- D) Bias² × Variance + Noise

<details><summary>Answer</summary>

**B) Bias² + Variance + Irreducible Noise**

$E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2_\epsilon$

The bias term is squared because we are working with MSE. The irreducible noise $\sigma^2_\epsilon$
is the variance of the data-generating noise that no model can eliminate.

</details>

---

**Q9.** A model has **high bias** (underfitting). What pattern do you expect in training and test errors?

- A) Low training error, high test error
- B) High training error, low test error
- C) Both training error and test error are high, and they are close to each other
- D) Both training error and test error are low

<details><summary>Answer</summary>

**C) Both training error and test error are high, and they are close to each other**

A high-bias model fails to capture the true underlying pattern, so it performs poorly even on
training data. Because the model is too simple to overfit, training and test error are similar —
both high. This contrasts with high variance, where training error is low but test error is much higher.

</details>

---

**Q10.** A model has **high variance** (overfitting). What pattern do you expect?

- A) Low training error, high test error (large gap between them)
- B) High training error, low test error
- C) Both errors are high and similar
- D) Both errors are low and similar

<details><summary>Answer</summary>

**A) Low training error, high test error (large gap between them)**

A high-variance model memorizes the training data, achieving very low training error, but fails
to generalize to new data, resulting in high test error. The large train-test gap is the hallmark
of overfitting / high variance.

</details>

---

**Q11.** As **model complexity increases** (e.g., increasing polynomial degree), what happens to bias and variance?

- A) Bias increases; variance decreases
- B) Both bias and variance increase
- C) Bias decreases; variance increases
- D) Both bias and variance decrease

<details><summary>Answer</summary>

**C) Bias decreases; variance increases**

A more complex model can fit more patterns, reducing bias (it can represent the true function better).
However, it becomes more sensitive to the specific training data it sees, increasing variance.
This is the essence of the bias-variance tradeoff — you generally cannot reduce both simultaneously
with the same model class.

</details>

---

**Q12.** How does **bagging** (e.g., Random Forest) reduce model error?

- A) By using a more complex base model to reduce bias
- B) By averaging predictions from multiple models trained on bootstrap samples, which reduces variance
- C) By sequentially correcting errors of weak learners, which reduces bias
- D) By removing noisy features, which reduces irreducible noise

<details><summary>Answer</summary>

**B) By averaging predictions from multiple models trained on bootstrap samples, which reduces variance**

Bagging (Bootstrap Aggregating) trains many models on different random subsets of the training data
and averages their predictions. Averaging reduces variance because random errors in individual models
cancel out. Bias is roughly preserved (each model has similar bias to the base learner).
This is why Random Forests outperform individual decision trees without increasing bias significantly.

</details>

---

**Q13.** How does **boosting** (e.g., AdaBoost, Gradient Boosting) primarily reduce model error?

- A) By reducing variance through averaging
- B) By reducing bias by sequentially training weak learners that correct previous errors
- C) By adding dropout to reduce overfitting
- D) By using larger training datasets

<details><summary>Answer</summary>

**B) By reducing bias by sequentially training weak learners that correct previous errors**

Boosting trains models sequentially, with each new model focusing on the examples the previous
models got wrong. This process drives down bias — even if each individual learner has high bias
(is weak), their combination has low bias. However, boosting can increase variance and overfit
if run for too many iterations, which is why early stopping or regularization is often used.

</details>

---

**Q14.** Can we ever reduce the **irreducible noise** term in the bias-variance decomposition?

- A) Yes, by using a more complex model
- B) Yes, by collecting more training data
- C) Yes, by using ensemble methods
- D) No — it is the inherent randomness in the data-generating process and cannot be reduced by any model

<details><summary>Answer</summary>

**D) No — it is the inherent randomness in the data-generating process and cannot be reduced by any model**

Irreducible noise $\sigma^2_\epsilon$ comes from the stochastic nature of the world — measurement
error, unmeasured confounders, or fundamentally random processes. No matter how good your model
is, it cannot predict this noise. The only way to reduce it is to collect better/more informative
features (which changes the problem, not the model), but no learning algorithm alone can eliminate it.

</details>

---

## Summary Table

| Concept | Key Formula | Range / Units |
|---------|------------|---------------|
| Population Variance | $\sigma^2 = \frac{1}{N}\sum(x_i-\mu)^2$ | $\geq 0$, squared units |
| Sample Variance | $s^2 = \frac{1}{n-1}\sum(x_i-\bar{x})^2$ | $\geq 0$, squared units |
| Standard Deviation | $\sigma = \sqrt{\text{Var}(X)}$ | $\geq 0$, same units as $X$ |
| Covariance | $\text{Cov}(X,Y) = \frac{1}{n-1}\sum(x_i-\bar{x})(y_i-\bar{y})$ | $(-\infty, +\infty)$ |
| Pearson Correlation | $r = \text{Cov}(X,Y)/(\sigma_X\sigma_Y)$ | $[-1, +1]$ |
| Bias | $E[\hat{f}(x)] - f^*(x)$ | $(-\infty, +\infty)$ |
| Expected MSE | $\text{Bias}^2 + \text{Variance} + \sigma^2_\epsilon$ | $\geq 0$ |
