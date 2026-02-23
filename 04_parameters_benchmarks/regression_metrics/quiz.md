# Regression Metrics — Quiz

Test your understanding of evaluation metrics for regression models.

---

**Q1.** A dataset contains house prices. Your model predicts $300,000 for a house worth $350,000. Calculate MSE, RMSE, and MAE for this single sample. Which is easiest to interpret, and why?

<details>
<summary>Answer</summary>

Error = 350,000 - 300,000 = 50,000

- **MSE** = 50,000² = **2,500,000,000** (in dollars²) — very hard to interpret.
- **RMSE** = sqrt(2,500,000,000) = **$50,000** — same unit as house price, directly interpretable.
- **MAE** = |50,000| = **$50,000** — also interpretable.

RMSE and MAE are equally interpretable here (one sample). RMSE is easiest in general because it shares units with the target, unlike MSE.
</details>

---

**Q2.** Your dataset has one outlier: a house worth $1,000,000 that your model predicts as $100,000 (error = $900,000). How does this outlier affect MSE vs MAE differently?

<details>
<summary>Answer</summary>

The outlier contributes:

- To **MSE**: 900,000² = 810,000,000,000 — a massive quadratic contribution that will dominate the overall MSE.
- To **MAE**: 900,000 — a large but linear contribution.

**MSE is far more sensitive to outliers** because errors are squared. A single large error can make MSE appear terrible even when most predictions are good.

**MAE** treats the outlier as just a large linear penalty — more robust to extreme values.

Practical implication: If your RMSE is much larger than your MAE, you likely have outliers or a heavy-tailed error distribution.
</details>

---

**Q3.** What does R² = 0.85 mean? What does R² = 0.0 mean? Can R² be negative?

<details>
<summary>Answer</summary>

**R² = 0.85:** The model explains **85% of the variance** in the target variable. The remaining 15% is unexplained (residual variance).

**R² = 0.0:** The model performs **no better than predicting the mean** of y_true for every sample. This is equivalent to the intercept-only (null) model.

**R² can be negative:** Yes. If the model's predictions are worse than simply predicting the mean (e.g., systematic bias, or evaluating on a different distribution than training), then SS_res > SS_tot, so R² = 1 - SS_res/SS_tot < 0.

Examples that lead to negative R²:
- A model trained on one city's house prices evaluated on another.
- A severely underfitted model evaluated on test data.
- Applying a regression model outside its training domain.
</details>

---

**Q4.** Why is RMSE more interpretable than MSE? Give a concrete example.

<details>
<summary>Answer</summary>

RMSE is interpretable because it is in the **same units as the target variable**.

**Example:** Predicting daily temperature in Celsius.
- MSE = 4.0 °C² — the unit "degrees squared" has no physical meaning.
- RMSE = sqrt(4.0) = 2.0 °C — you can say "predictions are off by 2°C on average," which everyone understands.

This is why RMSE is almost always reported instead of MSE, even though MSE is what's actually minimized during training (they are monotonically related, so minimizing one minimizes the other).
</details>

---

**Q5.** What is the mathematical connection between minimizing MSE (in regression) and maximizing log-likelihood? What distribution does this assume?

<details>
<summary>Answer</summary>

Minimizing MSE is equivalent to maximizing the **Gaussian log-likelihood**.

Under the assumption that errors follow a Gaussian (normal) distribution:
```
y_i = ŷ_i + ε_i,   where ε_i ~ N(0, σ²)
```

The log-likelihood is:
```
log L = -n/2 * log(2πσ²) - (1/2σ²) * sum((y_i - ŷ_i)²)
```

The first term is a constant. Maximizing log L with respect to model parameters is equivalent to minimizing `sum((y_i - ŷ_i)²)`, which is proportional to MSE.

**Connection to cross-entropy:** For binary classification with Bernoulli targets, the analogous result holds: minimizing binary cross-entropy = maximizing Bernoulli log-likelihood.
</details>

---

**Q6.** Why does MAPE fail when a target value is zero? Give an example. What alternative metrics exist?

<details>
<summary>Answer</summary>

MAPE formula:
```
MAPE = (1/n) * sum(|y_i - ŷ_i| / |y_i|)
```

When `y_i = 0`, the denominator is 0 → **division by zero** → MAPE is undefined or infinite.

**Example:** Sales forecasting where a product had zero sales on some days. Any prediction on those days blows up MAPE.

**Alternatives:**
- **sMAPE (Symmetric MAPE):** `2 * |y - ŷ| / (|y| + |ŷ|)` — defined even when y=0 (unless both are 0).
- **MASE:** Scales by naive baseline, suitable for time series.
- **RMSE/MAE:** Never have this zero-denominator problem.
- Add a small epsilon: `|y_i - ŷ_i| / max(|y_i|, ε)` — but this changes the metric's meaning.
</details>

---

**Q7.** A residual plot shows a fan shape: residuals are small for low fitted values and large for high fitted values. What does this indicate, and how do you fix it?

<details>
<summary>Answer</summary>

This pattern is called **heteroscedasticity** — the variance of the residuals is not constant but increases with the fitted value.

**What it means:** The model's error variance depends on the predicted level. This violates the OLS assumption of homoscedasticity.

**Consequences:**
- OLS coefficient estimates are still unbiased but are **no longer efficient** (not minimum variance).
- Standard errors are incorrect → hypothesis tests and confidence intervals are invalid.

**Fixes:**
1. **Log-transform the target:** `log(y)` often stabilizes variance (common for prices, counts, time data).
2. **Weighted Least Squares (WLS):** Weight observations inversely proportional to their variance.
3. **Robust standard errors:** Use heteroscedasticity-consistent (HC) standard errors without changing the model.
4. **Box-Cox transformation:** Find the optimal power transformation.
</details>

---

**Q8.** What is the difference between AIC and BIC? Which penalizes model complexity more heavily, and when?

<details>
<summary>Answer</summary>

```
AIC = 2k - 2 * log-likelihood
BIC = k * log(n) - 2 * log-likelihood
```

- **AIC penalty per parameter:** 2 (constant).
- **BIC penalty per parameter:** log(n) (grows with sample size).

**BIC penalizes complexity more heavily when n > e² ≈ 7.4** — which is almost always. With n = 1,000, BIC penalty is log(1000) ≈ 6.9 per parameter vs AIC's 2.

**When to prefer each:**
- **AIC:** When the goal is **prediction performance** on new data. AIC is asymptotically equivalent to leave-one-out cross-validation for linear models.
- **BIC:** When the goal is **model identification** (finding the true generating model). BIC is consistent — as n → ∞, it selects the correct model with probability 1.

In practice: BIC produces sparser models; AIC is more permissive. For large n, the difference can be dramatic.
</details>

---

**Q9.** What is Adjusted R² and why is it preferred over R² for comparing models with different numbers of features?

<details>
<summary>Answer</summary>

```
R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)
```

**Problem with R²:** Adding any feature to a linear model, even pure noise, will never decrease R² — it can only stay the same or increase. This makes R² a biased measure of model quality when comparing models of different sizes.

**Adjusted R²** penalizes for the number of predictors `p`. If a new feature doesn't explain enough additional variance to justify losing a degree of freedom, Adjusted R² will **decrease**.

**Comparison example:**
- Model A: 3 features, R²=0.85, R²_adj=0.84
- Model B: 10 features, R²=0.87, R²_adj=0.82

Model A is actually better (higher Adjusted R²) despite lower raw R².

Use R²_adj whenever comparing models with different numbers of features, or when doing feature selection.
</details>

---

**Q10.** Your linear regression residuals show a curved (U-shaped) pattern in the residuals-vs-fitted plot. What does this suggest, and what is the remedy?

<details>
<summary>Answer</summary>

A U-shaped or curved pattern in the residuals-vs-fitted plot indicates **non-linearity** — the model is missing a nonlinear relationship in the data. The model is systematically over- or underpredicting in certain ranges.

**Interpretation:** The true relationship between features and target is not linear. A linear model cannot capture the curvature.

**Remedies:**
1. **Add polynomial features:** Include x², x³ terms.
2. **Feature transformations:** Try log(x), sqrt(x), 1/x.
3. **Interaction terms:** The nonlinearity might come from interaction between features.
4. **Use a nonlinear model:** Decision tree, random forest, neural network, polynomial regression.
5. **Piecewise/spline regression:** Fit different linear pieces over different ranges.
</details>

---

**Q11.** When should you prefer MAE over MSE as your evaluation metric? Give two concrete scenarios.

<details>
<summary>Answer</summary>

**Prefer MAE when:**

1. **Your data contains outliers** that you don't want to dominate the metric. MAE weights all errors linearly. Example: Real estate price prediction where a few ultra-luxury properties have massive residuals — MAE gives a more representative picture of typical prediction error.

2. **The business cost of errors is linear** — a prediction that's twice as wrong is exactly twice as costly, not four times. Example: Supply chain inventory prediction where each unit of over/under-stock has a fixed linear cost.

**Prefer MSE/RMSE when:**
- Large errors are disproportionately costly (quadratic cost structure).
- You want a differentiable loss for gradient-based optimization.
- The Gaussian noise assumption is appropriate.
- You need to train via OLS (MSE has a closed-form solution; MAE does not).
</details>

---

**Q12.** A model has R² = 0.92 on the training set and R² = 0.45 on the test set. What does this tell you, and how would you diagnose and fix it?

<details>
<summary>Answer</summary>

This large gap (0.92 training vs 0.45 test) is a classic sign of **overfitting** — the model has memorized the training data including noise, but fails to generalize.

**Diagnosis:**
- Compute residuals on train vs test — train residuals will be small and random; test residuals will be large.
- Check if the model is too complex (e.g., high-degree polynomial, too many features, deep tree without pruning).
- Check if training data is small (easy to overfit).

**Fixes:**
1. **Regularization:** Ridge (L2), Lasso (L1), or Elastic Net add penalty for model complexity.
2. **Reduce model complexity:** Lower polynomial degree, fewer features, shallower tree.
3. **Feature selection:** Remove irrelevant or redundant features.
4. **Cross-validation:** Use k-fold CV instead of a single train/test split for more reliable estimates.
5. **Collect more data:** The most reliable solution when feasible.
6. **Dropout/early stopping:** For neural networks specifically.
</details>

---

## Summary Cheat Sheet

| Metric | Formula | Outlier Sensitivity | Units | Range |
|--------|---------|---------------------|-------|-------|
| MSE | mean((y - ŷ)²) | High | target² | [0, ∞) |
| RMSE | sqrt(MSE) | High | target | [0, ∞) |
| MAE | mean(\|y - ŷ\|) | Low | target | [0, ∞) |
| MAPE | mean(\|y - ŷ\| / \|y\|) * 100 | Medium | % | [0, ∞) |
| R² | 1 - SS_res/SS_tot | Medium | unitless | (-∞, 1] |
| Adj. R² | penalizes features | Medium | unitless | (-∞, 1] |
