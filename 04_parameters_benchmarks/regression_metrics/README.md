# Regression Metrics

A comprehensive guide to evaluating regression models. Unlike classification, regression predicts continuous values, so evaluation requires different tools.

---

## 1. Mean Squared Error (MSE)

**Vietnamese:** Độ lệch bình phương trung bình

**Formula:**

```
MSE = (1/n) * sum( (y_i - ŷ_i)² )
```

where `y_i` is the true value and `ŷ_i` is the predicted value.

### Properties

- Always non-negative; **lower is better**.
- Units are the **square of the target variable** (e.g., if predicting house prices in dollars, MSE is in dollars²).
- **Penalizes large errors quadratically** — an error of 10 contributes 100 to MSE, while an error of 1 contributes only 1. This means MSE is heavily influenced by outliers.

### When to Use

MSE is the default loss function for linear regression because it is differentiable and convex. Use it when **large errors are disproportionately bad** (e.g., trajectory prediction, structural engineering where a large deviation is catastrophic).

### Gradient Connection

MSE is the loss that, when minimized via ordinary least squares (OLS), yields the Maximum Likelihood Estimate under a Gaussian noise assumption.

---

## 2. Root Mean Squared Error (RMSE)

**Formula:**

```
RMSE = sqrt(MSE) = sqrt( (1/n) * sum( (y_i - ŷ_i)² ) )
```

### Why RMSE?

RMSE is simply the square root of MSE. Its key advantage: **same units as the target variable**.

If you're predicting house prices in dollars, RMSE is in dollars — you can say "on average, predictions are off by $15,000" instead of "$225,000,000 squared dollars."

### Comparison to MAE

- RMSE ≥ MAE always.
- The gap between RMSE and MAE indicates how much large errors are inflating the average — a large RMSE/MAE ratio suggests **outliers** or **heavy-tailed errors**.
- Tip: If RMSE >> MAE, investigate the largest residuals.

---

## 3. Mean Absolute Error (MAE)

**Formula:**

```
MAE = (1/n) * sum( |y_i - ŷ_i| )
```

### Properties

- Same units as the target.
- **Robust to outliers** — a prediction error of 100 contributes 100 to MAE (linearly), not 10,000.
- MAE corresponds to the L1 loss, while MSE corresponds to L2 loss.
- The optimal constant predictor under MAE is the **median** of y_true (versus the mean for MSE).

### When to Use

Use MAE when:
- Your data contains **outliers** you don't want to overweight.
- You want an easily interpretable average error magnitude.
- The business cost of errors is **linear** (e.g., a 2x larger error is exactly 2x worse).

### Disadvantage

MAE is not differentiable at zero, which can complicate gradient-based optimization. In practice, the Huber loss is used as a smooth compromise between MAE and MSE.

---

## 4. Mean Absolute Percentage Error (MAPE)

**Formula:**

```
MAPE = (100/n) * sum( |y_i - ŷ_i| / |y_i| )
```

Result is expressed as a percentage.

### Interpretation

MAPE = 10% means predictions are off by 10% on average relative to the true values. This makes it easy to communicate to non-technical stakeholders.

### Problems with MAPE

1. **Zero denominator:** MAPE is undefined when any `y_i = 0`. This is a hard failure.
2. **Asymmetry:** A 50% underprediction and a 200% overprediction have very different MAPE contributions even if the absolute error is the same.
3. **Bias toward underprediction:** MAPE penalizes overpredictions more than underpredictions.

### Alternatives

- **sMAPE (Symmetric MAPE):** `2 * |y - ŷ| / (|y| + |ŷ|)` — mitigates asymmetry.
- **MASE (Mean Absolute Scaled Error):** scales by a naive baseline, suitable for time series.

---

## 5. R² Score (Coefficient of Determination)

**Formula:**

```
SS_res = sum( (y_i - ŷ_i)² )          ← residual sum of squares
SS_tot = sum( (y_i - ȳ)² )            ← total sum of squares (variance * n)

R² = 1 - SS_res / SS_tot
```

where `ȳ` is the mean of `y_true`.

### Interpretation

| R²   | Meaning |
|------|---------|
| 1.0  | Perfect predictions — model explains all variance |
| 0.0  | Model performs no better than predicting the mean |
| <0.0 | Model performs **worse** than just predicting the mean |

R² is the **proportion of variance in y explained by the model**. It is scale-independent (unitless), which makes it useful for comparing models across different datasets.

### Can R² Be Negative?

Yes. If your model is so bad that its predictions are farther from the true values than simply predicting `ȳ` would be, then R² < 0. This often happens when:
- You evaluate on a different distribution than you trained on.
- The model is severely underfitted.

### Caution

R² **always increases** (or stays the same) when you add more features to a linear regression model, even if those features are pure noise. This is why Adjusted R² exists.

---

## 6. Adjusted R²

**Formula:**

```
R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)
```

where:
- `n` = number of samples
- `p` = number of predictors (features, excluding intercept)

### Why Adjusted R²?

Adjusted R² **penalizes adding irrelevant features**. Unlike R², it can decrease when a new feature doesn't explain enough variance to justify the loss in degrees of freedom.

Use Adjusted R² when:
- Comparing models with **different numbers of features**.
- Doing feature selection.

Adjusted R² ≤ R². As p approaches n, Adjusted R² drops sharply.

---

## 7. Log-Likelihood

**Maximum Likelihood Estimation perspective:**

Under the assumption that errors are Gaussian with standard deviation σ:

```
log p(y | X, θ) = -n/2 * log(2πσ²) - (1/2σ²) * sum( (y_i - ŷ_i)² )
```

### Connection to Loss Functions

Minimizing MSE is equivalent to **maximizing the Gaussian log-likelihood**:

```
argmin MSE  ⟺  argmax log-likelihood (Gaussian noise)
```

This gives OLS its theoretical justification.

### For Classification (Binary Cross-Entropy)

Under Bernoulli assumptions (binary classification):

```
log-likelihood = sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
```

This is the **negative** of binary cross-entropy loss. Minimizing cross-entropy = maximizing Bernoulli log-likelihood.

---

## 8. AIC and BIC (Model Selection Criteria)

Both AIC and BIC penalize model complexity to prevent overfitting and enable fair comparison across models of different sizes.

**Akaike Information Criterion (AIC):**

```
AIC = 2k - 2 * log-likelihood
```

**Bayesian Information Criterion (BIC):**

```
BIC = k * log(n) - 2 * log-likelihood
```

where:
- `k` = number of model parameters
- `n` = number of data points
- Lower is better for both.

### AIC vs BIC

| | AIC | BIC |
|--|-----|-----|
| Penalty per parameter | 2 | log(n) |
| Stricter? | Less strict | More strict when n > 7 |
| Aims to select | Best predictive model | True data-generating model |
| Asymptotic behavior | Can overfit | Consistent (selects correct model) |

**When n is large** (which is usually the case), BIC penalizes complexity much more than AIC, leading to sparser models. BIC is better for **inference** (identifying the true model); AIC is better for **prediction**.

Rule of thumb: Use AIC when your goal is prediction; use BIC when your goal is explanation/inference.

---

## 9. Residual Analysis

A residual is the error for a single prediction:

```
e_i = y_i - ŷ_i
```

### Residual Plots

A good regression model should produce **residuals that look like white noise**:

1. **Residuals vs Fitted Values:** Should show a random horizontal band around 0. Patterns indicate non-linearity (the model is missing structure).

2. **Q-Q Plot (Quantile-Quantile):** Should show residuals lying on a straight diagonal line. Deviations indicate non-Gaussian errors.

3. **Scale-Location Plot:** Should be flat. An upward slope indicates **heteroscedasticity** (variance increases with fitted values).

4. **Residuals vs Index (time):** For time-series data, should show no autocorrelation.

### Common Problems Detected by Residual Analysis

| Pattern | Problem | Remedy |
|---------|---------|--------|
| Curved residuals | Non-linearity | Add polynomial features, use nonlinear model |
| Fan-shaped (increasing spread) | Heteroscedasticity | Log-transform y, weighted regression |
| Heavy tails in Q-Q | Non-Gaussian errors | Robust regression, quantile regression |
| Systematic bias | Missing variable | Add features |

### Checking Assumptions

Linear regression assumes:
1. **Linearity** — E[e | X] = 0
2. **Homoscedasticity** — Var(e | X) = σ² (constant)
3. **Independence** — errors are uncorrelated
4. **Normality** — e ~ N(0, σ²) (for inference, not prediction)

Always check residuals before reporting results.

---

## Reference Links

- [scikit-learn Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Wikipedia: Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)
- [Wikipedia: Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [Wikipedia: Akaike Information Criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
- [Wikipedia: Residual Analysis](https://en.wikipedia.org/wiki/Errors_and_residuals)
