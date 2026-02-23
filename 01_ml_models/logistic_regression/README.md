# Logistic Regression

Logistic Regression is a **probabilistic classifier** that models the probability that a data point belongs to a particular class. Despite its name, it is used for **classification**, not regression.

---

## 1. What is Logistic Regression?

Logistic Regression extends linear regression to classification by squashing the linear output through a **sigmoid function** to produce a probability in `(0, 1)`.

For binary classification:

```
P(y=1 | x) = σ(wᵀx + b)
```

where `σ` is the sigmoid function and `w` are learned weights.

It is a **discriminative model** — it models `P(y | x)` directly rather than the joint distribution.

### Key Characteristics

- Output is always between 0 and 1 (a valid probability).
- Decision boundary is **linear** in feature space.
- Interpretable: weights correspond to **log-odds**.
- Efficient and fast to train.
- Works well as a baseline before trying complex models.

---

## 2. The Sigmoid Function

The sigmoid (logistic) function maps any real number to `(0, 1)`:

```
σ(z) = 1 / (1 + e^{-z})
```

### Properties

| Property | Value |
|----------|-------|
| σ(0) | 0.5 |
| σ(+∞) | → 1 |
| σ(−∞) | → 0 |
| Derivative | σ(z) · (1 − σ(z)) |

### Interpretation

- `σ(z) > 0.5` → predict class 1
- `σ(z) < 0.5` → predict class 0
- `σ(z) = 0.5` → decision boundary (z = 0)

The output `σ(wᵀx + b)` can be interpreted as the **probability** that the input `x` belongs to class 1.

---

## 3. Decision Boundary

The decision boundary is defined by:

```
wᵀx + b = 0
```

This is a **hyperplane** in feature space:
- In 2D: a straight line.
- In 3D: a flat plane.
- In n-D: an (n-1)-dimensional hyperplane.

Logistic Regression can only learn **linearly separable** boundaries. For non-linear boundaries, use polynomial features or switch to non-linear models.

---

## 4. Binary Cross-Entropy Loss

We use **binary cross-entropy** (log-loss) rather than Mean Squared Error (MSE) because:
- MSE produces a **non-convex** loss surface for classification — gradient descent can get stuck.
- Cross-entropy is **convex** for logistic regression, guaranteeing a global minimum.

### Formula

```
L(y, ŷ) = -[y · log(ŷ) + (1 − y) · log(1 − ŷ)]
```

Over a dataset of `n` samples:

```
J(w, b) = -(1/n) Σᵢ [yᵢ · log(ŷᵢ) + (1 − yᵢ) · log(1 − ŷᵢ)]
```

### Intuition

- If `y=1` and `ŷ→1`: loss → 0 (correct, confident).
- If `y=1` and `ŷ→0`: loss → ∞ (wrong, confident → heavily penalized).
- If `y=0` and `ŷ→0`: loss → 0 (correct).
- If `y=0` and `ŷ→1`: loss → ∞ (wrong, confident).

---

## 5. Gradient Descent for Logistic Regression

The gradient of the cross-entropy loss with respect to weights is:

```
∂J/∂w = (1/n) Xᵀ (ŷ - y)
∂J/∂b = (1/n) Σ (ŷᵢ - yᵢ)
```

where `ŷ = σ(Xw + b)`.

### Weight Update Rule

```
w ← w - α · (∂J/∂w)
b ← b - α · (∂J/∂b)
```

`α` (learning rate) controls the step size. Variants:
- **Batch GD:** use all n samples per update (slow but stable).
- **Stochastic GD (SGD):** use 1 sample per update (fast, noisy).
- **Mini-batch GD:** use a batch of m samples (best of both worlds).

---

## 6. Multiclass Extension

### Softmax (Multinomial Logistic Regression)

For `C` classes, compute a score for each class and normalize with Softmax:

```
P(y=c | x) = exp(wₒᵀx) / Σⱼ exp(wⱼᵀx)
```

- Uses **categorical cross-entropy** as the loss.
- All class probabilities sum to 1.
- A generalization of sigmoid (sigmoid is a special case with C=2).

### One-vs-Rest (OvR)

- Train `C` separate binary classifiers.
- Each classifier predicts "class c vs. all others".
- Predict the class with the highest output probability.
- Simpler but probabilities do not sum to 1.

---

## 7. Regularization in Logistic Regression

Regularization adds a penalty term to the loss to prevent overfitting (large weights that memorize training noise).

### L2 Regularization (Ridge)

```
J_reg = J + (λ/2) Σ wᵢ²
```

- Shrinks all weights toward zero but **does not zero them out**.
- Controlled by `C = 1/λ` in sklearn (smaller C = stronger regularization).
- Preferred when all features are expected to contribute.

### L1 Regularization (Lasso)

```
J_reg = J + λ Σ |wᵢ|
```

- Can shrink weights **exactly to zero** → feature selection.
- Produces **sparse** weight vectors.
- Useful when many features are irrelevant.

### Elastic Net

Combines L1 and L2:
```
J_reg = J + λ₁ Σ |wᵢ| + λ₂ Σ wᵢ²
```

---

## 8. Assumptions and Limitations

| Assumption / Limitation | Detail |
|--------------------------|--------|
| Linear decision boundary | Cannot capture non-linear relationships without feature engineering |
| Feature independence | Multicollinearity inflates weight variance |
| No missing values | Must handle NaN before training |
| Large sample size | Low-sample performance can be poor |
| Classes must be (nearly) linearly separable | Otherwise accuracy suffers |
| Outputs calibrated probabilities | Better calibrated than SVM, but may still need Platt scaling |

---

## 9. Reference Links

- [Scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Wikipedia: Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
