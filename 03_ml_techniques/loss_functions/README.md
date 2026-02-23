# Loss Functions

A comprehensive guide to loss functions — what they are, why they matter, and how to choose the right one for your task.

---

## 1. What is a Loss Function?

A **loss function** (also called a cost function or objective function) is a mathematical function that measures how far a model's predictions are from the true target values. It takes predictions and ground truth as inputs and outputs a single scalar value representing the prediction error.

### Role in Machine Learning

```
Model Output (ŷ) ──┐
                   ├──→ Loss Function L(y, ŷ) ──→ Scalar Loss ──→ Optimization
True Labels (y)  ──┘
```

1. **Guides optimization**: Gradient descent minimizes the loss to improve predictions.
2. **Defines what "better" means**: Choosing the wrong loss leads the model to optimize the wrong objective.
3. **Must be differentiable**: The optimizer needs gradients `∂L/∂θ` to update parameters.

### Loss vs Metric

| | Loss | Metric |
|---|------|--------|
| **Purpose** | Guides optimization | Evaluates model quality |
| **Property** | Must be differentiable | Can be non-differentiable |
| **Examples** | Cross-entropy, MSE | Accuracy, F1, AUC |
| **Audience** | Optimizer (math) | Humans (interpretation) |

Example: Accuracy is a great metric but cannot be used as a loss because its gradient is zero almost everywhere (step function). Cross-entropy is used instead — it is differentiable and its minimum corresponds to the model predicting correct probabilities.

---

## 2. Regression Losses

### Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/N) Σᵢ (yᵢ − ŷᵢ)²
```

**Properties:**
- Smooth and differentiable everywhere.
- **Sensitive to outliers**: A single large error is squared, dominating the total loss.
- Penalizes large errors disproportionately (squared penalty).
- Unique minimum — convex for linear models.

**When to use:**
- Target values have Gaussian noise.
- You want to penalize large deviations severely.
- No significant outliers in the data.

### Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/N) Σᵢ |yᵢ − ŷᵢ|
```

**Properties:**
- **Robust to outliers**: Large errors are penalized linearly, not quadratically.
- Not differentiable at zero — requires subgradient methods.
- Gradient is constant (±1/N) regardless of error magnitude.

**When to use:**
- Data has outliers you want to ignore.
- You care equally about all error magnitudes.
- Predicting median rather than mean.

### Huber Loss (Smooth L1)

**Formula:**
```
L_δ(y, ŷ) = {
    0.5 × (y − ŷ)²              if |y − ŷ| ≤ δ
    δ × (|y − ŷ| − 0.5 × δ)    otherwise
}
```

**Properties:**
- **Combines MSE + MAE**: Quadratic (smooth) near zero, linear (robust) for large errors.
- The parameter `δ` controls the threshold between quadratic and linear regimes.
- Differentiable everywhere — no subgradient needed.

**When to use:**
- You want robustness to outliers but also smooth gradients near the minimum.
- Object detection models (e.g., bounding box regression in Faster R-CNN uses smooth L1).
- Default `δ = 1.0`.

### Comparison: MSE vs MAE vs Huber

```
Error=1: MSE=1.0,  MAE=1.0,  Huber=0.5
Error=2: MSE=4.0,  MAE=2.0,  Huber=1.5
Error=5: MSE=25.0, MAE=5.0,  Huber=4.5
Error=10: MSE=100, MAE=10.0, Huber=9.5
```

As error grows, MSE explodes while MAE and Huber grow linearly.

---

## 3. Classification Losses

### Binary Cross-Entropy (Log Loss)

**Formula:**
```
BCE = −(1/N) Σᵢ [yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ)]
```

Where `ŷᵢ ∈ (0, 1)` is the predicted probability and `yᵢ ∈ {0, 1}` is the true label.

**Probabilistic interpretation:**
Binary cross-entropy is equivalent to the **negative log-likelihood** under a Bernoulli distribution. Minimizing BCE is equivalent to **maximum likelihood estimation** of the model parameters.

**Why not use MSE for classification?**
- MSE with sigmoid outputs has vanishing gradients when predictions are confident but wrong (sigmoid saturation).
- Cross-entropy gradient `ŷ − y` does not vanish at saturation — it provides strong learning signal for wrong confident predictions.

**Numerical stability:** Always clip `ŷ` to `[ε, 1−ε]` to avoid `log(0)`.

### Categorical Cross-Entropy

**Formula (one-hot y):**
```
CCE = −(1/N) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
```

Where `ŷᵢₖ` is the softmax probability for class `k` on example `i`.

**Simplification:** Since `y` is one-hot (only one class = 1), the sum over `k` reduces to just the log probability of the true class:
```
CCE = −(1/N) Σᵢ log(ŷᵢ[true_class])
```

**Relationship to binary cross-entropy:** BCE is a special case of CCE with 2 classes.

### Hinge Loss (SVM)

**Formula (binary, y ∈ {-1, +1}):**
```
Hinge = (1/N) Σᵢ max(0, 1 − yᵢ · ŷᵢ)
```

Where `ŷᵢ` is the raw model score (not a probability).

**Properties:**
- Zero loss when the prediction is correct **and** confident (margin ≥ 1).
- Penalizes predictions that are correct but not confident enough.
- Results in **support vectors** — only examples near the decision boundary contribute to the gradient.
- Not differentiable at `yᵢ · ŷᵢ = 1` — uses subgradient.

**When to use:**
- SVMs (natural loss function).
- Max-margin learning problems.

### Focal Loss

**Formula:**
```
FL(p_t) = −α_t × (1 − p_t)^γ × log(p_t)
```

Where:
- `p_t = ŷ` if `y = 1`, else `p_t = 1 − ŷ`
- `γ ≥ 0`: focusing parameter (typically 2)
- `α_t`: class weight

**Key idea:** Focal loss down-weights easy examples (high `p_t`) so that the model focuses training on hard, misclassified examples.

When `γ = 0`: Focal loss reduces to weighted binary cross-entropy.
When `γ > 0`: The modulating factor `(1 − p_t)^γ` is small for easy examples (correct, confident), pushing the model to focus on hard negatives.

**When to use:**
- **Severe class imbalance** (e.g., one-stage object detection where background:object ≈ 1000:1).
- RetinaNet (the paper that introduced focal loss).
- Any scenario where easy negatives dominate the loss.

---

## 4. Custom Losses

When standard losses don't capture the right objective, custom losses are needed:

### Examples

| Task | Problem | Custom Loss |
|------|---------|-------------|
| Metric learning | Learn embeddings where similar items are close | Contrastive Loss, Triplet Loss |
| GAN training | Generator vs discriminator | Adversarial Loss |
| Ranking | Order items by relevance | Listwise ranking loss |
| Structured prediction | Predict sequences or trees | CTC Loss, REINFORCE |
| Multi-task learning | Multiple objectives simultaneously | Weighted sum of task losses |

### Contrastive Loss (Siamese Networks)
```
L = y × d² + (1 − y) × max(margin − d, 0)²
```
Where `d = ||f(x₁) − f(x₂)||` is the embedding distance.

### Triplet Loss
```
L = max(d(a, p) − d(a, n) + margin, 0)
```
Ensures anchor-positive distance < anchor-negative distance by at least `margin`.

---

## 5. Loss vs Metric

### Why Loss Must Be Differentiable

Gradient descent requires computing `∂L/∂θ`. If the loss function has zero gradients everywhere (like accuracy, which is a step function), the optimizer receives no signal and cannot improve the model.

### Common Mismatches

| Task | Training Loss | Evaluation Metric |
|------|--------------|-------------------|
| Binary classification | Binary cross-entropy | Accuracy, F1, AUC |
| Multi-class | Categorical cross-entropy | Accuracy, macro-F1 |
| Regression | MSE | RMSE, MAE, R² |
| Object detection | Focal loss + Smooth L1 | mAP |
| Machine translation | Cross-entropy | BLEU score |

The loss optimizes a **surrogate** of the true metric. Good surrogates are:
1. Differentiable proxies for the metric.
2. Their minimum corresponds to the metric's optimum.
3. Their gradient provides useful signal throughout training.

---

## 6. Reference Links

- [PyTorch Loss Functions Documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Wikipedia: Loss function](https://en.wikipedia.org/wiki/Loss_function)
