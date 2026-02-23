# Regularization

A comprehensive guide to regularization techniques — methods that prevent overfitting and improve generalization in machine learning models.

---

## 1. What is Regularization?

**Regularization** is any technique that modifies a learning algorithm to prevent overfitting by penalizing model complexity. The core idea is to add a bias away from complex models even at the cost of slightly higher training error, in exchange for better generalization to unseen data.

### The Overfitting Problem

```
Without regularization (high capacity model):
    Training loss: very low (memorizes training data)
    Validation loss: high (fails to generalize)
    
With regularization:
    Training loss: slightly higher (some constraint on fitting)
    Validation loss: lower (better generalization)
```

### Regularized Loss Function

Most regularization methods add a **penalty term** to the loss:

```
L_regularized(θ) = L(θ) + λ × Ω(θ)
```

Where:
- `L(θ)`: Original loss (e.g., cross-entropy, MSE)
- `λ`: Regularization strength (hyperparameter)
- `Ω(θ)`: Complexity penalty (depends on the method)

---

## 2. L2 Regularization / Ridge Regression

### Formula

```
L₂ penalty: Ω(w) = ||w||² = Σᵢ wᵢ²
L_ridge(θ) = L(θ) + λ × ||w||²
```

Note: The bias term `b` is usually excluded from regularization.

### Gradient

```
∂L_ridge/∂w = ∂L/∂w + 2λ × w
```

### Effect on Weights

The L2 penalty shrinks all weights toward zero, proportionally to their magnitude:

```
Weight update: w ← w - α × (∂L/∂w + 2λ × w)
             = w × (1 - 2αλ) - α × ∂L/∂w
```

The factor `(1 - 2αλ)` is the **weight decay** factor — weights are multiplied by a value slightly less than 1 each step.

### Properties

- **All weights shrink** but rarely reach exactly zero.
- Prefers models with many small weights over models with few large weights.
- Equivalent to a Gaussian prior on weights (Bayesian interpretation: MAP estimation).
- Convex penalty — computationally friendly.

### When to Use

- Preventing overfitting in linear/logistic regression.
- Neural networks (as "weight decay" in optimizer settings).
- When you believe all features are relevant but should have moderate influence.

---

## 3. L1 Regularization / Lasso

### Formula

```
L₁ penalty: Ω(w) = ||w||₁ = Σᵢ |wᵢ|
L_lasso(θ) = L(θ) + λ × ||w||₁
```

### Gradient (Subgradient)

```
∂L_lasso/∂wᵢ = ∂L/∂wᵢ + λ × sign(wᵢ)
```

Where `sign(0) = 0` by convention (or set to any value in [-1, 1]).

### Why L1 Promotes Sparsity (Geometric Explanation)

The L1 ball (constraint region in parameter space) is a **diamond** with corners at the axes. When the unconstrained loss contours touch the L1 ball, they are most likely to touch at a **corner** — which corresponds to a solution where some weights are exactly zero.

In contrast, the L2 ball is a sphere with no corners — the optimal solution is on the curved surface where no weight is exactly zero.

### Properties

- **Produces sparse solutions**: Many weights become exactly zero → automatic feature selection.
- Robust to irrelevant features — irrelevant features are zeroed out.
- Non-differentiable at zero — requires proximal operators or subgradient methods.
- Equivalent to a Laplace prior on weights.

### When to Use

- High-dimensional data with many potentially irrelevant features.
- When interpretability is important (sparse models are more interpretable).
- Compressed sensing / signal recovery.

---

## 4. Elastic Net

### Formula

```
L_elastic(θ) = L(θ) + λ × [α × ||w||₁ + (1 - α) × ||w||²]
```

Where `α ∈ [0, 1]` controls the L1/L2 mix:
- `α = 1`: Pure Lasso
- `α = 0`: Pure Ridge
- `α = 0.5`: Equal mix

### Properties

- Combines sparsity (L1) with weight shrinkage (L2).
- Handles correlated features better than pure Lasso (Lasso arbitrarily picks one from a group; Elastic Net can include all with reduced coefficients).
- More stable than pure Lasso when features are highly correlated.

### When to Use

- High-dimensional datasets where some features are correlated.
- When you want sparsity but also need to handle feature groups.

---

## 5. Dropout

### Mechanism

During training, randomly set each neuron's output to zero with probability `p` (the dropout rate). The remaining neurons are scaled by `1/(1-p)` to maintain expected activation magnitude.

```
For each training forward pass:
    mask = Bernoulli(1-p)  # 1 with prob (1-p), 0 with prob p
    output = activation × mask × [1/(1-p)]   # inverted dropout
```

During inference (evaluation), dropout is disabled — all neurons are active.

### Effect on Ensemble

Dropout is equivalent to training an ensemble of `2^N` different neural networks (where N is the number of neurons), sharing weights. Each training step trains a different "thinned" network. At test time, using the full network with scaled weights approximates averaging the ensemble.

### Dropout Rate Trade-off

| Dropout Rate `p` | Effect |
|-----------------|--------|
| p = 0 | No regularization — standard training |
| p = 0.2-0.5 | Typical for fully connected layers |
| p = 0.5 | Original paper recommendation |
| p > 0.7 | Too aggressive — training becomes very slow |

### Important: train() vs eval() Mode

```python
model.train()  # Enables dropout (training phase)
model.eval()   # Disables dropout (inference phase)
```

Forgetting to call `model.eval()` during inference is a common bug — predictions will be noisy and inconsistent.

### When to Use

- Dense (fully connected) layers in neural networks.
- Applied after activation functions.
- Not commonly applied to convolutional layers (spatial dropout variants exist).
- Very effective for large models prone to overfitting.

---

## 6. Batch Normalization

### Mechanism

After each layer (or before activation), normalize the layer's inputs to have zero mean and unit variance, then apply learnable scale (γ) and shift (β) parameters:

```
Batch statistics (during training):
    μ_B = mean(x_B)           (batch mean)
    σ²_B = var(x_B)           (batch variance)

Normalized:
    x̂_i = (x_i - μ_B) / √(σ²_B + ε)

Output:
    y_i = γ × x̂_i + β        (learnable scale and shift)
```

During inference, use **running statistics** (exponential moving average of μ and σ² computed during training).

### Benefits

1. **Reduces internal covariate shift**: Stabilizes the distribution of activations between layers, allowing higher learning rates.
2. **Implicit regularization**: The normalization introduces noise (since batch statistics are noisy estimates), acting as a regularizer.
3. **Reduces sensitivity to initialization**: Networks with BN are less sensitive to weight initialization.
4. **Allows higher learning rates**: Gradients flow more easily through normalized layers.

### Batch Norm vs Dropout

| | Batch Norm | Dropout |
|---|-----------|---------|
| Primary purpose | Training stability + regularization | Regularization |
| Position | Before/after activation | After activation |
| Parameters | Learnable γ, β | p (fixed) |
| train/eval difference | Uses batch vs running stats | Active vs inactive |
| Common use | CNNs, deep networks | Fully connected layers |

---

## 7. Early Stopping

### Mechanism

Monitor validation loss during training. Stop when validation loss starts increasing (or stops improving) for a sustained number of epochs.

```
Training curve:
    Epoch 1-30:  Train loss ↓, Val loss ↓    (healthy learning)
    Epoch 30-50: Train loss ↓, Val loss →    (plateau, near optimum)
    Epoch 50+:   Train loss ↓, Val loss ↑    (overfitting — STOP!)
```

### Algorithm

```python
best_val_loss = infinity
patience_counter = 0

for epoch in training:
    val_loss = evaluate(model, val_data)
    
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        save_best_weights(model)
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        restore_best_weights(model)
        break
```

### Hyperparameters

- **patience**: How many epochs to wait for improvement before stopping (typically 5-20).
- **min_delta**: Minimum improvement to count as progress (typically 1e-4).
- **restore_best_weights**: Whether to revert to the best checkpoint when stopping.

### Why Early Stopping Works

It implicitly limits the "effective capacity" of the model by stopping before weights have time to memorize the training data.

---

## 8. Data Augmentation

### Concept

Artificially expand the training dataset by creating modified versions of existing examples. Each modification preserves the semantic label while changing the raw input.

### Examples by Modality

| Modality | Augmentation Techniques |
|----------|------------------------|
| Images | Random crop, flip, rotation, color jitter, cutout, mixup |
| Text | Back-translation, synonym replacement, random deletion |
| Audio | Time stretching, pitch shifting, noise injection |
| Tabular | SMOTE (for imbalanced classes), Gaussian noise on features |

### Why it Regularizes

- The model sees more diverse examples → harder to memorize any specific pattern.
- Encodes **inductive biases** (e.g., image classifiers should be translation-invariant → use random crops).
- Implicitly augments the dataset distribution to be more representative of the true data distribution.
- Often the most effective regularization for vision tasks.

---

## 9. Weight Decay vs L2 Regularization

### For SGD: Equivalent

```
SGD with L2 loss:        θ ← θ - α(∇L + 2λw) = θ(1 - 2αλ) - α∇L
SGD with weight decay:   θ ← θ(1 - λ_wd) - α∇L
```

Setting `λ_wd = 2αλ` makes them identical for SGD.

### For Adam: NOT Equivalent

Adam normalizes gradients by the second moment `√v̂`. With L2 in the loss:
```
The L2 gradient (2λw) is also normalized by √v̂
```
This means frequently-updated parameters get less regularization (their `v̂` is large).

**AdamW** (Loshchilov & Hutter, 2019) fixes this by applying weight decay directly to parameters, bypassing the adaptive scaling:
```
θ ← θ - α × m̂/(√v̂ + ε) - λ_wd × θ    (decoupled)
```

**Practical advice**: Always use AdamW (not Adam + L2 loss) for Transformer models and whenever you want proper regularization with Adam.

---

## 10. Reference Links

- [Wikipedia: Regularization (mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))
- [PyTorch Dropout Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
