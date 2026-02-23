# Gradient Descent

A comprehensive guide to gradient descent optimization — the backbone of training nearly all machine learning models.

---

## 1. Gradient Descent Intuition

### The Loss Landscape

When a model is trained, a **loss function** `L(θ)` measures how wrong the model's predictions are. This function is defined over the model's parameter space `θ` (all weights and biases). The loss landscape is a high-dimensional surface where:

- **Low points**: Good parameter configurations (low error)
- **High points**: Bad parameter configurations (high error)

### The Steepest Descent Idea

Gradient descent finds a local minimum of `L(θ)` by iteratively stepping in the direction opposite to the gradient:

```
θ ← θ - α · ∇L(θ)
```

Where:
- `θ` = current parameters
- `α` = learning rate (step size)
- `∇L(θ)` = gradient of loss with respect to parameters
- The **negative gradient** points in the direction of steepest descent

The analogy: imagine standing on a hilly landscape in fog, taking small steps downhill by feeling which direction slopes most steeply downward.

### Why Gradients Work

The gradient `∇L(θ)` tells us the direction and magnitude of steepest **increase** in loss. By moving in the **opposite** direction, we make the loss decrease as quickly as possible at the current point. This is a local, first-order approximation — it works well when the step size is small.

---

## 2. Batch Gradient Descent (BGD)

### Algorithm

Compute the gradient using the **entire training dataset** at each step:

```
∇L(θ) = (1/N) Σᵢ ∇L(θ; xᵢ, yᵢ)    (sum over all N training examples)
θ ← θ - α · ∇L(θ)
```

### Characteristics

| Property | Description |
|----------|-------------|
| Gradient estimate | Exact (no noise) |
| Update frequency | Once per epoch |
| Convergence | Smooth, stable |
| Speed per epoch | Slow (must process all N examples) |
| Memory | Must fit all data in memory |

### When to Use

- Small datasets that fit in memory
- When precise gradient estimates are critical
- Convex problems where guaranteed convergence matters

### Limitations

- Too slow for large datasets (millions of examples)
- Cannot benefit from hardware parallelism (GPU mini-batches)
- Cannot escape sharp local minima or saddle points due to noiseless updates

---

## 3. Stochastic Gradient Descent (SGD)

### Algorithm

Compute the gradient using a **single randomly chosen training example** at each step:

```
∇L(θ) ≈ ∇L(θ; xᵢ, yᵢ)    (one random example)
θ ← θ - α · ∇L(θ; xᵢ, yᵢ)
```

### Characteristics

| Property | Description |
|----------|-------------|
| Gradient estimate | Very noisy (high variance) |
| Update frequency | N times per epoch (once per example) |
| Convergence | Noisy, oscillating loss curve |
| Speed per update | Very fast |
| Memory | Only one example at a time |

### Why Noise Can Help

The noisy gradient estimates act as implicit regularization:
- Can escape sharp local minima (noise kicks the optimizer out)
- Can navigate saddle points more effectively than BGD
- Often converges to flatter minima that generalize better

### Limitations

- High variance makes convergence slow and unreliable
- Hard to fully leverage GPU parallelism (processes one example)
- Learning rate must be carefully decayed to converge

---

## 4. Mini-Batch Gradient Descent

### Algorithm

Compute the gradient using a **small random batch** of B examples:

```
∇L(θ) ≈ (1/B) Σᵢ∈batch ∇L(θ; xᵢ, yᵢ)
θ ← θ - α · ∇L(θ)
```

Typical batch sizes: 32, 64, 128, 256, 512.

### Characteristics

| Property | Description |
|----------|-------------|
| Gradient estimate | Low variance (better than SGD) |
| Update frequency | N/B times per epoch |
| Convergence | Reasonably smooth |
| Speed | Fast (GPU parallelism over B examples) |
| Memory | Only B examples at a time |

### Why Mini-Batch is Standard Practice

1. **GPU efficiency**: GPUs are designed for parallel matrix operations — processing B examples simultaneously is nearly as fast as one.
2. **Balanced variance**: Less noise than SGD, more updates than BGD.
3. **Better generalization**: Some noise from mini-batches helps regularization.
4. **Practical**: Works for arbitrarily large datasets (stream batches).

In practice, "SGD" often refers to mini-batch gradient descent in the deep learning literature.

---

## 5. Learning Rate

### Effect of Learning Rate

The learning rate `α` controls the size of each parameter update step:

```
Too high (α too large):
    Loss oscillates or diverges — steps overshoot the minimum

Too low (α too small):
    Convergence is extremely slow — training takes forever

Just right:
    Smooth, efficient convergence toward a minimum
```

### Learning Rate Schedules

Instead of a fixed learning rate, schedules adjust `α` during training:

#### Step Decay
```
α(epoch) = α₀ × drop^(epoch / epochs_drop)
```
Reduces learning rate by a fixed factor every N epochs.

#### Exponential Decay
```
α(epoch) = α₀ × e^(−decay_rate × epoch)
```
Continuously decays learning rate.

#### Cosine Annealing
```
α(epoch) = α_min + 0.5 × (α₀ − α_min) × (1 + cos(π × epoch / T_max))
```
Smoothly cycles learning rate from `α₀` down to `α_min`. Often combined with restarts (SGDR).

#### Warm-Up
Start with a very small learning rate, linearly increase to the target `α₀` over the first few epochs, then decay.

```
α(epoch) = α₀ × (epoch / warmup_epochs)    if epoch < warmup_epochs
           [then apply schedule]             otherwise
```

Warm-up prevents instability in the early training stages, especially important for Transformers and Adam optimizer.

---

## 6. Momentum

### Problem with Vanilla SGD

Standard SGD treats each gradient update independently, leading to:
- Slow progress along flat directions
- Oscillations in steep directions
- Difficulty escaping ravines in the loss surface

### Momentum Solution

Momentum maintains an **exponential moving average (EMA) of gradients** (velocity vector `v`):

```
v ← β × v + (1 − β) × ∇L(θ)
θ ← θ − α × v
```

Where `β` is the momentum coefficient (typically 0.9).

### Effect

- **Accelerates**: When gradients consistently point in the same direction, velocity builds up — faster convergence.
- **Dampens oscillations**: When gradients alternate direction (ravines), velocity cancels oscillation.
- **Smooths updates**: Recent gradients have more influence than older ones.

### Nesterov Momentum

A variant that computes the gradient at the "looked-ahead" position:

```
v ← β × v + ∇L(θ − α × β × v)
θ ← θ − α × v
```

Often converges faster than standard momentum.

---

## 7. Adaptive Optimizers

Adaptive optimizers adjust the learning rate **per parameter** based on historical gradient information.

### AdaGrad

**Accumulate sum of squared gradients:**

```
G ← G + (∇L)²            (element-wise, accumulated over all steps)
θ ← θ − (α / √(G + ε)) × ∇L
```

- Parameters with large historical gradients → smaller effective learning rate
- Parameters with small historical gradients → larger effective learning rate
- **Problem**: `G` grows monotonically → learning rate eventually shrinks to zero, stopping learning

### RMSProp

**Exponential moving average of squared gradients (fixes AdaGrad's decay problem):**

```
E[g²] ← β × E[g²] + (1 − β) × (∇L)²
θ ← θ − (α / √(E[g²] + ε)) × ∇L
```

- `β` ≈ 0.9 controls the forgetting rate
- Older gradients are discounted, preventing the learning rate from shrinking to zero
- Effective for non-stationary objectives (RNNs, noisy problems)

### Adam (Adaptive Moment Estimation)

**Combines momentum (first moment) + RMSProp (second moment):**

```
m ← β₁ × m + (1 − β₁) × ∇L          (first moment — mean of gradients)
v ← β₂ × v + (1 − β₂) × (∇L)²       (second moment — variance of gradients)

m̂ ← m / (1 − β₁ᵗ)                   (bias correction for first moment)
v̂ ← v / (1 − β₂ᵗ)                   (bias correction for second moment)

θ ← θ − α × m̂ / (√v̂ + ε)
```

**Default hyperparameters:**
- `β₁ = 0.9` (momentum decay)
- `β₂ = 0.999` (RMSProp decay)
- `ε = 1e-8` (numerical stability)
- `α = 0.001` (typical default learning rate)

**Why bias correction?** At step `t=1`, `m` and `v` are initialized at zero, so early estimates are biased toward zero. Dividing by `(1 − βᵗ)` corrects this.

**Why Adam is preferred in practice:**
- Works well across diverse architectures (CNNs, RNNs, Transformers)
- Less sensitive to learning rate than SGD
- Handles sparse gradients well (NLP tasks)
- Converges quickly with default hyperparameters

**Caution**: Adam can generalize slightly worse than tuned SGD+Momentum in some image classification tasks. SGD+Momentum often achieves better final accuracy when carefully tuned.

---

## 8. Gradient Clipping

### Problem: Exploding Gradients

In recurrent neural networks (RNNs) and very deep networks, gradients can **grow exponentially** through many layers (the "exploding gradient problem"), causing:
- NaN values in parameters
- Wildly unstable training
- Parameter values going to infinity

### Solution: Gradient Clipping

**Clip by value**: Set any gradient component exceeding threshold `c` to `c`:
```
g ← max(min(g, c), -c)    (element-wise)
```

**Clip by norm** (more common): Scale the entire gradient vector if its norm exceeds threshold `c`:
```
if ||g|| > c:
    g ← g × (c / ||g||)
```

Clipping by norm preserves the direction of the gradient while limiting its magnitude — generally preferred over clip by value.

### When to Use

- RNNs and LSTMs (almost always used)
- Very deep networks
- Transformers with large learning rates
- When you observe NaN losses or loss spikes during training

---

## 9. Reference Links

- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/)
- [PyTorch Optimization Documentation](https://pytorch.org/docs/stable/optim.html)
