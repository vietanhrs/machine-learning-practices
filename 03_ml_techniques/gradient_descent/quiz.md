# Quiz: Gradient Descent

Test your understanding of gradient descent variants, learning rates, and optimizers.

---

**Q1.** What are the three main variants of gradient descent, and how do they differ in how much data they use per update?

<details>
<summary>Answer</summary>

| Variant | Data per Update | Gradient Estimate |
|---------|----------------|-------------------|
| **Batch GD (BGD)** | Entire dataset (N examples) | Exact (no noise) |
| **Stochastic GD (SGD)** | 1 random example | Very noisy |
| **Mini-Batch GD** | B examples (e.g., 32–512) | Low noise (standard) |

**BGD**: Stable, smooth convergence but slow — must process all N examples before updating.

**SGD**: Very fast updates, high noise. Can escape local minima but has high variance. In practice, the term "SGD" in PyTorch and TensorFlow refers to mini-batch SGD.

**Mini-Batch**: Best of both worlds — leverages GPU parallelism, reasonable variance, and frequent updates. This is what is actually used in practice.

</details>

---

**Q2.** What happens when the learning rate is too high? What happens when it is too low? How would you detect each problem from a training loss curve?

<details>
<summary>Answer</summary>

**Learning Rate Too High:**
- Parameter updates overshoot the minimum.
- Loss oscillates wildly, fails to converge, or diverges to NaN.
- Detection: Loss curve shows high-frequency oscillation, upward trend, or sudden NaN/Inf values.
- Extreme case: Loss increases monotonically — the optimizer is "bouncing off the walls" of the loss landscape.

**Learning Rate Too Low:**
- Updates are very small; optimization crawls.
- Training takes an impractically long time.
- Detection: Loss curve decreases very slowly, essentially flat after many epochs.
- May falsely appear converged when actually far from the optimum.

**Just Right:**
- Loss decreases smoothly and stabilizes at a low value.
- Validation loss tracks training loss (no divergence = likely not too high).

**Practical tools**: Learning rate range test (Leslie Smith) — train for a few iterations while linearly increasing LR from very small to large, plot loss vs LR, and pick the LR just before loss starts rising.

</details>

---

**Q3.** Explain what momentum does in gradient descent. How does it help with oscillations in a ravine-shaped loss landscape?

<details>
<summary>Answer</summary>

**What Momentum Does:**
Momentum maintains an exponential moving average (velocity vector `v`) of past gradients:
```
v ← β·v + (1−β)·∇L
θ ← θ − α·v
```
With `β = 0.9`, 90% of the previous velocity is retained and 10% of the current gradient is added.

**Effect in a Ravine:**
A ravine is a region where the loss surface is steep in one direction (short axis) and shallow in another (long axis, toward minimum).

Without momentum: Gradients alternate direction across the ravine → zigzag pattern → slow progress.

With momentum:
- In the long axis: Gradients consistently point in the same direction → velocity builds up → accelerated progress.
- In the short axis: Gradients alternate + and − → velocity cancels out → oscillations are dampened.

**Result**: Momentum enables faster convergence along shallow directions (toward minimum) while suppressing oscillations across steep directions. It effectively "smooths" the optimization path.

</details>

---

**Q4.** What are the default hyperparameter values for Adam, and what does each control?

<details>
<summary>Answer</summary>

Adam default hyperparameters:

| Parameter | Default Value | Controls |
|-----------|--------------|----------|
| `α` (learning rate) | `0.001` | Overall step size |
| `β₁` | `0.9` | Decay rate for first moment (momentum term — EMA of gradients) |
| `β₂` | `0.999` | Decay rate for second moment (EMA of squared gradients — RMSProp term) |
| `ε` (epsilon) | `1e-8` | Numerical stability — prevents division by zero |

**β₁ = 0.9**: The gradient estimate is a 10-step exponential moving average — smooth momentum.

**β₂ = 0.999**: The squared gradient estimate uses a very long memory (~1000 steps) — slowly adapts per-parameter learning rates.

**ε**: Added to the denominator `√v̂ + ε` to prevent division by zero when `v̂ ≈ 0`. Sometimes tuned to `1e-7` or `1e-6` for stability.

**Bias correction**: Adam divides by `(1 − β₁ᵗ)` and `(1 − β₂ᵗ)` at step `t` to correct initialization bias (since `m` and `v` start at 0).

</details>

---

**Q5.** Why is Adam generally preferred over vanilla SGD in practice, especially for NLP and Transformer-based models?

<details>
<summary>Answer</summary>

**Reasons Adam is preferred:**

1. **Per-parameter adaptive learning rates**: Adam scales learning rates individually for each parameter. Parameters with sparse gradients (e.g., embedding weights for rare words) get larger effective learning rates, while frequently updated parameters get smaller rates. This is crucial for NLP where input is sparse.

2. **Less sensitive to learning rate choice**: Adam's default `lr=1e-3` works well across many tasks. SGD requires careful tuning of both learning rate and momentum.

3. **Fast early convergence**: Adam's combined momentum + adaptive learning rates lead to rapid loss reduction in the first few epochs, reducing iteration time for research.

4. **Handles sparse gradients naturally**: In word embeddings and attention mechanisms, many parameters receive zero gradients for any given batch. AdaGrad/Adam handle sparsity well.

5. **Scale invariant to gradient magnitude**: Normalizing by the second moment makes Adam robust to very large or very small gradient scales.

**Caveats:**
- On some image classification benchmarks (CIFAR-10, ImageNet), carefully tuned SGD+Momentum+cosine annealing achieves higher final accuracy than Adam.
- Adam with weight decay requires "decoupled weight decay" (AdamW) for correct L2 regularization.
- Adam can be memory-intensive (stores m and v for every parameter).

</details>

---

**Q6.** What is the purpose of learning rate warm-up? Which model architectures particularly benefit from it?

<details>
<summary>Answer</summary>

**Purpose of Warm-Up:**

At the start of training, model parameters are initialized randomly. The gradient estimates from early mini-batches are noisy and potentially misleading. Starting with a large learning rate can cause:
- Catastrophic initial parameter updates
- Divergence before the optimizer has seen enough data to form reliable estimates
- Poor initial representations that are hard to recover from

**Warm-up** starts with a very small learning rate (often 0) and linearly (or otherwise) increases it to the target learning rate over the first `W` steps:
```
α(t) = α_target × (t / W)    for t < W
α(t) = [schedule]             for t ≥ W
```

**Benefits:**
- Allows the model to establish initial representations before large updates
- Prevents the second moment estimate `v` in Adam from being unstable at initialization
- Reduces early loss spikes

**Architectures that benefit most:**

1. **Transformers (BERT, GPT, ViT)**: The original Transformer paper used warm-up + inverse square root decay. Without warm-up, early training is often unstable.
2. **Very deep networks**: Gradients can be ill-conditioned at initialization.
3. **Large batch training**: With large batches, a higher target LR is needed; warm-up helps reach it safely.
4. **Transfer learning / fine-tuning**: Prevents aggressive updates to pretrained weights in early steps.

</details>

---

**Q7.** A colleague's RNN model produces NaN losses after a few training steps. What is the most likely cause, and how would you fix it?

<details>
<summary>Answer</summary>

**Most Likely Cause: Exploding Gradients**

In RNNs, gradients are backpropagated through many time steps (BPTT — backpropagation through time). The gradient involves a product of many Jacobian matrices:

```
∂L/∂θ ≈ ∏ᵢ (∂hᵢ/∂hᵢ₋₁)  ×  ∂L/∂hₜ
```

If these Jacobians have eigenvalues > 1, the gradient norm grows exponentially with the sequence length → eventually becomes `Inf` or `NaN`.

**Fix: Gradient Clipping**

```python
# PyTorch example
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip by norm
optimizer.step()
```

**Additional fixes:**
1. **Reduce learning rate**: Smaller steps reduce the chance of NaN.
2. **Use LSTM/GRU instead of vanilla RNN**: Gating mechanisms mitigate gradient explosion.
3. **Gradient clipping by value**: Less common, but can also work.
4. **Check data for NaN/Inf values**: Bad input data can cause NaN.
5. **Reduce sequence length**: Shorter BPTT windows reduce gradient depth.
6. **Add gradient monitoring**: `torch.nn.utils.clip_grad_norm_` returns the gradient norm before clipping — monitor this to detect explosions early.

</details>

---

**Q8.** What is the key difference between a saddle point and a local minimum in a deep learning loss landscape? Why does this matter for gradient descent?

<details>
<summary>Answer</summary>

**Saddle Point:**
- A point where the gradient is zero (`∇L = 0`) but which is neither a local minimum nor a local maximum.
- In some directions, it is a minimum (loss increases); in other directions, it is a maximum (loss decreases if you move away).
- Named after a horse saddle: minimum along one axis, maximum along the perpendicular axis.

**Local Minimum:**
- A point where `∇L = 0` and the loss is lower than all nearby points.
- The Hessian (second derivative matrix) is positive definite.

**Why it Matters:**

Early deep learning theory worried about local minima trapping gradient descent. However, research by Dauphin et al. (2014) showed that:

1. **Saddle points are far more common than local minima** in high-dimensional spaces.
2. **Local minima in deep networks are usually not that bad** — they tend to be approximately as good as the global minimum when the network is sufficiently overparameterized.
3. **SGD noise helps escape saddle points**: Noisy gradient estimates help the optimizer move away from saddle points where the exact gradient is zero.
4. **Saddle points slow convergence**: Near a saddle point, gradients are small in the "escape" direction — the optimizer crawls.

**Adam and momentum** help escape saddle points by using gradient history (velocity) to maintain movement direction even when instantaneous gradients are near zero.

</details>

---

**Q9.** Compare first-order and second-order optimization methods. Why are second-order methods rarely used in deep learning despite being theoretically superior?

<details>
<summary>Answer</summary>

**First-Order Methods** (gradient descent family):
- Use only `∇L(θ)` — the gradient.
- Approximate the loss as linear at each step.
- Cost per step: O(N·P) where N = data size, P = parameters.
- Examples: SGD, Adam, RMSProp.

**Second-Order Methods** (Newton's method, natural gradient):
- Use `∇L(θ)` and `∇²L(θ)` — the Hessian (matrix of second derivatives).
- Approximate the loss as quadratic at each step — much more accurate.
- Update: `θ ← θ − H⁻¹ · ∇L` (multiply gradient by inverse Hessian).
- Examples: L-BFGS, Natural Gradient, K-FAC.

**Theoretical advantage**: Second-order methods can take much larger steps (scale with curvature), converge in far fewer iterations on convex problems, and are invariant to parameter reparameterization.

**Why not used in deep learning:**

1. **Memory cost**: The Hessian for a model with P parameters is P×P. For ResNet-50 (25M params), the Hessian has 6.25×10¹⁴ entries — completely infeasible.

2. **Computation cost**: Computing and inverting the Hessian costs O(P³) — astronomically expensive.

3. **Non-convexity**: Second-order convergence guarantees assume convexity. Deep learning loss landscapes are highly non-convex.

4. **Mini-batch noise**: Stochastic Hessian estimates are very noisy, reducing their advantage.

**Practical approximations**: K-FAC, Shampoo, and natural gradient methods approximate the Hessian structure (e.g., Kronecker factorization) and are used in some research settings, but rarely in standard training.

</details>

---

**Q10.** In AdaGrad, the learning rate for each parameter is divided by the square root of the sum of all squared gradients. What problem does this cause for long training runs, and how does RMSProp fix it?

<details>
<summary>Answer</summary>

**AdaGrad's Problem — Monotonically Decaying Learning Rate:**

```
G_t = G_{t-1} + (∇L_t)²    (accumulate squared gradients forever)
θ ← θ − (α / √(G_t + ε)) · ∇L_t
```

Since `G_t` only increases over time (each term adds a positive squared gradient), the effective learning rate `α / √G_t` **monotonically decreases toward zero**. After enough steps, the learning rate becomes so small that learning effectively stops — this is especially problematic in non-convex deep learning where we need to keep adapting.

AdaGrad was designed for convex sparse problems (like ad click prediction) where you want to shrink the LR as you approach the optimum. For deep learning, it's too aggressive.

**RMSProp's Fix — Exponential Moving Average:**

```
E[g²]_t = β · E[g²]_{t-1} + (1−β) · (∇L_t)²    (exponential moving average)
θ ← θ − (α / √(E[g²]_t + ε)) · ∇L_t
```

With `β = 0.9`, only the last ~10 gradient steps significantly influence the denominator. Old gradients are "forgotten" exponentially. This means:
- The effective learning rate doesn't shrink to zero.
- The optimizer keeps adapting to recent gradient magnitudes.
- Suitable for non-stationary and non-convex objectives.

</details>

---

**Q11.** What is the difference between weight decay and L2 regularization? Are they equivalent?

<details>
<summary>Answer</summary>

**L2 Regularization:**
Adds an L2 penalty to the loss function: `L_reg = L + (λ/2) · ||θ||²`

The gradient becomes: `∇L_reg = ∇L + λ·θ`

SGD update with L2 regularization:
```
θ ← θ − α·(∇L + λ·θ) = θ·(1 − α·λ) − α·∇L
```

**Weight Decay:**
Directly adds a decay term to the parameter update:
```
θ ← θ·(1 − λ_wd) − α·∇L
```

**Are they equivalent?**

- **For SGD**: Yes, equivalent with `λ_wd = α·λ`. The two formulations produce identical updates.

- **For Adam (and other adaptive optimizers): NO — they differ.**

Adam normalizes gradients by the second moment estimate. When L2 regularization is added to the loss, the regularization gradient `λ·θ` is also normalized by the second moment:

```
Effective regularization ≈ λ·θ / (√v̂ + ε)
```

This means frequently updated parameters (large `v̂`) get less regularization, and rarely updated parameters get more — this is undesirable. The intended uniform L2 penalty is distorted.

**AdamW** (Loshchilov & Hutter, 2019) fixes this by implementing "decoupled weight decay" — applying weight decay directly to parameters without normalizing:
```
θ ← θ − α · m̂/(√v̂ + ε) − λ_wd · θ
```
AdamW is now standard for Transformers (BERT, GPT use AdamW).

</details>

---

**Q12.** Describe the cosine annealing learning rate schedule. What makes it preferable to step decay for many deep learning tasks?

<details>
<summary>Answer</summary>

**Cosine Annealing:**

```
α(t) = α_min + 0.5 · (α₀ − α_min) · (1 + cos(π · t / T_max))
```

- At `t = 0`: `α = α₀` (maximum learning rate)
- At `t = T_max/2`: `α = (α₀ + α_min)/2`
- At `t = T_max`: `α = α_min` (minimum learning rate)

The schedule follows a cosine curve, smoothly decreasing from `α₀` to `α_min`.

**Variant — SGDR (Cosine Annealing with Warm Restarts):**
After reaching `α_min`, the learning rate is reset to `α₀` and the cycle repeats (often with increasing cycle length). Each restart allows the optimizer to explore different loss landscape regions.

**Why preferable to step decay:**

1. **Smooth transitions**: Step decay abruptly halves the learning rate — can cause training instability or loss spikes at step boundaries. Cosine annealing changes smoothly.

2. **No hyperparameter tuning for step timing**: Step decay requires choosing when to drop the learning rate (e.g., epoch 30, 60, 90). Cosine annealing only requires `T_max`.

3. **Warm restarts enable exploration**: SGDR periodically resets the LR, allowing the model to escape local minima and explore new regions — often finds better solutions.

4. **Better final accuracy**: Empirically, cosine annealing achieves higher accuracy on image classification benchmarks (ResNet on CIFAR/ImageNet) compared to step decay.

5. **High initial LR → fast convergence, low final LR → fine-grained optimization**: The shape naturally provides both coarse and fine optimization phases.

</details>

---

**Q13.** A training run with Adam shows that the training loss is 0.05 but the validation loss is 0.85. The team decides to switch from Adam to SGD+Momentum. Why might this help? What else could be done?

<details>
<summary>Answer</summary>

**Why SGD+Momentum might help:**

This scenario describes severe overfitting (training loss << validation loss). The switch from Adam to SGD+Momentum can help because:

1. **Adam can overfit more easily**: Adam's adaptive learning rates allow it to exploit the training data more aggressively. Each parameter finds its optimal learning rate for the training set, which can lead to memorization.

2. **SGD+Momentum finds flatter minima**: Research (e.g., Wilson et al., 2017) showed that adaptive optimizers tend to converge to sharper minima that generalize worse. SGD+Momentum with a decayed learning rate tends to find flatter minima that generalize better.

3. **Implicit regularization**: The noisier gradient estimates of SGD act as implicit regularization, preventing the model from perfectly memorizing the training set.

**Other solutions:**

1. **More regularization**: Add L2/weight decay, increase dropout rate, add batch normalization.
2. **More training data**: Or data augmentation.
3. **Reduce model capacity**: Fewer layers, smaller hidden dimensions.
4. **Early stopping**: Stop training when validation loss starts increasing.
5. **Learning rate tuning**: If Adam's LR is too high, it can memorize training data faster.
6. **AdamW**: Use decoupled weight decay instead of L2 regularization inside loss.

</details>

---

**Q14.** What is gradient clipping "by norm" vs "by value"? Why is clipping by norm generally preferred?

<details>
<summary>Answer</summary>

**Clip by Value:**
```
gᵢ = clip(gᵢ, -c, c)    for each gradient element gᵢ
```
Sets any individual gradient component that exceeds ±c to ±c.

**Clip by Norm:**
```
if ||g|| > c:
    g ← g × (c / ||g||)
```
If the global gradient norm `||g||` exceeds threshold `c`, scales the entire gradient vector down proportionally so its norm equals exactly `c`.

**Why clip by norm is preferred:**

1. **Preserves gradient direction**: Clip by norm scales the gradient uniformly — the direction (relative magnitudes across parameters) is preserved. The optimizer still moves in the correct direction, just with a smaller step.

2. **Clip by value distorts direction**: Setting individual components to ±c changes the relative ratios between gradient components — the resulting gradient vector points in a completely different direction. This is an unintended change in optimization direction.

3. **Interpretable threshold**: The gradient norm threshold `c` has a clear meaning (the maximum allowed step magnitude in parameter space). Clip by value threshold `c` is harder to interpret since it applies to each component independently.

4. **Stable behavior**: Clip by norm guarantees the update step never exceeds `α × c` in parameter space. Clip by value can still produce large updates if many components are at the clipping threshold.

**Typical values**: `max_norm = 1.0` for RNNs, `max_norm = 5.0` for some architectures. PyTorch provides `torch.nn.utils.clip_grad_norm_()`.

</details>

---

*End of Quiz — 14 Questions*
